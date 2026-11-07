"""Focused lease heartbeat and stale-recovery regressions (no live providers)."""

from __future__ import annotations

import multiprocessing
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base
from api.explanation_preparation import (
    ExplanationPreparationManager,
    PreparationJob,
    STAGE_EXPLAINING,
    STAGE_EXTRACTING,
    STAGE_FAILED,
    STAGE_TTS,
    _Lease,
)
from api.explanation_service import PreparationCancelledError
from api.models import Explanation, ExplanationPreparation, Intervention, Video


VIDEO_ID = "LeaseTest01"
INTERVENTION_ID = "lease-point"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _recover_stale_in_process(db_path, data_dir, start_event, result_queue):
    """Spawn-safe worker proving SQLite arbitrates recovery across processes."""
    engine = create_engine(
        f"sqlite:///{Path(db_path).as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    manager = ExplanationPreparationManager(
        session_factory=Session,
        orchestrator_factory=lambda: object(),
        data_dir=data_dir,
        generator=lambda **_kwargs: None,
        lease_timeout_seconds=0.2,
        maintenance_interval_seconds=0.03,
    )
    start_event.wait(5.0)
    try:
        result_queue.put(("ok", manager._recover_stale_preparations()))
    except Exception as exc:  # pragma: no cover - returned to the parent process
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        engine.dispose()


class ExplanationLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.db_path = self.root / "leases.sqlite"
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(Video(id=VIDEO_ID, title="Lease test", source_url="https://example.test"))
            db.add(Intervention(
                id=INTERVENTION_ID,
                video_id=VIDEO_ID,
                timestamp=1.0,
                frame_path="frames/1.jpg",
                content_type="diagram",
            ))
            db.commit()
        self.managers = []

    def manager(self, *, start=True, **kwargs):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
            lease_timeout_seconds=kwargs.pop("lease_timeout_seconds", 0.3),
            maintenance_interval_seconds=kwargs.pop(
                "maintenance_interval_seconds", 0.03
            ),
            **kwargs,
        )
        if start:
            manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def seed_generating(self, *, owner="foreign-owner", attempts=1, age=0.0):
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_ID,
                intervention_id=INTERVENTION_ID,
                output_mode="explanatory",
                status="generating",
                stage=STAGE_EXTRACTING,
                attempt_count=attempts,
                lease_owner=owner,
                lease_started_at=_utcnow() - timedelta(seconds=age),
            ))
            db.commit()

    def state(self):
        with self.Session() as db:
            return db.query(ExplanationPreparation).one()

    def test_fresh_generating_lease_survives_two_manager_startup_unchanged(self):
        self.seed_generating(owner="live-owner")
        calls = []
        first = self.manager(
            generator=lambda **kwargs: calls.append(("first", kwargs)) or None,
            lease_timeout_seconds=2.0,
            maintenance_interval_seconds=0.05,
        )
        second = self.manager(
            generator=lambda **kwargs: calls.append(("second", kwargs)) or None,
            lease_timeout_seconds=2.0,
            maintenance_interval_seconds=0.05,
        )
        time.sleep(0.15)

        state = self.state()
        self.assertEqual(state.status, "generating")
        self.assertEqual(state.lease_owner, "live-owner")
        self.assertEqual(state.attempt_count, 1)
        self.assertEqual(calls, [])
        self.assertTrue(first._maintenance_thread.is_alive())
        self.assertTrue(second._maintenance_thread.is_alive())

    def test_stale_lease_is_recovered_once_and_attempt_increments_once(self):
        self.seed_generating(attempts=1, age=2.0)
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def blocked(**_kwargs):
            calls.append("called")
            entered.set()
            release.wait(2.0)
            raise RuntimeError("controlled failure")

        self.addCleanup(release.set)
        self.manager(generator=blocked)
        self.manager(generator=blocked)
        self.assertTrue(entered.wait(2.0))
        time.sleep(0.1)

        state = self.state()
        self.assertEqual(calls, ["called"])
        self.assertEqual(state.status, "generating")
        self.assertEqual(state.attempt_count, 2)
        release.set()
        self.assertTrue(wait_until(lambda: self.state().status == "failed"))

    def test_previously_fresh_foreign_lease_recovers_after_expiry_without_request(self):
        self.seed_generating(attempts=1)
        entered = threading.Event()

        def recovered(**_kwargs):
            entered.set()
            raise RuntimeError("controlled recovered attempt")

        self.manager(
            generator=recovered,
            lease_timeout_seconds=0.2,
            maintenance_interval_seconds=0.03,
        )
        self.assertTrue(entered.wait(2.0))
        self.assertTrue(wait_until(lambda: self.state().status == "failed"))
        self.assertEqual(self.state().attempt_count, 2)

    def _assert_stage_heartbeat(self, stage):
        gate = threading.Event()
        entered = threading.Event()

        def evidence(**_kwargs):
            if stage == STAGE_EXTRACTING:
                entered.set()
                gate.wait(2.0)
            return None

        def generate(**kwargs):
            if stage == STAGE_TTS:
                kwargs["on_stage"](STAGE_TTS)
            if stage in {STAGE_EXPLAINING, STAGE_TTS}:
                entered.set()
                gate.wait(2.0)
            raise RuntimeError("controlled stage completion")

        self.addCleanup(gate.set)
        manager = self.manager(
            evidence_preparer=evidence,
            generator=generate,
            lease_timeout_seconds=0.18,
            maintenance_interval_seconds=0.03,
        )
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(entered.wait(2.0))
        initial = self.state().lease_started_at
        time.sleep(0.3)
        current = self.state()
        self.assertEqual(current.status, "generating")
        self.assertEqual(current.stage, stage)
        self.assertEqual(current.attempt_count, 1)
        self.assertGreater(current.lease_started_at, initial)
        gate.set()
        self.assertTrue(wait_until(lambda: self.state().status == "failed"))
        self.assertTrue(wait_until(lambda: not manager._tracked_leases))

    def test_vlm_stage_heartbeat_preserves_ownership(self):
        self._assert_stage_heartbeat(STAGE_EXTRACTING)

    def test_mistral_stage_heartbeat_preserves_ownership(self):
        self._assert_stage_heartbeat(STAGE_EXPLAINING)

    def test_tts_stage_heartbeat_preserves_ownership(self):
        self._assert_stage_heartbeat(STAGE_TTS)

    def test_heartbeat_cannot_renew_cancelled_or_superseded_owner(self):
        gate = threading.Event()
        entered = threading.Event()

        def evidence(**_kwargs):
            entered.set()
            gate.wait(2.0)
            return None

        self.addCleanup(gate.set)
        manager = self.manager(evidence_preparer=evidence, generator=lambda **_kwargs: None)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(entered.wait(2.0))
        manager.cancel_video(VIDEO_ID)
        time.sleep(0.1)
        cancelled = self.state()
        self.assertEqual(cancelled.status, "cancelled")
        self.assertIsNone(cancelled.lease_owner)
        self.assertIsNone(cancelled.lease_started_at)
        gate.set()
        time.sleep(0.05)
        self.assertEqual(self.state().status, "cancelled")

        # A separately seeded superseded lease cannot be renewed by the old token.
        with self.Session() as db:
            state = db.query(ExplanationPreparation).one()
            state.status = "generating"
            state.stage = STAGE_EXTRACTING
            state.attempt_count = 2
            state.lease_owner = "new-owner"
            state.lease_started_at = _utcnow()
            marker = state.lease_started_at
            db.commit()
        time.sleep(0.1)
        superseded = self.state()
        self.assertEqual(superseded.lease_owner, "new-owner")
        self.assertEqual(superseded.lease_started_at, marker)

    def test_exhausted_and_terminal_failed_rows_are_not_auto_retried(self):
        self.seed_generating(attempts=3, age=2.0)
        calls = []
        manager = self.manager(generator=lambda **kwargs: calls.append(kwargs))
        self.assertTrue(wait_until(lambda: self.state().status == "failed"))
        state = self.state()
        self.assertEqual(state.stage, STAGE_FAILED)
        self.assertEqual(state.error_code, "retry_exhausted")
        self.assertEqual(state.attempt_count, 3)
        time.sleep(0.1)
        self.assertEqual(calls, [])
        self.assertFalse(manager._scheduled)

    def test_heartbeat_stops_after_ready_and_on_shutdown(self):
        audio = self.data_dir / "ready.mp3"

        def successful(**kwargs):
            audio.write_bytes(b"ID3-ready")
            with kwargs["session_factory"]() as db:
                db.add(Explanation(
                    intervention_id=INTERVENTION_ID,
                    output_mode="explanatory",
                    text_explanation="Ready",
                    audio_file_path=str(audio),
                ))
                db.commit()
            return object()

        manager = self.manager(generator=successful)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.state().status == "ready"))
        self.assertTrue(wait_until(lambda: not manager._tracked_leases))
        manager.shutdown(timeout=1.0)
        self.assertIsNone(manager._maintenance_thread)

    def test_heartbeat_database_error_is_retried_then_sustained_loss_fences_owner(self):
        self.seed_generating(owner="owned", attempts=1)
        manager = self.manager(start=False)
        state = self.state()
        job = PreparationJob(VIDEO_ID, INTERVENTION_ID, "explanatory")
        lease = _Lease("owned", str(state.id), 1)
        manager._track_lease(job, lease)

        original_factory = manager._session_factory
        manager._session_factory = lambda: (_ for _ in ()).throw(
            RuntimeError("controlled database outage")
        )
        manager._renew_tracked_leases()  # A transient failure does not kill ownership.
        self.assertTrue(manager._local_lease_is_healthy("owned"))

        with manager._state_lock:
            manager._tracked_leases["owned"].last_renewed -= 1.0
        manager._renew_tracked_leases()  # The owner cannot resurrect after its deadline.
        self.assertFalse(manager._local_lease_is_healthy("owned"))
        manager._session_factory = original_factory
        with self.assertRaises(PreparationCancelledError):
            manager._require_lease(job, "owned")

    def test_two_processes_have_one_atomic_stale_recovery_winner(self):
        self.seed_generating(attempts=1, age=2.0)
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        args = (str(self.db_path), str(self.data_dir), start_event, result_queue)
        processes = [
            context.Process(target=_recover_stale_in_process, args=args)
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        start_event.set()
        results = [result_queue.get(timeout=10.0) for _ in processes]
        for process in processes:
            process.join(10.0)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual([status for status, _ in results], ["ok", "ok"])
        self.assertEqual(sorted(value for _, value in results), [0, 1])
        state = self.state()
        self.assertEqual(state.status, "pending")
        self.assertEqual(state.attempt_count, 1)


if __name__ == "__main__":
    unittest.main()
