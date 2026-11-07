"""Focused processing-lease recovery tests using only temporary SQLite data."""

from __future__ import annotations

import multiprocessing
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base
from api.models import ProcessingJob, Video
from api.processing_jobs import ProcessingJobManager


VIDEO_ID = "Recover1234"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def successful_result(root: Path, title="Recovered") -> dict:
    return {
        "video_id": VIDEO_ID,
        "video_dir": str(root / VIDEO_ID),
        "title": title,
        "duration_seconds": 30,
        "duration_formatted": "00:30",
        "metadata": {"total_keyframes": 1},
        "transcript": {
            "entries": [{"start": 0, "end": 5, "text": "Synthetic transcript"}]
        },
        "intervention_points": [
            SimpleNamespace(
                timestamp=3.0,
                frame_path="frames/3.jpg",
                content_type="diagram",
                complexity_score=0.5,
                clip_confidence=0.9,
                trigger_reason="Visual content",
                confidence="high",
                transcript_context="Synthetic transcript",
            )
        ],
    }


class ControlledOrchestrator:
    def __init__(self, result, *, gate=None):
        self.result = result
        self.gate = gate
        self.started = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def _extract_video_id(self, _url):
        return VIDEO_ID

    def process_detection_only(self, _url, **_kwargs):
        with self.lock:
            self.calls += 1
        self.started.set()
        if self.gate is not None:
            self.gate.wait(3.0)
        return self.result


class DynamicOrchestrator:
    def __init__(self, root):
        self.root = root
        self.calls = 0

    @staticmethod
    def _extract_video_id(url):
        return url.rsplit("=", 1)[-1]

    def process_detection_only(self, url, **_kwargs):
        self.calls += 1
        result = successful_result(self.root, f"Video {self.calls}")
        result["video_id"] = self._extract_video_id(url)
        return result


class TakeoverOrchestrator:
    """Old call waits; the replacement generation completes first."""

    def __init__(self, root):
        self.root = root
        self.old_started = threading.Event()
        self.new_started = threading.Event()
        self.release_old = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def _extract_video_id(self, _url):
        return VIDEO_ID

    def process_detection_only(self, _url, **_kwargs):
        with self.lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            self.old_started.set()
            self.release_old.wait(4.0)
            return successful_result(self.root, "OLD")
        self.new_started.set()
        return successful_result(self.root, "NEW")


def _claim_stale_in_process(db_path, job_id, start_event, result_queue):
    """Spawn-safe worker proving one SQLite generation claim wins."""
    engine = create_engine(
        f"sqlite:///{Path(db_path).as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    manager = ProcessingJobManager(
        Session,
        orchestrator_factory=lambda: object(),
        lease_seconds=1.0,
        heartbeat_seconds=0.1,
        recovery_interval_seconds=0.1,
    )
    start_event.wait(5.0)
    try:
        result_queue.put(("ok", manager._claim(job_id) is not None))
    except Exception as exc:  # pragma: no cover - returned to the parent process
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        manager.shutdown(wait=False)
        engine.dispose()


class ProcessingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "processing.sqlite"
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.managers = []

    def manager(self, orchestrator, *, start=True, **kwargs):
        manager = ProcessingJobManager(
            self.Session,
            orchestrator_factory=lambda: orchestrator,
            lease_seconds=kwargs.pop("lease_seconds", 0.3),
            heartbeat_seconds=kwargs.pop("heartbeat_seconds", 0.04),
            recovery_interval_seconds=kwargs.pop("recovery_interval_seconds", 0.03),
            **kwargs,
        )
        if start:
            manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown)
        return manager

    def seed_job(
        self,
        *,
        status="running",
        lease_age=-1.0,
        attempt_count=1,
        owner="foreign-owner",
        generation="foreign-generation",
    ):
        lease = _utcnow() - timedelta(seconds=lease_age)
        with self.Session() as db:
            job = ProcessingJob(
                id="11111111-1111-4111-8111-111111111111",
                video_id=VIDEO_ID,
                source_url=VIDEO_URL,
                requested_output_mode="explanatory",
                status=status,
                stage="processing" if status == "running" else status,
                attempt_count=attempt_count,
                lease_owner=owner if status == "running" else None,
                lease_expires_at=lease if status == "running" else None,
                generation_id=generation if status == "running" else None,
                active_slot=1 if status in {"running", "cancelling"} else None,
            )
            db.add(job)
            db.commit()
            return str(job.id)

    def job(self):
        with self.Session() as db:
            return db.query(ProcessingJob).one()

    def test_fresh_startup_lease_is_preserved_then_recovers_without_request(self):
        self.seed_job(lease_age=-0.25)
        gate = threading.Event()
        orchestrator = ControlledOrchestrator(successful_result(self.root), gate=gate)
        self.addCleanup(gate.set)
        manager = self.manager(orchestrator)

        time.sleep(0.12)
        fresh = self.job()
        self.assertEqual(fresh.status, "running")
        self.assertEqual(fresh.lease_owner, "foreign-owner")
        self.assertEqual(fresh.attempt_count, 1)
        self.assertEqual(orchestrator.calls, 0)

        self.assertTrue(orchestrator.started.wait(2.0))
        recovered = self.job()
        self.assertEqual(recovered.attempt_count, 2)
        self.assertEqual(recovered.lease_owner, manager.owner)
        gate.set()
        self.assertTrue(wait_until(lambda: self.job().status == "succeeded"))

    def test_heartbeat_renewed_live_job_is_never_stolen_or_dispatched_twice(self):
        gate = threading.Event()
        orchestrator = ControlledOrchestrator(successful_result(self.root), gate=gate)
        self.addCleanup(gate.set)
        first = self.manager(orchestrator)
        created = first.create_or_join(VIDEO_URL, "explanatory")
        self.assertTrue(orchestrator.started.wait(2.0))
        second = self.manager(orchestrator)

        time.sleep(0.55)
        live = self.job()
        self.assertEqual(orchestrator.calls, 1)
        self.assertEqual(live.attempt_count, 1)
        self.assertEqual(live.lease_owner, first.owner)
        self.assertNotEqual(live.lease_owner, second.owner)
        gate.set()
        self.assertEqual(first.wait(created["job_id"], timeout=3)["status"], "succeeded")

    def test_two_managers_recover_one_stale_job_once_across_repeated_scans(self):
        self.seed_job(lease_age=1.0)
        gate = threading.Event()
        orchestrator = ControlledOrchestrator(successful_result(self.root), gate=gate)
        self.addCleanup(gate.set)
        first = self.manager(orchestrator)
        second = self.manager(orchestrator)
        self.assertTrue(orchestrator.started.wait(2.0))
        time.sleep(0.2)

        state = self.job()
        self.assertEqual(orchestrator.calls, 1)
        self.assertEqual(state.attempt_count, 2)
        self.assertIn(state.lease_owner, {first.owner, second.owner})
        gate.set()
        self.assertTrue(wait_until(lambda: self.job().status == "succeeded"))

    def test_old_generation_cannot_persist_after_periodic_takeover(self):
        orchestrator = TakeoverOrchestrator(self.root)
        self.addCleanup(orchestrator.release_old.set)
        first = self.manager(orchestrator, start=False)
        # Deterministically simulate a dead heartbeat while its provider call remains late.
        first._heartbeat = (
            lambda _job, _generation, stop, _ownership_lost: stop.wait(4.0)
        )
        created = first.create_or_join(VIDEO_URL, "explanatory")
        first.start()
        self.assertTrue(orchestrator.old_started.wait(2.0))
        second = self.manager(orchestrator)

        with self.Session() as db:
            job = db.get(ProcessingJob, created["job_id"])
            job.lease_expires_at = _utcnow() - timedelta(seconds=1)
            db.commit()
        self.assertTrue(orchestrator.new_started.wait(2.0))
        self.assertTrue(wait_until(lambda: self.job().status == "succeeded"))
        orchestrator.release_old.set()
        self.assertTrue(wait_until(lambda: not first._futures))

        with self.Session() as db:
            job = db.get(ProcessingJob, created["job_id"])
            video = db.get(Video, VIDEO_ID)
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(job.attempt_count, 2)
            self.assertEqual(video.title, "NEW")
        self.assertEqual(orchestrator.calls, 2)
        self.assertNotEqual(first.owner, second.owner)

    def test_terminal_jobs_are_not_restarted_and_cancelling_uses_existing_policy(self):
        for status in ("failed", "cancelled", "cancelling"):
            with self.subTest(status=status):
                with self.Session() as db:
                    db.query(ProcessingJob).delete()
                    db.commit()
                self.seed_job(status=status)
                orchestrator = ControlledOrchestrator(successful_result(self.root))
                manager = self.manager(orchestrator)
                time.sleep(0.12)
                self.assertEqual(orchestrator.calls, 0)
                expected = "cancelled" if status == "cancelling" else status
                self.assertEqual(self.job().status, expected)
                manager.shutdown()

    def test_temporary_scan_failure_does_not_kill_eventual_recovery(self):
        self.seed_job(lease_age=-0.15)
        orchestrator = ControlledOrchestrator(successful_result(self.root))
        manager = self.manager(orchestrator)
        original_factory = manager.session_factory

        class FailOnce:
            def __init__(self):
                self.failed = False
                self.lock = threading.Lock()

            def __call__(self):
                with self.lock:
                    if not self.failed:
                        self.failed = True
                        raise RuntimeError("controlled database read failure")
                return original_factory()

        flaky = FailOnce()
        manager.session_factory = flaky
        self.assertTrue(orchestrator.started.wait(2.0))
        self.assertTrue(flaky.failed)
        self.assertTrue(wait_until(lambda: self.job().status == "succeeded"))
        self.assertEqual(orchestrator.calls, 1)
        self.assertEqual(self.job().attempt_count, 2)

    def test_shutdown_stops_scanning_and_prevents_later_dispatch(self):
        self.seed_job(lease_age=-0.15)
        orchestrator = ControlledOrchestrator(successful_result(self.root))
        manager = self.manager(orchestrator)
        manager.shutdown()
        self.assertIsNone(manager._maintenance_thread)
        time.sleep(0.3)
        state = self.job()
        self.assertEqual(orchestrator.calls, 0)
        self.assertEqual(state.status, "running")
        self.assertEqual(state.attempt_count, 1)

    def test_completion_releases_slot_and_allows_another_video(self):
        orchestrator = DynamicOrchestrator(self.root)
        manager = self.manager(orchestrator)
        first = manager.create_or_join(VIDEO_URL, "explanatory")
        self.assertEqual(manager.wait(first["job_id"], timeout=3)["status"], "succeeded")
        with self.Session() as db:
            self.assertEqual(
                db.query(ProcessingJob).filter(ProcessingJob.active_slot == 1).count(),
                0,
            )

        second_video = "Recover5678"
        second = manager.create_or_join(
            f"https://www.youtube.com/watch?v={second_video}", "explanatory"
        )
        self.assertEqual(manager.wait(second["job_id"], timeout=3)["status"], "succeeded")
        self.assertEqual(orchestrator.calls, 2)

    def test_two_processes_have_one_atomic_generation_winner(self):
        job_id = self.seed_job(lease_age=1.0)
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        args = (str(self.db_path), job_id, start_event, result_queue)
        processes = [
            context.Process(target=_claim_stale_in_process, args=args)
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
        self.assertEqual(sorted(value for _, value in results), [False, True])
        state = self.job()
        self.assertEqual(state.status, "running")
        self.assertEqual(state.attempt_count, 2)
        self.assertIsNotNone(state.generation_id)


if __name__ == "__main__":
    unittest.main()
