"""Focused B4 heartbeat resilience and bounded processing shutdown tests."""

from __future__ import annotations

import multiprocessing
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine
from api.models import ProcessingJob, Video
from api.processing_jobs import ProcessingJobManager


VIDEO_ID = "Shutdown123"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _result(root: Path, title="recovered"):
    return {
        "video_id": VIDEO_ID,
        "video_dir": str(root / VIDEO_ID),
        "title": title,
        "duration_seconds": 10,
        "duration_formatted": "00:10",
        "metadata": {"total_keyframes": 1},
        "transcript": {"entries": [{"start": 0, "end": 1, "text": "mock"}]},
        "intervention_points": [SimpleNamespace(
            timestamp=1.0, frame_path="frames/1.jpg", content_type="diagram",
            complexity_score=0.5, clip_confidence=0.9,
            trigger_reason="mock", confidence="high", transcript_context="mock",
        )],
    }


class GateOrchestrator:
    def __init__(self, result, *, obey_continue=False):
        self.result = result
        self.obey_continue = obey_continue
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    @staticmethod
    def _extract_video_id(_url):
        return VIDEO_ID

    def process_detection_only(self, _url, should_continue=None):
        self.calls += 1
        self.started.set()
        while not self.release.wait(0.005):
            if self.obey_continue and should_continue is not None and not should_continue():
                return {"error": "cooperatively stopped"}
        return self.result


class ImmediateOrchestrator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    @staticmethod
    def _extract_video_id(_url):
        return VIDEO_ID

    def process_detection_only(self, _url, should_continue=None):
        self.calls += 1
        return self.result


class SleepOrchestrator:
    def __init__(self, delay):
        self.delay = delay
        self.started = threading.Event()

    @staticmethod
    def _extract_video_id(_url):
        return VIDEO_ID

    def process_detection_only(self, _url, should_continue=None):
        self.started.set()
        time.sleep(self.delay)
        return {"error": "late blocked result"}


def _subprocess_shutdown(db_path, result_queue):
    engine = create_engine(
        f"sqlite:///{Path(db_path).as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    orchestrator = SleepOrchestrator(0.5)
    manager = ProcessingJobManager(
        Session, lambda: orchestrator,
        lease_seconds=1, heartbeat_seconds=0.05,
        recovery_interval_seconds=0.1, shutdown_grace_seconds=0.05,
    )
    manager.start()
    manager.create_or_join(VIDEO_URL, "explanatory")
    orchestrator.started.wait(2)
    started = time.monotonic()
    stopped = manager.shutdown()
    result_queue.put((time.monotonic() - started, stopped))
    # Function return does not make the executor's non-daemon worker vanish;
    # interpreter shutdown still waits for the sleeping synchronous call.


class ProcessingHeartbeatShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "shutdown.sqlite"
        self.engine = create_database_engine(f"sqlite:///{self.db_path.as_posix()}")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.managers = []

    def manager(self, orchestrator, *, start=False, **kwargs):
        manager = ProcessingJobManager(
            self.Session,
            lambda: orchestrator,
            lease_seconds=kwargs.pop("lease_seconds", 0.3),
            heartbeat_seconds=kwargs.pop("heartbeat_seconds", 0.02),
            recovery_interval_seconds=kwargs.pop("recovery_interval_seconds", 0.03),
            heartbeat_retry_backoff_seconds=kwargs.pop(
                "heartbeat_retry_backoff_seconds", 0.01
            ),
            shutdown_grace_seconds=kwargs.pop("shutdown_grace_seconds", 0.05),
            **kwargs,
        )
        if start:
            manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown, True, 0.5)
        return manager

    def seed_owned(self, manager, *, status="running", generation="generation-one"):
        job_id = str(uuid.uuid4())
        lease = _utcnow() + timedelta(seconds=0.2)
        with self.Session() as db:
            db.add(ProcessingJob(
                id=job_id, video_id=VIDEO_ID, source_url=VIDEO_URL,
                requested_output_mode="explanatory", status=status,
                stage={
                    "running": "processing", "succeeded": "complete",
                    "cancelled": "cancelled", "failed": "failed",
                }.get(status, "queued"),
                attempt_count=1, lease_owner=manager.owner if status == "running" else None,
                lease_expires_at=lease if status == "running" else None,
                generation_id=generation if status == "running" else None,
                active_slot=1 if status == "running" else None,
            ))
            db.commit()
        return job_id, lease

    def state(self):
        with self.Session() as db:
            return db.query(ProcessingJob).one()

    def test_single_transient_heartbeat_error_recovers_and_renews(self):
        manager = self.manager(object(), heartbeat_retry_limit=3)
        job_id, old_lease = self.seed_owned(manager)
        original = manager.session_factory
        failures = 0

        def flaky_factory():
            nonlocal failures
            if failures == 0:
                failures += 1
                raise OperationalError("heartbeat", {}, sqlite3.OperationalError("locked"))
            return original()

        manager.session_factory = flaky_factory
        stop = threading.Event()
        lost = threading.Event()
        thread = threading.Thread(
            target=manager._heartbeat, args=(job_id, "generation-one", stop, lost)
        )
        thread.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and self.state().lease_expires_at <= old_lease:
            time.sleep(0.005)
        stop.set()
        thread.join(1)
        self.assertEqual(failures, 1)
        self.assertFalse(lost.is_set())
        self.assertGreater(self.state().lease_expires_at, old_lease)

    def test_commit_operational_error_rolls_back_before_retry(self):
        manager = self.manager(object(), heartbeat_retry_limit=1)
        job_id, _ = self.seed_owned(manager)
        real = self.Session()
        rollback = mock.Mock(wraps=real.rollback)
        real.rollback = rollback
        real.commit = mock.Mock(side_effect=OperationalError(
            "heartbeat", {}, sqlite3.OperationalError("busy")
        ))

        class Context:
            def __enter__(self):
                return real

            def __exit__(self, *_args):
                real.close()

        manager.session_factory = Context
        lost = threading.Event()
        manager._heartbeat(job_id, "generation-one", threading.Event(), lost)
        self.assertTrue(lost.is_set())
        rollback.assert_called_once()

    def test_repeated_heartbeat_errors_are_bounded_and_fence_owner(self):
        manager = self.manager(
            object(), heartbeat_retry_limit=3,
            heartbeat_retry_backoff_seconds=0.01,
        )
        job_id, _ = self.seed_owned(manager)
        calls = 0

        def unavailable():
            nonlocal calls
            calls += 1
            raise OperationalError("heartbeat", {}, sqlite3.OperationalError("locked"))

        manager.session_factory = unavailable
        lost = threading.Event()
        started = time.monotonic()
        manager._heartbeat(job_id, "generation-one", threading.Event(), lost)
        elapsed = time.monotonic() - started
        self.assertEqual(calls, 3)
        self.assertTrue(lost.is_set())
        self.assertGreaterEqual(elapsed, 0.035)

    def test_confirmed_ownership_loss_never_renews_other_generation(self):
        manager = self.manager(object())
        job_id, old_lease = self.seed_owned(manager, generation="new-generation")
        lost = threading.Event()
        manager._heartbeat(job_id, "old-generation", threading.Event(), lost)
        self.assertTrue(lost.is_set())
        state = self.state()
        self.assertEqual(state.generation_id, "new-generation")
        self.assertEqual(state.lease_expires_at, old_lease)

    def test_heartbeat_stops_for_terminal_or_cancelled_state(self):
        for status in ("succeeded", "cancelled", "failed"):
            with self.subTest(status=status):
                with self.Session() as db:
                    db.query(ProcessingJob).delete()
                    db.commit()
                manager = self.manager(object())
                job_id, _ = self.seed_owned(manager, status=status)
                lost = threading.Event()
                manager._heartbeat(job_id, "generation-one", threading.Event(), lost)
                self.assertTrue(lost.is_set())
                self.assertEqual(self.state().status, status)

    def test_sustained_heartbeat_failure_discards_late_provider_result(self):
        orchestrator = GateOrchestrator(_result(self.root))
        manager = self.manager(
            orchestrator, start=True, heartbeat_retry_limit=2,
            lease_seconds=0.2, heartbeat_seconds=0.01,
        )
        original = manager.session_factory

        def heartbeat_unavailable():
            if threading.current_thread().name.startswith("processing-heartbeat-"):
                raise OperationalError("heartbeat", {}, sqlite3.OperationalError("locked"))
            return original()

        manager.session_factory = heartbeat_unavailable
        created = manager.create_or_join(VIDEO_URL, "explanatory")
        self.assertTrue(orchestrator.started.wait(1))
        time.sleep(0.08)
        orchestrator.release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and manager._futures:
            time.sleep(0.005)
        with self.Session() as db:
            job = db.get(ProcessingJob, created["job_id"])
            self.assertEqual(job.status, "running")
            self.assertIsNone(db.get(Video, VIDEO_ID))

    def test_shutdown_cancels_queued_processing_without_claiming_it(self):
        blocker = threading.Event()
        manager = self.manager(object(), shutdown_grace_seconds=0.02)
        manager._executor.submit(blocker.wait, 1)
        job_id, _ = self.seed_owned(manager, status="queued")
        self.assertTrue(manager._submit(job_id))
        stopped = manager.shutdown()
        blocker.set()
        self.assertFalse(stopped)
        self.assertEqual(self.state().status, "queued")
        self.assertEqual(self.state().attempt_count, 1)

    def test_blocked_detection_shutdown_respects_grace_and_late_result_is_fenced(self):
        orchestrator = GateOrchestrator(_result(self.root))
        manager = self.manager(orchestrator, start=True, shutdown_grace_seconds=0.05)
        created = manager.create_or_join(VIDEO_URL, "explanatory")
        self.assertTrue(orchestrator.started.wait(1))
        started = time.monotonic()
        stopped = manager.shutdown()
        elapsed = time.monotonic() - started
        self.assertFalse(stopped)
        self.assertLess(elapsed, 0.2)
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            manager.create_or_join(VIDEO_URL, "brief")
        orchestrator.release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and manager._futures:
            time.sleep(0.005)
        with self.Session() as db:
            job = db.get(ProcessingJob, created["job_id"])
            self.assertEqual(job.status, "running")
            self.assertIsNone(db.get(Video, VIDEO_ID))

    def test_cooperative_detection_stops_within_grace_and_shutdown_is_idempotent(self):
        orchestrator = GateOrchestrator(_result(self.root), obey_continue=True)
        manager = self.manager(orchestrator, start=True, shutdown_grace_seconds=0.3)
        manager.create_or_join(VIDEO_URL, "explanatory")
        self.assertTrue(orchestrator.started.wait(1))
        self.assertTrue(manager.shutdown())
        self.assertTrue(manager.shutdown())
        self.assertFalse(manager._submit("unused"))

    def test_new_manager_recovers_shutdown_interruption_after_lease_expiry(self):
        old_orchestrator = GateOrchestrator(_result(self.root, "OLD"))
        old = self.manager(
            old_orchestrator, start=True, lease_seconds=0.12,
            heartbeat_seconds=0.02, shutdown_grace_seconds=0.02,
        )
        created = old.create_or_join(VIDEO_URL, "explanatory")
        self.assertTrue(old_orchestrator.started.wait(1))
        self.assertFalse(old.shutdown())
        replacement_orchestrator = ImmediateOrchestrator(_result(self.root, "NEW"))
        replacement = self.manager(
            replacement_orchestrator, start=True, lease_seconds=0.12,
            heartbeat_seconds=0.02, recovery_interval_seconds=0.02,
            shutdown_grace_seconds=0.2,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.Session() as db:
                job = db.get(ProcessingJob, created["job_id"])
                if job.status == "succeeded":
                    break
            time.sleep(0.01)
        old_orchestrator.release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and old._futures:
            time.sleep(0.005)
        with self.Session() as db:
            job = db.get(ProcessingJob, created["job_id"])
            video = db.get(Video, VIDEO_ID)
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(job.attempt_count, 2)
            self.assertEqual(video.title, "NEW")

    def test_subprocess_proves_manager_deadline_not_interpreter_exit(self):
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        process_db = self.root / "subprocess.sqlite"
        process = context.Process(
            target=_subprocess_shutdown, args=(str(process_db), results)
        )
        started = time.monotonic()
        process.start()
        latency, stopped = results.get(timeout=10)
        self.assertFalse(stopped)
        self.assertLess(latency, 0.2)
        self.assertTrue(process.is_alive())
        process.join(5)
        self.assertEqual(process.exitcode, 0)
        self.assertGreater(time.monotonic() - started, latency)


if __name__ == "__main__":
    unittest.main()
