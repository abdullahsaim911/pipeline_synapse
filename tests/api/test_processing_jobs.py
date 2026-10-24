"""Durable processing-job concurrency, cache, failure, and recovery tests."""

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine, get_db
from api.models import (
    Intervention,
    ProcessingJob,
    ProcessingJobRequestedMode,
    Transcript,
    Video,
)
from api.processing_jobs import ProcessingJobManager
from api.routers import video as video_module


VIDEO_ID = "JobTest1234"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def successful_result(root: Path) -> dict:
    return {
        "video_id": VIDEO_ID,
        "video_dir": str(root / VIDEO_ID),
        "title": "Durable processing",
        "duration_seconds": 45,
        "duration_formatted": "00:45",
        "metadata": {"total_keyframes": 1},
        "transcript": {
            "entries": [{"start": 0, "end": 5, "text": "A valid transcript"}]
        },
        "intervention_points": [
            SimpleNamespace(
                timestamp=4.0,
                frame_path="frames/4.jpg",
                content_type="diagram",
                complexity_score=0.8,
                clip_confidence=0.9,
                trigger_reason="Visual content",
                confidence="high",
                transcript_context="A valid transcript",
            )
        ],
    }


class ControlledOrchestrator:
    def __init__(self, results, started=None, release=None):
        self.results = list(results)
        self.started = started
        self.release = release
        self.calls = 0
        self.lock = threading.Lock()

    def _extract_video_id(self, _url):
        return VIDEO_ID

    def process_detection_only(self, _url):
        with self.lock:
            index = self.calls
            self.calls += 1
        if self.started:
            self.started.set()
        if self.release:
            self.release.wait(3)
        return self.results[min(index, len(self.results) - 1)]


class RecordingPreparationManager:
    """Records scheduling without running explanation inference."""

    def __init__(self):
        self.calls = []
        self.scheduled = set()
        self.generation_starts = 0
        self.lock = threading.Lock()

    def schedule_video(self, video_id, output_mode):
        with self.lock:
            self.calls.append((video_id, output_mode))
            if (video_id, output_mode) not in self.scheduled:
                self.generation_starts += 1
            self.scheduled.add((video_id, output_mode))
        return 0


class ProcessingJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = create_database_engine(
            f"sqlite:///{(self.root / 'jobs.sqlite').as_posix()}"
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.managers = []

    def manager(self, orchestrator, **kwargs):
        manager = ProcessingJobManager(
            self.Session,
            orchestrator_factory=lambda: orchestrator,
            heartbeat_seconds=0.05,
            **kwargs,
        )
        manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown)
        return manager

    def seed_valid_cache(self):
        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="Cached lecture",
                source_url=VIDEO_URL,
                total_keyframes=1,
            ))
            db.add(Transcript(
                video_id=VIDEO_ID,
                start_time=0,
                end_time=5,
                text="Cached transcript",
            ))
            db.add(Intervention(
                id="cached-point",
                video_id=VIDEO_ID,
                timestamp=4,
                frame_path="frames/4.jpg",
            ))
            db.commit()

    def test_concurrent_managers_create_or_join_one_pipeline_run(self):
        started = threading.Event()
        release = threading.Event()
        orchestrator = ControlledOrchestrator(
            [successful_result(self.root)], started=started, release=release
        )
        first_manager = self.manager(orchestrator)
        second_manager = self.manager(orchestrator)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(manager.create_or_join, VIDEO_URL, "explanatory")
                for manager in (first_manager, second_manager)
            ]
            snapshots = [future.result(timeout=2) for future in futures]

        self.assertTrue(started.wait(1))
        self.assertEqual({item["job_id"] for item in snapshots}, {snapshots[0]["job_id"]})
        self.assertTrue(any(item["joined_existing"] for item in snapshots))
        release.set()
        final = first_manager.wait(snapshots[0]["job_id"], timeout=3)
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(orchestrator.calls, 1)
        with self.Session() as db:
            self.assertEqual(db.query(ProcessingJob).count(), 1)
            self.assertEqual(db.query(Video).count(), 1)
            self.assertEqual(db.query(Transcript).count(), 1)

    def test_concurrent_joiners_preserve_and_schedule_different_modes(self):
        started = threading.Event()
        release = threading.Event()
        preparation = RecordingPreparationManager()
        orchestrator = ControlledOrchestrator(
            [successful_result(self.root)], started=started, release=release
        )
        first_manager = self.manager(orchestrator, preparation_manager=preparation)
        second_manager = self.manager(orchestrator, preparation_manager=preparation)

        first = first_manager.create_or_join(VIDEO_URL, "brief")
        self.assertTrue(started.wait(1))
        second = second_manager.create_or_join(VIDEO_URL, "detailed")

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["joined_existing"])
        self.assertEqual(set(second["requested_modes"]), {"brief", "detailed"})
        release.set()
        final = first_manager.wait(first["job_id"], timeout=3)

        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(set(final["requested_modes"]), {"brief", "detailed"})
        self.assertEqual(
            preparation.scheduled,
            {(VIDEO_ID, "brief"), (VIDEO_ID, "detailed")},
        )
        self.assertEqual(orchestrator.calls, 1)
        with self.Session() as db:
            self.assertEqual(db.query(ProcessingJobRequestedMode).count(), 2)

    def test_mode_requested_after_processing_uses_cache_and_is_prepared(self):
        preparation = RecordingPreparationManager()
        orchestrator = ControlledOrchestrator([successful_result(self.root)])
        manager = self.manager(orchestrator, preparation_manager=preparation)

        first = manager.create_or_join(VIDEO_URL, "brief")
        manager.wait(first["job_id"], timeout=3)
        after = manager.create_or_join(VIDEO_URL, "explanatory")

        self.assertTrue(after["reused_cache"])
        self.assertEqual(set(after["requested_modes"]), {"brief", "explanatory"})
        self.assertEqual(
            preparation.scheduled,
            {(VIDEO_ID, "brief"), (VIDEO_ID, "explanatory")},
        )
        self.assertEqual(orchestrator.calls, 1)

    def test_complete_cache_is_reused_without_pipeline_call(self):
        self.seed_valid_cache()
        orchestrator = ControlledOrchestrator([successful_result(self.root)])
        preparation = RecordingPreparationManager()
        manager = self.manager(orchestrator, preparation_manager=preparation)

        snapshot = manager.create_or_join(VIDEO_URL, "brief")
        repeated = manager.create_or_join(VIDEO_URL, "brief")

        self.assertEqual(snapshot["status"], "succeeded")
        self.assertEqual(snapshot["stage"], "complete")
        self.assertTrue(snapshot["reused_cache"])
        self.assertEqual(snapshot["requested_modes"], ["brief"])
        self.assertEqual(repeated["job_id"], snapshot["job_id"])
        self.assertEqual(preparation.scheduled, {(VIDEO_ID, "brief")})
        self.assertEqual(preparation.generation_starts, 1)
        self.assertEqual(orchestrator.calls, 0)

    def test_incomplete_cache_is_replaced_atomically(self):
        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="Incomplete",
                source_url=VIDEO_URL,
                total_keyframes=1,
            ))
            db.commit()
        orchestrator = ControlledOrchestrator([successful_result(self.root)])
        manager = self.manager(orchestrator)

        created = manager.create_or_join(VIDEO_URL, "explanatory")
        final = manager.wait(created["job_id"], timeout=3)

        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(orchestrator.calls, 1)
        with self.Session() as db:
            self.assertEqual(db.query(Video).count(), 1)
            self.assertEqual(db.query(Transcript).count(), 1)
            self.assertEqual(db.query(Intervention).count(), 1)

    def test_failure_is_sanitized_and_retry_succeeds(self):
        orchestrator = ControlledOrchestrator([
            {"error": "secret provider traceback and local path"},
            successful_result(self.root),
        ])
        manager = self.manager(orchestrator)
        created = manager.create_or_join(VIDEO_URL, "detailed")
        failed = manager.wait(created["job_id"], timeout=3)

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "pipeline_failed")
        self.assertNotIn("secret", failed["error_message"])

        retried = manager.retry(created["job_id"])
        recovered = manager.wait(retried["job_id"], timeout=3)
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["attempt_count"], 2)
        self.assertEqual(orchestrator.calls, 2)

    def test_startup_recovers_an_abandoned_running_job(self):
        expired = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)
        with self.Session() as db:
            job = ProcessingJob(
                id="11111111-1111-4111-8111-111111111111",
                video_id=VIDEO_ID,
                source_url=VIDEO_URL,
                requested_output_mode="explanatory",
                status="running",
                stage="processing",
                attempt_count=1,
                lease_owner="dead-worker",
                lease_expires_at=expired,
            )
            db.add(job)
            db.commit()
            job_id = str(job.id)

        orchestrator = ControlledOrchestrator([successful_result(self.root)])
        manager = self.manager(orchestrator)
        recovered = manager.wait(job_id, timeout=3)

        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["attempt_count"], 2)
        self.assertEqual(orchestrator.calls, 1)

    def test_startup_does_not_steal_an_active_worker_lease(self):
        active_lease = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        with self.Session() as db:
            job = ProcessingJob(
                id="22222222-2222-4222-8222-222222222222",
                video_id=VIDEO_ID,
                source_url=VIDEO_URL,
                requested_output_mode="explanatory",
                status="running",
                stage="processing",
                attempt_count=1,
                lease_owner="another-live-worker",
                lease_expires_at=active_lease,
            )
            db.add(job)
            db.commit()
            job_id = str(job.id)

        orchestrator = ControlledOrchestrator([successful_result(self.root)])
        manager = self.manager(orchestrator)
        status = manager.get(job_id)

        self.assertEqual(status["status"], "running")
        self.assertEqual(status["attempt_count"], 1)
        self.assertEqual(orchestrator.calls, 0)

    def test_create_and_status_endpoints_expose_no_fabricated_percentage(self):
        started = threading.Event()
        release = threading.Event()
        orchestrator = ControlledOrchestrator(
            [successful_result(self.root)], started=started, release=release
        )
        manager = self.manager(orchestrator)
        app = FastAPI()
        app.state.processing_jobs = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with TestClient(app) as client:
            created = client.post(
                "/video/processing-jobs",
                json={"youtube_url": VIDEO_URL, "output_mode": "brief"},
            )
            self.assertEqual(created.status_code, 202, created.text)
            body = created.json()
            self.assertNotIn("progress", body)
            self.assertIn(body["stage"], ("queued", "processing"))
            status = client.get(f"/video/processing-jobs/{body['job_id']}")
            self.assertEqual(status.status_code, 200, status.text)
            self.assertEqual(status.json()["job_id"], body["job_id"])
            release.set()
            manager.wait(body["job_id"], timeout=3)

    def test_concurrent_legacy_process_requests_join_the_same_job(self):
        started = threading.Event()
        release = threading.Event()
        orchestrator = ControlledOrchestrator(
            [successful_result(self.root)], started=started, release=release
        )
        manager = self.manager(orchestrator)
        app = FastAPI()
        app.state.processing_jobs = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
            requests = [
                executor.submit(
                    client.post,
                    "/video/process",
                    json={"youtube_url": VIDEO_URL, "output_mode": "explanatory"},
                )
                for _ in range(2)
            ]
            self.assertTrue(started.wait(1))
            release.set()
            responses = [request.result(timeout=3) for request in requests]

        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(orchestrator.calls, 1)
        self.assertEqual(
            {item["id"] for item in responses[0].json()["interventions"]},
            {item["id"] for item in responses[1].json()["interventions"]},
        )


if __name__ == "__main__":
    unittest.main()
