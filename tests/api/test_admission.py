"""Focused B3 durable admission tests with temporary SQLite and fake providers."""

from __future__ import annotations

import multiprocessing
import tempfile
import threading
import time
import uuid
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine, get_db
from api.explanation_preparation import ExplanationPreparationManager
from api.explanation_service import ExplanationArtifact, PreparationCancelledError
from api.models import (
    Explanation, ExplanationPreparation, Intervention, ProcessingJob, Transcript, Video,
)
import api.processing_jobs as processing_jobs_module
from api.processing_jobs import (
    ProcessingJobBusyError,
    ProcessingJobManager,
    commit_active_slot,
)
from api.routers import video as video_module


VIDEO_A = "AdmissionA1x"
VIDEO_B = "AdmissionB2x"


class DynamicOrchestrator:
    @staticmethod
    def _extract_video_id(url):
        return url.rsplit("=", 1)[-1]

    def process_detection_only(self, _url):  # pragma: no cover - must stay unused
        raise AssertionError("provider pipeline must not run in admission tests")


def _processing_race_worker(db_path, video_id, start, results, existing=False):
    engine = create_engine(
        f"sqlite:///{Path(db_path).as_posix()}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    manager = ProcessingJobManager(Session, lambda: DynamicOrchestrator())
    manager._submit = lambda _job_id: False
    start.wait(5)
    try:
        snapshot = manager.create_or_join(
            f"https://www.youtube.com/watch?v={video_id}", "explanatory"
        )
        results.put(("won", snapshot["video_id"], snapshot["job_id"]))
    except ProcessingJobBusyError as exc:
        results.put(("busy", exc.video_id, exc.job_id))
    except Exception as exc:  # pragma: no cover - reported for useful diagnosis
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        manager.shutdown()
        engine.dispose()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "admission.sqlite"
        self.engine = create_database_engine(f"sqlite:///{self.db_path.as_posix()}")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _seed_video(self, video_id, *, intervention=True):
        with self.Session() as db:
            db.add(Video(
                id=video_id,
                title=f"Video {video_id}",
                source_url=f"https://www.youtube.com/watch?v={video_id}",
            ))
            if intervention:
                db.add(Intervention(
                    id=f"point-{video_id}", video_id=video_id,
                    timestamp=1.0, frame_path="frames/1.jpg",
                ))
            db.commit()

    def _seed_job(self, video_id, *, status="failed", active_slot=None):
        job_id = str(uuid.uuid4())
        with self.Session() as db:
            db.add(ProcessingJob(
                id=job_id, video_id=video_id,
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                requested_output_mode="explanatory", status=status,
                stage={
                    "running": "processing", "cancelled": "cancelled",
                    "succeeded": "complete",
                }.get(status, "failed"),
                active_slot=active_slot,
            ))
            db.commit()
        return job_id

    def _manager(self, preparation_manager=None):
        manager = ProcessingJobManager(
            self.Session, lambda: DynamicOrchestrator(),
            preparation_manager=preparation_manager,
        )
        manager._submit = lambda _job_id: False
        self.addCleanup(manager.shutdown)
        return manager

    def _artifact_generator(self, calls, started=None, release=None, failure=None):
        def generate(**kwargs):
            calls.append(kwargs["video_id"])
            if started is not None:
                started.set()
            if release is not None:
                release.wait(3)
            if failure is not None:
                raise failure
            audio = self.root / f"{kwargs['intervention_id']}.mp3"
            audio.write_bytes(b"ID3-test")
            with self.Session() as db:
                db.add(Explanation(
                    intervention_id=kwargs["intervention_id"],
                    text_explanation="mock explanation",
                    audio_file_path=str(audio),
                    output_mode=kwargs["output_mode"],
                ))
                db.commit()
            return ExplanationArtifact(
                kwargs["intervention_id"], "mock explanation", str(audio),
                None, None, kwargs["output_mode"],
            )
        return generate

    def _preparation_manager(self, generator):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.root),
            evidence_preparer=lambda **_kwargs: None,
            generator=generator,
            lease_timeout_seconds=2,
            maintenance_interval_seconds=0.2,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1)
        return manager

    def test_two_process_new_video_race_has_one_winner_and_real_busy_owner(self):
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_processing_race_worker,
                args=(str(self.db_path), video_id, start, results),
            )
            for video_id in (VIDEO_A, VIDEO_B)
        ]
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(15)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual([item[0] for item in outcomes].count("won"), 1, outcomes)
        self.assertEqual([item[0] for item in outcomes].count("busy"), 1, outcomes)
        winner = next(item for item in outcomes if item[0] == "won")
        blocked = next(item for item in outcomes if item[0] == "busy")
        self.assertEqual(blocked[1:], winner[1:])

    def test_two_process_existing_row_activation_race_has_one_winner(self):
        self._seed_job(VIDEO_A)
        self._seed_job(VIDEO_B)
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_processing_race_worker,
                args=(str(self.db_path), video_id, start, results, True),
            )
            for video_id in (VIDEO_A, VIDEO_B)
        ]
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(item[0] for item in outcomes), ["busy", "won"], outcomes)

    def test_same_video_duplicates_join_one_durable_row(self):
        first = self._manager()
        second = self._manager()
        url = f"https://www.youtube.com/watch?v={VIDEO_A}"
        with ThreadPoolExecutor(max_workers=2) as pool:
            snapshots = list(pool.map(
                lambda manager: manager.create_or_join(url, "explanatory"),
                (first, second),
            ))
        self.assertEqual({item["job_id"] for item in snapshots}, {snapshots[0]["job_id"]})
        with self.Session() as db:
            self.assertEqual(db.query(ProcessingJob).count(), 1)

    def test_cached_activation_racing_new_submission_has_one_owner(self):
        self._seed_video(VIDEO_A)
        with self.Session() as db:
            db.get(Video, VIDEO_A).total_keyframes = 1
            db.add(Transcript(
                video_id=VIDEO_A, start_time=0, end_time=1, text="cached",
            ))
            db.commit()
        session_factory = self.Session

        class PendingPreparation:
            def schedule_video(self, video_id, output_mode):
                with session_factory() as db:
                    if db.query(ExplanationPreparation).filter_by(
                        video_id=video_id, output_mode=output_mode
                    ).first() is None:
                        db.add(ExplanationPreparation(
                            video_id=video_id,
                            intervention_id=f"point-{video_id}",
                            output_mode=output_mode,
                            status="pending",
                            stage="pending_extraction",
                            attempt_count=0,
                        ))
                        db.commit()
                return 1

        first = self._manager(PendingPreparation())
        second = self._manager()
        barrier = threading.Barrier(2)
        original = processing_jobs_module.commit_active_slot
        lock = threading.Lock()
        calls = 0

        def synchronized_commit(db, video_id):
            nonlocal calls
            with lock:
                calls += 1
                synchronize = calls <= 2
            if synchronize:
                barrier.wait(3)
            return original(db, video_id)

        with mock.patch.object(
            processing_jobs_module, "commit_active_slot", synchronized_commit
        ), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(first.create_or_join,
                            f"https://www.youtube.com/watch?v={VIDEO_A}", "brief"),
                pool.submit(second.create_or_join,
                            f"https://www.youtube.com/watch?v={VIDEO_B}", "brief"),
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(("won", future.result(timeout=5)["video_id"]))
                except ProcessingJobBusyError as exc:
                    outcomes.append(("busy", exc.video_id))
        self.assertEqual(sorted(item[0] for item in outcomes), ["busy", "won"])
        winner = next(item[1] for item in outcomes if item[0] == "won")
        self.assertEqual(next(item[1] for item in outcomes if item[0] == "busy"), winner)

    def test_resume_racing_new_submission_has_one_owner(self):
        cancelled_id = self._seed_job(VIDEO_A, status="cancelled")
        first = self._manager()
        second = self._manager()
        barrier = threading.Barrier(2)
        original = processing_jobs_module.commit_active_slot
        lock = threading.Lock()
        calls = 0

        def synchronized_commit(db, video_id):
            nonlocal calls
            with lock:
                calls += 1
                synchronize = calls <= 2
            if synchronize:
                barrier.wait(3)
            return original(db, video_id)

        with mock.patch.object(
            processing_jobs_module, "commit_active_slot", synchronized_commit
        ), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(first.resume, cancelled_id),
                pool.submit(second.create_or_join,
                            f"https://www.youtube.com/watch?v={VIDEO_B}", "brief"),
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(("won", future.result(timeout=5)["video_id"]))
                except ProcessingJobBusyError as exc:
                    outcomes.append(("busy", exc.video_id))
        self.assertEqual(sorted(item[0] for item in outcomes), ["busy", "won"])
        winner = next(item[1] for item in outcomes if item[0] == "won")
        self.assertEqual(next(item[1] for item in outcomes if item[0] == "busy"), winner)

    def test_unrelated_integrity_error_is_not_converted_to_busy(self):
        with self.Session() as db:
            db.add(ProcessingJob(
                id=str(uuid.uuid4()), video_id=VIDEO_A, source_url=None,
                requested_output_mode="explanatory", active_slot=1,
            ))
            with self.assertRaises(IntegrityError):
                commit_active_slot(db, VIDEO_A)

    def test_http_conflict_is_409_and_identifies_durable_winner(self):
        winner_id = self._seed_job(VIDEO_A, status="running", active_slot=1)
        manager = self._manager()
        app = FastAPI()
        app.include_router(video_module.router)
        app.state.processing_jobs = manager

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with TestClient(app) as client:
            response = client.post("/video/processing-jobs", json={
                "youtube_url": f"https://www.youtube.com/watch?v={VIDEO_B}",
                "output_mode": "explanatory",
            })
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], {
            "code": "video_processing_busy",
            "active_video_id": VIDEO_A,
            "active_job_id": winner_id,
        })

    def test_manual_other_video_is_blocked_before_any_provider_call(self):
        self._seed_video(VIDEO_A)
        self._seed_video(VIDEO_B)
        self._seed_job(VIDEO_A, status="running", active_slot=1)
        calls = []
        manager = self._preparation_manager(self._artifact_generator(calls))
        with self.assertRaises(ProcessingJobBusyError) as raised:
            manager.generate_manual(VIDEO_B, f"point-{VIDEO_B}", "explanatory")
        self.assertEqual(raised.exception.video_id, VIDEO_A)
        self.assertEqual(calls, [])

    def test_explanation_retry_is_blocked_before_provider_call(self):
        self._seed_video(VIDEO_A)
        self._seed_video(VIDEO_B)
        self._seed_job(VIDEO_A, status="running", active_slot=1)
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_B,
                intervention_id=f"point-{VIDEO_B}",
                output_mode="explanatory",
                status="failed",
                stage="failed",
                attempt_count=1,
                error_code="transient_failure",
            ))
            db.commit()
        calls = []
        manager = self._preparation_manager(self._artifact_generator(calls))
        with self.assertRaises(ProcessingJobBusyError):
            manager.retry(VIDEO_B, f"point-{VIDEO_B}", "explanatory")
        self.assertEqual(calls, [])
        with self.Session() as db:
            self.assertEqual(
                db.query(ExplanationPreparation).filter_by(video_id=VIDEO_B).one().status,
                "failed",
            )

    def test_manual_same_active_video_is_allowed_without_releasing_processing_slot(self):
        self._seed_video(VIDEO_A)
        job_id = self._seed_job(VIDEO_A, status="running", active_slot=1)
        calls = []
        manager = self._preparation_manager(self._artifact_generator(calls))
        result = manager.generate_manual(VIDEO_A, f"point-{VIDEO_A}", "explanatory")
        self.assertEqual(result.text_explanation, "mock explanation")
        self.assertEqual(calls, [VIDEO_A])
        with self.Session() as db:
            job = db.get(ProcessingJob, job_id)
            self.assertEqual(job.active_slot, 1)
            self.assertEqual(job.status, "running")

    def test_legacy_manual_row_blocks_submission_then_releases_on_success(self):
        self._seed_video(VIDEO_A)
        started = threading.Event()
        release = threading.Event()
        calls = []
        manager = self._preparation_manager(
            self._artifact_generator(calls, started=started, release=release)
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                manager.generate_manual, VIDEO_A, f"point-{VIDEO_A}", "explanatory"
            )
            self.assertTrue(started.wait(2))
            with self.Session() as db:
                legacy = db.query(ProcessingJob).filter_by(video_id=VIDEO_A).one()
                self.assertEqual((legacy.status, legacy.stage, legacy.active_slot),
                                 ("succeeded", "complete", 1))
            with self.assertRaises(ProcessingJobBusyError):
                self._manager().create_or_join(
                    f"https://www.youtube.com/watch?v={VIDEO_B}", "explanatory"
                )
            release.set()
            future.result(timeout=3)
        with self.Session() as db:
            self.assertIsNone(db.query(ProcessingJob).filter_by(video_id=VIDEO_A).one().active_slot)

    def test_restart_recovery_finishes_legacy_preparation_and_releases_slot(self):
        self._seed_video(VIDEO_A)
        self._seed_job(VIDEO_A, status="succeeded", active_slot=1)
        stale = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_A,
                intervention_id=f"point-{VIDEO_A}",
                output_mode="explanatory",
                status="generating",
                stage="generating_explanation",
                attempt_count=1,
                lease_owner="dead-process",
                lease_started_at=stale,
            ))
            db.commit()
        calls = []
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.root),
            evidence_preparer=lambda **_kwargs: None,
            generator=self._artifact_generator(calls),
            lease_timeout_seconds=0.2,
            maintenance_interval_seconds=0.05,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.Session() as db:
                state = db.query(ExplanationPreparation).filter_by(video_id=VIDEO_A).one()
                slot = db.query(ProcessingJob).filter_by(video_id=VIDEO_A).one().active_slot
                if state.status == "ready" and slot is None:
                    break
            time.sleep(0.02)
        self.assertEqual(calls, [VIDEO_A])
        with self.Session() as db:
            self.assertEqual(
                db.query(ExplanationPreparation).filter_by(video_id=VIDEO_A).one().status,
                "ready",
            )
            self.assertIsNone(
                db.query(ProcessingJob).filter_by(video_id=VIDEO_A).one().active_slot
            )

    def test_manual_failure_and_cancellation_do_not_strand_legacy_slot(self):
        for index, failure in enumerate((RuntimeError("mock failure"),
                                         PreparationCancelledError("mock cancel"))):
            video_id = f"Failure{index:04d}"
            self._seed_video(video_id)
            manager = self._preparation_manager(
                self._artifact_generator([], failure=failure)
            )
            with self.assertRaises(type(failure)):
                manager.generate_manual(video_id, f"point-{video_id}", "explanatory")
            with self.Session() as db:
                self.assertIsNone(
                    db.query(ProcessingJob).filter_by(video_id=video_id).one().active_slot
                )

    def test_concurrent_same_video_manual_requests_deduplicate_provider(self):
        self._seed_video(VIDEO_A)
        started = threading.Event()
        release = threading.Event()
        calls = []
        manager = self._preparation_manager(
            self._artifact_generator(calls, started=started, release=release)
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(
                manager.generate_manual, VIDEO_A, f"point-{VIDEO_A}", "explanatory"
            ) for _ in range(2)]
            self.assertTrue(started.wait(2))
            release.set()
            results = [future.result(timeout=3) for future in futures]
        self.assertEqual(len(calls), 1)
        self.assertEqual({result.text_explanation for result in results}, {"mock explanation"})


if __name__ == "__main__":
    unittest.main()
