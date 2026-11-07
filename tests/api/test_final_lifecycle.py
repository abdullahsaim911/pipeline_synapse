"""Focused REL-008/009 slot reconciliation and preparation shutdown tests."""

import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine
from api.explanation_preparation import (
    ExplanationPreparationManager,
    STAGE_TTS,
)
from api.explanation_service import EvidenceArtifact, ExplanationArtifact
from api.models import Explanation, ExplanationPreparation, Intervention, ProcessingJob, Video
from api.processing_jobs import ProcessingJobManager, release_terminal_video_slot


VIDEO_ID = "FinalLife01"
VIDEO_TWO = "FinalLife02"


class FinalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = create_database_engine(
            f"sqlite:///{(self.root / 'final.sqlite').as_posix()}"
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.managers = []

    def seed(self, statuses, *, job_status="succeeded", video_id=VIDEO_ID):
        with self.Session() as db:
            db.add(Video(
                id=video_id, title="Lifecycle",
                source_url=f"https://www.youtube.com/watch?v={video_id}",
            ))
            db.add(ProcessingJob(
                id=str(uuid.uuid4()), video_id=video_id,
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                requested_output_mode="explanatory", status=job_status,
                stage={"succeeded": "complete", "running": "processing",
                       "queued": "queued", "cancelled": "cancelled",
                       "failed": "failed"}[job_status],
                active_slot=1,
            ))
            db.flush()
            for index, status in enumerate(statuses):
                intervention_id = f"point-{index}"
                db.add(Intervention(
                    id=intervention_id, video_id=video_id,
                    timestamp=float(index), frame_path=f"frames/{index}.jpg",
                ))
            db.flush()
            for index, status in enumerate(statuses):
                intervention_id = f"point-{index}"
                db.add(ExplanationPreparation(
                    video_id=video_id, intervention_id=intervention_id,
                    output_mode="explanatory", status=status,
                    stage=status, attempt_count=1,
                    lease_owner="owner" if status == "generating" else None,
                ))
            db.commit()

    def active_slot(self, video_id=VIDEO_ID):
        with self.Session() as db:
            return db.query(ProcessingJob).filter_by(video_id=video_id).one().active_slot

    def manager(self, *, evidence, generator):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.root),
            evidence_preparer=evidence,
            generator=generator,
            lease_timeout_seconds=0.3,
            maintenance_interval_seconds=0.05,
            shutdown_grace_seconds=0.03,
        )
        manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown, 0.5)
        return manager

    def test_all_terminal_statuses_release_slot_immediately(self):
        for status in ("ready", "failed", "cancelled"):
            with self.subTest(status=status):
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                self.seed([status])
                self.assertTrue(release_terminal_video_slot(self.Session, VIDEO_ID))
                self.assertIsNone(self.active_slot())

    def test_pending_or_generating_preparation_retains_slot(self):
        for status in ("pending", "generating"):
            with self.subTest(status=status):
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                self.seed(["ready", status])
                self.assertFalse(release_terminal_video_slot(self.Session, VIDEO_ID))
                self.assertEqual(self.active_slot(), 1)

    def test_running_or_queued_processing_retains_slot(self):
        for status in ("running", "queued"):
            with self.subTest(status=status):
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                self.seed(["ready"], job_status=status)
                self.assertFalse(release_terminal_video_slot(self.Session, VIDEO_ID))
                self.assertEqual(self.active_slot(), 1)

    def test_last_concurrent_terminal_transition_releases_without_premature_clear(self):
        self.seed(["generating", "generating"])
        first_done = threading.Event()
        allow_second = threading.Event()

        def finish(intervention_id, wait_for_second=False):
            if wait_for_second:
                first_done.wait(1)
                allow_second.wait(1)
            with self.Session() as db:
                state = db.query(ExplanationPreparation).filter_by(
                    intervention_id=intervention_id
                ).one()
                state.status = "failed"
                state.stage = "failed"
                state.lease_owner = None
                db.commit()
            released = release_terminal_video_slot(self.Session, VIDEO_ID)
            if not wait_for_second:
                first_done.set()
            return released

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(finish, "point-0")
            second = pool.submit(finish, "point-1", True)
            self.assertTrue(first_done.wait(1))
            self.assertFalse(first.result())
            self.assertEqual(self.active_slot(), 1)
            allow_second.set()
            self.assertTrue(second.result())
        self.assertIsNone(self.active_slot())

    def test_resume_race_cannot_clear_newly_acquired_slot(self):
        self.seed(["ready"], job_status="cancelled")
        with self.Session() as db:
            job_id = str(db.query(ProcessingJob).one().id)
        manager = ProcessingJobManager(self.Session, lambda: object())
        manager._submit = lambda _job_id: False
        self.addCleanup(manager.shutdown)
        barrier = threading.Barrier(2)

        def resume():
            barrier.wait()
            manager.resume(job_id)

        def release():
            barrier.wait()
            release_terminal_video_slot(self.Session, VIDEO_ID)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(resume), pool.submit(release)]
            for future in futures:
                future.result()
        with self.Session() as db:
            job = db.query(ProcessingJob).one()
            self.assertEqual((job.status, job.active_slot), ("queued", 1))

    def test_shutdown_during_blocked_vlm_is_bounded_and_late_result_is_fenced(self):
        self.seed(["pending"])
        entered = threading.Event()
        release = threading.Event()

        def evidence(**_kwargs):
            entered.set()
            release.wait(2)
            return EvidenceArtifact(None, None, False)

        calls = []
        manager = self.manager(evidence=evidence, generator=lambda **kwargs: calls.append(kwargs))
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        self.assertFalse(manager.shutdown())
        self.assertLess(time.monotonic() - started, 0.2)
        with self.assertRaisesRegex(RuntimeError, "not running"):
            manager.schedule_video(VIDEO_ID, "explanatory")
        release.set()
        self.assertTrue(manager.shutdown(timeout=1))
        self.assertEqual(calls, [])
        with self.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).one().status, "generating")
            self.assertEqual(db.query(Explanation).count(), 0)

    def test_shutdown_lease_recovers_once_after_expiry(self):
        self.seed(["pending"])
        entered = threading.Event()
        release = threading.Event()

        def blocked_evidence(**_kwargs):
            entered.set()
            release.wait(2)
            return EvidenceArtifact(None, None, False)

        original = self.manager(
            evidence=blocked_evidence,
            generator=lambda **_kwargs: None,
        )
        original.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(entered.wait(1))
        self.assertFalse(original.shutdown())
        release.set()
        self.assertTrue(original.shutdown(timeout=1))

        calls = []

        def recovered(**_kwargs):
            calls.append("recovered")
            audio = self.root / "recovered.mp3"
            audio.write_bytes(b"ID3-recovered")
            with self.Session() as db:
                db.add(Explanation(
                    intervention_id="point-0",
                    text_explanation="recovered",
                    audio_file_path=str(audio),
                    output_mode="explanatory",
                ))
                db.commit()
            return ExplanationArtifact(
                "point-0", "recovered", str(audio), None, None, "explanatory"
            )

        self.manager(
            evidence=lambda **_kwargs: EvidenceArtifact(None, None, False),
            generator=recovered,
        )
        self.manager(
            evidence=lambda **_kwargs: EvidenceArtifact(None, None, False),
            generator=recovered,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.Session() as db:
                state = db.query(ExplanationPreparation).one()
                if state.status == "ready":
                    break
            time.sleep(0.02)

        with self.Session() as db:
            state = db.query(ExplanationPreparation).one()
            # Attempt 2 was the interrupted owner; the recovered owner is 3.
            self.assertEqual((state.status, state.attempt_count), ("ready", 3))
        self.assertEqual(calls, ["recovered"])
        self.assertIsNone(self.active_slot())

    def test_shutdown_during_blocked_mistral_or_tts_fences_late_result(self):
        for stage in ("mistral", "tts"):
            with self.subTest(stage=stage):
                Base.metadata.drop_all(self.engine)
                Base.metadata.create_all(self.engine)
                self.seed(["pending"])
                entered = threading.Event()
                release = threading.Event()

                def generator(**kwargs):
                    if stage == "tts":
                        kwargs["on_stage"](STAGE_TTS)
                    entered.set()
                    release.wait(2)
                    return ExplanationArtifact(
                        "point-0", "late", None, None, None, "explanatory"
                    )

                manager = self.manager(
                    evidence=lambda **_kwargs: EvidenceArtifact(None, None, False),
                    generator=generator,
                )
                manager.schedule_video(VIDEO_ID, "explanatory")
                self.assertTrue(entered.wait(1))
                self.assertFalse(manager.shutdown())
                release.set()
                self.assertTrue(manager.shutdown(timeout=1))
                with self.Session() as db:
                    self.assertEqual(db.query(ExplanationPreparation).one().status,
                                     "generating")
                    self.assertEqual(db.query(Explanation).count(), 0)

    def test_queued_preparation_remains_durable_and_shutdown_is_idempotent(self):
        self.seed(["pending", "pending"])
        entered = threading.Event()
        release = threading.Event()

        def evidence(**_kwargs):
            entered.set()
            release.wait(2)
            return EvidenceArtifact(None, None, False)

        manager = self.manager(evidence=evidence, generator=lambda **_kwargs: None)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(entered.wait(1))
        self.assertFalse(manager.shutdown())
        release.set()
        self.assertTrue(manager.shutdown(timeout=1))
        self.assertTrue(manager.shutdown(timeout=0.1))
        with self.Session() as db:
            statuses = sorted(row.status for row in db.query(ExplanationPreparation).all())
            self.assertEqual(statuses, ["generating", "pending"])


if __name__ == "__main__":
    unittest.main()
