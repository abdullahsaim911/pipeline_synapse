from explanation_cache import generation_manifest, write_manifest
"""Deterministic video cancellation, fencing, resume, and admission tests."""

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine, get_db
from api.explanation_preparation import (
    ExplanationPreparationManager, PreparationJob, _Lease,
)
from api.explanation_service import (
    EvidenceArtifact, ExplanationArtifact, PreparationCancelledError,
)
from api.models import Explanation, ExplanationPreparation, Intervention, ProcessingJob, Transcript, Video
from api.processing_jobs import ProcessingJobBusyError, ProcessingJobManager
from api.routers import video as video_module
from orchestrator.orchestrator import PipelineOrchestrator
from tests.test_evidence_reconciliation import _element, _evidence


VIDEO_A = "CancelVid01"
VIDEO_B = "CancelVid02"


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def detection_result(root: Path, video_id: str) -> dict:
    return {
        "video_id": video_id,
        "video_dir": str(root / video_id),
        "title": "Cancellation test",
        "duration_seconds": 30,
        "duration_formatted": "00:30",
        "metadata": {"total_keyframes": 1},
        "transcript": {"entries": [{"start": 0, "end": 4, "text": "Transcript"}]},
        "intervention_points": [SimpleNamespace(
            timestamp=2.0, frame_path="frames/2.jpg", content_type="diagram",
            complexity_score=0.8, clip_confidence=0.9,
            trigger_reason="Visual", confidence="high", transcript_context="Transcript",
        )],
    }


class FencedOrchestrator:
    def __init__(self, root: Path):
        self.root = root
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    @staticmethod
    def _extract_video_id(url):
        return url.rsplit("=", 1)[-1]

    def process_detection_only(self, url, should_continue=None):
        self.calls += 1
        video_id = self._extract_video_id(url)
        self.started.set()
        self.release.wait(3)
        # The manager owns the authoritative post-call fence.
        return detection_result(self.root, video_id)


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = create_database_engine(
            f"sqlite:///{(self.root / 'cancel.sqlite').as_posix()}"
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.managers = []

    def processing_manager(self, orchestrator, preparation=None):
        manager = ProcessingJobManager(
            self.Session, lambda: orchestrator, preparation_manager=preparation,
            heartbeat_seconds=0.02,
        )
        manager.start()
        self.managers.append(manager)
        self.addCleanup(manager.shutdown)
        return manager

    def seed_interventions(self, count=2):
        with self.Session() as db:
            db.add(Video(id=VIDEO_A, title="Lecture", source_url=f"https://youtu.be/{VIDEO_A}"))
            for index in range(count):
                db.add(Intervention(
                    id=f"ip-{index}", video_id=VIDEO_A, timestamp=float(index),
                    frame_path=f"frames/{index}.jpg", content_type="diagram",
                ))
            db.commit()

    def preparation_manager(self, evidence_preparer, generator):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.root),
            evidence_preparer=evidence_preparer,
            generator=generator,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def test_cancel_inflight_processing_discards_late_result_and_is_idempotent(self):
        orchestrator = FencedOrchestrator(self.root)
        manager = self.processing_manager(orchestrator)
        created = manager.create_or_join(
            f"https://www.youtube.com/watch?v={VIDEO_A}", "brief"
        )
        self.assertTrue(orchestrator.started.wait(1))

        first = manager.cancel(created["job_id"])
        second = manager.cancel(created["job_id"])
        self.assertEqual(first["status"], "cancelling")
        self.assertEqual(second["status"], "cancelling")
        orchestrator.release.set()
        self.assertTrue(wait_until(lambda: manager.get(created["job_id"])["status"] == "cancelled"))

        with self.Session() as db:
            self.assertIsNone(db.get(Video, VIDEO_A))
            self.assertEqual(db.query(Intervention).count(), 0)

    def test_cancel_then_immediate_resume_fences_old_attempt(self):
        orchestrator = FencedOrchestrator(self.root)
        manager = self.processing_manager(orchestrator)
        created = manager.create_or_join(
            f"https://www.youtube.com/watch?v={VIDEO_A}", "explanatory"
        )
        self.assertTrue(orchestrator.started.wait(1))
        manager.cancel(created["job_id"])
        resumed = manager.resume(created["job_id"])
        self.assertEqual(resumed["status"], "queued")
        orchestrator.release.set()

        final = manager.wait(created["job_id"], timeout=4)
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(final["attempt_count"], 2)
        self.assertEqual(orchestrator.calls, 2)
        with self.Session() as db:
            self.assertEqual(db.query(Video).count(), 1)

    def test_different_video_is_rejected_while_first_is_active(self):
        orchestrator = FencedOrchestrator(self.root)
        manager = self.processing_manager(orchestrator)
        manager.create_or_join(f"https://www.youtube.com/watch?v={VIDEO_A}", "brief")
        self.assertTrue(orchestrator.started.wait(1))
        with self.assertRaises(ProcessingJobBusyError) as raised:
            manager.create_or_join(f"https://www.youtube.com/watch?v={VIDEO_B}", "brief")
        self.assertEqual(raised.exception.video_id, VIDEO_A)
        orchestrator.release.set()

    def test_concurrent_different_video_admission_has_one_durable_winner(self):
        orchestrator = FencedOrchestrator(self.root)
        first = self.processing_manager(orchestrator)
        second = self.processing_manager(orchestrator)

        def submit(manager, video_id):
            try:
                return manager.create_or_join(
                    f"https://www.youtube.com/watch?v={video_id}", "brief"
                )
            except ProcessingJobBusyError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [
                executor.submit(submit, first, VIDEO_A),
                executor.submit(submit, second, VIDEO_B),
            ]
            values = [future.result(timeout=3) for future in results]

        self.assertEqual(sum(isinstance(value, dict) for value in values), 1)
        self.assertEqual(sum(isinstance(value, ProcessingJobBusyError) for value in values), 1)
        with self.Session() as db:
            self.assertEqual(db.query(ProcessingJob).filter_by(active_slot=1).count(), 1)
        orchestrator.release.set()

    def test_cancel_and_resume_routes_expose_durable_states(self):
        orchestrator = FencedOrchestrator(self.root)
        manager = self.processing_manager(orchestrator)
        app = FastAPI()
        app.state.processing_jobs = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with TestClient(app) as client:
            created = client.post("/video/processing-jobs", json={
                "youtube_url": f"https://www.youtube.com/watch?v={VIDEO_A}",
                "output_mode": "brief",
            }).json()
            self.assertTrue(orchestrator.started.wait(1))
            cancelled = client.post(f"/video/processing-jobs/{created['job_id']}/cancel")
            self.assertEqual(cancelled.status_code, 202)
            self.assertEqual(cancelled.json()["status"], "cancelling")
            resumed = client.post(f"/video/processing-jobs/{created['job_id']}/resume")
            self.assertEqual(resumed.status_code, 202)
            self.assertEqual(resumed.json()["status"], "queued")
            orchestrator.release.set()

    def test_processing_route_returns_conflict_for_second_video(self):
        orchestrator = FencedOrchestrator(self.root)
        manager = self.processing_manager(orchestrator)
        app = FastAPI()
        app.state.processing_jobs = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with TestClient(app) as client:
            first = client.post("/video/processing-jobs", json={
                "youtube_url": f"https://www.youtube.com/watch?v={VIDEO_A}",
                "output_mode": "brief",
            })
            self.assertEqual(first.status_code, 202)
            self.assertTrue(orchestrator.started.wait(1))
            conflict = client.post("/video/processing-jobs", json={
                "youtube_url": f"https://www.youtube.com/watch?v={VIDEO_B}",
                "output_mode": "brief",
            })
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["detail"]["code"], "video_processing_busy")
            self.assertEqual(conflict.json()["detail"]["active_video_id"], VIDEO_A)
            orchestrator.release.set()

    def test_cancel_before_next_vlm_dispatch_and_late_vlm_is_discarded(self):
        self.seed_interventions(2)
        started = threading.Event()
        release = threading.Event()
        calls = []

        def evidence(**kwargs):
            calls.append(kwargs["intervention_id"])
            started.set()
            release.wait(3)
            return EvidenceArtifact({"id": kwargs["intervention_id"]}, "snapshot.json", False)

        def generator(**kwargs):
            raise AssertionError("Cancelled evidence must not reach Mistral")

        manager = self.preparation_manager(evidence, generator)
        manager.schedule_video(VIDEO_A, "brief")
        self.assertTrue(started.wait(1))
        self.assertEqual(manager.cancel_video(VIDEO_A), 2)
        release.set()
        self.assertTrue(wait_until(lambda: not manager.has_inflight_video(VIDEO_A)))

        self.assertEqual(calls, ["ip-0"])
        with self.Session() as db:
            states = db.query(ExplanationPreparation).order_by(
                ExplanationPreparation.intervention_id
            ).all()
            self.assertEqual([state.status for state in states], ["cancelled", "cancelled"])
            self.assertEqual(db.query(Explanation).count(), 0)

    def test_cancel_during_mistral_discards_late_generation(self):
        self.seed_interventions(1)
        started = threading.Event()
        release = threading.Event()

        def evidence(**kwargs):
            return EvidenceArtifact({"id": "ip-0"}, "snapshot.json", False)

        def generator(**kwargs):
            started.set()
            release.wait(3)
            return ExplanationArtifact("ip-0", "late text", None, None, None, "brief")

        manager = self.preparation_manager(evidence, generator)
        manager.schedule_video(VIDEO_A, "brief")
        self.assertTrue(started.wait(1))
        manager.cancel_video(VIDEO_A)
        release.set()
        self.assertTrue(wait_until(lambda: not manager.has_inflight_video(VIDEO_A)))
        with self.Session() as db:
            state = db.query(ExplanationPreparation).one()
            self.assertEqual(state.status, "cancelled")
            self.assertEqual(db.query(Explanation).count(), 0)

    def test_old_mistral_after_resume_cannot_strand_fresh_attempt(self):
        self.seed_interventions(1)
        first_started = threading.Event()
        release_first = threading.Event()
        calls = 0

        def evidence(**kwargs):
            return EvidenceArtifact({"id": "ip-0"}, "snapshot.json", False)

        def generator(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                release_first.wait(3)
            else:
                audio = self.root / "resumed.mp3"
                audio.write_bytes(b"ID3-resumed")
                write_manifest(audio, generation_manifest("Fresh result", audio))
                with self.Session() as db:
                    db.add(Explanation(
                        intervention_id="ip-0", text_explanation="Fresh result",
                        audio_file_path=str(audio), output_mode="brief",
                    ))
                    db.commit()
            return ExplanationArtifact(
                "ip-0", "Fresh result", str(self.root / "resumed.mp3"),
                None, None, "brief",
            )

        manager = self.preparation_manager(evidence, generator)
        manager.schedule_video(VIDEO_A, "brief")
        self.assertTrue(first_started.wait(1))
        manager.cancel_video(VIDEO_A)
        manager.resume_video(VIDEO_A, ["brief"])
        release_first.set()

        self.assertTrue(wait_until(
            lambda: manager.get_status(VIDEO_A, "brief")[0].status == "ready",
            timeout=4,
        ))
        self.assertEqual(calls, 2)

    def test_valid_ready_audio_is_preserved_by_video_cancel(self):
        self.seed_interventions(1)
        audio = self.root / "ready.mp3"
        audio.write_bytes(b"ID3-valid")
        write_manifest(audio, generation_manifest("Complete", audio))
        with self.Session() as db:
            db.add(Explanation(
                intervention_id="ip-0", text_explanation="Complete",
                audio_file_path=str(audio), output_mode="brief",
            ))
            db.add(ExplanationPreparation(
                video_id=VIDEO_A, intervention_id="ip-0", output_mode="brief",
                status="ready", stage="ready", attempt_count=1,
            ))
            db.commit()
        manager = self.preparation_manager(lambda **kwargs: None, lambda **kwargs: None)

        self.assertEqual(manager.cancel_video(VIDEO_A), 0)
        self.assertTrue(audio.is_file())
        with self.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).one().status, "ready")
            self.assertEqual(db.query(Explanation).count(), 1)

    def test_cancellation_winning_checkpoint_race_prevents_file_commit(self):
        self.seed_interventions(1)
        manager = ExplanationPreparationManager(
            session_factory=self.Session, orchestrator_factory=lambda: object(),
            data_dir=str(self.root), generator=lambda **kwargs: None,
            evidence_preparer=lambda **kwargs: None,
        )
        with self.Session() as db:
            state = ExplanationPreparation(
                video_id=VIDEO_A, intervention_id="ip-0", output_mode="brief",
                status="generating", stage="extracting_evidence", attempt_count=1,
                lease_owner="old-owner",
            )
            db.add(state)
            db.commit()
            db.refresh(state)
            lease = _Lease("old-owner", str(state.id), 1)
        manager.cancel_video(VIDEO_A)
        target = self.root / "snapshot.json"

        with self.assertRaises(PreparationCancelledError):
            manager._commit_checkpoint(
                PreparationJob(VIDEO_A, "ip-0", "brief"), lease,
                lambda: target.write_text("late", encoding="utf-8"),
            )
        self.assertFalse(target.exists())

    def test_checkpoint_winning_race_is_preserved_before_cancel_accepts(self):
        self.seed_interventions(1)
        manager = ExplanationPreparationManager(
            session_factory=self.Session, orchestrator_factory=lambda: object(),
            data_dir=str(self.root), generator=lambda **kwargs: None,
            evidence_preparer=lambda **kwargs: None,
        )
        with self.Session() as db:
            state = ExplanationPreparation(
                video_id=VIDEO_A, intervention_id="ip-0", output_mode="brief",
                status="generating", stage="extracting_evidence", attempt_count=1,
                lease_owner="owner",
            )
            db.add(state)
            db.commit()
            db.refresh(state)
            lease = _Lease("owner", str(state.id), 1)
        target = self.root / "snapshot.json"
        action_started = threading.Event()
        release_action = threading.Event()

        def action():
            action_started.set()
            release_action.wait(2)
            target.write_text("valid", encoding="utf-8")

        commit_thread = threading.Thread(target=lambda: manager._commit_checkpoint(
            PreparationJob(VIDEO_A, "ip-0", "brief"), lease, action
        ))
        commit_thread.start()
        self.assertTrue(action_started.wait(1))
        cancel_attempted = threading.Event()

        def cancel_video():
            cancel_attempted.set()
            manager.cancel_video(VIDEO_A)

        cancel_thread = threading.Thread(target=cancel_video)
        cancel_thread.start()
        self.assertTrue(cancel_attempted.wait(1))
        self.assertTrue(cancel_thread.is_alive())
        release_action.set()
        commit_thread.join(2)
        cancel_thread.join(2)

        self.assertEqual(target.read_text(encoding="utf-8"), "valid")
        with self.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).one().status, "cancelled")

    def test_restart_settles_durable_pending_cancellation(self):
        self.seed_interventions(1)
        job_id = "11111111-1111-4111-8111-111111111111"
        with self.Session() as db:
            db.add(ProcessingJob(
                id=job_id, video_id=VIDEO_A,
                source_url=f"https://www.youtube.com/watch?v={VIDEO_A}",
                requested_output_mode="brief", status="cancelling", stage="cancelling",
                attempt_count=1,
                cancel_requested_at=datetime.now(timezone.utc).replace(tzinfo=None),
            ))
            db.add(ExplanationPreparation(
                video_id=VIDEO_A, intervention_id="ip-0", output_mode="brief",
                status="generating", stage="generating_explanation", attempt_count=1,
                lease_owner="dead-worker",
            ))
            db.commit()
        preparation = ExplanationPreparationManager(
            session_factory=self.Session, orchestrator_factory=lambda: object(),
            data_dir=str(self.root), generator=lambda **kwargs: None,
            evidence_preparer=lambda **kwargs: None,
        )
        preparation.start()
        self.addCleanup(preparation.shutdown, 1.0)
        manager = self.processing_manager(FencedOrchestrator(self.root), preparation)

        self.assertEqual(manager.get(job_id)["status"], "cancelled")
        with self.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).one().status, "cancelled")

    def test_cancel_during_tts_removes_temporary_audio_and_keeps_text_checkpoint(self):
        orchestrator = PipelineOrchestrator(base_dir=str(self.root))
        (self.root / VIDEO_A / "interventions" / "ip-tts").mkdir(parents=True)
        snapshot = {
            "visual_evidence": _evidence(
                "CT-06", elements=[_element("label", "Visible label", "CT-06")]
            ),
            "detected_types": ["text"],
        }

        class FakeSynthesizer:
            def __init__(self, **kwargs):
                pass

            def weave(self, **kwargs):
                return "A complete saved text explanation."

            def cleanup(self):
                pass

        class FakeTTS:
            def generate(self, _text, path):
                Path(path).write_bytes(b"ID3-incomplete-uncommitted")

            def cleanup(self):
                pass

        class ReadyOllama:
            def ensure_ready(self):
                return None

        commits = 0

        def checkpoint(action):
            nonlocal commits
            commits += 1
            if commits == 3:
                raise PreparationCancelledError("cancel won audio commit")
            action()

        with (
            mock.patch("synthesizer.LLMSynthesizer", FakeSynthesizer),
            mock.patch("tts_engine.TTSEngine", FakeTTS),
            mock.patch("orchestrator.orchestrator.get_ollama_service", return_value=ReadyOllama()),
        ):
            with self.assertRaises(PreparationCancelledError):
                orchestrator.synthesize_intervention_explanation(
                    video_id=VIDEO_A,
                    intervention_id="ip-tts",
                    intervention_timestamp=1.0,
                    content_type="text",
                    trigger_reason="Visual content",
                    transcript_entries=[],
                    output_mode="brief",
                    vlm_snapshot=snapshot,
                    ownership_check=lambda: None,
                    checkpoint_commit=checkpoint,
                )

        directory = self.root / VIDEO_A / "interventions" / "ip-tts"
        self.assertTrue((directory / "brief.txt").is_file())
        self.assertFalse((directory / "brief.mp3").exists())
        self.assertFalse((directory / "brief.mp3.tmp.mp3").exists())


if __name__ == "__main__":
    unittest.main()
