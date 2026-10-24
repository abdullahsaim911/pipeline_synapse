"""Persistent P2 explanation status, recovery, retry, and coordination tests."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.explanation_preparation import (
    ExplanationPreparationManager,
    PreparationRetryExhaustedError,
)
from api.explanation_service import ExplanationArtifact
from api.models import Explanation, ExplanationPreparation, Intervention, Video
from api.routers import video as video_module


VIDEO_ID = "P2Status001"
OTHER_VIDEO_ID = "P2Status002"


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class RecordingGenerator:
    """Fake generator that writes the same cache artifacts as production."""

    def __init__(self, data_dir, fail_ids=(), block_first=False):
        self.data_dir = Path(data_dir)
        self.fail_ids = set(fail_ids)
        self.block_first = block_first
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __call__(self, **kwargs):
        intervention_id = kwargs["intervention_id"]
        output_mode = kwargs["output_mode"]
        with self._lock:
            self.calls.append(intervention_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_number = len(self.calls)
        try:
            self.started.set()
            if self.block_first and call_number == 1:
                self.release.wait(3.0)
            if intervention_id in self.fail_ids:
                raise RuntimeError("private path C:/secret must never reach clients")

            audio = (
                self.data_dir / VIDEO_ID / intervention_id / f"{output_mode}.mp3"
            )
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"ID3-p2-fixture")
            with kwargs["session_factory"]() as db:
                row = db.query(Explanation).filter_by(
                    intervention_id=intervention_id,
                    output_mode=output_mode,
                ).first()
                if row is None:
                    row = Explanation(
                        intervention_id=intervention_id,
                        output_mode=output_mode,
                    )
                row.text_explanation = f"Prepared {intervention_id}"
                row.audio_file_path = str(audio)
                db.add(row)
                db.commit()
            return ExplanationArtifact(
                intervention_id=intervention_id,
                text_explanation=f"Prepared {intervention_id}",
                audio_file_path=str(audio),
                text_file_path=None,
                vlm_snapshot_path=None,
                output_mode=output_mode,
            )
        finally:
            with self._lock:
                self.active -= 1


class ControlledOrchestrator:
    """Orchestrator fake used with the real shared generation service."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_intervention_explanation(self, **kwargs):
        self.calls += 1
        self.started.set()
        self.release.wait(3.0)
        audio = (
            self.data_dir
            / kwargs["video_id"]
            / kwargs["intervention_id"]
            / f"{kwargs['output_mode']}.mp3"
        )
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"ID3-coordinated")
        return {
            "intervention_id": kwargs["intervention_id"],
            "text_explanation": "One coordinated explanation",
            "audio_file_path": str(audio),
            "text_file_path": None,
            "vlm_snapshot_path": None,
            "output_mode": kwargs["output_mode"],
        }


class PersistentPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"
        self.data_dir.mkdir()
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'p2.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def seed(self, timestamps=(30.0, 10.0, 20.0), video_id=VIDEO_ID):
        with self.Session() as db:
            db.add(Video(
                id=video_id,
                title="P2 lecture",
                source_url=f"https://www.youtube.com/watch?v={video_id}",
            ))
            for timestamp in timestamps:
                db.add(Intervention(
                    id=f"point-{video_id[-1]}-{int(timestamp)}",
                    video_id=video_id,
                    timestamp=timestamp,
                    frame_path=f"frames/{timestamp}.jpg",
                    content_type="diagram",
                ))
            db.commit()

    def manager(self, generator, **kwargs):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
            generator=generator,
            **kwargs,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def states(self, mode="explanatory"):
        with self.Session() as db:
            rows = db.query(ExplanationPreparation).filter_by(
                video_id=VIDEO_ID,
                output_mode=mode,
            ).all()
            return {row.intervention_id: (row.status, row.error_code, row.attempt_count) for row in rows}

    def add_valid_cache(self, intervention_id, mode="explanatory"):
        audio = self.data_dir / VIDEO_ID / intervention_id / f"{mode}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"ID3-valid-cache")
        with self.Session() as db:
            db.add(Explanation(
                intervention_id=intervention_id,
                text_explanation="Cached explanation",
                audio_file_path=str(audio),
                output_mode=mode,
            ))
            db.commit()
        return audio

    def test_new_jobs_persist_pending_then_transition_generating_to_ready_sequentially(self):
        self.seed((10.0, 20.0))
        generator = RecordingGenerator(self.data_dir, block_first=True)
        manager = self.manager(generator)

        self.assertEqual(manager.schedule_video(VIDEO_ID, "explanatory"), 2)
        self.assertTrue(generator.started.wait(1.0))
        current = self.states()
        self.assertEqual(current["point-1-10"][0], "generating")
        self.assertEqual(current["point-1-20"][0], "pending")
        generator.release.set()

        self.assertTrue(wait_until(
            lambda: all(value[0] == "ready" for value in self.states().values())
        ))
        self.assertEqual(generator.calls, ["point-1-10", "point-1-20"])
        self.assertEqual(generator.max_active, 1)
        self.assertEqual(
            [value[2] for value in self.states().values()], [1, 1]
        )

    def test_failure_is_safe_and_does_not_block_next_intervention(self):
        self.seed((10.0, 20.0))
        generator = RecordingGenerator(self.data_dir, fail_ids={"point-1-10"})
        manager = self.manager(generator)
        manager.schedule_video(VIDEO_ID, "explanatory")

        self.assertTrue(wait_until(
            lambda: len(self.states()) == 2
            and self.states()["point-1-10"][0] == "failed"
            and self.states()["point-1-20"][0] == "ready"
        ))
        failed = self.states()["point-1-10"]
        self.assertEqual(failed[1], "generation_failed")
        self.assertNotIn("secret", failed[1])

    def test_valid_cache_is_reused_but_missing_mp3_is_not_ready(self):
        self.seed((10.0, 20.0))
        self.add_valid_cache("point-1-10")
        with self.Session() as db:
            db.add(Explanation(
                intervention_id="point-1-20",
                text_explanation="Text without playable audio",
                audio_file_path=str(self.data_dir / "missing.mp3"),
                output_mode="explanatory",
            ))
            db.commit()

        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
            generator=mock.Mock(),
        )
        status = {item.intervention_id: item for item in manager.get_status(VIDEO_ID, "explanatory")}
        self.assertEqual(status["point-1-10"].status, "ready")
        self.assertEqual(status["point-1-20"].status, "failed")
        self.assertEqual(status["point-1-20"].error_code, "missing_audio")

        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        self.assertEqual(manager.schedule_video(VIDEO_ID, "explanatory"), 1)
        self.assertTrue(wait_until(lambda: manager._generator.call_count == 1))

    def test_duplicate_scheduling_creates_one_state_and_one_active_job(self):
        self.seed((10.0,))
        generator = RecordingGenerator(self.data_dir, block_first=True)
        manager = self.manager(generator)
        self.assertEqual(manager.schedule_video(VIDEO_ID, "brief"), 1)
        self.assertTrue(generator.started.wait(1.0))
        self.assertEqual(manager.schedule_video(VIDEO_ID, "brief"), 0)
        with self.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).count(), 1)
        generator.release.set()
        self.assertTrue(wait_until(lambda: self.states("brief")["point-1-10"][0] == "ready"))
        self.assertEqual(generator.calls, ["point-1-10"])

    def test_manual_and_background_share_one_inference(self):
        self.seed((10.0,))
        orchestrator = ControlledOrchestrator(self.data_dir)
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: orchestrator,
            data_dir=str(self.data_dir),
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(orchestrator.started.wait(1.0))

        result = {}
        manual = threading.Thread(
            target=lambda: result.setdefault(
                "artifact",
                manager.generate_manual(
                    VIDEO_ID, "point-1-10", "explanatory"
                ),
            )
        )
        manual.start()
        time.sleep(0.05)
        self.assertTrue(manual.is_alive())
        orchestrator.release.set()
        manual.join(2.0)

        self.assertFalse(manual.is_alive())
        self.assertEqual(orchestrator.calls, 1)
        self.assertEqual(result["artifact"].text_explanation, "One coordinated explanation")
        self.assertTrue(wait_until(lambda: self.states()["point-1-10"][0] == "ready"))

    def test_explicit_retry_succeeds_and_limit_is_enforced(self):
        self.seed((10.0,))
        generator = RecordingGenerator(self.data_dir, fail_ids={"point-1-10"})
        manager = self.manager(generator, max_attempts=2)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.states()["point-1-10"][0] == "failed"))

        generator.fail_ids.clear()
        retry = manager.retry(VIDEO_ID, "point-1-10", "explanatory")
        self.assertTrue(retry.queued)
        self.assertEqual(retry.status, "pending")
        self.assertTrue(wait_until(lambda: self.states()["point-1-10"][0] == "ready"))
        self.assertEqual(self.states()["point-1-10"][2], 2)

        # A separate exhausted key cannot be queued beyond its cap.
        self.seed((10.0,), video_id=OTHER_VIDEO_ID)
        exhausted_generator = RecordingGenerator(
            self.data_dir, fail_ids={"point-2-10"}
        )
        exhausted = self.manager(exhausted_generator, max_attempts=1)
        exhausted.schedule_video(OTHER_VIDEO_ID, "brief")
        self.assertTrue(wait_until(lambda: any(
            item.status == "failed"
            for item in exhausted.get_status(OTHER_VIDEO_ID, "brief")
        )))
        with self.assertRaises(PreparationRetryExhaustedError):
            exhausted.retry(OTHER_VIDEO_ID, "point-2-10", "brief")

    def test_startup_recovers_only_tracked_interrupted_work(self):
        self.seed((10.0, 20.0))
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_ID,
                intervention_id="point-1-10",
                output_mode="detailed",
                status="generating",
                attempt_count=1,
                lease_owner="dead-worker",
                lease_started_at=(
                    datetime.now(timezone.utc).replace(tzinfo=None)
                    - timedelta(hours=1)
                ),
            ))
            db.commit()

        generator = RecordingGenerator(self.data_dir)
        manager = self.manager(generator)
        self.assertTrue(wait_until(lambda: self.states("detailed")["point-1-10"][0] == "ready"))
        self.assertEqual(generator.calls, ["point-1-10"])
        self.assertEqual(self.states("detailed")["point-1-10"][2], 2)
        self.assertNotIn("point-1-20", self.states("detailed"))

    def test_new_table_is_added_without_altering_existing_tables(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite"
        legacy_engine = create_engine(f"sqlite:///{legacy_path}")
        self.addCleanup(legacy_engine.dispose)
        Video.__table__.create(legacy_engine)
        Intervention.__table__.create(legacy_engine)
        Explanation.__table__.create(legacy_engine)
        LegacySession = sessionmaker(bind=legacy_engine)
        with LegacySession() as db:
            db.add(Video(
                id="Legacy00001",
                title="Existing database row",
                source_url="https://www.youtube.com/watch?v=Legacy00001",
            ))
            db.commit()

        Base.metadata.create_all(legacy_engine)
        tables = set(inspect(legacy_engine).get_table_names())
        self.assertIn("videos", tables)
        self.assertIn("interventions", tables)
        self.assertIn("explanations", tables)
        self.assertIn("explanation_preparations", tables)
        with LegacySession() as db:
            self.assertEqual(db.get(Video, "Legacy00001").title, "Existing database row")


class PreparationEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"
        self.data_dir.mkdir()
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'endpoints.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def seed(self, timestamps=(30.0, 10.0, 20.0)):
        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="P2 endpoint lecture",
                source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            ))
            for timestamp in timestamps:
                db.add(Intervention(
                    id=f"point-1-{int(timestamp)}",
                    video_id=VIDEO_ID,
                    timestamp=timestamp,
                    frame_path=f"frames/{timestamp}.jpg",
                    content_type="diagram",
                ))
            db.commit()

    def manager(self, generator):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
            generator=generator,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def add_valid_cache(self, intervention_id, mode="explanatory"):
        audio = self.data_dir / VIDEO_ID / intervention_id / f"{mode}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"ID3-valid-cache")
        with self.Session() as db:
            db.add(Explanation(
                intervention_id=intervention_id,
                text_explanation="Cached explanation",
                audio_file_path=str(audio),
                output_mode=mode,
            ))
            db.commit()

    def make_client(self, manager):
        app = FastAPI()
        app.state.explanation_preparation = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        app.dependency_overrides[video_module.get_data_dir] = lambda: str(self.data_dir)
        return TestClient(app, raise_server_exceptions=False)

    def test_status_endpoint_is_ordered_mode_specific_and_small(self):
        self.seed((30.0, 10.0, 20.0))
        self.add_valid_cache("point-1-20", mode="explanatory")
        with self.Session() as db:
            db.add_all([
                ExplanationPreparation(
                    video_id=VIDEO_ID,
                    intervention_id="point-1-10",
                    output_mode="explanatory",
                    status="failed",
                    attempt_count=2,
                    error_code="generation_failed",
                ),
                ExplanationPreparation(
                    video_id=VIDEO_ID,
                    intervention_id="point-1-30",
                    output_mode="brief",
                    status="failed",
                    attempt_count=1,
                    error_code="generation_failed",
                ),
            ])
            db.commit()
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
        )
        with self.make_client(manager) as client:
            response = client.get(
                f"/video/{VIDEO_ID}/explanations/status?output_mode=explanatory"
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["video_id"], VIDEO_ID)
        self.assertEqual(body["output_mode"], "explanatory")
        self.assertEqual(
            [item["timestamp"] for item in body["items"]], [10.0, 20.0, 30.0]
        )
        self.assertEqual(
            [item["status"] for item in body["items"]],
            ["failed", "ready", "pending"],
        )
        self.assertNotIn("text_explanation", response.text)
        self.assertNotIn("audio", response.text)

    def test_retry_endpoint_is_nonblocking_and_validates_inputs(self):
        self.seed((10.0,))
        generator = RecordingGenerator(self.data_dir, block_first=True)
        manager = self.manager(generator)
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_ID,
                intervention_id="point-1-10",
                output_mode="explanatory",
                status="failed",
                attempt_count=1,
                error_code="generation_failed",
            ))
            db.commit()

        with self.make_client(manager) as client:
            before = time.monotonic()
            response = client.post(
                f"/video/{VIDEO_ID}/interventions/point-1-10/explanations/retry"
                "?output_mode=explanatory"
            )
            elapsed = time.monotonic() - before
            invalid_mode = client.get(
                f"/video/{VIDEO_ID}/explanations/status?output_mode=standard"
            )
            invalid_video = client.get(
                "/video/not-valid/explanations/status?output_mode=brief"
            )
            invalid_intervention = client.post(
                f"/video/{VIDEO_ID}/interventions/bad$id/explanations/retry"
            )
            unknown_video = client.get(
                "/video/Missing0001/explanations/status?output_mode=brief"
            )
            unknown_intervention = client.post(
                f"/video/{VIDEO_ID}/interventions/not-a-real-id/explanations/retry"
            )

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "pending")
        self.assertLess(elapsed, 0.5)
        self.assertEqual(invalid_mode.status_code, 422)
        self.assertEqual(invalid_video.status_code, 422)
        self.assertEqual(invalid_intervention.status_code, 422)
        self.assertEqual(unknown_video.status_code, 404)
        self.assertEqual(unknown_intervention.status_code, 404)
        self.assertTrue(generator.started.wait(1.0))
        generator.release.set()

    def test_retry_endpoint_reports_exhaustion_without_queueing(self):
        self.seed((10.0,))
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=str(self.data_dir),
            generator=mock.Mock(),
            max_attempts=1,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        with self.Session() as db:
            db.add(ExplanationPreparation(
                video_id=VIDEO_ID,
                intervention_id="point-1-10",
                output_mode="brief",
                status="failed",
                attempt_count=1,
                error_code="generation_failed",
            ))
            db.commit()

        with self.make_client(manager) as client:
            response = client.post(
                f"/video/{VIDEO_ID}/interventions/point-1-10/explanations/retry"
                "?output_mode=brief"
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], {
            "code": "retry_exhausted",
            "attempt_count": 1,
        })
        manager._generator.assert_not_called()

    def test_manual_failure_records_safe_state_and_hides_internal_details(self):
        self.seed((10.0,))
        generator = RecordingGenerator(
            self.data_dir, fail_ids={"point-1-10"}
        )
        manager = self.manager(generator)

        with self.make_client(manager) as client:
            response = client.post("/video/intervention/explain", json={
                "video_id": VIDEO_ID,
                "intervention_id": "point-1-10",
                "output_mode": "explanatory",
            })

        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["detail"], "Explanation generation failed.")
        self.assertNotIn("secret", response.text)
        status = manager.get_status(VIDEO_ID, "explanatory")[0]
        self.assertEqual(status.status, "failed")
        self.assertEqual(status.error_code, "generation_failed")


if __name__ == "__main__":
    unittest.main()
