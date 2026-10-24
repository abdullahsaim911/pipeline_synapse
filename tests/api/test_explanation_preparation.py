"""Deterministic P1 background explanation preparation tests."""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.explanation_preparation import ExplanationPreparationManager
from api.models import Explanation, Intervention, Transcript, Video
from api.routers import video as video_module


VIDEO_ID = "Prepare1234"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeExplanationOrchestrator:
    def __init__(self, data_dir):
        self.calls = []
        self.data_dir = Path(data_dir)

    def generate_intervention_explanation(self, **kwargs):
        self.calls.append(kwargs)
        mode = kwargs["output_mode"]
        intervention_id = kwargs["intervention_id"]
        audio = self.data_dir / VIDEO_ID / intervention_id / f"{mode}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"ID3-synthetic")
        return {
            "intervention_id": intervention_id,
            "text_explanation": f"Prepared {intervention_id}",
            "audio_file_path": f"data/{VIDEO_ID}/{intervention_id}/{mode}.mp3",
            "text_file_path": f"data/{VIDEO_ID}/{intervention_id}/{mode}.txt",
            "vlm_snapshot_path": f"data/{VIDEO_ID}/{intervention_id}/snapshot.json",
            "output_mode": mode,
        }


class ExplanationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"
        transcript_dir = self.data_dir / VIDEO_ID
        transcript_dir.mkdir(parents=True)
        (transcript_dir / "transcript.json").write_text(
            json.dumps({"entries": [{"start": 0, "duration": 5, "text": "Synthetic lecture"}]}),
            encoding="utf-8",
        )

        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'test.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def seed(self, timestamps=(30.0, 10.0, 20.0)):
        with self.Session() as db:
            db.add(Video(id=VIDEO_ID, title="Prepared lecture", source_url=VIDEO_URL))
            for timestamp in timestamps:
                db.add(Intervention(
                    id=f"point-{int(timestamp)}",
                    video_id=VIDEO_ID,
                    timestamp=timestamp,
                    frame_path=f"frames/{timestamp}.jpg",
                    content_type="diagram",
                    clip_confidence=0.9,
                ))
            db.commit()

    def manager(self, generator, orchestrator_factory=lambda: object()):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=orchestrator_factory,
            data_dir=str(self.data_dir),
            generator=generator,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def test_fifo_timestamp_order_single_concurrency_dedup_and_failure_continuation(self):
        self.seed()
        calls = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def generate(**kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(kwargs["intervention_id"])
            try:
                time.sleep(0.02)
                if kwargs["intervention_id"] == "point-20":
                    raise RuntimeError("controlled failure")
            finally:
                with lock:
                    active -= 1

        manager = self.manager(generate)
        self.assertEqual(manager.schedule_video(VIDEO_ID, "explanatory"), 3)
        self.assertEqual(manager.schedule_video(VIDEO_ID, "explanatory"), 0)
        self.assertTrue(wait_until(lambda: len(calls) == 3))
        self.assertEqual(calls, ["point-10", "point-20", "point-30"])
        self.assertEqual(max_active, 1)

    def test_real_service_persists_cache_and_reopen_schedules_nothing(self):
        self.seed((10.0,))
        orchestrator = FakeExplanationOrchestrator(self.data_dir)
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: orchestrator,
            data_dir=str(self.data_dir),
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)

        self.assertEqual(manager.schedule_video(VIDEO_ID, "brief"), 1)
        self.assertTrue(wait_until(lambda: len(orchestrator.calls) == 1))
        self.assertTrue(wait_until(lambda: self._explanation_count() == 1))
        self.assertEqual(manager.schedule_video(VIDEO_ID, "brief"), 0)
        with self.Session() as db:
            stored = db.query(Explanation).one()
            self.assertEqual(stored.intervention_id, "point-10")
            self.assertEqual(stored.output_mode, "brief")
        self.assertEqual(len(orchestrator.calls), 1)

    def _explanation_count(self):
        with self.Session() as db:
            return db.query(Explanation).count()

    def test_cached_mode_is_skipped_but_another_mode_is_scheduled(self):
        self.seed((10.0,))
        cached_audio = self.data_dir / "cached.mp3"
        cached_audio.write_bytes(b"ID3-cached")
        with self.Session() as db:
            db.add(Explanation(
                intervention_id="point-10",
                text_explanation="Cached",
                audio_file_path=str(cached_audio),
                output_mode="explanatory",
            ))
            db.commit()

        calls = []
        manager = self.manager(lambda **kwargs: calls.append(kwargs["output_mode"]))
        self.assertEqual(manager.schedule_video(VIDEO_ID, "explanatory"), 0)
        self.assertEqual(manager.schedule_video(VIDEO_ID, "detailed"), 1)
        self.assertTrue(wait_until(lambda: calls == ["detailed"]))

    def test_shutdown_is_bounded_while_generation_is_active(self):
        self.seed((10.0, 20.0))
        started = threading.Event()
        release = threading.Event()

        def slow_generate(**_kwargs):
            started.set()
            release.wait(2.0)

        manager = self.manager(slow_generate)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(started.wait(1.0))
        before = time.monotonic()
        manager.shutdown(timeout=0.05)
        self.assertLess(time.monotonic() - before, 0.5)
        release.set()

    def test_manual_endpoint_generates_once_and_keeps_response_contract(self):
        self.seed((10.0,))
        orchestrator = FakeExplanationOrchestrator(self.data_dir)
        app = FastAPI()
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        app.dependency_overrides[video_module.get_data_dir] = lambda: str(self.data_dir)
        with mock.patch.object(video_module, "get_orchestrator", return_value=orchestrator):
            with TestClient(app) as client:
                payload = {
                    "video_id": VIDEO_ID,
                    "intervention_id": "point-10",
                    "output_mode": "detailed",
                }
                first = client.post("/video/intervention/explain", json=payload)
                second = client.post("/video/intervention/explain", json=payload)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(len(orchestrator.calls), 1)
        self.assertEqual(first.json()["output_mode"], "detailed")
        self.assertEqual(
            first.json()["audio_url"],
            "/video/intervention/point-10/audio?output_mode=detailed",
        )


class ProcessSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'process.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def test_process_commits_before_nonblocking_schedule_with_selected_mode(self):
        point = SimpleNamespace(
            timestamp=12.0,
            frame_path="frames/12.jpg",
            content_type="diagram",
            complexity_score=0.7,
            clip_confidence=0.8,
            trigger_reason="Visual content",
            confidence="high",
            transcript_context="Synthetic context",
        )
        result = {
            "video_id": VIDEO_ID,
            "video_dir": self.temp.name,
            "title": "Synthetic lecture",
            "duration_seconds": 60,
            "duration_formatted": "01:00",
            "metadata": {"total_keyframes": 1},
            "transcript": {"entries": [{"start": 0, "duration": 5, "text": "Lecture"}]},
            "intervention_points": [point],
        }
        detector = SimpleNamespace(
            _extract_video_id=lambda _url: VIDEO_ID,
            process_detection_only=lambda _url: result,
        )
        release = threading.Event()
        started = threading.Event()
        observed = {}

        def background_generate(**kwargs):
            with self.Session() as db:
                observed["video"] = db.get(Video, VIDEO_ID) is not None
                observed["interventions"] = db.query(Intervention).filter_by(video_id=VIDEO_ID).count()
            observed["mode"] = kwargs["output_mode"]
            started.set()
            release.wait(2.0)

        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: object(),
            data_dir=self.temp.name,
            generator=background_generate,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        self.addCleanup(release.set)

        app = FastAPI()
        app.state.explanation_preparation = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with mock.patch.object(video_module, "get_orchestrator", return_value=detector):
            with TestClient(app) as client:
                before = time.monotonic()
                response = client.post("/video/process", json={
                    "youtube_url": VIDEO_URL,
                    "output_mode": "detailed",
                })
                elapsed = time.monotonic() - before

        self.assertEqual(response.status_code, 200, response.text)
        self.assertLess(elapsed, 0.75)
        self.assertTrue(started.wait(1.0))
        self.assertEqual(observed, {"video": True, "interventions": 1, "mode": "detailed"})
        release.set()

    def test_process_mode_defaults_to_explanatory(self):
        from api.schemas import ProcessVideoRequest

        parsed = ProcessVideoRequest(youtube_url=VIDEO_URL)
        self.assertEqual(parsed.output_mode, "explanatory")

    def test_already_processed_video_schedules_only_missing_work(self):
        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="Existing lecture",
                source_url=VIDEO_URL,
                total_keyframes=1,
            ))
            db.add(Transcript(
                video_id=VIDEO_ID,
                start_time=0,
                end_time=5,
                text="Existing transcript",
            ))
            db.add(Intervention(
                id="existing-point",
                video_id=VIDEO_ID,
                timestamp=8.0,
                frame_path="frames/8.jpg",
                content_type="diagram",
            ))
            db.commit()

        scheduled = []
        manager = SimpleNamespace(
            schedule_video=lambda video_id, mode: scheduled.append((video_id, mode))
        )
        detector = SimpleNamespace(_extract_video_id=lambda _url: VIDEO_ID)
        app = FastAPI()
        app.state.explanation_preparation = manager
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        with mock.patch.object(video_module, "get_orchestrator", return_value=detector):
            with TestClient(app) as client:
                response = client.post("/video/process", json={
                    "youtube_url": VIDEO_URL,
                    "output_mode": "brief",
                })

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(scheduled, [(VIDEO_ID, "brief")])


if __name__ == "__main__":
    unittest.main()
