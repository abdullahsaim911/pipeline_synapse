"""Transactional persistence and complete intervention retrieval tests."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.models import Explanation, Intervention, Transcript, Video
from api.routers import video as video_module
from qa.transcript import load_transcript


VIDEO_ID = "Persist1234"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def point(timestamp, content_type="diagram"):
    return SimpleNamespace(
        timestamp=timestamp,
        frame_path=f"frames/{timestamp}.jpg",
        content_type=content_type,
        complexity_score=0.8,
        clip_confidence=0.9,
        trigger_reason="Visual content",
        confidence="high",
        transcript_context=f"Context at {timestamp}",
    )


class FakeOrchestrator:
    def __init__(self, result):
        self.result = result
        self.process_calls = 0

    def _extract_video_id(self, _url):
        return VIDEO_ID

    def process_detection_only(self, _url):
        self.process_calls += 1
        return self.result


class VideoPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        database_path = os.path.join(self.temp.name, "test.sqlite")
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.video_dir = os.path.join(self.temp.name, VIDEO_ID)
        os.makedirs(self.video_dir)
        self.result = {
            "video_id": VIDEO_ID,
            "video_dir": self.video_dir,
            "title": "Persistence lecture",
            "duration_seconds": 90,
            "duration_formatted": "01:30",
            "metadata": {"total_keyframes": 3},
            "transcript": {
                "entries": [
                    {"start": 0.0, "end": 4.0, "text": "First concept"},
                    {"start": 5.0, "duration": 3.0, "text": "Second concept"},
                ]
            },
            # Deliberately out of order to verify retrieval ordering.
            "intervention_points": [point(40.0, "graph"), point(10.0)],
        }
        self.orchestrator = FakeOrchestrator(self.result)

        app = FastAPI()
        app.include_router(video_module.router)

        def test_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = test_db
        self.patch = mock.patch.object(
            video_module, "get_orchestrator", return_value=self.orchestrator
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def process(self):
        return self.client.post("/video/process", json={"youtube_url": VIDEO_URL})

    def test_success_is_committed_before_response_and_ids_match(self):
        response = self.process()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        with self.Session() as db:
            video = db.get(Video, VIDEO_ID)
            stored = db.query(Intervention).filter_by(video_id=VIDEO_ID).all()
            transcripts = (
                db.query(Transcript)
                .filter_by(video_id=VIDEO_ID)
                .order_by(Transcript.start_time)
                .all()
            )
            loaded = load_transcript(VIDEO_ID, db, self.temp.name)

        self.assertIsNotNone(video)
        self.assertEqual(video.title, "Persistence lecture")
        self.assertEqual(video.source_url, VIDEO_URL)
        self.assertEqual(len(stored), 2)
        self.assertEqual(
            {item["id"] for item in body["interventions"]},
            {item.id for item in stored},
        )
        self.assertEqual([(row.start_time, row.end_time) for row in transcripts], [(0.0, 4.0), (5.0, 8.0)])
        self.assertEqual(loaded.source, "database")
        self.assertEqual([segment.text for segment in loaded.segments], ["First concept", "Second concept"])

    def test_retrieval_includes_unexplained_interventions_in_timestamp_order(self):
        self.assertEqual(self.process().status_code, 200)
        with self.Session() as db:
            self.assertEqual(db.query(Explanation).count(), 0)

        response = self.client.get(f"/video/{VIDEO_ID}/interventions")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["total_interventions"], 2)
        self.assertEqual(
            [item["timestamp"] for item in body["interventions"]],
            [10.0, 40.0],
        )

    def test_returned_interventions_are_immediately_available_for_explanation_lookup(self):
        response = self.process()
        self.assertEqual(response.status_code, 200, response.text)

        with self.Session() as db:
            for item in response.json()["interventions"]:
                stored = (
                    db.query(Intervention)
                    .filter(
                        Intervention.id == item["id"],
                        Intervention.video_id == VIDEO_ID,
                    )
                    .one_or_none()
                )
                self.assertIsNotNone(stored)

    def test_reprocessing_existing_video_returns_stored_rows_without_duplicates(self):
        first = self.process()
        second = self.process()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(self.orchestrator.process_calls, 1)
        self.assertEqual(
            {item["id"] for item in first.json()["interventions"]},
            {item["id"] for item in second.json()["interventions"]},
        )
        with self.Session() as db:
            self.assertEqual(db.query(Video).count(), 1)
            self.assertEqual(db.query(Transcript).count(), 2)
            self.assertEqual(db.query(Intervention).count(), 2)

    def test_persistence_failure_rolls_back_every_required_record(self):
        self.result["transcript"]["entries"][1]["text"] = None
        with self.assertLogs("api.routers.video", level="ERROR"):
            response = self.process()

        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["detail"], "Processing results could not be saved.")
        with self.Session() as db:
            self.assertEqual(db.query(Video).count(), 0)
            self.assertEqual(db.query(Transcript).count(), 0)
            self.assertEqual(db.query(Intervention).count(), 0)


if __name__ == "__main__":
    unittest.main()
