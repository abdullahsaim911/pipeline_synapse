"""Secure explanation-audio delivery and response compatibility tests."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.models import Explanation, Intervention, Video
from api.routers import video as video_module


VIDEO_ID = "AudioTest01"
INTERVENTION_ID = "11111111-1111-1111-1111-111111111111"


class ExplanationAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"
        self.data_dir.mkdir()

        engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'test.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="Audio lecture",
                source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            ))
            db.add(Intervention(
                id=INTERVENTION_ID,
                video_id=VIDEO_ID,
                timestamp=12.0,
                frame_path="frame.jpg",
            ))
            db.commit()

        app = FastAPI()
        app.include_router(video_module.router)

        def test_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = test_db
        app.dependency_overrides[video_module.get_data_dir] = lambda: str(self.data_dir)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def add_explanation(self, audio_path, mode="explanatory"):
        with self.Session() as db:
            db.add(Explanation(
                intervention_id=INTERVENTION_ID,
                text_explanation="Readable explanation text.",
                audio_file_path=str(audio_path) if audio_path is not None else None,
                text_file_path="data/text.txt",
                vlm_snapshot_path="data/snapshot.json",
                output_mode=mode,
            ))
            db.commit()

    def audio_endpoint(self, intervention_id=INTERVENTION_ID, mode="explanatory"):
        return f"/video/intervention/{intervention_id}/audio?output_mode={mode}"

    def test_serves_generated_mp3_with_correct_content_type(self):
        audio = self.data_dir / VIDEO_ID / "interventions" / "11111111" / "explanatory.mp3"
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"ID3\x04synapse-test")
        self.add_explanation(audio)

        response = self.client.get(self.audio_endpoint())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"ID3\x04synapse-test")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")

    def test_missing_audio_file_returns_404(self):
        self.add_explanation(self.data_dir / "missing.mp3")
        response = self.client.get(self.audio_endpoint())
        self.assertEqual(response.status_code, 404)

    def test_invalid_identifier_is_rejected(self):
        response = self.client.get(self.audio_endpoint("bad$id"))
        self.assertEqual(response.status_code, 422)

    def test_traversal_path_is_not_served(self):
        secret = Path(self.temp.name) / "secret.mp3"
        secret.write_bytes(b"not public")
        self.add_explanation(Path("..") / "secret.mp3")

        response = self.client.get(self.audio_endpoint())

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"not public", response.content)

    def test_absolute_file_outside_data_directory_is_not_served(self):
        outside = Path(self.temp.name) / "outside.mp3"
        outside.write_bytes(b"not public")
        self.add_explanation(outside)

        response = self.client.get(self.audio_endpoint())

        self.assertEqual(response.status_code, 404)

    def test_non_mp3_database_path_is_not_served(self):
        text = self.data_dir / "explanation.txt"
        text.write_text("not audio", encoding="utf-8")
        self.add_explanation(text)
        self.assertEqual(self.client.get(self.audio_endpoint()).status_code, 404)

    def test_cached_explanation_response_keeps_paths_and_adds_audio_url(self):
        audio = self.data_dir / VIDEO_ID / "interventions" / "11111111" / "explanatory.mp3"
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"ID3")
        self.add_explanation(audio)

        with mock.patch.object(video_module, "get_orchestrator") as get_orchestrator:
            response = self.client.post("/video/intervention/explain", json={
                "video_id": VIDEO_ID,
                "intervention_id": INTERVENTION_ID,
                "output_mode": "explanatory",
            })

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["text_explanation"], "Readable explanation text.")
        self.assertEqual(body["audio_file_path"], str(audio))
        self.assertEqual(body["output_mode"], "explanatory")
        self.assertEqual(
            body["audio_url"],
            self.audio_endpoint(),
        )
        get_orchestrator.assert_not_called()

    def test_unknown_mode_is_rejected_without_generation(self):
        with mock.patch.object(video_module, "get_orchestrator") as get_orchestrator:
            response = self.client.post("/video/intervention/explain", json={
                "video_id": VIDEO_ID,
                "intervention_id": INTERVENTION_ID,
                "output_mode": "standard",
            })
        self.assertEqual(response.status_code, 422)
        get_orchestrator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
