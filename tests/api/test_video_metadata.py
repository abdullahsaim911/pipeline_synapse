"""Focused contract tests for the video metadata endpoint."""

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.models import Video
from api.routers import video as video_module


VIDEO_ID = "aircAruvnKk"


class VideoMetadataEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        engine = create_engine(
            f"sqlite:///{os.path.join(self.temp.name, 'test.sqlite')}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

        app = FastAPI()
        app.include_router(video_module.router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        app.dependency_overrides[video_module.get_data_dir] = lambda: self.temp.name
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_returns_persisted_metadata_and_optional_file_channel(self):
        with self.Session() as db:
            db.add(Video(
                id=VIDEO_ID,
                title="Neural Networks",
                source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
                duration_seconds=1153,
                duration_formatted="19:13",
            ))
            db.commit()

        video_dir = os.path.join(self.temp.name, VIDEO_ID)
        os.makedirs(video_dir)
        with open(os.path.join(video_dir, "video_meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"channel_name": "3Blue1Brown"}, handle)

        response = self.client.get(f"/video/meta/{VIDEO_ID}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {
            "video_id": VIDEO_ID,
            "title": "Neural Networks",
            "channel_name": "3Blue1Brown",
            "duration_seconds": 1153,
            "duration_formatted": "19:13",
            "thumbnail_url": f"https://i.ytimg.com/vi/{VIDEO_ID}/hqdefault.jpg",
            "exists": True,
            "processed": True,
        })

    def test_reports_an_unknown_video_without_fabricating_metadata(self):
        response = self.client.get(f"/video/meta/{VIDEO_ID}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["exists"], False)
        self.assertIsNone(response.json()["title"])
        self.assertIsNone(response.json()["thumbnail_url"])

    def test_rejects_invalid_video_ids(self):
        response = self.client.get("/video/meta/not-valid")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
