"""Ownership, retention, deduplication, and CRUD tests for local history."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine, get_db
from api.models import (
    Explanation,
    Intervention,
    LectureActivity,
    LibraryItem,
    LOCAL_DEVELOPMENT_PROFILE_KEY,
    LOCAL_DEVELOPMENT_USER_ID,
    PlaybackProgress,
    RecentExplanation,
    User,
    Video,
)
from api.routers.learning_history import router


VIDEO_ID = "History1234"
OTHER_VIDEO_ID = "OtherVid123"


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LearningHistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_environment = os.environ.get("SYNAPSE_ENV")
        os.environ["SYNAPSE_ENV"] = "test"
        self.addCleanup(self._restore_environment)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        database_path = Path(self.temp.name) / "history.sqlite"
        self.engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._seed()

        app = FastAPI()
        app.include_router(router)

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _restore_environment(self):
        if self.previous_environment is None:
            os.environ.pop("SYNAPSE_ENV", None)
        else:
            os.environ["SYNAPSE_ENV"] = self.previous_environment

    def _seed(self):
        with self.Session() as db:
            db.add_all([
                User(
                    id=LOCAL_DEVELOPMENT_USER_ID,
                    profile_key=LOCAL_DEVELOPMENT_PROFILE_KEY,
                    display_name="Local Synapse User",
                    is_local=True,
                ),
                User(
                    id="11111111-1111-4111-8111-111111111111",
                    profile_key="another-user",
                    display_name="Another user",
                    is_local=False,
                ),
                Video(
                    id=VIDEO_ID,
                    title="History lecture",
                    source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
                    duration_seconds=120,
                    total_keyframes=1,
                ),
                Video(
                    id=OTHER_VIDEO_ID,
                    title="Other lecture",
                    source_url=f"https://www.youtube.com/watch?v={OTHER_VIDEO_ID}",
                    duration_seconds=90,
                    total_keyframes=1,
                ),
            ])
            db.flush()
            intervention = Intervention(
                id="history-point",
                video_id=VIDEO_ID,
                timestamp=12.0,
                frame_path="frames/12.jpg",
            )
            db.add(intervention)
            db.flush()
            db.add(Explanation(
                id=501,
                intervention_id=intervention.id,
                text_explanation="A cached explanation",
                audio_file_path="History1234/history-point/explanatory.mp3",
                output_mode="explanatory",
            ))
            db.commit()

    def test_crud_upserts_prevent_duplicates_and_return_youtube_metadata(self):
        opened = self.client.put(
            f"/me/lectures/{VIDEO_ID}/opened",
            json={"last_intervention_id": "history-point"},
        )
        reopened = self.client.put(f"/me/lectures/{VIDEO_ID}/opened", json={})
        self.assertEqual(opened.status_code, 200, opened.text)
        self.assertEqual(reopened.status_code, 200, reopened.text)
        self.assertEqual(reopened.json()["open_count"], 2)
        self.assertEqual(reopened.json()["youtube_url"], f"https://www.youtube.com/watch?v={VIDEO_ID}")
        self.assertIsNotNone(reopened.json()["processed_at"])

        first_library = self.client.put(
            f"/me/library/{VIDEO_ID}", json={"preferred_mode": "brief"}
        )
        updated_library = self.client.put(
            f"/me/library/{VIDEO_ID}", json={"preferred_mode": "detailed"}
        )
        self.assertEqual(first_library.status_code, 200, first_library.text)
        self.assertEqual(updated_library.json()["preferred_mode"], "detailed")

        first_progress = self.client.put(
            f"/me/playback/{VIDEO_ID}",
            json={
                "position_seconds": 14,
                "duration_seconds": 120,
                "last_intervention_id": "history-point",
            },
        )
        updated_progress = self.client.put(
            f"/me/playback/{VIDEO_ID}",
            json={"position_seconds": 60, "duration_seconds": 120, "completed": False},
        )
        self.assertEqual(first_progress.status_code, 200, first_progress.text)
        self.assertEqual(updated_progress.json()["position_seconds"], 60)
        self.assertEqual(updated_progress.json()["last_intervention_id"], "history-point")

        first_view = self.client.put("/me/explanations/501/viewed")
        repeated_view = self.client.put("/me/explanations/501/viewed")
        self.assertEqual(first_view.status_code, 200, first_view.text)
        self.assertEqual(repeated_view.status_code, 200, repeated_view.text)
        self.assertEqual(first_view.json()["id"], repeated_view.json()["id"])
        self.assertIn("output_mode=explanatory", repeated_view.json()["audio_url"])

        self.assertEqual(len(self.client.get("/me/lectures/recent").json()), 1)
        self.assertEqual(len(self.client.get("/me/library").json()), 1)
        self.assertEqual(len(self.client.get("/me/explanations/recent").json()), 1)
        self.assertEqual(self.client.get(f"/me/playback/{VIDEO_ID}").status_code, 200)
        with self.Session() as db:
            self.assertEqual(db.query(LectureActivity).count(), 1)
            self.assertEqual(db.query(LibraryItem).count(), 1)
            self.assertEqual(db.query(PlaybackProgress).count(), 1)
            self.assertEqual(db.query(RecentExplanation).count(), 1)

    def test_request_rejects_client_user_id_and_queries_only_local_owner(self):
        rejected = self.client.put(
            f"/me/library/{VIDEO_ID}",
            json={
                "preferred_mode": "brief",
                "user_id": "11111111-1111-4111-8111-111111111111",
            },
        )
        self.assertEqual(rejected.status_code, 422, rejected.text)

        now = utcnow()
        with self.Session() as db:
            db.add_all([
                LectureActivity(
                    user_id="11111111-1111-4111-8111-111111111111",
                    video_id=OTHER_VIDEO_ID,
                    first_opened_at=now,
                    last_opened_at=now,
                ),
                LibraryItem(
                    user_id="11111111-1111-4111-8111-111111111111",
                    video_id=OTHER_VIDEO_ID,
                    added_at=now,
                    last_accessed_at=now,
                ),
            ])
            db.commit()

        self.assertEqual(self.client.get("/me/lectures/recent").json(), [])
        self.assertEqual(self.client.get("/me/library").json(), [])

    def test_retention_removes_only_personal_history_and_keeps_library_and_cache(self):
        old = utcnow() - timedelta(days=31)
        with self.Session() as db:
            db.add_all([
                LectureActivity(
                    user_id=LOCAL_DEVELOPMENT_USER_ID,
                    video_id=VIDEO_ID,
                    first_opened_at=old,
                    last_opened_at=old,
                ),
                LibraryItem(
                    user_id=LOCAL_DEVELOPMENT_USER_ID,
                    video_id=VIDEO_ID,
                    added_at=old,
                    last_accessed_at=old,
                ),
                RecentExplanation(
                    user_id=LOCAL_DEVELOPMENT_USER_ID,
                    video_id=VIDEO_ID,
                    intervention_id="history-point",
                    explanation_id=501,
                    output_mode="explanatory",
                    viewed_at=old,
                ),
            ])
            db.commit()

        self.assertEqual(self.client.get("/me/lectures/recent").json(), [])
        self.assertEqual(self.client.get("/me/explanations/recent").json(), [])
        self.assertEqual(len(self.client.get("/me/library").json()), 1)

        with self.Session() as db:
            self.assertEqual(db.query(LectureActivity).count(), 0)
            self.assertEqual(db.query(RecentExplanation).count(), 0)
            self.assertEqual(db.query(LibraryItem).count(), 1)
            self.assertIsNotNone(db.get(Video, VIDEO_ID))
            self.assertIsNotNone(db.get(Intervention, "history-point"))
            explanation = db.get(Explanation, 501)
            self.assertIsNotNone(explanation)
            self.assertEqual(
                explanation.audio_file_path,
                "History1234/history-point/explanatory.mp3",
            )

    def test_delete_endpoints_remove_personal_rows_not_shared_cache(self):
        self.client.put(f"/me/lectures/{VIDEO_ID}/opened", json={})
        self.client.put(f"/me/library/{VIDEO_ID}", json={})
        self.client.put(f"/me/playback/{VIDEO_ID}", json={"position_seconds": 5})
        viewed = self.client.put("/me/explanations/501/viewed").json()

        self.assertEqual(self.client.delete(f"/me/lectures/{VIDEO_ID}/recent").status_code, 204)
        self.assertEqual(self.client.delete(f"/me/library/{VIDEO_ID}").status_code, 204)
        self.assertEqual(self.client.delete(f"/me/playback/{VIDEO_ID}").status_code, 204)
        self.assertEqual(self.client.delete(f"/me/explanations/recent/{viewed['id']}").status_code, 204)
        with self.Session() as db:
            self.assertEqual(db.query(LectureActivity).count(), 0)
            self.assertEqual(db.query(LibraryItem).count(), 0)
            self.assertEqual(db.query(PlaybackProgress).count(), 0)
            self.assertEqual(db.query(RecentExplanation).count(), 0)
            self.assertEqual(db.query(Video).count(), 2)
            self.assertEqual(db.query(Intervention).count(), 1)
            self.assertEqual(db.query(Explanation).count(), 1)

    def test_personal_routes_are_hidden_outside_local_development(self):
        os.environ["SYNAPSE_ENV"] = "production"
        response = self.client.get("/me/library")
        self.assertEqual(response.status_code, 404)

    def test_user_deletion_cascades_personal_rows_but_not_shared_cache(self):
        self.client.put(f"/me/lectures/{VIDEO_ID}/opened", json={})
        self.client.put(f"/me/library/{VIDEO_ID}", json={})
        self.client.put(f"/me/playback/{VIDEO_ID}", json={"position_seconds": 5})
        self.client.put("/me/explanations/501/viewed")

        with self.Session() as db:
            db.query(User).filter(User.id == LOCAL_DEVELOPMENT_USER_ID).delete(
                synchronize_session=False
            )
            db.commit()
            self.assertEqual(db.query(LectureActivity).count(), 0)
            self.assertEqual(db.query(LibraryItem).count(), 0)
            self.assertEqual(db.query(PlaybackProgress).count(), 0)
            self.assertEqual(db.query(RecentExplanation).count(), 0)
            self.assertEqual(db.query(Video).count(), 2)
            self.assertEqual(db.query(Intervention).count(), 1)
            self.assertEqual(db.query(Explanation).count(), 1)


if __name__ == "__main__":
    unittest.main()
