"""Personal media boundaries, immutable versions, upload recovery and context."""
import io
import json
import os
from pathlib import Path
import tempfile
import shutil
import sqlite3
import unittest
from unittest.mock import patch
import wave
from contextlib import closing

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.database import Base, create_database_engine, get_db
from api.models import (Explanation, Intervention, LibraryItem, MediaDeletion, SavedExplanation,
                        User, Video, VoiceNote, ExplanationPreparation, LOCAL_DEVELOPMENT_USER_ID, LOCAL_DEVELOPMENT_PROFILE_KEY)
from api.personal_storage import media_path, snapshot
from api.personal_storage import reconcile_media
from api.routers.personal_storage import router
from api.routers.learning_history import router as history_router


def wav(seconds=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x01\x00" * int(16000 * seconds))
    return stream.getvalue()


class PersonalStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage_root = patch("api.personal_storage.data_root", return_value=self.root)
        self.storage_root.start()
        self.addCleanup(self.storage_root.stop)
        self.env = patch.dict(os.environ, SYNAPSE_ENV="test", SYNAPSE_LOCAL_PROFILE_COMPAT="1")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.engine = create_database_engine(f"sqlite:///{(self.root / 'test.sqlite').as_posix()}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.addCleanup(self.engine.dispose)
        with self.Session() as db:
            db.add(User(id=LOCAL_DEVELOPMENT_USER_ID, profile_key=LOCAL_DEVELOPMENT_PROFILE_KEY, display_name="Local", is_local=True))
            db.add(Video(id="abcdefghijk", title="Lecture", source_url="https://www.youtube.com/watch?v=abcdefghijk", duration_seconds=600))
            db.add(Video(id="other-video", title="Other", source_url="https://example.com/other"))
            db.flush()
            for index, seconds in enumerate([39, 40, 100, 160, 161]):
                audio = self.root / f"{index}.mp3"
                audio.write_bytes(f"audio {index}".encode())
                frame = self.root / f"{index}.jpg"
                frame.write_bytes(b"frame")
                point = Intervention(id=f"point-{index}", video_id="abcdefghijk", timestamp=seconds,
                    timestamp_formatted=f"{seconds // 60}:{seconds % 60:02}", frame_path=str(frame), content_type="diagram")
                db.add(point)
                db.flush()
                db.add(Explanation(id=index + 1, intervention_id=point.id, output_mode="explanatory",
                    text_explanation=f"original text {index}", audio_file_path=str(audio)))
            db.commit()
        app = FastAPI()
        app.include_router(router)
        app.include_router(history_router)
        def database():
            with self.Session() as db:
                yield db
        app.dependency_overrides[get_db] = database
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def upload(self, identity="one", seconds=100, data=None, **extra):
        metadata = {"id": identity, "videoId": "abcdefghijk", "lectureSeconds": seconds,
                    "createdAt": 1000000, **extra}
        return self.client.post("/me/voice-notes", data={"metadata": json.dumps(metadata)},
                                files={"file": ("note.wav", data or wav(), "audio/wav")})

    def test_real_audio_validation_idempotency_and_inclusive_context(self):
        first = self.upload()
        self.assertEqual(first.status_code, 200, first.text)
        note = first.json()
        self.assertEqual({entry["lectureSeconds"] for entry in note["explanations"]}, {40, 100, 160})
        self.assertEqual(note["durationMs"], 1000)
        again = self.upload()
        self.assertEqual(again.json()["id"], note["id"])
        self.assertEqual(self.upload(seconds=101).status_code, 409)
        audio = self.client.get(note["audioUrl"])
        self.assertEqual(audio.content, wav())
        with self.Session() as db:
            self.assertEqual(db.query(VoiceNote).count(), 1)
        # New client simulates reopening after frontend restart.
        self.assertEqual(len(self.client.get("/me/voice-notes").json()), 1)

    def test_note_without_context_and_invalid_uploads(self):
        self.assertEqual(self.upload(seconds=400).json()["explanations"], [])
        self.assertEqual(self.upload("invalid", data=b"fake audio").status_code, 422)
        self.assertEqual(self.upload("beyond", seconds=601).status_code, 422)
        self.assertEqual(self.upload("nan", seconds=float("nan")).status_code, 422)
        self.assertEqual(self.upload("owner", user_id="attacker").status_code, 422)
        self.assertEqual(self.upload("long", data=wav(301)).status_code, 422)

    def test_snapshot_does_not_change_when_cache_or_source_file_changes(self):
        self.assertEqual(self.client.put("/me/saved-library/abcdefghijk").status_code, 200)
        original = self.client.get("/me/saved-library").json()[0]["explanations"]
        chosen = next(item for item in original if item["sourceExplanationId"] == 3)
        self.client.put(f"/me/saved-explanations/{chosen['id']}/bookmark")
        with self.Session() as db:
            row = db.get(Explanation, 3)
            Path(row.audio_file_path).write_bytes(b"replacement audio")
            row.text_explanation = "replacement text"
            db.commit()
        saved = self.client.get(f"/me/saved-explanations/{chosen['id']}").json()
        self.assertEqual(saved["text"], "original text 2")
        self.assertEqual(self.client.get(saved["audioUrl"]).content, b"audio 2")
        self.client.delete("/me/library/abcdefghijk")
        self.assertEqual(self.client.get("/me/saved-library").json(), [])
        self.assertEqual(self.client.get("/me/saved-explanations?kind=bookmarks").json()[0]["id"], chosen["id"])

    def test_modes_and_owners_are_separate(self):
        with self.Session() as db:
            owner = User(profile_key="other", display_name="Other", is_local=False)
            db.add(owner)
            db.flush()
            row = snapshot(db, db.get(Explanation, 1), owner.id)
            db.commit()
            identity = row.id
            source = db.get(Explanation, 1)
            alternate = Explanation(intervention_id=source.intervention_id, output_mode="detailed",
                text_explanation="Detailed text", audio_file_path=source.audio_file_path)
            db.add(alternate)
            db.commit()
        self.assertEqual(self.client.get(f"/me/saved-explanations/{identity}").status_code, 404)
        self.client.put("/me/saved-library/abcdefghijk")
        rows = self.client.get("/me/saved-lectures/abcdefghijk").json()
        self.assertEqual(len([row for row in rows if row["pointId"] == "point-0"]), 2)

    def test_deletion_outbox_survives_filesystem_failure(self):
        note = self.upload().json()
        with patch.object(Path, "unlink", side_effect=PermissionError("busy")):
            self.assertEqual(self.client.delete(f"/me/voice-notes/{note['id']}").status_code, 204)
        with self.Session() as db:
            key = db.query(MediaDeletion).one().media_key
            self.assertTrue(media_path(key).is_file())
        self.assertEqual(self.client.get("/me/voice-notes").json(), [])
        self.assertFalse(media_path(key).exists())

    def test_origin_environment_and_path_boundaries(self):
        self.assertEqual(self.client.get("/me/voice-notes", headers={"Origin": "https://evil.example"}).status_code, 403)
        with patch.dict(os.environ, SYNAPSE_ENV="production"):
            self.assertEqual(self.client.get("/me/voice-notes").status_code, 404)
        with self.assertRaises(Exception):
            media_path("../escape")

    def test_backup_restores_the_database_and_matching_recording(self):
        from scripts.backup_learning_storage import create_backup, verify_backup
        note = self.upload().json()
        with tempfile.TemporaryDirectory() as directory:
            backup = create_backup(Path(directory) / "backup", str(self.engine.url), self.root)
            verify_backup(backup)
            restored = Path(directory) / "restored"
            shutil.copytree(backup, restored)
            with closing(sqlite3.connect(restored / "synapse.db")) as db:
                key = db.execute("select media_key from voice_notes where id = ?", [note["id"]]).fetchone()[0]
            self.assertEqual((restored / "data" / key).read_bytes(), wav())
            (backup / "data" / key).write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                verify_backup(backup)

    def test_reconciliation_only_removes_unreferenced_managed_files(self):
        self.upload()
        orphan = self.root / "users" / "orphan-user" / "voice_notes" / "orphan" / "old.webm"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"orphan")
        unrelated = self.root / "users" / "settings.json"
        unrelated.write_text("{}")
        with self.Session() as db:
            key = db.query(VoiceNote).one().media_key
            reconcile_media(db, grace_seconds=0)
        self.assertFalse(orphan.exists())
        self.assertTrue(media_path(key).is_file())
        self.assertTrue(unrelated.is_file())

    def test_delayed_progress_request_cannot_replace_a_newer_position(self):
        from datetime import datetime, timedelta, timezone
        newest = datetime.now(timezone.utc) - timedelta(seconds=1)
        path = "/me/playback/abcdefghijk"
        first = self.client.put(path, json={"position_seconds": 100, "observed_at": newest.isoformat()})
        self.assertEqual(first.status_code, 200, first.text)
        delayed = self.client.put(path, json={"position_seconds": 50, "observed_at": (newest - timedelta(seconds=15)).isoformat()})
        self.assertEqual(delayed.json()["position_seconds"], 100)

    def test_library_and_note_context_survive_shared_cache_deletion(self):
        self.client.put("/me/saved-library/abcdefghijk")
        with self.Session() as db:
            db.query(Explanation).delete()
            db.query(Intervention).delete()
            db.query(Video).filter_by(id="abcdefghijk").delete()
            db.commit()
        lectures = self.client.get("/me/saved-library").json()
        self.assertEqual(lectures[0]["videoTitle"], "Lecture")
        self.assertEqual(len(lectures[0]["explanations"]), 5)
        note = self.upload()
        self.assertEqual(note.status_code, 200, note.text)
        self.assertEqual({item["lectureSeconds"] for item in note.json()["explanations"]}, {40, 100, 160})

    def generate_detailed(self):
        from api.explanation_service import generate_explanation_from_evidence
        audio = self.root / "new-detailed.mp3"
        audio.write_bytes(b"ID3-ready-generated-audio")
        class Orchestrator:
            def synthesize_intervention_explanation(self, **_kwargs):
                return {"text_explanation": "New detailed explanation", "audio_file_path": str(audio)}
        return generate_explanation_from_evidence(session_factory=self.Session,
            orchestrator_factory=Orchestrator, data_dir=str(self.root), video_id="abcdefghijk",
            intervention_id="point-0", output_mode="detailed")

    def test_completed_generation_preserves_subscribed_content_without_opening_the_point(self):
        self.client.put("/me/saved-library/abcdefghijk")
        result = self.generate_detailed()
        with self.Session() as db:
            saved = db.query(SavedExplanation).filter_by(output_mode="detailed").one()
            self.assertEqual(json.loads(saved.payload)["sourceExplanationId"], result.explanation_id)

    def test_storage_failure_does_not_turn_successful_inference_into_failure(self):
        self.client.put("/me/saved-library/abcdefghijk")
        with patch("api.personal_storage.capture_subscribers", side_effect=OSError("disk busy")):
            result = self.generate_detailed()
        self.assertEqual(result.text_explanation, "New detailed explanation")
        rows = self.client.get("/me/saved-library").json()[0]["explanations"]
        self.assertTrue(any(row["mode"] == "detailed" for row in rows))

    def test_a_concurrent_preparation_cannot_mix_old_text_with_new_audio(self):
        from api.personal_storage import preserve_path
        with self.Session() as db:
            db.add(ExplanationPreparation(video_id="abcdefghijk", intervention_id="point-0", output_mode="explanatory", status="ready"))
            db.commit()
        changed = False
        def copy_then_start_generation(stored, suffix):
            nonlocal changed
            key = preserve_path(stored, suffix)
            if not changed:
                changed = True
                with self.Session() as db:
                    db.query(ExplanationPreparation).update({"status": "generating"})
                    db.commit()
            return key
        with patch("api.personal_storage.preserve_path", side_effect=copy_then_start_generation):
            response = self.client.post("/me/saved-explanations", json={"explanation_id": 1})
        self.assertEqual(response.status_code, 409)
        with self.Session() as db:
            self.assertEqual(db.query(SavedExplanation).count(), 0)
