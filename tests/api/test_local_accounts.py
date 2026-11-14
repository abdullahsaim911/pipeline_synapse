"""Account lifecycle and two-user authorization against real SQLite/media."""
from datetime import timedelta
import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from api.auth import router, _attempts, digest, utcnow
from api.models import AccountCredential, AccountSession, SavedExplanation, VoiceNote, User
from api.security import configure_cors
from tests.api import test_personal_storage as fixtures
wav = fixtures.wav


class LocalAccountTests(unittest.TestCase):
    upload = fixtures.PersonalStorageTests.upload

    def setUp(self):
        fixtures.PersonalStorageTests.setUp(self)
        self.client.app.include_router(router)
        self.origins = patch.dict(os.environ, SYNAPSE_CORS_ORIGINS="http://127.0.0.1:5174")
        self.origins.start()
        self.addCleanup(self.origins.stop)
        configure_cors(self.client.app)
        _attempts.clear()

    def signup(self, username="learner", password="correct horse battery"):
        result = self.client.post("/auth/signup", json={"username": username, "password": password})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def use(self, account):
        self.client.headers["Authorization"] = "Bearer " + account["token"]

    def test_signup_hashes_password_and_login_persists_across_clients(self):
        account = self.signup()
        self.use(account)
        self.assertFalse(account["user"]["isLocalProfile"])
        with self.Session() as db:
            credential = db.query(AccountCredential).one()
            self.assertTrue(credential.password_hash.startswith("$argon2id$"))
            self.assertNotIn("correct horse", credential.password_hash)
            self.assertEqual(db.query(AccountSession).one().token_digest, digest(account["token"]))
            self.assertNotEqual(db.query(AccountSession).one().token_digest, account["token"])
        from fastapi.testclient import TestClient
        with TestClient(self.client.app) as reopened:
            response = reopened.get("/auth/session", headers={"Authorization": "Bearer " + account["token"]})
            self.assertEqual(response.json()["user"]["id"], account["user"]["id"])
        login = self.client.post("/auth/login", json={"username": "LEARNER", "password": "correct horse battery"})
        self.assertEqual(login.status_code, 200, login.text)
        self.assertNotEqual(login.json()["token"], account["token"])

    def test_lecture_interest_is_account_scoped_and_logout_preserves_admitted_work(self):
        from unittest.mock import Mock
        from api.on_demand import OnDemandExplanations
        from api.routers.explanation_requests import router as requests_router
        from api.models import LectureSessionInterest
        self.client.app.include_router(requests_router)
        self.client.app.state.on_demand_explanations = OnDemandExplanations(self.Session,
            Mock(max_attempts=3), Mock())
        alice = self.signup("alice")
        self.use(alice)
        opened = self.client.post("/me/lecture-sessions", json={"video_id": "abcdefghijk"})
        self.assertEqual(opened.status_code, 201, opened.text)
        identity = opened.json()["id"]
        with self.Session() as db:
            self.assertEqual(db.get(LectureSessionInterest, identity).account_digest, digest(alice["token"]))
        bob = self.signup("bob")
        self.use(bob)
        self.assertEqual(self.client.put(f"/me/lecture-sessions/{identity}/heartbeat").status_code, 404)
        self.use(alice)
        self.assertEqual(self.client.post("/auth/logout").status_code, 204)
        with self.Session() as db:
            self.assertEqual(db.get(LectureSessionInterest, identity).state, "continuing")

    def test_duplicate_invalid_credentials_and_no_client_controlled_ownership(self):
        self.signup()
        self.assertEqual(self.client.post("/auth/signup", json={"username": "LEARNER", "password": "12345678"}).status_code, 409)
        with self.Session() as db:
            self.assertEqual(db.query(AccountCredential).count(), 1)
            self.assertEqual(db.query(User).count(), 2)  # no orphan from duplicate signup
        for username in ["learner", "missing"]:
            result = self.client.post("/auth/login", json={"username": username, "password": "bad password"})
            self.assertEqual(result.status_code, 401)
            self.assertEqual(result.json()["detail"], "Invalid username or password.")
        for extra in [{"user_id": "attacker"}, {"is_local": True}]:
            self.assertEqual(self.client.post("/auth/signup", json={"username": "other", "password": "12345678", **extra}).status_code, 422)

    def test_logout_revokes_session_and_invalid_never_falls_back(self):
        account = self.signup()
        self.use(account)
        self.assertEqual(self.client.post("/auth/logout").status_code, 204)
        self.assertEqual(self.client.post("/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/auth/session").status_code, 401)
        self.assertEqual(self.client.get("/me/saved-library").status_code, 401)
        for token in ["Bearer invalid", "Basic credentials", "Bearer " + "a" * 64]:
            self.assertEqual(self.client.get("/me/voice-notes", headers={"Authorization": token}).status_code, 401)
        del self.client.headers["Authorization"]
        self.assertTrue(self.client.get("/auth/session").json()["user"]["isLocalProfile"])
        with patch.dict(os.environ, SYNAPSE_LOCAL_PROFILE_COMPAT="0"):
            self.assertEqual(self.client.get("/me/saved-library").status_code, 401)

    def test_expiry_and_missing_session(self):
        account = self.signup()
        self.use(account)
        with self.Session() as db:
            db.query(AccountSession).update({AccountSession.expires_at: utcnow() - timedelta(seconds=1)})
            db.commit()
        self.assertEqual(self.client.get("/me/saved-library").status_code, 401)

    def test_missing_compatibility_setting_requires_sign_in(self):
        with patch.dict(os.environ):
            os.environ.pop("SYNAPSE_LOCAL_PROFILE_COMPAT", None)
            self.assertEqual(self.client.get("/auth/session").status_code, 401)
            self.assertEqual(self.client.get("/me/saved-library").status_code, 401)
            account = self.signup()
            self.use(account)
            self.assertEqual(self.client.get("/auth/session").status_code, 200)
            self.assertEqual(self.client.post("/auth/logout").status_code, 204)
            self.assertEqual(self.client.get("/me/saved-library").status_code, 401)

    def test_two_users_cannot_access_private_versions_or_media(self):
        first = self.signup("first")
        second = self.signup("second")
        self.use(first)
        saved = self.client.post("/me/saved-explanations", json={"explanation_id": 1}).json()
        self.assertEqual(self.client.put(f"/me/saved-explanations/{saved['id']}/bookmark").status_code, 200)
        self.assertEqual(self.client.put("/me/saved-library/abcdefghijk").status_code, 200)
        note = self.upload().json()
        self.assertEqual(self.client.get(note["audioUrl"]).content, wav())
        self.assertEqual(self.client.get(saved["audioUrl"]).status_code, 200)
        self.use(second)
        self.assertEqual(self.client.get("/me/saved-library").json(), [])
        self.assertEqual(self.client.get("/me/saved-explanations?kind=bookmarks").json(), [])
        self.assertEqual(self.client.get("/me/voice-notes").json(), [])
        for path in [f"/me/saved-explanations/{saved['id']}", saved["audioUrl"], saved["frameUrl"], note["audioUrl"]]:
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.put(f"/me/saved-explanations/{saved['id']}/bookmark").status_code, 404)
        self.client.delete(f"/me/voice-notes/{note['id']}")
        # Same retry ID belongs to each user independently.
        own_note = self.upload().json()
        self.assertNotEqual(own_note["id"], note["id"])
        self.use(first)
        self.assertEqual(self.client.get(note["audioUrl"]).content, wav())
        self.assertEqual(len(self.client.get("/me/saved-library").json()), 1)

    def test_progress_history_are_account_scoped(self):
        first, second = self.signup("first"), self.signup("second")
        self.use(first)
        response = self.client.put("/me/playback/abcdefghijk", json={"position_seconds": 99})
        self.assertEqual(response.status_code, 200, response.text)
        self.use(second)
        self.assertEqual(self.client.get("/me/playback/abcdefghijk").status_code, 404)
        self.client.put("/me/playback/abcdefghijk", json={"position_seconds": 12})
        self.use(first)
        self.assertEqual(self.client.get("/me/playback/abcdefghijk").json()["position_seconds"], 99)

    def test_revocation_during_slow_upload_prevents_commit(self):
        account = self.signup()
        self.use(account)
        from api.personal_storage import validate_recording
        def revoke(data):
            with self.Session() as db:
                db.query(AccountSession).update({AccountSession.revoked_at: utcnow()})
                db.commit()
            return validate_recording(data)
        with patch("api.routers.personal_storage.validate_recording", side_effect=revoke):
            result = self.upload()
        self.assertEqual(result.status_code, 401, result.text)
        with self.Session() as db:
            self.assertEqual(db.query(VoiceNote).count(), 0)

    def test_commit_guard_rejects_revoked_session(self):
        account = self.signup()
        with self.Session() as db:
            db.info["account_session_digest"] = digest(account["token"])
            with self.Session() as revoker:
                revoker.query(AccountSession).update({AccountSession.revoked_at: utcnow()})
                revoker.commit()
            with self.assertRaises(HTTPException):
                db.commit()
            db.rollback()

    def test_origin_environment_and_cors_boundaries(self):
        result = self.client.post("/auth/signup", json={"username": "first", "password": "12345678"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(result.status_code, 403)
        response = self.client.options("/me/saved-library", headers={"Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "Authorization"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Authorization", response.headers["access-control-allow-headers"])
        with patch.dict(os.environ, SYNAPSE_ENV="production"):
            self.assertEqual(self.client.get("/auth/session").status_code, 404)

    def test_login_throttling_is_bounded_and_temporary(self):
        for _ in range(10):
            self.client.post("/auth/login", json={"username": "missing", "password": "bad password"})
        result = self.client.post("/auth/login", json={"username": "missing", "password": "bad password"})
        self.assertEqual(result.status_code, 429)
        self.assertEqual(result.headers["retry-after"], "60")
        _attempts.clear()
        self.assertEqual(self.client.post("/auth/login", json={"username": "missing", "password": "bad password"}).status_code, 401)

    def test_oversized_signup_is_rejected_before_hashing(self):
        with patch("api.auth.passwords") as hashed:
            response = self.client.post("/auth/signup", json={"username": "first", "password": "x" * 9000})
        self.assertEqual(response.status_code, 413)
        hashed.hash.assert_not_called()
