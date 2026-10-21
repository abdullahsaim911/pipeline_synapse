"""POST /video/quiz through the real router, schemas and an in-memory database. Fake provider only."""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from tests.qa.helpers import FakeProvider
from tests.quiz.helpers import PASSAGE_TEXTS, PROJECT_ROOT, mcq, tf
from tests.quiz.helpers_api import (
    SECRET_PROMPT_WORD,
    TEST_LIMITS,
    VIDEO_ID,
    ScriptedProvider,
    good_candidates,
    reply,
)
from llm_interface.providers import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api import models
    from api.database import Base, get_db
    from api.routers import quiz as quiz_module
    from api.routers import video as video_module
    from api.security import ORIGINS_ENV_VAR, configure_cors
    from qa.service import AskService
    from quiz.config import QUIZ_ENABLED_ENV_VAR
    from quiz.schemas import QuizResponse
    from quiz.service import QuizService
    HAVE_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_STACK = False

VIDEO_B = "BBBBBBBBBBB"
SHORT_VIDEO = "CCCCCCCCCCC"
EMPTY_VIDEO = "DDDDDDDDDDD"
BROKEN_VIDEO = "EEEEEEEEEEE"
NO_TRANSCRIPT_VIDEO = "FFFFFFFFFFF"


def write_transcript_file(data_dir, video_id, texts, **extra):
    folder = os.path.join(data_dir, video_id)
    os.makedirs(folder, exist_ok=True)
    entries = [{"start": i * 100.0, "end": i * 100.0 + 90.0, "text": t} for i, t in enumerate(texts)]
    with open(os.path.join(folder, "transcript.json"), "w", encoding="utf-8") as handle:
        json.dump({"video_id": video_id, "entries": entries, **extra}, handle)


@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class QuizEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = self.tmp.name

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.Session = sessionmaker(bind=engine)

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ORIGINS_ENV_VAR, None)
        os.environ.pop(QUIZ_ENABLED_ENV_VAR, None)
        os.environ["SYNAPSE_ENV"] = "production"      # the quiz is off by default in production; development defaults are tested below

        self.provider = ScriptedProvider()
        self.service = QuizService(provider=self.provider, limits=TEST_LIMITS)
        self.ask_provider = FakeProvider()
        self.ask_service = AskService(provider=self.ask_provider)
        self.client = TestClient(self.build_app(), raise_server_exceptions=False)

        with self.Session() as db:
            db.add(models.Video(id=VIDEO_ID, title="A", source_url="http://x"))
            for i, text in enumerate(PASSAGE_TEXTS):
                db.add(models.Transcript(video_id=VIDEO_ID, start_time=i * 100.0, end_time=i * 100.0 + 90.0, text=text))
            db.add(models.Video(id=SHORT_VIDEO, title="C", source_url="http://x"))
            db.add(models.Transcript(video_id=SHORT_VIDEO, start_time=0.0, end_time=5.0, text="far too short"))
            db.add(models.Video(id=NO_TRANSCRIPT_VIDEO, title="F", source_url="http://x"))
            db.commit()
        write_transcript_file(self.data, VIDEO_B, PASSAGE_TEXTS)
        write_transcript_file(self.data, EMPTY_VIDEO, [], error="failed")
        os.makedirs(os.path.join(self.data, BROKEN_VIDEO))
        with open(os.path.join(self.data, BROKEN_VIDEO, "transcript.json"), "w") as handle:
            handle.write("{broken")

    def build_app(self):
        app = FastAPI()
        configure_cors(app)
        app.include_router(video_module.router)
        app.include_router(quiz_module.router)

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[quiz_module.get_quiz_service] = lambda: self.service
        app.dependency_overrides[video_module.get_ask_service] = lambda: self.ask_service
        app.dependency_overrides[video_module.get_answer_gate] = lambda: None
        app.dependency_overrides[video_module.get_data_dir] = lambda: self.data
        return app

    def enable(self):
        os.environ[QUIZ_ENABLED_ENV_VAR] = "1"

    def quiz(self, **overrides):
        body = {"video_id": VIDEO_ID}
        body.update(overrides)
        return self.client.post("/video/quiz", json=body)

    # -- feature flag -----------------------------------------------------------

    def test_disabled_by_default(self):
        self.provider.script = [reply(good_candidates(8))]
        response = self.quiz()
        self.assertEqual((response.status_code, response.json()), (404, {"detail": "Quiz is not available."}))
        self.assertEqual(self.provider.calls, [])

    def test_disabled_hides_everything_else(self):
        for response in (self.quiz(video_id="bad"), self.quiz(question_count=99), self.quiz(video_id=VIDEO_ID, extra=1)):
            self.assertEqual((response.status_code, response.json()["detail"]), (404, "Quiz is not available."))
        evil = self.client.post("/video/quiz", json={"video_id": VIDEO_ID}, headers={"Origin": "http://evil.example"})
        self.assertEqual(evil.status_code, 404)

    def test_only_exact_values_enable_it(self):
        self.provider.script = [reply(good_candidates(8))]
        for value in ("", "0", "false", "TRUE", "True", "yes", "on", "2", "1 1", "enabled"):
            os.environ[QUIZ_ENABLED_ENV_VAR] = value
            self.assertEqual(self.quiz().status_code, 404, repr(value))
        for value in ("1", "true", " 1 ", "true\n"):
            os.environ[QUIZ_ENABLED_ENV_VAR] = value
            self.provider.script = [reply(good_candidates(8))]
            self.assertEqual(self.quiz().status_code, 200, repr(value))

    def test_unset_flag_defaults_to_on_in_development_and_off_in_production(self):
        from quiz.config import quiz_enabled
        cases = [(None, True), ("", True), ("development", True), ("anything", True), ("production", False), (" PRODUCTION ", False)]
        for env_value, expected in cases:
            with mock.patch.dict(os.environ):
                os.environ.pop(QUIZ_ENABLED_ENV_VAR, None)
                os.environ.pop("SYNAPSE_ENV", None)
                if env_value is not None:
                    os.environ["SYNAPSE_ENV"] = env_value
                self.assertEqual(quiz_enabled(), expected, repr(env_value))

    def test_an_explicit_flag_always_wins_over_the_environment(self):
        from quiz.config import quiz_enabled
        for env_value in (None, "development", "production"):
            for flag, expected in (("1", True), ("true", True), ("0", False), ("false", False), ("", False), ("yes", False)):
                with mock.patch.dict(os.environ):
                    os.environ[QUIZ_ENABLED_ENV_VAR] = flag
                    os.environ.pop("SYNAPSE_ENV", None)
                    if env_value is not None:
                        os.environ["SYNAPSE_ENV"] = env_value
                    self.assertEqual(quiz_enabled(), expected, (env_value, flag))

    def test_the_real_app_in_development_answers_but_makes_no_model_call_for_an_unknown_video(self):
        code = (
            "import json\n"
            "from fastapi.testclient import TestClient\n"
            "import api.main as m\n"
            "r = TestClient(m.app).post('/video/quiz', json={'video_id': 'AAAAAAAAAAA'})\n"
            "print('RESULT ' + json.dumps({'status': r.status_code, 'detail': r.json()}))\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in (QUIZ_ENABLED_ENV_VAR, "SYNAPSE_ENV")}
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120)
        line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, done.stderr[-500:])
        self.assertEqual(json.loads(line[-1][len("RESULT "):]), {"status": 404, "detail": {"detail": "Video not found"}})   # on, and stopped before any model

    def test_the_real_app_registers_the_route_and_stays_disabled(self):
        code = (
            "import json\n"
            "from fastapi.testclient import TestClient\n"
            "import api.main as m\n"
            "paths = sorted(m.app.openapi()['paths'])\n"
            "r = TestClient(m.app).post('/video/quiz', json={'video_id': 'AAAAAAAAAAA'})\n"
            "print('RESULT ' + json.dumps({'route': '/video/quiz' in paths, 'ask': '/video/ask' in paths,"
            " 'status': r.status_code, 'detail': r.json()}))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != QUIZ_ENABLED_ENV_VAR}
        env["SYNAPSE_ENV"] = "production"
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                              capture_output=True, text=True, timeout=120)
        line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, done.stderr[-500:])
        result = json.loads(line[-1][len("RESULT "):])
        self.assertEqual(result, {"route": True, "ask": True, "status": 404, "detail": {"detail": "Quiz is not available."}})

    # -- request validation and origin -------------------------------------------

    def test_valid_request_defaults_to_six_questions(self):
        self.enable()
        self.provider.script = [reply(good_candidates(8))]
        body = self.quiz().json()
        self.assertEqual((body["requested_count"], body["returned_count"], body["status"]), (6, 6, "complete"))
        self.assertEqual(set(body), {"contract_version", "quiz_id", "video_id", "generated_at", "prompt_version", "model",
                                     "requested_count", "returned_count", "status", "reason", "notice", "omitted", "questions"})
        QuizResponse.model_validate(body)

    def test_question_counts_five_and_ten(self):
        self.enable()
        for count in (5, 10):
            self.provider.script = [reply(good_candidates(16)[:12])]
            body = self.quiz(question_count=count).json()
            self.assertEqual((body["requested_count"], len(body["questions"])), (count, count))

    def test_invalid_bodies_are_422(self):
        self.enable()
        bad = [{}, {"video_id": ""}, {"video_id": "short"}, {"video_id": VIDEO_ID + "x"}, {"video_id": VIDEO_ID + "\n"},
               {"video_id": "../../etc/pa"}, {"video_id": VIDEO_ID, "question_count": 4},
               {"video_id": VIDEO_ID, "question_count": 11}, {"video_id": VIDEO_ID, "question_count": "6"},
               {"video_id": VIDEO_ID, "question_count": 6.5}, {"video_id": VIDEO_ID, "question_count": True},
               {"video_id": VIDEO_ID, "question_count": None}, {"video_id": VIDEO_ID, "types": ["short"]},
               {"video_id": VIDEO_ID, "include_visual": True}, {"video_id": 12345678901}]
        for body in bad:
            self.assertEqual(self.client.post("/video/quiz", json=body).status_code, 422, body)
        self.assertEqual(self.client.post("/video/quiz", content="not json",
                                          headers={"Content-Type": "application/json"}).status_code, 422)
        self.assertEqual(self.provider.calls, [])

    def test_origin_rules(self):
        self.enable()
        self.provider.script = [reply(good_candidates(8))]
        rejected = self.client.post("/video/quiz", json={"video_id": VIDEO_ID}, headers={"Origin": "http://evil.example"})
        self.assertEqual((rejected.status_code, rejected.json()), (403, {"detail": "Origin not allowed"}))
        self.assertEqual(self.provider.calls, [])
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            self.provider.script = [reply(good_candidates(8))]
            ok = self.client.post("/video/quiz", json={"video_id": VIDEO_ID}, headers={"Origin": origin})
            self.assertEqual(ok.status_code, 200, origin)
        self.provider.script = [reply(good_candidates(8))]
        self.assertEqual(self.quiz().status_code, 200)       # no Origin header (scripts, Electron main)
        self.assertEqual(self.client.post("/video/quiz", json={"video_id": VIDEO_ID}, headers={"Origin": "null"}).status_code, 403)

    # -- videos and transcripts ------------------------------------------------------

    def test_unknown_video_is_404_and_skips_the_model(self):
        self.enable()
        response = self.quiz(video_id="ZZZZZZZZZZZ")
        self.assertEqual((response.status_code, response.json()), (404, {"detail": "Video not found"}))
        self.assertEqual(self.provider.calls, [])

    def test_file_only_video_works(self):
        self.enable()
        self.provider.script = [reply(good_candidates(8))]
        self.assertEqual(self.quiz(video_id=VIDEO_B).json()["status"], "complete")

    def test_video_without_a_usable_transcript_is_insufficient_without_a_model_call(self):
        self.enable()
        for video in (NO_TRANSCRIPT_VIDEO, EMPTY_VIDEO, BROKEN_VIDEO):
            response = self.quiz(video_id=video)
            body = response.json()
            self.assertEqual(response.status_code, 200, video)
            self.assertEqual((body["status"], body["reason"], body["questions"], body["returned_count"]),
                             ("insufficient_material", "no_transcript", [], 0), video)
            self.assertTrue(body["notice"])
        self.assertEqual(self.provider.calls, [])

    def test_too_short_transcript_is_insufficient_without_a_model_call(self):
        self.enable()
        body = self.quiz(video_id=SHORT_VIDEO).json()
        self.assertEqual((body["status"], body["reason"], body["model"]), ("insufficient_material", "too_short", "none"))
        self.assertEqual(self.provider.calls, [])

    def test_questions_are_grounded_in_the_requested_video_only(self):
        self.enable()
        write_transcript_file(self.data, "GGGGGGGGGGG", [t.replace("ball", "marble") for t in PASSAGE_TEXTS])
        self.provider.script = [reply(good_candidates(8))]
        self.quiz(video_id="GGGGGGGGGGG")
        prompt = self.provider.calls[0]["prompt"]
        self.assertIn("marble", prompt)
        self.assertNotIn(" the ball ", prompt)

    # -- outcomes and evidence mapping --------------------------------------------------

    def test_evidence_ids_become_server_owned_timestamps_and_quotes(self):
        self.enable()
        self.provider.script = [reply([mcq(2), mcq(4), tf(0), tf(6), mcq(7), tf(1)])]
        body = self.quiz().json()
        by_prompt = {q["prompt"]: q for q in body["questions"]}
        photosynthesis = by_prompt[mcq(2)["prompt"]]
        self.assertEqual([(e["start_seconds"], e["end_seconds"]) for e in photosynthesis["evidence"]], [(200.0, 290.0)])
        self.assertIn(photosynthesis["evidence"][0]["quote"], PASSAGE_TEXTS[2])
        for q in body["questions"]:
            for e in q["evidence"]:
                self.assertEqual(e["kind"], "transcript")
                self.assertTrue(any(e["quote"] in t for t in PASSAGE_TEXTS))
        starts = [q["evidence"][0]["start_seconds"] for q in body["questions"]]
        self.assertEqual(starts, sorted(starts))

    def test_timestamps_supplied_by_the_model_are_never_used(self):
        self.enable()
        forged = [dict(c, start_seconds=9999.0, end_seconds=10000.0) for c in good_candidates(8)]
        self.provider.script = [reply(forged), reply(forged)]
        body = self.quiz().json()
        self.assertEqual((body["status"], body["reason"]), ("insufficient_material", "validation_failed"))
        self.assertNotIn("9999", json.dumps(body))

    def test_partial_response(self):
        self.enable()
        self.provider.script = [reply(good_candidates(8)[:4]), reply([mcq(5)])]
        response = self.quiz()
        body = response.json()
        self.assertEqual((response.status_code, body["status"], body["returned_count"], body["reason"]), (200, "partial", 5, None))
        self.assertEqual(body["notice"], "Only 5 of the 6 questions you asked for could be made and checked from this lecture.")
        self.assertEqual(len(self.provider.calls), 2)

    def test_insufficient_after_screening_is_never_padded(self):
        self.enable()
        junk = [mcq(0, evidence_ids=["S99"]), mcq(1, type="short"), mcq(2, correct_index=9), mcq(3, hints=["a", "b", "c"]),
                mcq(4, evidence_ids=["M1"]), tf(5, answer="yes"), mcq(6, depends_on_visual=True), mcq(7)]
        self.provider.script = [reply(junk), reply(junk)]
        body = self.quiz().json()
        self.assertEqual((body["status"], body["reason"], body["questions"]), ("insufficient_material", "validation_failed", []))
        reasons = {o["reason"] for o in body["omitted"]}
        self.assertTrue({"unknown_evidence_id", "unsupported_question_type", "invalid_answer_key", "too_many_hints",
                         "unsupported_moment_evidence", "depends_on_visual"} <= reasons)
        self.assertNotIn("Concept", json.dumps(body["omitted"]))

    def test_unsupported_question_types_never_reach_the_response(self):
        self.enable()
        short = {"type": "short", "prompt": "Explain it.", "reference_answer": "x", "explanation": "y",
                 "concept": {"id": "c", "label": "C"}, "evidence_ids": ["S1"],
                 "relies_on_exact_figure": False, "depends_on_visual": False}
        self.provider.script = [reply(good_candidates(8)[:5] + [short] * 3), reply([])]
        body = self.quiz().json()
        self.assertTrue(all(q["type"] in ("mcq", "tf") for q in body["questions"]))

    # -- provider behaviour and errors ----------------------------------------------------

    def test_unusable_output_is_retried_once_then_502(self):
        self.enable()
        self.provider.script = ["garbage", "more garbage", reply(good_candidates(8))]
        response = self.quiz()
        self.assertEqual((response.status_code, response.json()["detail"]),
                         (502, "The AI service returned an unusable response. Please try again."))
        self.assertEqual(len(self.provider.calls), 2)

    def test_unusable_then_valid_succeeds(self):
        self.enable()
        self.provider.script = ['{"questions": [', reply(good_candidates(8))]
        self.assertEqual(self.quiz().json()["status"], "complete")
        self.assertEqual(len(self.provider.calls), 2)

    def test_provider_failures_map_to_http_errors_without_leaking_details(self):
        self.enable()
        leak = "sk-SECRET key=abc transcript text LIATE"
        cases = [
            (LLMRateLimitError(leak), 429), (LLMTimeoutError(leak), 504), (LLMConnectionError(leak), 503),
            (LLMUnavailableError(leak), 503), (LLMConfigurationError(leak), 500), (LLMAuthenticationError(leak), 500),
            (LLMRequestError(leak), 500), (LLMProviderError(leak), 500), (RuntimeError(leak), 500),
        ]
        for error, status in cases:
            self.provider.script = [error, error]
            self.provider.calls.clear()
            response = self.quiz()
            self.assertEqual(response.status_code, status, type(error).__name__)
            self.assertNotIn("SECRET", response.text)
            self.assertNotIn("LIATE", response.text)
            self.assertIsInstance(response.json()["detail"], str)
            self.assertLessEqual(len(self.provider.calls), 2)

    def test_unexpected_errors_are_logged_by_type_and_place_only(self):
        self.enable()
        leak = "SECRET-candidate-or-transcript-text"
        self.provider.script = [RuntimeError(leak)]
        with self.assertLogs("api.routers.quiz", level=logging.DEBUG) as captured:
            response = self.quiz()
        self.assertEqual((response.status_code, response.json()), (500, {"detail": "Could not create the quiz."}))
        text = " ".join(captured.output)
        self.assertIn("RuntimeError", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("Traceback", text)

    def test_provider_errors_other_than_schema_rejection_are_not_retried(self):
        self.enable()
        self.provider.script = [LLMRateLimitError("quota"), reply(good_candidates(8))]
        self.assertEqual(self.quiz().status_code, 429)
        self.assertEqual(len(self.provider.calls), 1)

    def test_quiz_is_usable_again_after_a_failure(self):
        self.enable()
        self.provider.script = [LLMTimeoutError("t")]
        self.assertEqual(self.quiz().status_code, 504)
        self.provider.script = [reply(good_candidates(8))]
        self.assertEqual(self.quiz().status_code, 200)

    def test_missing_provider_configuration_is_a_500_only_when_the_model_is_needed(self):
        self.enable()

        def broken():
            raise LLMConfigurationError("GEMINI_API_KEY is not set (sk-SECRET)")
        self.service = QuizService(provider_factory=broken, limits=TEST_LIMITS)
        self.assertEqual(self.quiz(video_id=SHORT_VIDEO).status_code, 200)       # decided without a model
        response = self.quiz()
        self.assertEqual((response.status_code, response.json()["detail"]), (500, "Quiz generation is not configured on this server."))
        self.assertNotIn("SECRET", response.text)

    # -- concurrency ---------------------------------------------------------------------

    def test_concurrent_request_for_the_same_video_is_429(self):
        self.enable()
        started, release = threading.Event(), threading.Event()
        results = {}

        def slow():
            started.set()
            self.assertTrue(release.wait(10))
            return reply(good_candidates(8))

        self.provider.script = [slow, reply(good_candidates(8))]
        thread = threading.Thread(target=lambda: results.update(first=self.quiz()))
        thread.start()
        try:
            self.assertTrue(started.wait(10))
            busy = self.quiz()
            self.assertEqual(busy.status_code, 429)
            self.assertIn("already being created", busy.json()["detail"])
            other = self.quiz(video_id=VIDEO_B)                  # a different video is not blocked
            self.assertEqual(other.status_code, 200)
        finally:
            release.set()
            thread.join(10)
        self.assertEqual(results["first"].status_code, 200)
        self.provider.script = [reply(good_candidates(8))]
        self.assertEqual(self.quiz().status_code, 200)           # lock released

    # -- logging -------------------------------------------------------------------------

    def test_logs_never_contain_prompts_replies_transcript_or_candidate_text(self):
        self.enable()
        hostile = good_candidates(8)
        hostile[0]["prompt"] = SECRET_PROMPT_WORD + " " + hostile[0]["prompt"]
        self.provider.script = [reply(hostile)]
        with self.assertLogs("quiz", level=logging.DEBUG) as captured:
            self.quiz()
        text = "\n".join(captured.output)
        for forbidden in (SECRET_PROMPT_WORD, "LIATE", "integration", "marble", "Concept"):
            self.assertNotIn(forbidden, text)

    # -- Ask is unchanged ---------------------------------------------------------------------

    def test_ask_still_works_beside_the_quiz_router(self):
        response = self.client.post("/video/ask", json={"video_id": VIDEO_ID, "question": "What is the formula?"})
        self.assertIn(response.status_code, (200,))
        self.assertEqual(set(response.json()), {"status", "scope", "answer_text", "sources"})

    def test_the_quiz_route_lives_on_its_own_router(self):
        self.assertNotIn("/video/quiz", {r.path for r in video_module.router.routes})
        self.assertIn("/video/ask", {r.path for r in video_module.router.routes})
        self.assertEqual({r.path for r in quiz_module.router.routes}, {"/video/quiz"})


if __name__ == "__main__":
    unittest.main()
