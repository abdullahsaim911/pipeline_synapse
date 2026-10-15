"""POST /video/ask through the real router, schemas and a real (in-memory) database."""

import json
import os
import tempfile
import unittest
from unittest import mock

from tests.qa.helpers import (
    FakeProvider,
    LECTURE_LINES,
    OTHER_LECTURE_LINES,
    forbidden_provider_factory,
)
from tests.qa.test_moment import ERROR_SNAPSHOT, GOOD_SNAPSHOT
from tests.qa.test_transcript import write_transcript
from llm_interface.providers import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMMalformedResponseError,
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
    from api.routers import video as video_module
    from api.schemas import AskResponse, AskSource
    from api.security import DEFAULT_ORIGINS, ORIGINS_ENV_VAR, configure_cors, get_allowed_origins
    from qa.service import AskConfig, AskService
    HAVE_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_STACK = False

VIDEO_A = "AAAAAAAAAAA"
VIDEO_B = "BBBBBBBBBBB"
SECRET_TEXT = "sk-SECRET-raw-provider-text"


@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class AskEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = self.tmp.name

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.Session = sessionmaker(bind=engine)

        self.provider = FakeProvider()
        self.service = AskService(provider=self.provider)

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ORIGINS_ENV_VAR, None)

        self.app = self.build_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

        # Lecture A in the database + file; lecture B in files only.
        with self.Session() as db:
            db.add(models.Video(id=VIDEO_A, title="A", source_url="http://x"))
            for i, line in enumerate(LECTURE_LINES):
                db.add(models.Transcript(video_id=VIDEO_A, start_time=i * 6.0,
                                         end_time=i * 6.0 + 9.0, text=line))
            db.commit()
        write_transcript(self.data, VIDEO_B, OTHER_LECTURE_LINES)

    # -- helpers ----------------------------------------------------------

    def build_app(self):
        """The Ask router with the same CORS setup as api.main, reading the current environment."""
        app = FastAPI()
        configure_cors(app)
        app.include_router(video_module.router)

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[video_module.get_ask_service] = lambda: self.service
        app.dependency_overrides[video_module.get_data_dir] = lambda: self.data
        return app

    def ask(self, **overrides):
        body = {"video_id": VIDEO_A, "question": "What did the instructor say about LIATE?"}
        body.update(overrides)
        return self.client.post("/video/ask", json=body)

    def add_intervention(self, iid, video_id=VIDEO_A, snapshot=None, explanation=None):
        path = None
        with self.Session() as db:
            db.add(models.Intervention(id=iid, video_id=video_id, timestamp=126.0, frame_path="f.jpg",
                                       content_type="graph", transcript_context="u equals x so du equals dx"))
            if explanation:
                if snapshot is not None:
                    folder = os.path.join(self.data, video_id, "interventions", iid[:8])
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.join(video_id, "interventions", iid[:8], "vlm_snapshot.json")
                    with open(os.path.join(self.data, path), "w", encoding="utf-8") as handle:
                        json.dump(snapshot, handle)
                db.add(models.Explanation(intervention_id=iid, text_explanation=explanation,
                                          output_mode="explanatory", vlm_snapshot_path=path))
            db.commit()

    # -- success ----------------------------------------------------------

    def test_answered_response_shape_and_sources(self):
        response = self.ask()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"status", "scope", "answer_text", "sources"})
        self.assertEqual((body["status"], body["scope"]), ("answered", "lecture"))
        self.assertTrue(body["answer_text"])
        self.assertGreaterEqual(len(body["sources"]), 1)
        for source in body["sources"]:
            self.assertEqual(set(source), {"start_seconds", "end_seconds", "intervention_id"})
            self.assertIsNone(source["intervention_id"])
            self.assertLess(source["start_seconds"], source["end_seconds"])
        self.assertTrue(any(s["start_seconds"] <= 66 <= s["end_seconds"] for s in body["sources"]))

    def test_question_is_trimmed(self):
        self.assertEqual(self.ask(question="   What is LIATE?   ").status_code, 200)
        self.assertIn('"""\nWhat is LIATE?\n"""', self.provider.calls[0]["prompt"])

    def test_optional_fields_may_be_null(self):
        response = self.ask(current_time_seconds=None, intervention_id=None)
        self.assertEqual(response.status_code, 200)

    def test_file_only_video_without_database_row(self):
        response = self.ask(video_id=VIDEO_B, question="What do mitochondria produce?")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "answered")

    def test_time_question_uses_requested_range(self):
        response = self.ask(question="what is said between 2:00 and 2:30")
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertTrue(all(105 <= s["start_seconds"] and s["end_seconds"] <= 160 for s in body["sources"]),
                        body["sources"])

    def test_relative_time_uses_current_time(self):
        response = self.ask(question="explain the last minute", current_time_seconds=150.0)
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertTrue(all(s["end_seconds"] <= 156 for s in body["sources"]))

    # -- deterministic non-answers ---------------------------------------------------

    def test_relative_time_without_position_is_insufficient_context(self):
        self.service = AskService(provider_factory=forbidden_provider_factory)
        body = self.ask(question="explain the last five minutes").json()
        self.assertEqual((body["status"], body["sources"]), ("insufficient_context", []))
        self.assertTrue(body["answer_text"])

    def test_known_video_without_transcript_skips_the_llm(self):
        with self.Session() as db:
            db.add(models.Video(id="CCCCCCCCCCC", title="C", source_url="http://x"))
            db.commit()
        self.service = AskService(provider_factory=forbidden_provider_factory)
        response = self.ask(video_id="CCCCCCCCCCC")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["status"], body["sources"]), ("insufficient_context", []))

    def test_failed_transcription_file_skips_the_llm(self):
        folder = os.path.join(self.data, "DDDDDDDDDDD")
        os.makedirs(folder)
        with open(os.path.join(folder, "transcript.json"), "w") as handle:
            json.dump({"video_id": "DDDDDDDDDDD", "entries": [], "error": "failed"}, handle)
        self.service = AskService(provider_factory=forbidden_provider_factory)
        body = self.ask(video_id="DDDDDDDDDDD").json()
        self.assertEqual(body["status"], "insufficient_context")

    def test_malformed_transcript_file_skips_the_llm(self):
        folder = os.path.join(self.data, "EEEEEEEEEEE")
        os.makedirs(folder)
        with open(os.path.join(folder, "transcript.json"), "w") as handle:
            handle.write("{broken")
        self.service = AskService(provider_factory=forbidden_provider_factory)
        body = self.ask(video_id="EEEEEEEEEEE").json()
        self.assertEqual(body["status"], "insufficient_context")

    def test_cross_video_isolation(self):
        # 'zebrafish' exists only in lecture B. Lecture A is short, so it is sent whole; the
        # model sees all of A and none of B, and (given the complete transcript) may say so.
        self.provider.reply = "[[NOT_IN_LECTURE]] This lecture does not discuss that."
        body = self.ask(video_id=VIDEO_A, question="What about the zebrafish genome?").json()
        self.assertEqual((body["status"], body["sources"]), ("not_in_lecture", []))
        prompt_a = self.provider.calls[-1]["prompt"]
        self.assertIn("integration by parts", prompt_a)
        for line in OTHER_LECTURE_LINES:
            self.assertNotIn(line, prompt_a)
        self.assertNotIn("zebrafish", prompt_a.split("LECTURE EXCERPTS:")[1])   # the question itself mentions it

        self.provider.reply = "Mitochondria are described. SOURCES: S1"
        self.ask(video_id=VIDEO_B, question="What about the zebrafish genome?")
        prompt_b = self.provider.calls[-1]["prompt"]
        self.assertIn("zebrafish", prompt_b)
        self.assertNotIn("integration by parts", prompt_b)

    def test_unmatched_question_on_a_long_lecture_is_insufficient_not_absent(self):
        self.service = AskService(provider_factory=forbidden_provider_factory,
                                  config=AskConfig(full_transcript_max_tokens=1))
        body = self.ask(video_id=VIDEO_A, question="What about the zebrafish genome?").json()
        self.assertEqual((body["status"], body["sources"]), ("insufficient_context", []))

    def test_model_declared_not_in_lecture_has_no_sources(self):
        self.provider.reply = "[[NOT_IN_LECTURE]] The lecture does not discuss that."
        body = self.ask(question="Explain the tabular method in baking").json()
        self.assertEqual((body["status"], body["sources"]), ("not_in_lecture", []))

    # -- moment context --------------------------------------------------------------

    def test_valid_intervention_adds_moment_context(self):
        self.add_intervention("int-1", snapshot=GOOD_SNAPSHOT,
                              explanation="The graph shows x times e to the x growing quickly.")
        body = self.ask(question="What does this graph represent?", intervention_id="int-1",
                        current_time_seconds=130.0).json()
        self.assertEqual((body["status"], body["scope"]), ("answered", "lecture_with_moment"))
        self.assertIn("int-1", [s["intervention_id"] for s in body["sources"]])
        prompt = self.provider.calls[-1]["prompt"]
        self.assertIn("rising steeply", prompt)
        self.assertIn("x times e to the x growing quickly", prompt)

    def test_failed_vlm_snapshot_withholds_its_explanation_but_not_the_transcript(self):
        self.add_intervention("int-1", snapshot=ERROR_SNAPSHOT,
                              explanation="Choosing u well makes the remaining integral simpler, as the arrows show.")
        body = self.ask(question="What does this show?", intervention_id="int-1").json()
        prompt = self.provider.calls[-1]["prompt"]
        self.assertNotIn("WinError", prompt)
        self.assertNotIn("Choosing u well makes", prompt)               # explanation from a failed analysis
        self.assertIn("Visual analysis: unavailable", prompt)
        self.assertIn("Caption spoken at this moment: u equals x so du equals dx", prompt)
        self.assertIn("plug into the formula", prompt)                  # transcript evidence is intact
        self.assertEqual((body["status"], body["scope"]), ("answered", "lecture_with_moment"))
        self.assertTrue(body["sources"])
        self.assertTrue(all(s["intervention_id"] is None for s in body["sources"]))

    def test_scaffolding_never_reaches_the_model(self):
        self.add_intervention("int-1", snapshot=GOOD_SNAPSHOT, explanation=(
            "YOUR EXPLANATION:\n\n1. COMPLETE VISUAL DESCRIPTION:\n   - The curve rises steeply.\n\n"
            "2. WHAT THE TEACHER IS EXPLAINING:\nChoosing u with LIATE."))
        self.ask(question="What does this show?", intervention_id="int-1")
        prompt = self.provider.calls[-1]["prompt"]
        self.assertIn("The curve rises steeply.", prompt)
        self.assertIn("Choosing u with LIATE.", prompt)
        for artifact in ("YOUR EXPLANATION", "COMPLETE VISUAL DESCRIPTION", "WHAT THE TEACHER IS EXPLAINING"):
            self.assertNotIn(artifact, prompt)

    def test_intervention_without_current_time_still_works(self):
        self.add_intervention("int-1")
        response = self.ask(question="Why is this step needed?", intervention_id="int-1")
        self.assertEqual(response.json()["status"], "answered")
        self.assertIn("PLAYBACK POSITION: about 02:06", self.provider.calls[-1]["prompt"])

    def test_unknown_intervention_is_404(self):
        response = self.ask(intervention_id="does-not-exist")
        self.assertEqual((response.status_code, response.json()["detail"]), (404, "Intervention not found"))
        self.assertEqual(self.provider.calls, [])

    def test_intervention_from_another_video_is_404(self):
        with self.Session() as db:
            db.add(models.Video(id=VIDEO_B, title="B", source_url="http://x"))
            db.commit()
        self.add_intervention("int-b", video_id=VIDEO_B)
        response = self.ask(video_id=VIDEO_A, intervention_id="int-b")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.provider.calls, [])

    # -- request validation -------------------------------------------------------------

    def test_unknown_video_is_404(self):
        response = self.ask(video_id="ZZZZZZZZZZZ")
        self.assertEqual((response.status_code, response.json()["detail"]), (404, "Video not found"))

    def test_validation_errors_are_422(self):
        cases = {
            "missing video_id": {"question": "hi there"},
            "missing question": {"video_id": VIDEO_A},
            "empty question": {"video_id": VIDEO_A, "question": ""},
            "blank question": {"video_id": VIDEO_A, "question": "    "},
            "long question": {"video_id": VIDEO_A, "question": "x" * 501},
            "short video id": {"video_id": "abc", "question": "hi there"},
            "traversal video id": {"video_id": "../../etc/pw", "question": "hi there"},
            "non-ascii video id": {"video_id": "AAAAAAAAAAé", "question": "hi there"},
            "negative time": {"video_id": VIDEO_A, "question": "hi there", "current_time_seconds": -1},
            "string time": {"video_id": VIDEO_A, "question": "hi there", "current_time_seconds": "soon"},
            "absurd time": {"video_id": VIDEO_A, "question": "hi there", "current_time_seconds": 1e12},
            "bad intervention id": {"video_id": VIDEO_A, "question": "hi there", "intervention_id": "x/../y"},
            "empty intervention id": {"video_id": VIDEO_A, "question": "hi there", "intervention_id": ""},
        }
        for name, body in cases.items():
            with self.subTest(name):
                response = self.client.post("/video/ask", json=body)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.provider.calls, [])

    def test_boundary_question_lengths_are_accepted(self):
        self.assertEqual(self.ask(question="x" * 500).status_code, 200)
        self.assertEqual(self.ask(question="  " + "y" * 500 + "  ").status_code, 200)

    # -- provider failures ---------------------------------------------------------------

    def test_provider_failures_map_to_http_errors_without_leaking_details(self):
        cases = [
            (LLMConfigurationError(f"GEMINI_API_KEY missing {SECRET_TEXT}"), 500, "not configured"),
            (LLMRateLimitError(SECRET_TEXT, status_code=429), 429, "busy"),
            (LLMTimeoutError(SECRET_TEXT), 504, "too long"),
            (LLMConnectionError(SECRET_TEXT), 503, "unavailable"),
            (LLMUnavailableError(SECRET_TEXT, status_code=503), 503, "unavailable"),
            (LLMAuthenticationError(SECRET_TEXT, status_code=401), 502, "credentials"),
            (LLMRequestError(SECRET_TEXT, status_code=400), 502, "could not produce"),
            (LLMMalformedResponseError(SECRET_TEXT), 502, "unusable"),
            (LLMProviderError(SECRET_TEXT), 502, "could not produce"),
        ]
        for error, status, fragment in cases:
            with self.subTest(type(error).__name__):
                self.provider.error = error
                response = self.ask()
                self.assertEqual(response.status_code, status, response.text)
                self.assertNotIn(SECRET_TEXT, response.text)
                self.assertNotIn("GEMINI_API_KEY", response.text)
                self.assertIn(fragment, response.json()["detail"].lower())

    def test_unexpected_error_is_a_generic_500(self):
        self.provider.error = RuntimeError(SECRET_TEXT)
        response = self.ask()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(SECRET_TEXT, response.text)

    def test_default_service_reports_missing_configuration(self):
        """No override of the service: the real wiring must fail cleanly without a key."""
        self.app.dependency_overrides.pop(video_module.get_ask_service)
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(video_module, "_ask_service", None):
            response = self.ask()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Question answering is not configured on this server.")

    def test_empty_transcript_does_not_need_configuration(self):
        self.app.dependency_overrides.pop(video_module.get_ask_service)
        with self.Session() as db:
            db.add(models.Video(id="CCCCCCCCCCC", title="C", source_url="http://x"))
            db.commit()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(video_module, "_ask_service", None):
            response = self.ask(video_id="CCCCCCCCCCC")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_context")

    # -- malformed model output --------------------------------------------------------

    def test_malformed_status_marker_is_a_502_and_is_never_echoed(self):
        for reply in ("Sure. [[NOT_IN_LECTURE]] nope", "[NOT_IN_LECTURE] nope", "[[ANSWERED]] x",
                      "[[NOT_IN_LECTURE]] [[INSUFFICIENT_CONTEXT]] x"):
            with self.subTest(reply=reply):
                self.provider.reply = reply
                response = self.ask()
                self.assertEqual(response.status_code, 502, response.text)
                self.assertIn("unusable", response.json()["detail"].lower())
                self.assertNotIn("NOT_IN_LECTURE", response.text)
                self.assertNotIn("nope", response.text)

    def test_well_formed_marker_from_the_model_is_not_an_error(self):
        self.provider.reply = "[[INSUFFICIENT_CONTEXT]] The excerpts do not explain that."
        body = self.ask().json()
        self.assertEqual((body["status"], body["sources"]), ("insufficient_context", []))
        self.assertEqual(body["answer_text"], "The excerpts do not explain that.")

    # -- transcript identity and database errors ---------------------------------------------

    def test_case_variant_of_a_known_video_id_is_not_found(self):
        response = self.ask(video_id=VIDEO_B.lower(), question="What do mitochondria produce?")
        self.assertEqual((response.status_code, response.json()["detail"]), (404, "Video not found"))
        self.assertEqual(self.provider.calls, [])

    def test_video_id_with_trailing_newline_is_rejected(self):
        for bad in (VIDEO_A + "\n", " " + VIDEO_A, VIDEO_A + " "):
            response = self.client.post("/video/ask", json={"video_id": bad, "question": "What is LIATE?"})
            self.assertEqual(response.status_code, 422, repr(bad))
        self.assertEqual(self.provider.calls, [])

    def test_database_failure_is_a_generic_500_not_a_silent_fallback(self):
        # lecture B has a transcript file, but a broken database must not be papered over
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        self.addCleanup(engine.dispose)
        no_tables = sessionmaker(bind=engine)

        def broken_db():
            db = no_tables()
            try:
                yield db
            finally:
                db.close()

        self.app.dependency_overrides[get_db] = broken_db
        with self.assertLogs("api.routers.video", level="ERROR"):
            response = self.ask(video_id=VIDEO_B, question="What do mitochondria produce?")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Could not answer the question.")
        self.assertNotIn("no such table", response.text)
        self.assertNotIn("SELECT", response.text)
        self.assertEqual(self.provider.calls, [])

    # -- browser origin controls ---------------------------------------------------------------

    def test_allowed_origin_is_served_with_cors_headers(self):
        origin = DEFAULT_ORIGINS[0]
        response = self.client.post("/video/ask", headers={"Origin": origin},
                                    json={"video_id": VIDEO_A, "question": "What is LIATE?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_disallowed_origin_is_rejected_before_any_work(self):
        for origin in ("https://evil.example", "http://localhost:5174", "http://localhost:5173.evil.example",
                       "null", "file://"):
            with self.subTest(origin=origin):
                response = self.client.post("/video/ask", headers={"Origin": origin},
                                            json={"video_id": VIDEO_A, "question": "What is LIATE?"})
                self.assertEqual((response.status_code, response.json()["detail"]), (403, "Origin not allowed"))
                self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertEqual(self.provider.calls, [])

    def test_request_without_origin_header_is_still_served(self):
        # curl, scripts and the Electron main process send no Origin header
        self.assertEqual(self.ask().status_code, 200)

    def test_preflight_from_allowed_origin(self):
        origin = DEFAULT_ORIGINS[1]
        response = self.client.options("/video/ask", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)
        methods = response.headers["access-control-allow-methods"]
        self.assertIn("POST", methods)
        for forbidden in ("DELETE", "PUT", "PATCH"):
            self.assertNotIn(forbidden, methods)
        self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_preflight_from_disallowed_origin_fails(self):
        response = self.client.options("/video/ask", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_other_endpoints_do_not_grant_cross_origin_reads(self):
        response = self.client.get(f"/video/meta/{VIDEO_A}", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", response.headers)
        allowed = self.client.get(f"/video/meta/{VIDEO_A}", headers={"Origin": DEFAULT_ORIGINS[0]})
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), DEFAULT_ORIGINS[0])

    def test_origins_can_be_configured_through_the_environment(self):
        custom = "https://app.example.com"
        with mock.patch.dict(os.environ, {ORIGINS_ENV_VAR: f"{custom}, http://localhost:3000/"}):
            client = TestClient(self.build_app(), raise_server_exceptions=False)
            body = {"video_id": VIDEO_A, "question": "What is LIATE?"}
            ok = client.post("/video/ask", headers={"Origin": custom}, json=body)
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.headers["access-control-allow-origin"], custom)
            self.assertEqual(client.post("/video/ask", headers={"Origin": "http://localhost:3000"},
                                         json=body).status_code, 200)
            # the development defaults are no longer allowed once configured
            self.assertEqual(client.post("/video/ask", headers={"Origin": DEFAULT_ORIGINS[0]},
                                         json=body).status_code, 403)

    def test_null_origin_is_only_allowed_when_explicitly_configured(self):
        with mock.patch.dict(os.environ, {ORIGINS_ENV_VAR: "null"}):
            client = TestClient(self.build_app(), raise_server_exceptions=False)
            response = client.post("/video/ask", headers={"Origin": "null"},
                                   json={"video_id": VIDEO_A, "question": "What is LIATE?"})
        self.assertEqual(response.status_code, 200)

    def test_real_application_cors_configuration(self):
        from api.main import app as real_app
        cors = [m for m in real_app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
        self.assertEqual(len(cors), 1)
        options = cors[0].kwargs
        self.assertNotIn("*", options["allow_origins"])
        self.assertTrue(options["allow_origins"])
        self.assertIs(options["allow_credentials"], False)
        self.assertNotIn("*", options["allow_methods"])
        self.assertNotIn("*", options["allow_headers"])



@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class ResponseSchemaTests(unittest.TestCase):
    def test_answered_requires_a_source(self):
        with self.assertRaises(ValueError):
            AskResponse(status="answered", scope="lecture", answer_text="x", sources=[])

    def test_non_answered_must_not_have_sources(self):
        source = AskSource(start_seconds=1, end_seconds=2)
        for status in ("not_in_lecture", "insufficient_context"):
            with self.assertRaises(ValueError):
                AskResponse(status=status, scope="lecture", answer_text="x", sources=[source])

    def test_answer_text_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            AskResponse(status="not_in_lecture", scope="lecture", answer_text="", sources=[])

    def test_status_and_scope_are_restricted(self):
        with self.assertRaises(ValueError):
            AskResponse(status="maybe", scope="lecture", answer_text="x", sources=[])
        with self.assertRaises(ValueError):
            AskResponse(status="not_in_lecture", scope="galaxy", answer_text="x", sources=[])

    def test_source_times_must_be_ordered(self):
        with self.assertRaises(ValueError):
            AskSource(start_seconds=5, end_seconds=4)

    def test_valid_response(self):
        response = AskResponse(status="answered", scope="lecture_with_moment", answer_text="ok",
                               sources=[AskSource(start_seconds=42.0, end_seconds=80.0)])
        self.assertIsNone(response.sources[0].intervention_id)


@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class AllowedOriginsTests(unittest.TestCase):
    def origins(self, value):
        env = {} if value is None else {ORIGINS_ENV_VAR: value}
        with mock.patch.dict(os.environ, env, clear=True):
            return get_allowed_origins()

    def test_defaults(self):
        for unset in (None, "", "   ", ","):
            self.assertEqual(self.origins(unset), list(DEFAULT_ORIGINS), repr(unset))

    def test_custom_list_is_trimmed_normalised_and_deduplicated(self):
        self.assertEqual(
            self.origins(" https://a.example.com/ , http://localhost:3000,https://a.example.com"),
            ["https://a.example.com", "http://localhost:3000"])

    def test_wildcards_and_malformed_entries_are_ignored(self):
        self.assertEqual(self.origins("*, https://ok.example, ftp://x, http://a/b, javascript:alert(1)"),
                         ["https://ok.example"])
        self.assertEqual(self.origins("*"), list(DEFAULT_ORIGINS))   # never falls back to allow-all

    def test_null_origin_requires_explicit_opt_in(self):
        self.assertNotIn("null", self.origins(None))
        self.assertEqual(self.origins("null"), ["null"])


if __name__ == "__main__":
    unittest.main()
