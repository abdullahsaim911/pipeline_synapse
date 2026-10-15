"""POST /video/ask with the optional faithfulness guardrail and hybrid retrieval wired in through the real router."""

import os
import tempfile
import unittest
from unittest import mock

from tests.qa.helpers import FakeProvider, LECTURE_LINES

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api import models
    from api.database import Base, get_db
    from api.routers import video as video_module
    from api.security import ORIGINS_ENV_VAR, configure_cors
    from qa.faithfulness import GATE_MESSAGE, FaithfulnessConfig, FaithfulnessResult, apply_faithfulness_gate
    from qa.hybrid import HybridSelector
    from qa.service import AskService
    HAVE_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_STACK = False

VIDEO_A = "AAAAAAAAAAA"
FAITHFUL = "The instructor says to use the acronym LIATE."
UNFAITHFUL = "The instructor says photosynthesis happens in chloroplasts using sunlight energy."


class StubChecker:
    def __init__(self, score):
        self.score, self.calls = score, 0

    def check(self, answer, passages):
        self.calls += 1
        return FaithfulnessResult(self.score, "stub", 2, 2, 1)


@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class AskEndpointGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.Session = sessionmaker(bind=engine)
        with self.Session() as db:
            db.add(models.Video(id=VIDEO_A, title="A", source_url="http://x"))
            for i, line in enumerate(LECTURE_LINES):
                db.add(models.Transcript(video_id=VIDEO_A, start_time=i * 6.0, end_time=i * 6.0 + 9.0, text=line))
            db.commit()

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in [k for k in os.environ if k.startswith("ASK_")] + [ORIGINS_ENV_VAR]:
            os.environ.pop(name, None)

        # The gate and the service are cached at module level; restore them so tests cannot leak into each other.
        saved = (video_module._answer_gate, video_module._answer_gate_built, video_module._ask_service)
        self.addCleanup(lambda: (setattr(video_module, "_answer_gate", saved[0]), setattr(video_module, "_answer_gate_built", saved[1]),
                                 setattr(video_module, "_ask_service", saved[2])))
        video_module._answer_gate, video_module._answer_gate_built, video_module._ask_service = None, False, None

        self.provider = FakeProvider(reply=FAITHFUL)
        self.service = AskService(provider=self.provider)
        self.app = FastAPI()
        configure_cors(self.app)
        self.app.include_router(video_module.router)

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        self.app.dependency_overrides[get_db] = override_db
        self.app.dependency_overrides[video_module.get_ask_service] = lambda: self.service
        self.app.dependency_overrides[video_module.get_data_dir] = lambda: self.tmp.name
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def ask(self, question="What did the instructor say about LIATE?"):
        return self.client.post("/video/ask", json={"video_id": VIDEO_A, "question": question})

    def set_gate(self, gate):
        self.app.dependency_overrides[video_module.get_answer_gate] = lambda: gate

    # -- default ----------------------------------------------------------------------------

    def test_gate_is_off_by_default_and_the_response_is_unchanged(self):
        self.assertIsNone(video_module.get_answer_gate())
        body = self.ask().json()
        self.assertEqual((body["status"], body["answer_text"]), ("answered", FAITHFUL))
        self.assertEqual(set(body), {"status", "scope", "answer_text", "sources"})          # response schema is unchanged
        self.assertTrue(body["sources"])

    # -- injected gates -------------------------------------------------------------------

    def test_enforce_turns_a_low_score_into_insufficient_context(self):
        cfg = FaithfulnessConfig(mode="enforce", threshold=0.6)
        self.set_gate(lambda outcome: apply_faithfulness_gate(outcome, StubChecker(0.1), cfg))
        r = self.ask()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual((body["status"], body["answer_text"], body["sources"]), ("insufficient_context", GATE_MESSAGE, []))
        self.assertEqual(len(self.provider.calls), 1)                                     # the model was still called once

    def test_shadow_scores_but_returns_the_model_answer(self):
        cfg = FaithfulnessConfig(mode="shadow", threshold=0.6)
        checker = StubChecker(0.1)
        self.set_gate(lambda outcome: apply_faithfulness_gate(outcome, checker, cfg))
        body = self.ask().json()
        self.assertEqual((body["status"], body["answer_text"]), ("answered", FAITHFUL))
        self.assertEqual(checker.calls, 1)

    def test_a_passing_score_is_returned_unchanged(self):
        cfg = FaithfulnessConfig(mode="enforce", threshold=0.6)
        self.set_gate(lambda outcome: apply_faithfulness_gate(outcome, StubChecker(0.9), cfg))
        self.assertEqual(self.ask().json()["status"], "answered")

    def test_non_answered_responses_pass_through_the_gate(self):
        self.service = AskService(provider=FakeProvider(reply="[[INSUFFICIENT_CONTEXT]] Not covered."))
        checker = StubChecker(0.0)
        self.set_gate(lambda outcome: apply_faithfulness_gate(outcome, checker, FaithfulnessConfig(mode="enforce")))
        self.assertEqual(self.ask().json()["status"], "insufficient_context")
        self.assertEqual(checker.calls, 0)

    # -- environment-driven, with the real lexical checker --------------------------------

    def test_env_enforce_blocks_an_unfaithful_answer(self):
        os.environ.update({"ASK_FAITHFULNESS_MODE": "enforce", "ASK_FAITHFULNESS_THRESHOLD": "0.9"})
        self.service = AskService(provider=FakeProvider(reply=UNFAITHFUL))
        body = self.ask().json()
        self.assertEqual((body["status"], body["answer_text"], body["sources"]), ("insufficient_context", GATE_MESSAGE, []))

    def test_env_enforce_lets_a_faithful_answer_through(self):
        os.environ.update({"ASK_FAITHFULNESS_MODE": "enforce", "ASK_FAITHFULNESS_THRESHOLD": "0.9"})
        body = self.ask().json()
        self.assertEqual((body["status"], body["answer_text"]), ("answered", FAITHFUL))

    def test_env_shadow_logs_but_does_not_block_an_unfaithful_answer(self):
        os.environ.update({"ASK_FAITHFULNESS_MODE": "shadow", "ASK_FAITHFULNESS_THRESHOLD": "0.9"})
        self.service = AskService(provider=FakeProvider(reply=UNFAITHFUL))
        with self.assertLogs("qa.faithfulness", level="INFO") as logs:
            body = self.ask().json()
        self.assertEqual(body["status"], "answered")
        self.assertIn("would_block", logs.output[0])

    # -- hybrid wiring --------------------------------------------------------------------

    def test_get_ask_service_builds_no_selector_by_default(self):
        self.assertIsNone(self._real_service()._evidence_selector)

    def _real_service(self):
        video_module._ask_service = None
        return video_module.get_ask_service()

    def test_hybrid_flag_builds_a_lazy_selector_into_the_service(self):
        os.environ["ASK_HYBRID_RETRIEVAL"] = "1"
        service = self._real_service()
        self.assertIsInstance(service._evidence_selector, HybridSelector)
        self.assertIsNone(service._evidence_selector.embedder._model)                    # nothing has been loaded


if __name__ == "__main__":
    unittest.main()
