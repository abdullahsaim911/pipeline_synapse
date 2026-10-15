"""AskService hooks for the optional pipeline stages: evidence selector, ASR sanitising, exposed passages. Defaults are unchanged."""

import os
import unittest
from unittest import mock

from tests.qa.helpers import FakeProvider, LECTURE_LINES, make_segments
from tests.qa.test_hybrid import FakeEmbedder, FakeReranker, lecture_lines
from qa import retrieval as R
from qa import service as S
from qa.hybrid import HybridConfig, HybridSelector
from qa.moment import MomentContext
from qa.service import AskConfig, AskService
from qa.transcript import normalize_segments

LIATE_Q = "What did the instructor say about LIATE?"


class RecordingSelector:
    def __init__(self, replace_plan=None):
        self.calls, self.replace_plan = [], replace_plan

    def __call__(self, question, segments, current_time=None, has_moment=False, config=None):
        self.calls.append({"question": question, "current_time": current_time, "has_moment": has_moment, "config": config})
        return self.replace_plan or R.select_evidence(question, segments, current_time=current_time, has_moment=has_moment, config=config)


class SelectorInjectionTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.segs = make_segments()

    def test_default_service_still_resolves_the_module_level_select_evidence_at_call_time(self):
        calls = []
        real = S.select_evidence

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(S, "select_evidence", spy):
            AskService(provider=self.provider).answer(LIATE_Q, self.segs)
        self.assertTrue(calls)

    def test_injected_selector_replaces_select_evidence(self):
        selector = RecordingSelector()
        with mock.patch.object(S, "select_evidence", side_effect=AssertionError("must not be used")):
            out = AskService(provider=self.provider, evidence_selector=selector).answer(LIATE_Q, self.segs, current_time=42.0)
        self.assertEqual(out.status, "answered")
        self.assertEqual(selector.calls[0]["question"], LIATE_Q)
        self.assertEqual(selector.calls[0]["current_time"], 42.0)
        self.assertFalse(selector.calls[0]["has_moment"])
        self.assertIsInstance(selector.calls[0]["config"], R.EvidenceConfig)

    def test_the_selected_evidence_is_what_the_model_sees_and_what_the_sources_report(self):
        only = R.EvidencePlan("ok", [R.EvidenceBlock(start=30.0, end=40.0, text="ONLY THIS PASSAGE", first_index=5, last_index=6, id="S1")],
                              complete=False, mode="retrieval", coverage=1.0)
        out = AskService(provider=self.provider, evidence_selector=RecordingSelector(only)).answer(LIATE_Q, self.segs)
        prompt = self.provider.calls[0]["prompt"]
        self.assertIn("ONLY THIS PASSAGE", prompt)
        self.assertNotIn("welcome to this lecture", prompt)
        self.assertEqual([(s.start_seconds, s.end_seconds) for s in out.sources], [(30.0, 40.0)])
        self.assertEqual(out.evidence_passages, ("ONLY THIS PASSAGE",))

    def test_a_selector_that_finds_nothing_resolves_without_the_model(self):
        none = R.EvidencePlan("no_evidence")
        out = AskService(provider=self.provider, evidence_selector=RecordingSelector(none)).answer(LIATE_Q, self.segs)
        self.assertEqual(out.status, "insufficient_context")
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(out.evidence_passages, ())

    def test_hybrid_selector_end_to_end_with_fake_models(self):
        segs = make_segments(lecture_lines())
        selector = HybridSelector(HybridConfig(enabled=True, rerank_top_k=1, candidates=6), embedder=FakeEmbedder(), reranker=FakeReranker())
        config = AskConfig(full_transcript_max_tokens=1)             # force retrieval mode instead of sending the whole short lecture
        out = AskService(provider=self.provider, config=config, evidence_selector=selector).answer("What is an eigenvalue?", segs)
        prompt = self.provider.calls[0]["prompt"]
        self.assertIn("determinant", prompt)                         # the reranker's choice
        self.assertNotIn("an eigenvalue is a scalar", prompt)
        self.assertEqual(out.status, "answered")
        self.assertTrue(all(s.start_seconds >= 36 * 6.0 - 30 for s in out.sources))

    def test_hybrid_selector_failure_leaves_the_answer_working(self):
        segs = make_segments(lecture_lines())
        selector = HybridSelector(HybridConfig(enabled=True, rerank_top_k=1, candidates=6), embedder=FakeEmbedder(fail=True), reranker=FakeReranker())
        out = AskService(provider=self.provider, config=AskConfig(full_transcript_max_tokens=1), evidence_selector=selector).answer("What is an eigenvalue?", segs)
        self.assertEqual(out.status, "answered")
        self.assertIn("an eigenvalue is a scalar", self.provider.calls[0]["prompt"])


class AsrSanitiseTests(unittest.TestCase):
    ROWS = [
        {"start": 0.0, "end": 6.0, "text": "welcome um to the lecture on the anti derivative"},
        {"start": 6.0, "end": 12.0, "text": "uh"},
        {"start": 12.0, "end": 18.0, "text": "remember the the acronym li8 for choosing u"},
        {"start": 18.0, "end": 24.0, "text": "then differentiate u and integrate dv"},
    ]

    def ask(self, **config):
        provider = FakeProvider()
        out = AskService(provider=provider, config=AskConfig(**config)).answer("What is the acronym for choosing u?", normalize_segments(self.ROWS))
        return provider.calls[0]["prompt"], out

    def test_default_leaves_the_captions_exactly_as_they_were(self):
        self.assertFalse(AskConfig().asr_sanitize)
        prompt, _ = self.ask()
        self.assertIn("anti derivative", prompt)
        self.assertIn("li8", prompt)
        self.assertIn(" um ", prompt)

    def test_enabled_cleans_the_text_the_model_sees(self):
        prompt, out = self.ask(asr_sanitize=True)
        self.assertIn("antiderivative", prompt)
        self.assertIn("acronym LIATE for choosing u", prompt)
        self.assertNotIn(" um ", prompt)
        self.assertNotIn("the the", prompt)
        self.assertEqual(out.status, "answered")

    def test_source_time_ranges_are_unaffected_by_cleaning(self):
        _, plain = self.ask()
        _, clean = self.ask(asr_sanitize=True)
        self.assertEqual([(s.start_seconds, s.end_seconds) for s in clean.sources], [(0.0, 24.0)])
        self.assertEqual(clean.sources[0].start_seconds, plain.sources[0].start_seconds)

    def test_flag_is_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {"ASK_ASR_SANITIZE": "1"}):
            self.assertTrue(AskConfig.from_env().asr_sanitize)
        with mock.patch.dict(os.environ, {"ASK_ASR_SANITIZE": "no"}):
            self.assertFalse(AskConfig.from_env().asr_sanitize)
        with mock.patch.dict(os.environ):
            os.environ.pop("ASK_ASR_SANITIZE", None)
            self.assertFalse(AskConfig.from_env().asr_sanitize)

    def test_a_transcript_that_cleans_to_nothing_is_reported_as_empty(self):
        segs = normalize_segments([{"start": 0.0, "end": 5.0, "text": "um"}, {"start": 5.0, "end": 9.0, "text": "uh, er"}])
        out = AskService(provider=FakeProvider(), config=AskConfig(asr_sanitize=True)).answer("What is said?", segs)
        self.assertEqual(out.status, "insufficient_context")


class EvidencePassageTests(unittest.TestCase):
    def test_answered_outcomes_carry_the_passages_the_model_was_given(self):
        provider = FakeProvider()
        out = AskService(provider=provider).answer(LIATE_Q, make_segments())
        self.assertTrue(out.evidence_passages)
        for passage in out.evidence_passages:
            self.assertIn(passage, provider.calls[0]["prompt"])

    def test_the_attached_moment_text_is_included(self):
        moment = MomentContext(intervention_id="int-1", timestamp=60.0, content_type="equation", caption="CAPTION TEXT",
                               explanation="EXPLANATION TEXT", visual=[("Equation", "VISUAL TEXT")])
        out = AskService(provider=FakeProvider()).answer("What does this equation mean?", make_segments(), moment=moment)
        for text in ("CAPTION TEXT", "EXPLANATION TEXT", "VISUAL TEXT"):
            self.assertIn(text, out.evidence_passages)

    def test_non_answered_outcomes_have_no_passages(self):
        provider = FakeProvider(reply="[[INSUFFICIENT_CONTEXT]] The excerpts do not cover this.")
        out = AskService(provider=provider).answer(LIATE_Q, make_segments())
        self.assertEqual((out.status, out.evidence_passages), ("insufficient_context", ()))

    def test_field_defaults_keep_existing_constructors_valid(self):
        out = S.AskOutcome("insufficient_context", "lecture", "text", [])
        self.assertEqual(out.evidence_passages, ())


if __name__ == "__main__":
    unittest.main()
