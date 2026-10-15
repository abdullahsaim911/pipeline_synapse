"""qa.faithfulness: sentence support checks and the shadow / enforce gate."""

import os
import unittest
from unittest import mock

from qa.faithfulness import (
    GATE_MESSAGE,
    spoken_numbers,
    FaithfulnessConfig,
    FaithfulnessResult,
    LexicalSupportChecker,
    NLIChecker,
    apply_faithfulness_gate,
    build_gate_from_env,
    checkable_sentences,
    split_sentences,
)
from qa.service import STATUS_ANSWERED, STATUS_INSUFFICIENT, STATUS_NOT_IN_LECTURE, AskOutcome, AskSourceInfo

PASSAGES = [
    "a helpful rule of thumb is the acronym LIATE L stands for logarithmic functions and I for inverse trigonometric functions",
    "pick u as the function that comes first in LIATE the rest of the integrand becomes dv",
    "u equals x so du equals dx dv equals e to the x dx so v equals e to the x",
    "this confirms that integration by parts reverses the product rule",
]


class SentenceHandlingTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_sentences("One thing. Another thing! A third?"), ["One thing.", "Another thing!", "A third?"])
        self.assertEqual(split_sentences(""), [])

    def test_absence_statements_and_short_sentences_are_not_checked(self):
        text = "The excerpts do not state the formula. Yes. The instructor says that you pick u as the function that comes first in LIATE."
        sentences, checked = checkable_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(len(checked), 1)
        self.assertIn("LIATE", checked[0][0])

    def test_attribution_prefix_is_removed_before_terms_are_taken(self):
        _, checked = checkable_sentences("The instructor says that integration reverses the product rule.")
        self.assertNotIn("instructor", checked[0][1])
        _, checked = checkable_sentences("From these steps, it seems that du equals dx.")
        self.assertFalse(any(t in ("seem", "step") for t in checked[0][1]))


class LexicalCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = LexicalSupportChecker(FaithfulnessConfig(mode="shadow"))

    def test_supported_answer_scores_one(self):
        r = self.checker.check("The instructor says that you pick u as the function that comes first in LIATE.", PASSAGES)
        self.assertEqual((r.score, r.n_checked, r.n_supported, r.unsupported), (1.0, 1, 1, ()))

    def test_unrelated_claim_is_unsupported(self):
        r = self.checker.check("The derivative of sine is cosine and the graph is a parabola with a vertex.", PASSAGES)
        self.assertEqual((r.score, r.n_checked), (0.0, 1))
        self.assertEqual(len(r.unsupported), 1)

    def test_mixed_answer_scores_the_supported_share(self):
        text = ("You pick u as the function that comes first in LIATE. "
                "Photosynthesis converts sunlight into chemical energy inside chloroplasts.")
        r = self.checker.check(text, PASSAGES)
        self.assertEqual((r.n_checked, r.n_supported, r.score), (2, 1, 0.5))

    def test_only_absence_statements_pass_vacuously(self):
        r = self.checker.check("The excerpts do not state the full formula.", PASSAGES)
        self.assertEqual((r.score, r.n_checked), (1.0, 0))

    def test_derived_sentence_with_label_is_checked_against_the_stated_steps(self):
        r = self.checker.check("From this, it seems that integration by parts reverses the product rule.", PASSAGES)
        self.assertEqual(r.score, 1.0)

    def test_digits_must_appear_in_the_passages_but_spoken_numbers_count(self):
        self.assertEqual(self.checker.check("Then du equals 2 dx here.", ["so du equals two dx here"]).score, 1.0)
        self.assertEqual(self.checker.check("Then du equals 7 dx here.", ["so du equals two dx here"]).score, 0.0)

    def test_a_wrong_number_fails_the_sentence_even_when_the_other_words_match(self):
        r = self.checker.check("Then du equals 7 dx here and v equals e to the x.", ["so du equals two dx here and v equals e to the x"])
        self.assertEqual(r.score, 0.0)

    def test_compound_spoken_numbers_support_digits(self):
        self.assertEqual(spoken_numbers("forty six and two hundred and seven".split()), {46, 200, 7})
        self.assertEqual(spoken_numbers("twenty nineteen".split()), {20, 19})
        self.assertEqual(self.checker.check("There are 46 chromosomes in a human cell.", ["humans have forty six chromosomes in each cell"]).score, 1.0)

    def test_no_passages_means_nothing_is_supported(self):
        self.assertEqual(self.checker.check("You pick u from LIATE first.", []).score, 0.0)

    def test_empty_answer_has_nothing_to_check(self):
        r = self.checker.check("", PASSAGES)
        self.assertEqual((r.score, r.n_sentences, r.n_checked), (1.0, 0, 0))

    def test_is_deterministic(self):
        text = "You pick u as the function that comes first in LIATE. Chloroplasts convert sunlight."
        self.assertEqual(self.checker.check(text, PASSAGES), self.checker.check(text, PASSAGES))

    def test_support_ratio_setting_matters(self):
        text = "You pick u as the function from LIATE using chloroplast sunlight rules."
        strict = LexicalSupportChecker(FaithfulnessConfig(mode="shadow", term_support_ratio=0.9)).check(text, PASSAGES)
        lenient = LexicalSupportChecker(FaithfulnessConfig(mode="shadow", term_support_ratio=0.2)).check(text, PASSAGES)
        self.assertLess(strict.score, lenient.score)


class FakeNLIModel:
    """Returns logits [contradiction, entailment, neutral]; entails when the premise contains the sentence's marker word."""

    class _Cfg:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    config = _Cfg()

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [[0.0, 5.0, 0.0] if "liate" in premise.lower() and "liate" in hyp.lower() else [4.0, 0.0, 1.0] for premise, hyp in pairs]


class NLICheckerTests(unittest.TestCase):
    def test_entailed_and_unentailed_sentences(self):
        model = FakeNLIModel()
        checker = NLIChecker(FaithfulnessConfig(mode="shadow", method="nli"), model=model)
        r = checker.check("You pick u from LIATE first. Chloroplasts convert sunlight into energy.", PASSAGES)
        self.assertEqual((r.method, r.n_checked, r.n_supported, r.score), ("nli", 2, 1, 0.5))
        self.assertEqual(len(model.pairs), 2)

    def test_premise_uses_the_most_overlapping_passages_and_stays_short(self):
        model = FakeNLIModel()
        NLIChecker(FaithfulnessConfig(mode="shadow", method="nli"), model=model).check("You pick u from LIATE first.", PASSAGES * 10)
        premise, hypothesis = model.pairs[0]
        self.assertIn("LIATE", premise)
        self.assertLessEqual(len(premise), 2 * 900 + 1)
        self.assertEqual(hypothesis, "You pick u from LIATE first.")

    def test_probabilities_accept_logits_or_probabilities(self):
        self.assertAlmostEqual(sum(NLIChecker._probabilities([0.0, 5.0, 0.0])), 1.0)
        self.assertEqual(NLIChecker._probabilities([0.1, 0.8, 0.1]), [0.1, 0.8, 0.1])

    def test_entailment_index_is_read_from_the_model_labels(self):
        class Swapped(FakeNLIModel):
            class _Cfg:
                id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
            config = _Cfg()

            def predict(self, pairs):
                return [[5.0, 0.0, 0.0] for _ in pairs]

        r = NLIChecker(FaithfulnessConfig(mode="shadow", method="nli"), model=Swapped()).check("You pick u from LIATE first.", PASSAGES)
        self.assertEqual(r.score, 1.0)

    def test_nothing_to_check_does_not_load_a_model(self):
        checker = NLIChecker(FaithfulnessConfig(mode="shadow", method="nli"))
        self.assertEqual(checker.check("The excerpts do not state it.", PASSAGES).score, 1.0)
        self.assertIsNone(checker._model)


def outcome(status=STATUS_ANSWERED, text="You pick u from LIATE first.", passages=tuple(PASSAGES)):
    sources = [AskSourceInfo(1.0, 5.0)] if status == STATUS_ANSWERED else []
    return AskOutcome(status, "lecture", text, sources, estimated_prompt_tokens=123, evidence_mode="retrieval", evidence_passages=passages)


class FixedChecker:
    def __init__(self, score=0.0, fail=False):
        self.score, self.fail, self.calls = score, fail, 0

    def check(self, answer, passages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return FaithfulnessResult(self.score, "fixed", 3, 2, 1)


class GateTests(unittest.TestCase):
    def cfg(self, mode, threshold=0.6):
        return FaithfulnessConfig(mode=mode, threshold=threshold)

    def test_off_does_nothing_and_never_calls_the_checker(self):
        checker, o = FixedChecker(0.0), outcome()
        self.assertIs(apply_faithfulness_gate(o, checker, self.cfg("off")), o)
        self.assertEqual(checker.calls, 0)

    def test_shadow_logs_but_never_changes_the_response(self):
        checker, o = FixedChecker(0.1), outcome()
        with self.assertLogs("qa.faithfulness", level="INFO") as logs:
            result = apply_faithfulness_gate(o, checker, self.cfg("shadow"))
        self.assertIs(result, o)
        self.assertEqual(checker.calls, 1)
        self.assertIn("would_block", logs.output[0])
        self.assertNotIn("LIATE", logs.output[0])               # answer text is never logged

    def test_enforce_blocks_a_low_score_with_insufficient_context(self):
        o = outcome()
        r = apply_faithfulness_gate(o, FixedChecker(0.2), self.cfg("enforce"))
        self.assertEqual((r.status, r.answer_text, r.sources), (STATUS_INSUFFICIENT, GATE_MESSAGE, []))
        self.assertEqual((r.scope, r.estimated_prompt_tokens, r.evidence_mode), (o.scope, 123, "retrieval"))

    def test_blocked_responses_are_never_not_in_lecture(self):
        for score in (0.0, 0.3, 0.59):
            self.assertNotEqual(apply_faithfulness_gate(outcome(), FixedChecker(score), self.cfg("enforce")).status, STATUS_NOT_IN_LECTURE)

    def test_score_at_or_above_the_threshold_passes(self):
        for score in (0.6, 0.8, 1.0):
            o = outcome()
            self.assertIs(apply_faithfulness_gate(o, FixedChecker(score), self.cfg("enforce", 0.6)), o)

    def test_only_answered_responses_are_checked(self):
        for status in (STATUS_INSUFFICIENT, STATUS_NOT_IN_LECTURE):
            checker, o = FixedChecker(0.0), outcome(status, "That is not covered.")
            self.assertIs(apply_faithfulness_gate(o, checker, self.cfg("enforce")), o)
            self.assertEqual(checker.calls, 0)

    def test_missing_passages_leave_the_response_alone(self):
        checker, o = FixedChecker(0.0), outcome(passages=())
        self.assertIs(apply_faithfulness_gate(o, checker, self.cfg("enforce")), o)
        self.assertEqual(checker.calls, 0)

    def test_a_failing_checker_fails_open(self):
        o = outcome()
        with self.assertLogs("qa.faithfulness", level="WARNING") as logs:
            self.assertIs(apply_faithfulness_gate(o, FixedChecker(fail=True), self.cfg("enforce")), o)
        self.assertIn("response left unchanged", logs.output[0])

    def test_no_checker_means_no_change(self):
        o = outcome()
        self.assertIs(apply_faithfulness_gate(o, None, self.cfg("enforce")), o)

    def test_end_to_end_with_the_real_lexical_checker(self):
        checker = LexicalSupportChecker(self.cfg("enforce"))
        good = outcome(text="You pick u as the function that comes first in LIATE.")
        bad = outcome(text="Photosynthesis converts sunlight into chemical energy inside chloroplasts.")
        self.assertIs(apply_faithfulness_gate(good, checker, self.cfg("enforce")), good)
        self.assertEqual(apply_faithfulness_gate(bad, checker, self.cfg("enforce")).status, STATUS_INSUFFICIENT)


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"mode": "block"}, {"method": "magic"}, {"threshold": 1.5}, {"term_support_ratio": -0.1}, {"min_terms": 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                FaithfulnessConfig(**kwargs)

    def test_default_is_off(self):
        self.assertEqual(FaithfulnessConfig().mode, "off")

    def test_from_env(self):
        env = {"ASK_FAITHFULNESS_MODE": "ENFORCE", "ASK_FAITHFULNESS_METHOD": "nli", "ASK_FAITHFULNESS_THRESHOLD": "0.75", "ASK_NLI_MODEL": "x/y"}
        with mock.patch.dict(os.environ, env):
            cfg = FaithfulnessConfig.from_env()
        self.assertEqual((cfg.mode, cfg.method, cfg.threshold, cfg.nli_model), ("enforce", "nli", 0.75, "x/y"))

    def test_invalid_env_values_fall_back_to_safe_defaults(self):
        env = {"ASK_FAITHFULNESS_MODE": "maybe", "ASK_FAITHFULNESS_METHOD": "??", "ASK_FAITHFULNESS_THRESHOLD": "7"}
        with mock.patch.dict(os.environ, env):
            cfg = FaithfulnessConfig.from_env()
        self.assertEqual((cfg.mode, cfg.method, cfg.threshold), ("off", "lexical", 0.6))

    def test_build_gate_from_env(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("ASK_FAITHFULNESS_MODE", None)
            self.assertIsNone(build_gate_from_env())
            os.environ["ASK_FAITHFULNESS_MODE"] = "shadow"
            gate = build_gate_from_env()
        self.assertTrue(callable(gate))
        o = outcome()
        self.assertIs(gate(o), o)                              # shadow never changes the response

    def test_nli_method_is_selected_without_loading_anything(self):
        with mock.patch.dict(os.environ, {"ASK_FAITHFULNESS_MODE": "shadow", "ASK_FAITHFULNESS_METHOD": "nli"}):
            self.assertTrue(callable(build_gate_from_env()))


if __name__ == "__main__":
    unittest.main()
