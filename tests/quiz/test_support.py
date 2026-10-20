"""Unit tests for the lexical and numeric screening helpers (quiz.support)."""

import unittest
from decimal import Decimal

from tests.quiz.helpers import LECTURE
from quiz.limits import QuizLimits
from quiz.reasons import Reason
from quiz.support import analyze, compare_quantity, evaluate_support, operator_conflict, stem, window_texts


def quantity(text):
    found = analyze(text).quantities
    assert len(found) == 1, found
    return found[0]


class StemTests(unittest.TestCase):
    def test_inflections_collapse(self):
        groups = [
            ("integration", "integrating", "integrate"),
            ("choose", "choosing", "chooses"),
            ("increase", "increases", "increasing"),
            ("part", "parts"),
            ("reaction", "reactions"),
            ("cell", "cells"),
            ("released", "release"),
            ("run", "running"),
        ]
        for group in groups:
            self.assertEqual(len({stem(w) for w in group}), 1, group)

    def test_unrelated_words_stay_apart_and_antonyms_are_not_detected(self):
        self.assertNotEqual(stem("increase"), stem("decrease"))   # known: no antonym handling
        self.assertNotEqual(stem("stroma"), stem("thylakoid"))

    def test_short_words_are_untouched(self):
        for word in ("ph", "dv", "gas", "mass"):
            self.assertEqual(stem(word), word)


class NumberParsingTests(unittest.TestCase):
    def test_digits_and_words_agree(self):
        for text in ("fifty ohms", "50 ohms", "50 ohm"):
            q = quantity(text)
            self.assertEqual((q.value, q.unit), (Decimal(50), "ohm"), text)
        self.assertEqual(analyze("one hundred twenty three").quantities[0].value, Decimal(123))
        self.assertEqual(analyze("two thousand five hundred").quantities[0].value, Decimal(2500))
        self.assertEqual(analyze("twenty-five").quantities[0].value, Decimal(25))
        self.assertEqual(analyze("1,200 meters").quantities[0].value, Decimal(1200))

    def test_decimals(self):
        for text in ("nine point eight", "9.8", "9.80"):
            q = analyze(text).quantities[0]
            self.assertEqual((q.value, q.is_decimal), (Decimal("9.8"), True), text)
        self.assertEqual(analyze("three point one four").quantities[0].value, Decimal("3.14"))

    def test_signs_percent_and_degrees(self):
        self.assertEqual(analyze("negative fifty ohms").quantities[0].value, Decimal(-50))
        self.assertEqual(analyze("-5 degrees").quantities[0].value, Decimal(-5))
        self.assertEqual(analyze("the value is minus three").quantities[0].value, Decimal(-3))
        self.assertEqual(analyze("u v minus integral").quantities, ())          # operator, not a sign
        for text in ("forty percent", "40%", "40 percent"):
            q = analyze(text).quantities[0]
            self.assertEqual((q.value, q.unit), (Decimal(40), "percent"), text)
        self.assertEqual(analyze("ninety degrees").quantities[0].unit, "degree")

    def test_leading_decimals(self):
        q = analyze("point five meters").quantities[0]
        self.assertEqual((q.value, q.is_decimal, q.unit), (Decimal("0.5"), True, "m"))

    def test_compound_units(self):
        for text in ("five meters per second squared", "5 m/s squared", "5 meters / second squared"):
            self.assertEqual(quantity(text).unit, "m/s^2", text)
        self.assertEqual(quantity("ten kilometers per hour").unit, "km/h")

    def test_bare_one_and_ordinary_words_are_not_numbers(self):
        for text in ("one of the students", "the second law", "a hundred", "point of view"):
            self.assertEqual(analyze(text).quantities, (), text)

    def test_checkable_rule(self):
        self.assertFalse(quantity("three").checkable)            # unitless 1 to 9 stays unchecked
        self.assertTrue(quantity("ten").checkable)
        self.assertTrue(quantity("0.5").checkable)
        self.assertTrue(quantity("three meters").checkable)
        self.assertTrue(quantity("negative three").checkable)

    def test_number_tokens_use_a_canonical_form(self):
        self.assertIn("#100", analyze("one hundred volts").tokens)
        self.assertEqual(analyze("9.8").tokens, analyze("nine point eight").tokens)
        self.assertEqual(analyze("9.80").tokens, analyze("9.8").tokens)


class CompareQuantityTests(unittest.TestCase):
    def evidence(self, text):
        return analyze(text).quantities

    def test_equal_values_are_supported_across_forms(self):
        ev = self.evidence("a resistance of fifty ohms")
        for claim in ("50 ohms", "fifty ohm", "50"):
            self.assertEqual(compare_quantity(quantity(claim), ev), "supported", claim)

    def test_different_value_with_the_same_unit_conflicts(self):
        ev = self.evidence("a resistance of fifty ohms")
        self.assertEqual(compare_quantity(quantity("sixty ohms"), ev), "conflict")

    def test_unit_change_conflicts_even_when_the_number_repeats_without_a_unit(self):
        ev = self.evidence("nine point eight meters per second squared and later still nine point eight")
        self.assertEqual(compare_quantity(quantity("9.8 meters per second squared"), ev), "supported")
        self.assertEqual(compare_quantity(quantity("9.8 kilometers per second squared"), ev), "conflict")
        self.assertEqual(compare_quantity(quantity("9.8"), ev), "supported")

    def test_sign_change_conflicts(self):
        ev = self.evidence("a resistance of fifty ohms")
        self.assertEqual(compare_quantity(quantity("negative fifty ohms"), ev), "conflict")

    def test_sign_flip_alone_conflicts_without_any_unit_or_context(self):
        self.assertEqual(compare_quantity(quantity("negative fifty"), self.evidence("fifty")), "conflict")
        self.assertEqual(compare_quantity(quantity("fifty"), self.evidence("negative fifty")), "conflict")

    def test_unrelated_value_is_absent(self):
        ev = self.evidence("a resistance of fifty ohms")
        self.assertEqual(compare_quantity(quantity("forty percent"), ev), "absent")
        self.assertEqual(compare_quantity(quantity("forty percent"), ()), "absent")

    def test_shared_context_word_counts_as_comparable(self):
        ev = self.evidence("the speed is twelve")
        self.assertEqual(compare_quantity(quantity("the speed is fifteen"), ev), "conflict")


class OperatorTests(unittest.TestCase):
    def check(self, claim, evidence):
        return operator_conflict(analyze(claim).words, analyze(evidence).words)

    def test_opposite_operator_only_in_the_evidence_conflicts(self):
        self.assertTrue(self.check("u v plus integral", "u v minus integral"))
        self.assertTrue(self.check("current divided by resistance", "current times resistance"))
        self.assertTrue(self.check("a positive charge", "a negative charge"))

    def test_matching_or_absent_operators_do_not_conflict(self):
        self.assertFalse(self.check("u v minus integral", "u v minus integral"))
        self.assertFalse(self.check("u v plus integral", "u v integral"))          # not detectable
        self.assertFalse(self.check("add them", "add them and subtract others"))   # claimed group present
        self.assertFalse(self.check("plus", "plus and minus"))


class WindowTests(unittest.TestCase):
    def test_sentences_are_kept_whole_when_short(self):
        self.assertEqual(window_texts("One short one. Another short one.", QuizLimits()),
                         ["One short one.", "Another short one."])

    def test_unpunctuated_text_is_cut_into_overlapping_windows_covering_the_tail(self):
        words = [f"w{i}" for i in range(60)]
        out = window_texts(" ".join(words), QuizLimits(window_words=24, window_step=12))
        self.assertTrue(all(len(w.split()) <= 24 for w in out))
        self.assertEqual(out[0].split()[0], "w0")
        self.assertEqual(out[-1].split()[-1], "w59")
        covered = {t for w in out for t in w.split()}
        self.assertEqual(covered, set(words))

    def test_empty_text_has_no_windows(self):
        self.assertEqual(window_texts("   ", QuizLimits()), [])


class EvaluateSupportTests(unittest.TestCase):
    P1 = LECTURE.passages[0]

    def run_mcq(self, prompt, options, key, passages=None):
        return evaluate_support(qtype="mcq", prompt=prompt, options=options, correct_index=key, answer=False,
                                passages=passages or [self.P1], limits=QuizLimits())

    def test_supported_question_has_no_reasons_and_a_primary_passage(self):
        e = self.run_mcq("Which rule helps you choose u in integration by parts?",
                         ["The LIATE rule", "The chain rule", "The quotient rule"], 0)
        self.assertEqual(e.reasons, ())
        self.assertEqual(e.primary_index, 1)
        self.assertIn(1, e.best_by_passage)

    def test_answer_coverage_boundary(self):
        base = ("Which rule helps you choose u in integration by parts?",)
        half = self.run_mcq(*base, ["Liate zebra", "The chain rule", "The quotient rule"], 0)
        third = self.run_mcq(*base, ["Liate zebra yak", "The chain rule", "The quotient rule"], 0)
        self.assertEqual(half.reasons, ())                                        # 1 of 2 = 0.5 passes
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, third.reasons)               # 1 of 3 fails

    def test_question_and_key_must_share_one_window(self):
        # topic words from the start of the passage, key word from the end: never in one 24-word window
        far = " ".join(["integration by parts"] + ["filler"] * 40 + ["liate"])
        passage = LECTURE.passages[0].__class__(id="S1", index=1, start=0.0, end=10.0, text=far, tokens=50)
        e = evaluate_support(qtype="mcq", prompt="What is integration by parts?",
                             options=["Liate", "Chain", "Quotient"], correct_index=0, answer=False,
                             passages=[passage], limits=QuizLimits())
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, e.reasons)

    def test_uncheckable_inputs_are_rejected_not_waved_through(self):
        e = self.run_mcq("What is it?", ["The LIATE rule", "chain", "quotient"], 0)
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, e.reasons)       # no content words in the question
        e = self.run_mcq("Which rule helps you choose u in integration by parts?", ["It is", "chain", "quotient"], 0)
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, e.reasons)       # no content words in the key

    def test_true_false_needs_most_of_the_statement(self):
        ok = evaluate_support(qtype="tf", prompt="The LIATE rule puts logarithmic functions first", options=[],
                              correct_index=-1, answer=True, passages=[self.P1], limits=QuizLimits())
        bad = evaluate_support(qtype="tf", prompt="The LIATE rule makes exponential growth harmless to tigers",
                               options=[], correct_index=-1, answer=True, passages=[self.P1], limits=QuizLimits())
        self.assertEqual(ok.reasons, ())
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, bad.reasons)


if __name__ == "__main__":
    unittest.main()
