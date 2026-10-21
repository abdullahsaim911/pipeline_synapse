"""
Two problems seen in the first real run (one model, four lectures, 2026-10-20):

1. A caption error became a question. The captions said "the acronym li8" (the mnemonic is LIATE) and the model wrote a
   question whose key was "li8". Screening now rejects a letter-and-digit blend that is not a recognisable formula.
2. Two questions tested one fact. Two true/false questions rested on the same supporting sentence about adding "plus C",
   and two questions shared one concept label. Screening now rejects the second.

The strings below are copied from that real run's output; no model is called here.
"""

import unittest

from tests.quiz.helpers import LECTURE, QUIZ_ID, concept, mcq, tf
from quiz.limits import QuizLimits
from quiz.reasons import Reason
from quiz.screening import _suspect_caption_token, screen_candidates


def screen(candidates, requested=6, limits=QuizLimits()):
    return screen_candidates(candidates, LECTURE, quiz_id=QUIZ_ID, requested_count=requested, limits=limits)


def reasons_of(result, index=0):
    return [r for rej in result.rejected if rej.index == index for r in rej.reasons]


class SuspectCaptionTokenTests(unittest.TestCase):
    def test_the_token_from_the_real_run_is_suspect(self):
        self.assertTrue(_suspect_caption_token("li8"))
        self.assertTrue(_suspect_caption_token("In the acronym li8, what does the letter l stand for?"))
        self.assertTrue(_suspect_caption_token("Li8"))

    def test_the_real_questions_that_were_fine_are_not_suspect(self):
        for text in (
            "What acronym is used to help select the part of the function that equals U?",
            "Peroxisomes accumulate and degrade which dangerous substance?",
            "If a fish pushes the water back with a force of ten newtons, what force does the water push back on the fish with?",
        ):
            self.assertFalse(_suspect_caption_token(text), text)

    def test_ordinary_formulas_units_and_variables_are_not_suspect(self):
        for text in ("water is h2o", "carbon dioxide co2", "ammonia nh3", "oxygen is o2", "glucose c6h12o6", "sulfuric acid h2so4",
                     "the value of x2 and v0", "a 3d model", "the 2nd law", "utf8 and base64 and ipv6", "an mp3 file"):
            self.assertFalse(_suspect_caption_token(text), text)

    def test_gibberish_blends_and_lone_element_counts_are_suspect(self):
        for text in ("abc3", "the zq9 rule", "cl8", "na4 reacts", "ph7x"):
            self.assertTrue(_suspect_caption_token(text), text)

    def test_the_check_ignores_case_and_punctuation(self):
        self.assertTrue(_suspect_caption_token("LI8!"))
        self.assertFalse(_suspect_caption_token("H2O."))


class SuspectTokenScreeningTests(unittest.TestCase):
    def test_a_key_or_distractor_or_prompt_with_a_suspect_token_is_rejected(self):
        base = mcq(0)
        self.assertEqual(len(screen([base]).accepted), 1)
        for field, value in (("prompt", base["prompt"].rstrip("?") + " li8?"),
                             ("choices", [base["choices"][0], "li8", base["choices"][2]][: len(base["choices"])])):
            result = screen([mcq(0, **{field: value})])
            self.assertEqual(len(result.accepted), 0, field)
            self.assertIn(Reason.SUSPECT_CAPTION_TOKEN, reasons_of(result), field)

    def test_true_false_prompts_are_checked_too(self):
        result = screen([tf(0, prompt="The li8 rule puts logarithmic functions first")])
        self.assertEqual(reasons_of(result), [Reason.SUSPECT_CAPTION_TOKEN])


class SameFactDuplicateTests(unittest.TestCase):
    def test_the_same_concept_label_is_a_duplicate_even_with_different_wording(self):
        # real run: two physics questions were both labelled "Forces on Different Objects"
        a = tf(0, concept={"id": "two-different-objects", "label": "Forces on Different Objects"})
        b = tf(1, concept={"id": "forces-on-different-objects", "label": "Forces on Different Objects"})
        result = screen([a, b])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(reasons_of(result, 1), [Reason.DUPLICATE_QUESTION])

    def test_label_comparison_ignores_case_and_punctuation(self):
        a = tf(0, concept={"id": "alpha-one", "label": "Forces on Different Objects"})
        b = tf(1, concept={"id": "alpha-two", "label": "forces on different objects!"})
        self.assertEqual(reasons_of(screen([a, b]), 1), [Reason.DUPLICATE_QUESTION])

    def test_the_same_concept_id_is_a_duplicate(self):
        a = tf(0, concept={"id": "same-id", "label": "One label"})
        b = tf(1, concept={"id": "same-id", "label": "Another label"})
        self.assertEqual(reasons_of(screen([a, b]), 1), [Reason.DUPLICATE_QUESTION])

    def test_two_questions_on_the_same_supporting_text_are_a_duplicate(self):
        # real run: two true/false questions about adding "plus C" had different concept ids but the same supporting sentence.
        # In the fixture, topic 2's MCQ and true/false rest on the same supporting window.
        a = mcq(2, concept=concept("first", 2))
        b = tf(2, concept=concept("second", 2))
        result = screen([a, b])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(reasons_of(result, 1), [Reason.DUPLICATE_QUESTION])
        # it is the supporting-text rule, not the wording or the concept, that rejects it
        self.assertEqual(len(screen([a, b], limits=QuizLimits(duplicate_support_overlap=2.0)).accepted), 2)

    def test_questions_on_different_supporting_text_in_one_passage_are_both_kept(self):
        a = mcq(0, concept=concept("first"))
        b = tf(0, concept=concept("second"))
        self.assertEqual(len(screen([a, b]).accepted), 2)         # topic 0's two questions rest on different windows

    def test_distinct_topics_are_unaffected(self):
        self.assertEqual(len(screen([mcq(i) for i in range(4)]).accepted), 4)
        self.assertEqual(len(screen([mcq(0), mcq(1), tf(3), tf(4)]).accepted), 4)

    def test_the_overlap_threshold_is_a_provisional_limit_in_one_place(self):
        self.assertEqual(QuizLimits().duplicate_support_overlap, 0.6)


if __name__ == "__main__":
    unittest.main()
