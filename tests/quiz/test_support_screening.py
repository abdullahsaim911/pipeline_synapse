"""
Adversarial tests for the evidence-support screening, run through screen_candidates.

These tests document what the heuristics catch AND what they do not. A passing
case is "not rejected by these heuristics"; it is never "verified correct".
"""

import unittest
from unittest import mock

from tests.quiz.helpers import concept, LECTURE, PASSAGE_TEXTS, QUIZ_ID, REVIEW_TF, TOPICS, lecture_from, mcq, tf
from quiz.passages import EvidenceRange, map_evidence_ids, verify_evidence_ranges
from quiz.reasons import Reason
from quiz.limits import QuizLimits
from quiz.screening import screen_candidates

RESISTANCE = "A circuit can have a resistance of "


def screen(candidates, requested=6, passages=LECTURE, limits=QuizLimits()):
    return screen_candidates(candidates, passages, quiz_id=QUIZ_ID, requested_count=requested, limits=limits)


def reasons_of(result, index=0):
    return [r for rej in result.rejected if rej.index == index for r in rej.reasons]


class IrrelevantEvidenceTests(unittest.TestCase):
    def test_real_but_irrelevant_passage_is_rejected_for_every_topic(self):
        for n in range(8):
            wrong = f"S{(n + 3) % 8 + 1}"
            for candidate in (mcq(n, evidence_ids=[wrong]), tf(n, evidence_ids=[wrong])):
                result = screen([candidate])
                self.assertEqual(result.accepted, (), (n, candidate["type"]))
                self.assertIn(Reason.NO_SUPPORTING_SENTENCE, reasons_of(result))

    def test_padding_with_an_irrelevant_second_passage_is_rejected(self):
        result = screen([mcq(0, evidence_ids=["S1", "S5"])])
        self.assertEqual(result.accepted, ())
        self.assertEqual(reasons_of(result), [Reason.CITED_PASSAGE_NOT_RELEVANT])

    def test_a_relevant_second_passage_is_allowed(self):
        result = screen([mcq(0, evidence_ids=["S1", "S9"])])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(len(result.accepted[0].evidence), 2)

    def test_question_unrelated_to_the_lecture_is_rejected(self):
        result = screen([mcq(0, prompt="Which tropical bird migrates furthest each winter?",
                             choices=["The arctic tern", "The emu", "The kiwi"])])
        self.assertEqual(result.accepted, ())


class KeyConflictTests(unittest.TestCase):
    def test_key_pointing_at_a_distractor_that_the_evidence_does_not_support(self):
        result = screen([mcq(1, correct_index=1)])      # key: "Zero because the ball stops"
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.KEY_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_key_absent_from_the_evidence_while_a_distractor_is_present(self):
        result = screen([mcq(2, correct_index=1)])      # key: "Nitrogen"; "Oxygen" is in the text
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, reasons_of(result))
        self.assertIn(Reason.KEY_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_two_equally_supported_options_are_ambiguous(self):
        """Key and distractor differ only in words that are both in the text; the position is not compared."""
        tie = mcq(0, prompt="Which functions does the LIATE rule say to pick as u?",
                  choices=["Logarithmic functions first", "Exponential functions first", "Algebraic functions"])
        result = screen([tie])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.ANSWER_MARGIN_TOO_SMALL, reasons_of(result))

    def test_distractor_that_shares_the_supported_word_but_adds_an_unsupported_one_passes(self):
        """'LIATE method' is judged wrong only because 'method' is absent: a lexical call, not proof."""
        result = screen([mcq(0, choices=["The LIATE rule", "The LIATE method", "The chain rule"])])
        self.assertEqual(len(result.accepted), 1)

    def test_known_limitation_same_sentence_distractors_are_rejected(self):
        """A good question whose distractors come from the same sentence is lost (false rejection)."""
        phases = mcq(3, prompt="During which phase of mitosis do chromosomes line up in the middle of the cell?",
                     choices=["Metaphase", "Prophase", "Anaphase", "Telophase"])
        result = screen([phases])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.ANSWER_MARGIN_TOO_SMALL, reasons_of(result))

    def test_distractors_outside_the_passage_pass(self):
        phases = mcq(3, prompt="During which phase of mitosis do chromosomes line up in the middle of the cell?",
                     choices=["Metaphase", "Interphase", "Cytokinesis", "Meiosis"])
        self.assertEqual(len(screen([phases]).accepted), 1)

    def test_known_limitation_negation_flip_is_not_detected(self):
        flipped = tf(1, prompt="A ball thrown straight up has no zero velocity at the top of its path")
        self.assertEqual(len(screen([flipped]).accepted), 1)


class ExactValueTests(unittest.TestCase):
    def test_changed_number_in_the_question(self):
        prompt = "A circuit has a current of three amperes and a resistance of fifty ohms. What is the voltage?"
        result = screen([mcq(4, prompt=prompt)])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_changed_number_in_the_key(self):
        result = screen([mcq(4, choices=["Two hundred volts", "Twenty five volts", "Fifty two volts", "Four volts"])])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_changed_decimal(self):
        gravity = "Gravity pulls the ball down with an acceleration of {} meters per second squared"
        for form in ("nine point six", "9.6", "9.79"):
            result = screen([tf(1, prompt=gravity.format(form))])
            self.assertIn(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE, reasons_of(result), form)

    def test_same_value_in_a_different_written_form_passes(self):
        gravity = "Gravity pulls the ball down with an acceleration of {} meters per second squared"
        for form in ("nine point eight", "9.8", "9.80"):
            self.assertEqual(len(screen([tf(1, prompt=gravity.format(form))]).accepted), 1, form)
        digits = "A circuit with a current of 2 amperes and a resistance of 50 ohms has a voltage of 100 volts"
        self.assertEqual(len(screen([tf(4, prompt=digits)]).accepted), 1)

    def test_changed_unit(self):
        prompt = "Gravity pulls the ball down with an acceleration of nine point eight kilometers per second squared"
        result = screen([tf(1, prompt=prompt)])
        self.assertIn(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_changed_sign(self):
        result = screen([tf(4, prompt=RESISTANCE + "negative fifty ohms")])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE, reasons_of(result))

    def test_value_with_nothing_comparable_in_the_evidence(self):
        prompt = "Binary search discards half of the list each time and runs in forty percent of the time"
        for answer in (True, False):
            result = screen([tf(6, prompt=prompt, answer=answer)])
            self.assertEqual(result.accepted, (), answer)
            self.assertIn(Reason.NUMBER_NOT_IN_EVIDENCE, reasons_of(result), answer)

    def test_false_statement_may_carry_a_different_comparable_value(self):
        prompt = "Gravity pulls the ball down with an acceleration of nine point six meters per second squared"
        result = screen([tf(1, prompt=prompt, answer=False)])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].correct_choice_id, "false")

    def test_changed_operator(self):
        for prompt, topic in (
            ("Integration by parts says the integral of u dv equals u v plus the integral of v du", 0),
            ("Ohms law says that voltage equals current divided by resistance", 4),
        ):
            result = screen([tf(topic, prompt=prompt)])
            self.assertEqual(result.accepted, (), prompt)
            self.assertIn(Reason.OPERATOR_CONFLICTS_WITH_EVIDENCE, reasons_of(result), prompt)
        correct = screen([tf(4, prompt="Ohms law says that voltage equals current times resistance")])
        self.assertEqual(len(correct.accepted), 1)

    def test_a_false_statement_may_flip_an_operator(self):
        prompt = "Integration by parts says the integral of u dv equals u v plus the integral of v du"
        result = screen([tf(0, prompt=prompt, answer=False)])
        self.assertEqual(len(result.accepted), 1)

    def test_known_limitation_small_integers_are_not_checked(self):
        """Contract: unitless integers 1 to 9 cannot be checked, so this wrong figure passes."""
        wrong = tf(3, prompt="Mitosis divides a cell into three identical daughter cells")
        self.assertEqual(len(screen([wrong]).accepted), 1)

    def test_known_limitation_derived_numbers_are_rejected(self):
        derived = mcq(7, choices=["Thirty meters per second squared", "Five meters per second squared",
                                  "Sixteen meters per second squared", "Eighty meters per second squared"])
        result = screen([derived])
        self.assertEqual(result.accepted, ())


class ParaphraseTests(unittest.TestCase):
    def test_reworded_questions_that_keep_the_key_terms_pass(self):
        cases = [
            mcq(0, prompt="Which mnemonic helps you choose u when integrating by parts?",
                choices=["The LIATE mnemonic", "The chain rule", "The quotient rule", "The ratio test"]),
            mcq(2, prompt="What gas does the light reaction release as a byproduct?"),
            tf(0, prompt="According to integration by parts, the integral of u dv equals u times v minus the integral of v du"),
            tf(7, prompt="A 20 newton force on a 4 kilogram mass gives an acceleration of 5 meters per second squared"),
            tf(6, prompt="The list that binary search discards half of each time"),
        ]
        for candidate in cases:
            result = screen([candidate])
            self.assertEqual(len(result.accepted), 1, (candidate["prompt"], reasons_of(result)))

    def test_known_limitation_a_paraphrase_sharing_no_content_word_is_rejected(self):
        result = screen([mcq(0, prompt="Which mnemonic assists with the substitution selection?",
                             choices=["The LIATE mnemonic", "The chain rule", "The quotient rule"])])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, reasons_of(result))

    def test_lexical_overlap_is_not_a_correctness_signal(self):
        """A confidently wrong TF key with full word overlap is still accepted: screening is not proof."""
        wrong_but_overlapping = tf(0, prompt="Integration by parts says the integral of u dv equals u v minus "
                                             "the integral of v du", answer=False)
        self.assertEqual(len(screen([wrong_but_overlapping]).accepted), 1)

    def test_answer_boundary_is_inclusive(self):
        base = "Which rule helps you choose u in integration by parts?"
        half = mcq(0, prompt=base, choices=["Liate zebra", "The chain rule", "The quotient rule"])
        third = mcq(0, prompt=base, choices=["Liate zebra yak", "The chain rule", "The quotient rule"])
        self.assertEqual(len(screen([half]).accepted), 1)
        self.assertEqual(screen([third]).accepted, ())


class QuoteAndTimestampConsistencyTests(unittest.TestCase):
    def evidence_consistent(self, question, passage_ids):
        ranges, _ = map_evidence_ids(passage_ids, LECTURE)
        self.assertEqual(len(ranges), len(question.evidence))
        for item, mapped in zip(question.evidence, ranges):
            self.assertEqual((item.start_seconds, item.end_seconds), (mapped.start, mapped.end))
            self.assertTrue(any(item.quote in LECTURE.passages[i - 1].text for i in mapped.passage_indexes))

    def test_every_fixture_question_has_server_times_and_a_quote_inside_its_range(self):
        for n in range(8):
            for candidate in (mcq(n), tf(n)):
                question = screen([candidate]).accepted[0]
                self.evidence_consistent(question, candidate["evidence_ids"])

    def test_quote_is_the_supporting_window_not_the_passage_opening(self):
        question = screen([mcq(1)]).accepted[0]
        self.assertIn("nine point eight meters", question.evidence[0].quote)
        self.assertFalse(LECTURE.passages[1].text.startswith(question.evidence[0].quote))
        self.assertLessEqual(len(question.evidence[0].quote), 240)
        key = next(c.text for c in question.choices if c.id == question.correct_choice_id)
        self.assertEqual(key, "Still nine point eight meters per second squared")

    def test_each_range_of_a_multi_passage_question_has_its_own_quote(self):
        question = screen([mcq(0, evidence_ids=["S1", "S9"])]).accepted[0]
        self.evidence_consistent(question, ["S1", "S9"])
        first, second = question.evidence
        self.assertIn(first.quote, LECTURE.passages[0].text)
        self.assertIn(second.quote, LECTURE.passages[8].text)
        self.assertLess(first.end_seconds, second.start_seconds)

    def test_adjacent_passages_merge_into_one_range_with_a_quote_from_inside_it(self):
        lecture = lecture_from([PASSAGE_TEXTS[0], PASSAGE_TEXTS[8]])
        question = screen([mcq(0, evidence_ids=["S1", "S2"])], passages=lecture).accepted[0]
        self.assertEqual(len(question.evidence), 1)
        self.assertEqual((question.evidence[0].start_seconds, question.evidence[0].end_seconds), (0.0, 200.0))
        self.assertTrue(any(question.evidence[0].quote in p.text for p in lecture.passages))

    def test_review_question_cites_only_its_own_passage_range(self):
        question = screen([tf(0, prompt=REVIEW_TF[0], evidence_ids=["S9"])]).accepted[0]
        passage = LECTURE.passages[8]
        self.assertEqual((question.evidence[0].start_seconds, question.evidence[0].end_seconds),
                         (passage.start, passage.end))

    def test_verifier_rejects_inconsistent_evidence(self):
        good = EvidenceRange(0.0, 100.0, "u dv equals u v minus", (1,))
        self.assertTrue(verify_evidence_ranges([good], LECTURE))
        bad = {
            "quote not in range": EvidenceRange(0.0, 100.0, "text from another passage entirely", (1,)),
            "wrong start": EvidenceRange(5.0, 100.0, "u dv equals", (1,)),
            "wrong end": EvidenceRange(0.0, 99.0, "u dv equals", (1,)),
            "past the lecture": EvidenceRange(800.0, 1000.0, "x", (9,)),
            "no passages": EvidenceRange(0.0, 100.0, "u dv equals", ()),
            "index out of range": EvidenceRange(0.0, 100.0, "u dv equals", (10,)),
            "non-adjacent indexes": EvidenceRange(0.0, 300.0, "u dv equals", (1, 3)),
            "empty quote": EvidenceRange(0.0, 100.0, "", (1,)),
            "over-long quote": EvidenceRange(0.0, 100.0, LECTURE.passages[0].text + " " * 0, (1,)),
        }
        for name, item in bad.items():
            self.assertFalse(verify_evidence_ranges([item], LECTURE), name)
        self.assertFalse(verify_evidence_ranges([], LECTURE))
        unsorted = [EvidenceRange(100.0, 200.0, "ball", (2,)), EvidenceRange(0.0, 100.0, "u dv equals", (1,))]
        self.assertFalse(verify_evidence_ranges(unsorted, LECTURE))

    def test_inconsistent_evidence_is_rejected_by_the_self_check(self):
        """If trusted code ever produced a quote outside its own range, the question must not be accepted."""
        broken = [EvidenceRange(100.0, 200.0, "text that is not in the cited passage", (2,))]
        with mock.patch("quiz.screening._requote", return_value=broken):
            result = screen([mcq(1)])
        self.assertEqual(result.accepted, ())
        self.assertEqual(reasons_of(result), [Reason.EVIDENCE_INCONSISTENT])

    def test_model_supplied_quote_or_time_never_replaces_the_server_values(self):
        for extra in ({"quote": "invented quote"}, {"start_seconds": 1.0, "end_seconds": 2.0}):
            self.assertEqual(screen([mcq(1, **extra)]).accepted, ())


# See tests/quiz/test_screening.py: the cap is tested alone, with the support-duplicate rule switched off.
ISOLATE_CAP = QuizLimits(duplicate_support_overlap=2.0)


class PassageReuseTests(unittest.TestCase):
    def test_central_concept_questions_are_capped_by_policy_not_correctness(self):
        questions = [
            mcq(0), tf(0),
            tf(0, prompt="The LIATE rule puts logarithmic functions first", concept=concept("b")),
            tf(0, prompt="Exponential functions come last in the LIATE rule", concept=concept("c")),
        ]
        for q in questions:
            self.assertEqual(len(screen([q]).accepted), 1)           # each is individually fine
        capped = screen(questions, requested=6, limits=ISOLATE_CAP)
        self.assertEqual(len(capped.accepted), 2)
        self.assertEqual({r for rej in capped.rejected for r in rej.reasons}, {Reason.PASSAGE_OVERUSED})

    def test_cap_scales_with_the_requested_count(self):
        questions = [
            mcq(0), tf(0),
            tf(0, prompt="The LIATE rule puts logarithmic functions first", concept=concept("b")),
            tf(0, prompt="Exponential functions come last in the LIATE rule", concept=concept("c")),
        ]
        self.assertEqual(len(screen(questions, requested=9, limits=ISOLATE_CAP).accepted), 3)
        self.assertEqual(len(screen(questions, requested=5, limits=ISOLATE_CAP).accepted), 2)

    def test_short_lectures_can_still_reach_the_requested_count(self):
        one = lecture_from([PASSAGE_TEXTS[0]])
        questions = [
            mcq(0), tf(0),
            tf(0, prompt="The LIATE rule puts logarithmic functions first", concept=concept("b")),
            tf(0, prompt="Exponential functions come last in the LIATE rule", concept=concept("c")),
            tf(0, prompt="With x times e to the x the rule says pick u equal to x", concept=concept("d")),
        ]
        result = screen(questions, requested=5, passages=one, limits=ISOLATE_CAP)
        self.assertEqual(len(result.accepted), 5)  # cap = max(ceil(5/3), ceil(5/1)) = 5


class TopicTableSanityTests(unittest.TestCase):
    def test_fixture_is_synthetic_and_self_consistent(self):
        self.assertEqual(len(TOPICS), 8)
        self.assertEqual(len(LECTURE.passages), 9)


if __name__ == "__main__":
    unittest.main()
