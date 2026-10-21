"""Deterministic, reject-only screening of untrusted candidates (no model involved)."""

import json
import unittest

from tests.quiz.helpers import concept, LECTURE, QUIZ_ID, REVIEW_TF, lecture_from, mcq, tf, without
from quiz.limits import QuizLimits
from quiz.passages import build_passages
from quiz.reasons import Reason
from quiz.schemas import QuizQuestion, QuizResponse
from quiz.screening import screen_candidates

TOPIC0_CHOICES = ["The LIATE rule", "The chain rule", "The quotient rule", "The ratio test"]


def screen(candidates, requested=6, passages=LECTURE, limits=QuizLimits(), quiz_id=QUIZ_ID):
    return screen_candidates(candidates, passages, quiz_id=quiz_id, requested_count=requested, limits=limits)


# The passage cap is a separate rule from the support-duplicate rule; cap tests stack several questions on one passage,
# so they switch the duplicate rule off to keep testing the cap alone.
ISOLATE_CAP = QuizLimits(duplicate_support_overlap=2.0)


def reasons_of(result, index=0):
    return [r for rej in result.rejected if rej.index == index for r in rej.reasons]


class AcceptanceTests(unittest.TestCase):
    def test_valid_mcq_is_accepted_and_rebuilt_from_trusted_parts(self):
        result = screen([mcq(0)])
        self.assertEqual((len(result.accepted), len(result.rejected)), (1, 0))
        q = result.accepted[0]
        self.assertIsInstance(q, QuizQuestion)
        self.assertEqual((q.id, q.type, q.grounding), ("q1", "mcq", "screened"))
        texts = {c.id: c.text for c in q.choices}
        self.assertEqual(texts[q.correct_choice_id], "The LIATE rule")  # key follows the shuffle
        self.assertEqual(sorted(texts.values()), sorted(TOPIC0_CHOICES))
        self.assertEqual([c.id for c in q.choices], ["a", "b", "c", "d"])

    def test_evidence_times_are_server_owned(self):
        q = screen([mcq(1)]).accepted[0]
        passage = LECTURE.passages[1]
        self.assertEqual((q.evidence[0].start_seconds, q.evidence[0].end_seconds), (passage.start, passage.end))
        self.assertIn(q.evidence[0].quote, passage.text)

    def test_shuffle_is_deterministic_per_quiz_id(self):
        a = screen([mcq(0)]).accepted[0].model_dump()
        b = screen([mcq(0)]).accepted[0].model_dump()
        self.assertEqual(a, b)

    def test_valid_true_false(self):
        for answer, key in ((True, "true"), (False, "false")):
            q = screen([tf(0, answer=answer)]).accepted[0]
            self.assertEqual(q.correct_choice_id, key)
            self.assertEqual([c.text for c in q.choices], ["True", "False"])

    def test_three_choice_mcq_boundary(self):
        q = screen([mcq(0, choices=TOPIC0_CHOICES[:3])]).accepted[0]
        self.assertEqual([c.id for c in q.choices], ["a", "b", "c"])

    def test_no_type_mix_is_enforced_only_recorded(self):
        result = screen([tf(i) for i in range(4)])
        self.assertEqual(len(result.accepted), 4)
        self.assertEqual(result.type_mix, {"mcq": 0, "tf": 4})
        mixed = screen([mcq(0), tf(1), tf(2)])
        self.assertEqual(mixed.type_mix, {"mcq": 1, "tf": 2})

    def test_screened_questions_form_a_valid_response(self):
        result = screen([mcq(i) if i % 2 else tf(i) for i in range(6)])
        self.assertEqual(len(result.accepted), 6)
        QuizResponse(
            contract_version=1, quiz_id=QUIZ_ID, video_id="5ugDJhmmkFM", generated_at="2026-01-01T12:00:00Z",
            prompt_version="quiz-v1", model="test", requested_count=6, returned_count=6, status="complete",
            questions=list(result.accepted),
        )

    def test_every_fixture_topic_passes_both_question_types(self):
        # Each topic's MCQ and true/false rest on the same fact. With the support-duplicate rule off, all 16 are fine ...
        result = screen([mcq(i) for i in range(8)] + [tf(i) for i in range(8)], requested=10,
                        limits=QuizLimits(max_candidates=16, duplicate_support_overlap=2.0))
        self.assertEqual(len(result.accepted), 16)
        self.assertEqual(result.rejected, ())
        # ... and with the default limits the second question on the same supporting text is a duplicate (one fact, two questions).
        default = screen([mcq(i) for i in range(8)] + [tf(i) for i in range(8)], requested=10, limits=QuizLimits(max_candidates=16))
        self.assertLess(len(default.accepted), 16)
        self.assertGreaterEqual(len(default.accepted), 8)
        self.assertEqual({r for rej in default.rejected for r in rej.reasons}, {Reason.DUPLICATE_QUESTION})


class EvidenceTests(unittest.TestCase):
    def test_unknown_and_missing_evidence_ids(self):
        total = len(LECTURE.passages)
        cases = {
            "unknown": (mcq(0, evidence_ids=[f"S{total + 1}"]), Reason.UNKNOWN_EVIDENCE_ID),
            "empty": (mcq(0, evidence_ids=[]), Reason.NO_EVIDENCE),
            "missing key": (without(mcq(0), "evidence_ids"), Reason.MISSING_FIELD),
            "not a list": (mcq(0, evidence_ids="S1"), Reason.NO_EVIDENCE),
            "malformed": (mcq(0, evidence_ids=["s1"]), Reason.INVALID_EVIDENCE_ID),
            "null": (mcq(0, evidence_ids=None), Reason.NO_EVIDENCE),
        }
        for name, (candidate, reason) in cases.items():
            result = screen([candidate])
            self.assertEqual(result.accepted, (), name)
            self.assertIn(reason, reasons_of(result), name)

    def test_unsupported_moment_evidence(self):
        for ids in (["M1"], ["S1", "M1"], ["m3"]):
            result = screen([mcq(0, evidence_ids=ids)])
            self.assertEqual(result.accepted, ())
            self.assertIn(Reason.UNSUPPORTED_MOMENT_EVIDENCE, reasons_of(result))
        as_object = mcq(0, evidence=[{"kind": "moment", "intervention_id": "x", "start_seconds": 3}])
        result = screen([as_object])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.MODEL_SUPPLIED_TIMESTAMP, reasons_of(result))

    def test_exact_figure_needs_two_distinct_passages(self):
        one = screen([mcq(0, relies_on_exact_figure=True)])
        self.assertIn(Reason.EXACT_FIGURE_NEEDS_TWO_PASSAGES, reasons_of(one))
        two = screen([mcq(0, relies_on_exact_figure=True, evidence_ids=["S1", "S9"])])
        self.assertEqual(len(two.accepted), 1)

    def test_evidence_id_limit_boundary(self):
        three = screen([mcq(0, evidence_ids=["S1", "S3", "S5"])])
        self.assertNotIn(Reason.TOO_MANY_EVIDENCE_IDS, reasons_of(three))
        four = screen([mcq(0, evidence_ids=["S1", "S3", "S5", "S7"])])
        self.assertEqual(four.accepted, ())
        self.assertIn(Reason.TOO_MANY_EVIDENCE_IDS, reasons_of(four))

    def test_empty_or_malformed_transcript_accepts_nothing(self):
        for passages in (build_passages([]), build_passages([None, "x", {"start": 1}])):
            result = screen([mcq(0), tf(1)], passages=passages)
            self.assertEqual(result.accepted, ())
            self.assertTrue(all(Reason.UNKNOWN_EVIDENCE_ID in r.reasons for r in result.rejected))


class TimestampInjectionTests(unittest.TestCase):
    def test_timestamp_fields_are_rejected_not_used(self):
        for key in ("start_seconds", "end_seconds", "timestamp", "Timestamp", " start ", "timestamps",
                    "time", "quote", "evidence", "intervention_id", "seconds", "range"):
            result = screen([mcq(0, **{key: 12.5})])
            self.assertEqual(result.accepted, (), key)
            self.assertIn(Reason.MODEL_SUPPLIED_TIMESTAMP, reasons_of(result), key)

    def test_times_written_in_text_are_rejected(self):
        for field, candidate in (
            ("prompt", mcq(0, prompt="What is said at 3:21 about the limit?")),
            ("explanation", mcq(0, explanation="See 1:02:03 in the lecture.")),
            ("choice", mcq(0, choices=["at 0:45 yes", "no", "maybe so"], correct_index=0)),
        ):
            result = screen([candidate])
            self.assertEqual(result.accepted, (), field)
            self.assertIn(Reason.TIME_IN_TEXT, reasons_of(result), field)

    def test_time_in_a_hint_drops_the_hint(self):
        result = screen([mcq(0, hints=["Listen around 3:21.", "Think about the setup."])])
        self.assertEqual(result.accepted[0].hints, ["Think about the setup."])
        self.assertEqual([d.reason for d in result.dropped_hints], [Reason.HINT_INVALID])

    def test_injected_fields_never_reach_accepted_output(self):
        result = screen([mcq(0)])
        self.assertNotIn("INJECTED", json.dumps(result.accepted[0].model_dump()))
        injected = screen([mcq(0, start_seconds=1, note="INJECTED")])
        self.assertEqual(injected.accepted, ())


class ChoiceAndKeyTests(unittest.TestCase):
    def test_choice_count_boundaries(self):
        pool = TOPIC0_CHOICES + ["The root test"]
        for count, ok in ((0, False), (2, False), (3, True), (4, True), (5, False)):
            result = screen([mcq(0, choices=pool[:count])])
            self.assertEqual(len(result.accepted) == 1, ok, count)
            if not ok:
                self.assertIn(Reason.INVALID_CHOICE_COUNT, reasons_of(result))

    def test_invalid_answer_keys(self):
        for key in (-1, 4, 99, True, False, "1", 1.0, None, [1]):
            result = screen([mcq(0, correct_index=key)])
            self.assertEqual(result.accepted, (), repr(key))
            self.assertIn(Reason.INVALID_ANSWER_KEY, reasons_of(result), repr(key))
        self.assertIn(Reason.MISSING_FIELD, reasons_of(screen([without(mcq(0), "correct_index")])))
        for answer in ("true", 1, 0, None, "yes"):
            result = screen([tf(0, answer=answer)])
            self.assertIn(Reason.INVALID_ANSWER_KEY, reasons_of(result), repr(answer))

    def test_boundary_answer_keys_are_accepted(self):
        first = screen([mcq(0, choices=TOPIC0_CHOICES, correct_index=0)])
        last = screen([mcq(0, choices=list(reversed(TOPIC0_CHOICES)), correct_index=3)])
        self.assertEqual((len(first.accepted), len(last.accepted)), (1, 1))

    def test_duplicate_choices_after_folding(self):
        result = screen([mcq(0, choices=["The limit", "the LIMIT!", "other thing"], correct_index=2)])
        self.assertIn(Reason.DUPLICATE_CHOICES, reasons_of(result))

    def test_banned_choice_patterns(self):
        for banned in ("All of the above", "none of the above.", "Both A and B"):
            result = screen([mcq(0, choices=["first thing", "second thing", banned], correct_index=0)])
            self.assertIn(Reason.BANNED_CHOICE_PATTERN, reasons_of(result), banned)

    def test_wrong_fields_for_the_type(self):
        self.assertIn(Reason.UNKNOWN_FIELD, reasons_of(screen([tf(0, choices=["a", "b"])])))
        self.assertIn(Reason.UNKNOWN_FIELD, reasons_of(screen([mcq(0, answer=True)])))

    def test_choices_must_be_strings(self):
        for choices in ([1, 2, 3], [{"id": "a"}, "b", "c"], "abc", None):
            result = screen([mcq(0, choices=choices)])
            self.assertEqual(result.accepted, (), repr(choices))


class TypeTests(unittest.TestCase):
    def test_unsupported_types(self):
        for qtype in ("short", "short_answer", "MCQ", "mcq ", "", None, 3, ["mcq"], {"a": 1}):
            result = screen([mcq(0, type=qtype)])
            self.assertEqual(result.accepted, (), repr(qtype))
            self.assertIn(Reason.UNSUPPORTED_QUESTION_TYPE, reasons_of(result), repr(qtype))

    def test_missing_type(self):
        self.assertIn(Reason.MISSING_FIELD, reasons_of(screen([without(mcq(0), "type")])))

    def test_non_objects(self):
        for item in (None, "text", 5, [], [mcq(0)], True):
            result = screen([item])
            self.assertEqual(reasons_of(result), [Reason.NOT_AN_OBJECT], repr(item))


class HintTests(unittest.TestCase):
    def test_zero_one_two_hints_are_accepted(self):
        hints = ["Think about the setup.", "Recall how the speaker framed the first step."]
        for count in (0, 1, 2):
            result = screen([mcq(0, hints=hints[:count])])
            self.assertEqual(result.accepted[0].hints, hints[:count])
            self.assertEqual(result.dropped_hints, ())

    def test_three_hints_reject_the_question_instead_of_truncating(self):
        result = screen([mcq(0, hints=["one hint", "two hint", "three hint"])])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.TOO_MANY_HINTS, reasons_of(result))

    def test_malformed_hint_containers(self):
        for hints in ("a hint", {"a": 1}, [1], [None], [["x"]]):
            self.assertEqual(screen([mcq(0, hints=hints)]).accepted, (), repr(hints))

    def test_hint_containing_the_correct_choice_is_dropped(self):
        result = screen([mcq(0, hints=["Consider the LIATE rule carefully.", "Think about the setup."])])
        self.assertEqual(result.accepted[0].hints, ["Think about the setup."])
        self.assertEqual([d.reason for d in result.dropped_hints], [Reason.HINT_REVEALS_ANSWER])
        self.assertEqual(result.dropped_hints[0].candidate_index, 0)

    def test_hints_naming_an_option_are_dropped(self):
        for text in ("The answer is b.", "Option C looks right.", "Choose the second option.",
                     "It is the correct answer you want.", "Pick choice 2."):
            result = screen([mcq(0, hints=[text])])
            self.assertEqual(result.accepted[0].hints, [], text)
            self.assertEqual([d.reason for d in result.dropped_hints], [Reason.HINT_REVEALS_ANSWER], text)

    def test_true_false_hints_must_not_state_the_verdict(self):
        for text in ("This statement is true.", "That is False, listen again.", "It is correct as stated."):
            result = screen([tf(0, hints=[text])])
            self.assertEqual(result.accepted[0].hints, [], text)
        clean = screen([tf(0, hints=["Think about what the speaker claimed first."])])
        self.assertEqual(len(clean.accepted[0].hints), 1)

    def test_hint_with_formatting_or_visual_reference_is_dropped(self):
        result = screen([mcq(0, hints=["Use **bold** thinking.", "Look at this graph."])])
        self.assertEqual(result.accepted[0].hints, [])
        self.assertEqual([d.reason for d in result.dropped_hints], [Reason.HINT_INVALID] * 2)
        mixed = screen([mcq(0, hints=["Look at this graph.", "Recall the setup."])])
        self.assertEqual(mixed.accepted[0].hints, ["Recall the setup."])

    def test_hint_length_boundary(self):
        self.assertEqual(len(screen([mcq(0, hints=["h" * 200])]).accepted[0].hints), 1)
        self.assertEqual(screen([mcq(0, hints=["h" * 201])]).accepted[0].hints, [])

    def test_dropped_hints_are_not_recorded_for_rejected_questions(self):
        result = screen([mcq(0, hints=["The answer is b."], correct_index=9)])
        self.assertEqual(result.dropped_hints, ())


class TextRuleTests(unittest.TestCase):
    def test_length_boundaries(self):
        self.assertNotIn(Reason.TEXT_LENGTH, reasons_of(screen([mcq(0, prompt="p" * 400)])))
        for prompt in ("p" * 401, "", "   ", "???"):
            self.assertIn(Reason.TEXT_LENGTH, reasons_of(screen([mcq(0, prompt=prompt)])), repr(prompt))
        self.assertEqual(len(screen([mcq(0, explanation="e" * 600)]).accepted), 1)
        self.assertIn(Reason.TEXT_LENGTH, reasons_of(screen([mcq(0, explanation="e" * 601)])))

    def test_formatting_and_control_characters(self):
        for text in ("Use $x$ here", "a \\frac b", "x^2 is", "x_{1} is", "# Heading", "- bullet", "`code`",
                     "<b>bold</b>", "[link](http://x)", "**bold**", "smile \U0001F600"):
            self.assertIn(Reason.FORMATTING_NOT_ALLOWED, reasons_of(screen([mcq(0, prompt=text)])), text)
        for text in ("line one\nline two", "tab\there", "zero​width"):
            self.assertIn(Reason.CONTROL_CHARACTER, reasons_of(screen([mcq(0, prompt=text)])), repr(text))

    def test_wrong_types(self):
        for field in ("prompt", "explanation"):
            self.assertIn(Reason.WRONG_TYPE, reasons_of(screen([mcq(0, **{field: 5})])), field)

    def test_visual_dependence(self):
        self.assertIn(Reason.DEPENDS_ON_VISUAL, reasons_of(screen([mcq(0, depends_on_visual=True)])))
        for bad in (None, "false", 0):
            self.assertIn(Reason.WRONG_TYPE, reasons_of(screen([mcq(0, depends_on_visual=bad)])), repr(bad))
        self.assertIn(Reason.MISSING_FIELD, reasons_of(screen([without(mcq(0), "depends_on_visual")])))
        for phrase in ("As shown above, what is the slope?", "What does this graph say?", "What is on the screen?"):
            self.assertIn(Reason.VISUAL_REFERENCE, reasons_of(screen([mcq(0, prompt=phrase)])), phrase)
        choices = ["see this diagram", "plain choice", "another choice"]
        self.assertIn(Reason.VISUAL_REFERENCE, reasons_of(screen([mcq(0, choices=choices, correct_index=1)])))

    def test_concept_rules(self):
        for concept in ({"id": "Bad ID", "label": "x"}, {"id": "ok", "label": ""}, {"id": "ok"},
                        {"id": "ok", "label": "x", "extra": 1}, "text", None, {"id": 5, "label": "x"}):
            self.assertIn(Reason.INVALID_CONCEPT, reasons_of(screen([mcq(0, concept=concept)])), repr(concept))


def same_passage_questions():
    """Four distinct, supported questions whose best support is passage S1."""
    return [
        mcq(0),
        tf(0),
        tf(0, prompt="The LIATE rule puts logarithmic functions first", concept=concept("b")),
        tf(0, prompt="Exponential functions come last in the LIATE rule", concept=concept("c")),
    ]


class BatchTests(unittest.TestCase):
    def test_duplicate_questions(self):
        result = screen([mcq(0), mcq(0)])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(reasons_of(result, 1), [Reason.DUPLICATE_QUESTION])

    def test_near_duplicate_by_token_overlap(self):
        near = mcq(0, prompt=mcq(0)["prompt"].replace("Which", "What"))
        self.assertEqual(reasons_of(screen([mcq(0), near]), 1), [Reason.DUPLICATE_QUESTION])

    def test_case_and_punctuation_do_not_hide_a_duplicate(self):
        shouted = mcq(0, prompt=mcq(0)["prompt"].upper().rstrip("?") + "!!")
        self.assertEqual(reasons_of(screen([mcq(0), shouted]), 1), [Reason.DUPLICATE_QUESTION])

    def test_rejected_candidates_do_not_block_later_ones(self):
        result = screen([mcq(0, correct_index=9), mcq(0)])
        self.assertEqual(len(result.accepted), 1)

    def test_repeated_use_of_one_passage(self):
        questions = same_passage_questions()
        result = screen(questions, requested=6, limits=ISOLATE_CAP)  # cap = max(ceil(6 / 3), ceil(6 / 9)) = 2
        self.assertEqual(len(result.accepted), 2)
        self.assertEqual([reasons_of(result, i) for i in (2, 3)], [[Reason.PASSAGE_OVERUSED]] * 2)
        self.assertEqual(len(screen(questions, requested=9, limits=ISOLATE_CAP).accepted), 3)   # cap = 3
        self.assertEqual(len(screen(questions, requested=5, limits=ISOLATE_CAP).accepted), 2)   # cap = ceil(5 / 3) = 2

    def test_secondary_citations_do_not_count_toward_the_cap(self):
        # S1 is already at its cap of 2. This question's best support is S9; S1 is only corroboration.
        cands = [mcq(0), tf(0), tf(0, prompt=REVIEW_TF[0], evidence_ids=["S9", "S1"], concept=concept("e"))]
        result = screen(cands, requested=6, limits=ISOLATE_CAP)
        self.assertEqual(len(result.accepted), 3)
        self.assertEqual(len(result.accepted[2].evidence), 2)

    def test_cap_has_a_capacity_floor_on_short_lectures(self):
        two_passages = lecture_from([LECTURE.passages[0].text, LECTURE.passages[1].text])
        result = screen(same_passage_questions()[:3], requested=6, passages=two_passages, limits=ISOLATE_CAP)
        self.assertEqual(len(result.accepted), 3)  # ceil(6 / 2) = 3 beats ceil(6 / 3) = 2

    def test_candidate_limit_boundaries(self):
        limits = QuizLimits(max_candidates=12)
        twelve = screen([tf(0)] * 12, limits=limits, requested=10)
        self.assertNotIn(Reason.OVER_CANDIDATE_LIMIT, [r for rej in twelve.rejected for r in rej.reasons])
        thirteen = screen([tf(0)] * 13, limits=limits, requested=10)
        self.assertEqual(reasons_of(thirteen, 12), [Reason.OVER_CANDIDATE_LIMIT])
        self.assertEqual(thirteen.candidates_seen, 13)
        self.assertEqual(len(thirteen.accepted) + len(thirteen.rejected), 13)

    def test_custom_candidate_limit(self):
        result = screen([tf(i) for i in range(4)], limits=QuizLimits(max_candidates=2))
        self.assertEqual(len(result.accepted), 2)
        self.assertEqual(reasons_of(result, 2), [Reason.OVER_CANDIDATE_LIMIT])

    def test_empty_candidate_list(self):
        result = screen([])
        self.assertEqual((result.accepted, result.rejected, result.candidates_seen), ((), (), 0))

    def test_accepted_ids_are_sequential_over_accepted_only(self):
        result = screen([mcq(0), mcq(1, correct_index=9), mcq(2)])
        self.assertEqual([q.id for q in result.accepted], ["q1", "q2"])

    def test_caller_bugs_raise(self):
        with self.assertRaises(TypeError):
            screen({"a": 1})
        with self.assertRaises(TypeError):
            screen("text")
        for bad in (4, 11, 0):
            with self.assertRaises(ValueError):
                screen([mcq(0)], requested=bad)
        with self.assertRaises(ValueError):
            screen([mcq(0)], quiz_id="")

    def test_candidates_are_not_mutated(self):
        candidates = [mcq(0, hints=["The answer is b."]), tf(1)]
        before = json.dumps(candidates, sort_keys=True)
        screen(candidates)
        self.assertEqual(json.dumps(candidates, sort_keys=True), before)


class NoLeakageTests(unittest.TestCase):
    SENTINEL = "UNTRUSTED-SENTINEL-TEXT"

    def test_rejections_hold_only_reason_codes(self):
        bad = mcq(0, prompt=self.SENTINEL + " at 3:21", correct_index=9, evidence_ids=["S9999"],
                  hints=["x", "y", "z"], start_seconds=1, **{self.SENTINEL: self.SENTINEL})
        result = screen([bad, "free text " + self.SENTINEL, {"type": self.SENTINEL}])
        self.assertTrue(result.rejected)
        for rejection in result.rejected:
            self.assertTrue(all(isinstance(r, Reason) for r in rejection.reasons))
        self.assertNotIn(self.SENTINEL, repr(result))
        self.assertEqual(result.accepted, ())

    def test_support_rejections_hold_no_candidate_text(self):
        unsupported = mcq(0, prompt=f"Which {self.SENTINEL} unrelated topic is discussed?",
                          choices=[self.SENTINEL + " one", "other two", "other three"])
        result = screen([unsupported])
        self.assertEqual(result.accepted, ())
        self.assertIn(Reason.NO_SUPPORTING_SENTENCE, reasons_of(result))
        self.assertNotIn(self.SENTINEL, repr(result))

    def test_dropped_hints_hold_no_hint_text(self):
        result = screen([mcq(0, hints=[self.SENTINEL + " the answer is b"])])
        self.assertEqual(len(result.dropped_hints), 1)
        self.assertNotIn(self.SENTINEL, repr(result.dropped_hints))

    def test_rejection_counts_summarise_by_first_reason(self):
        result = screen([mcq(0, correct_index=9), mcq(1, correct_index=9), mcq(2, type="short")])
        self.assertEqual(result.rejection_counts, {"invalid_answer_key": 2, "unsupported_question_type": 1})


if __name__ == "__main__":
    unittest.main()
