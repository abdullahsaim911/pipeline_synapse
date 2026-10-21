"""Strict request/response/question schemas (contract_version 1)."""

import copy
import math
import unittest

from pydantic import ValidationError

from tests.quiz.helpers import QUIZ_ID
from quiz.schemas import QuizQuestion, QuizRequest, QuizResponse

VIDEO = "5ugDJhmmkFM"


def question(n=1, **overrides):
    data = {
        "id": f"q{n}", "type": "mcq", "prompt": "Which one?",
        "choices": [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}, {"id": "c", "text": "three"}],
        "correct_choice_id": "b", "explanation": "Because.", "hints": [],
        "concept": {"id": "c-1", "label": "C"},
        "evidence": [{"kind": "transcript", "start_seconds": 1.0, "end_seconds": 5.0, "quote": "text"}],
        "grounding": "screened",
    }
    data.update(overrides)
    return data


def response(n=6, requested=6, status="complete", **overrides):
    data = {
        "contract_version": 1, "quiz_id": QUIZ_ID, "video_id": VIDEO,
        "generated_at": "2026-01-01T12:00:00Z", "prompt_version": "quiz-v1", "model": "m",
        "requested_count": requested, "returned_count": n, "status": status,
        "reason": None, "notice": None, "omitted": [],
        "questions": [question(i) for i in range(1, n + 1)],
    }
    data.update(overrides)
    return data


class RequestTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        self.assertEqual(QuizRequest(video_id=VIDEO).question_count, 6)
        for ok in (5, 10):
            self.assertEqual(QuizRequest(video_id=VIDEO, question_count=ok).question_count, ok)
        for bad in (4, 11, 0, -1, "6", 6.0, True, None):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                QuizRequest(video_id=VIDEO, question_count=bad)

    def test_video_id_and_unknown_fields(self):
        for bad in ("short", "has space!!", VIDEO + "x", VIDEO + "\n", "../etc/passwd", ""):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                QuizRequest(video_id=bad)
        for extra in ({"types": ["mcq"]}, {"include_visual": True}):
            with self.assertRaises(ValidationError):
                QuizRequest(video_id=VIDEO, **extra)


class QuestionTests(unittest.TestCase):
    def test_valid_mcq_and_tf(self):
        QuizQuestion(**question())
        QuizQuestion(**question(
            type="tf", choices=[{"id": "true", "text": "True"}, {"id": "false", "text": "False"}],
            correct_choice_id="false"))

    def test_choice_shapes(self):
        two = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
        five = [{"id": i, "text": "x"} for i in "abcde"]
        out_of_order = [{"id": "b", "text": "x"}, {"id": "a", "text": "y"}, {"id": "c", "text": "z"}]
        for choices in (two, five, out_of_order):
            with self.assertRaises(ValidationError):
                QuizQuestion(**question(choices=choices))
        with self.assertRaises(ValidationError):  # true/false with MCQ-style choices
            QuizQuestion(**question(type="tf", correct_choice_id="a"))
        with self.assertRaises(ValidationError):  # key not among choices
            QuizQuestion(**question(correct_choice_id="d"))

    def test_hint_limit(self):
        for count in (0, 1, 2):
            QuizQuestion(**question(hints=["hint"] * count))
        with self.assertRaises(ValidationError):
            QuizQuestion(**question(hints=["hint"] * 3))
        with self.assertRaises(ValidationError):
            QuizQuestion(**question(hints=[""]))

    def test_short_answer_and_moment_evidence_are_unrepresentable(self):
        with self.assertRaises(ValidationError):
            QuizQuestion(**question(type="short"))
        moment = {"kind": "moment", "start_seconds": 3.0, "end_seconds": 3.0, "quote": "q", "intervention_id": "x"}
        with self.assertRaises(ValidationError):
            QuizQuestion(**question(evidence=[moment]))
        with self.assertRaises(ValidationError):
            QuizQuestion(**question(evidence=[{"kind": "moment", "start_seconds": 3.0, "end_seconds": 3.0, "quote": "q"}]))

    def test_evidence_bounds(self):
        base = {"kind": "transcript", "quote": "q"}
        bad_sets = [
            [],
            [dict(base, start_seconds=5.0, end_seconds=1.0)],
            [dict(base, start_seconds=-1.0, end_seconds=1.0)],
            [dict(base, start_seconds=math.nan, end_seconds=1.0)],
            [dict(base, start_seconds=0.0, end_seconds=math.inf)],
            [dict(base, start_seconds=0.0, end_seconds=1.0)] * 4,
            [dict(base, start_seconds=5.0, end_seconds=6.0), dict(base, start_seconds=1.0, end_seconds=2.0)],
            [dict(base, start_seconds=0.0, end_seconds=1.0, extra=1)],
        ]
        for evidence in bad_sets:
            with self.assertRaises(ValidationError):
                QuizQuestion(**question(evidence=evidence))

    def test_grounding_is_never_verified_and_extras_are_forbidden(self):
        for overrides in ({"grounding": "verified"}, {"extra": 1}, {"id": "q0"}, {"id": "x1"}):
            with self.assertRaises(ValidationError, msg=str(overrides)):
                QuizQuestion(**question(**overrides))


class ResponseTests(unittest.TestCase):
    def test_complete(self):
        QuizResponse(**response())

    def test_complete_rejects_notice_reason_or_short_count(self):
        for overrides in ({"notice": "n"}, {"reason": "too_short"}):
            with self.assertRaises(ValidationError):
                QuizResponse(**response(**overrides))
        with self.assertRaises(ValidationError):
            QuizResponse(**response(n=5, requested=6))

    def test_partial(self):
        QuizResponse(**response(n=5, requested=6, status="partial", notice="Only 5 of 6."))
        QuizResponse(**response(n=3, requested=6, status="partial", notice="Only 3 of 6."))
        with self.assertRaises(ValidationError):  # no notice
            QuizResponse(**response(n=5, requested=6, status="partial"))
        with self.assertRaises(ValidationError):  # below the usable minimum
            QuizResponse(**response(n=2, requested=6, status="partial", notice="x"))
        with self.assertRaises(ValidationError):  # nothing missing
            QuizResponse(**response(n=6, requested=6, status="partial", notice="x"))

    def test_insufficient_material(self):
        for reason in ("no_transcript", "too_short", "validation_failed"):
            QuizResponse(**response(n=0, status="insufficient_material", reason=reason, notice="n"))
        with self.assertRaises(ValidationError):
            QuizResponse(**response(n=0, status="insufficient_material", notice="n"))      # no reason
        with self.assertRaises(ValidationError):
            QuizResponse(**response(n=0, status="insufficient_material", reason="too_short"))  # no notice
        with self.assertRaises(ValidationError):  # padded with questions
            QuizResponse(**response(n=2, status="insufficient_material", reason="too_short", notice="n"))
        with self.assertRaises(ValidationError):
            QuizResponse(**response(n=0, status="insufficient_material", reason="other", notice="n"))

    def test_counts_ids_and_version(self):
        with self.assertRaises(ValidationError):
            QuizResponse(**response(returned_count=5))
        bad_ids = response()
        bad_ids["questions"][2]["id"] = "q9"
        with self.assertRaises(ValidationError):
            QuizResponse(**bad_ids)
        with self.assertRaises(ValidationError):
            QuizResponse(**response(contract_version=2))
        with self.assertRaises(ValidationError):
            QuizResponse(**response(extra="x"))
        with self.assertRaises(ValidationError):
            QuizResponse(**response(quiz_id="not-a-uuid"))

    def test_more_than_ten_questions_can_never_form_a_response(self):
        with self.assertRaises(ValidationError):
            QuizResponse(**response(n=11, requested=10))

    def test_input_is_not_mutated(self):
        data = response()
        before = copy.deepcopy(data)
        QuizResponse(**data)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
