"""
Strict schemas for POST /video/quiz (contract_version 1). PROPOSED: no endpoint exists.

Only auto-scored question types (MCQ, true/false) and transcript evidence are
representable; short-answer questions and "moment" evidence fail validation.
"""

import re
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quiz.limits import (
    DEFAULT_QUESTION_COUNT,
    MAX_EVIDENCE_IDS,
    MAX_HINTS,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    MIN_USABLE_QUESTIONS,
)

CONTRACT_VERSION = 1
VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
QUIZ_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
CONCEPT_ID_PATTERN = r"^[a-z0-9-]{1,40}$"
_QUESTION_ID_RE = re.compile(r"^q([1-9][0-9]?)$")
CHOICE_IDS = ("a", "b", "c", "d")
TF_CHOICES = (("true", "True"), ("false", "False"))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuizRequest(_Strict):
    video_id: str = Field(..., pattern=VIDEO_ID_PATTERN)
    question_count: int = Field(default=DEFAULT_QUESTION_COUNT, ge=MIN_QUESTION_COUNT, le=MAX_QUESTION_COUNT, strict=True)


class QuizChoice(_Strict):
    id: Literal["a", "b", "c", "d", "true", "false"]
    text: str = Field(..., min_length=1, max_length=200)


class QuizConcept(_Strict):
    id: str = Field(..., pattern=CONCEPT_ID_PATTERN)
    label: str = Field(..., min_length=1, max_length=60)


class QuizEvidence(_Strict):
    """Transcript evidence only. `moment` evidence is reserved and rejected in v1."""
    kind: Literal["transcript"]
    start_seconds: float = Field(..., ge=0, allow_inf_nan=False)
    end_seconds: float = Field(..., ge=0, allow_inf_nan=False)
    quote: str = Field(..., min_length=1, max_length=240)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must not be before start_seconds")
        return self

class QuizQuestion(_Strict):
    id: str
    type: Literal["mcq", "tf"]
    prompt: str = Field(..., min_length=1, max_length=400)
    choices: List[QuizChoice]
    correct_choice_id: str
    explanation: str = Field(..., min_length=1, max_length=600)
    hints: List[str] = Field(default_factory=list, max_length=MAX_HINTS)
    concept: QuizConcept
    evidence: List[QuizEvidence] = Field(..., min_length=1, max_length=MAX_EVIDENCE_IDS)
    grounding: Literal["screened"]  # never "verified": screening is not proof

    @field_validator("id")
    @classmethod
    def _question_id(cls, value):
        if not _QUESTION_ID_RE.fullmatch(value):
            raise ValueError("id must look like q1..q10")
        return value

    @field_validator("hints")
    @classmethod
    def _hint_text(cls, value):
        for hint in value:
            if not hint or len(hint) > 200:
                raise ValueError("each hint must be 1 to 200 characters")
        return value

    @model_validator(mode="after")
    def _shape(self):
        ids = [c.id for c in self.choices]
        if self.type == "tf":
            if [(c.id, c.text) for c in self.choices] != list(TF_CHOICES):
                raise ValueError("a true/false question has exactly the choices True and False")
        else:
            if not 3 <= len(ids) <= 4:
                raise ValueError("a multiple-choice question has 3 or 4 choices")
            if ids != list(CHOICE_IDS[: len(ids)]):
                raise ValueError("multiple-choice choice ids must be a, b, c[, d] in order")
        if self.correct_choice_id not in ids:
            raise ValueError("correct_choice_id must be one of the choices")
        starts = [e.start_seconds for e in self.evidence]
        if starts != sorted(starts):
            raise ValueError("evidence must be sorted by start time")
        return self


class QuizOmitted(_Strict):
    reason: str = Field(..., pattern=r"^[a-z_]{1,40}$")
    count: int = Field(..., ge=1, le=100)


class QuizResponse(_Strict):
    contract_version: Literal[1]
    quiz_id: str = Field(..., pattern=QUIZ_ID_PATTERN)
    video_id: str = Field(..., pattern=VIDEO_ID_PATTERN)
    generated_at: datetime
    prompt_version: str = Field(..., min_length=1, max_length=40)
    model: str = Field(..., min_length=1, max_length=100)
    requested_count: int = Field(..., ge=MIN_QUESTION_COUNT, le=MAX_QUESTION_COUNT, strict=True)
    returned_count: int = Field(..., ge=0, le=MAX_QUESTION_COUNT, strict=True)
    status: Literal["complete", "partial", "insufficient_material"]
    reason: Optional[Literal["no_transcript", "too_short", "validation_failed"]] = None
    notice: Optional[str] = Field(default=None, min_length=1, max_length=300)
    omitted: List[QuizOmitted] = Field(default_factory=list)
    questions: List[QuizQuestion]

    @model_validator(mode="after")
    def _invariants(self):
        n = len(self.questions)
        if self.returned_count != n:
            raise ValueError("returned_count must equal the number of questions")
        if n > self.requested_count:
            raise ValueError("more questions than requested")
        expected = [f"q{i}" for i in range(1, n + 1)]
        if [q.id for q in self.questions] != expected:
            raise ValueError("question ids must be q1..qN in order")
        if self.status == "complete":
            if n != self.requested_count or self.reason is not None or self.notice is not None:
                raise ValueError("a complete quiz has all requested questions and no reason or notice")
        elif self.status == "partial":
            if not MIN_USABLE_QUESTIONS <= n < self.requested_count:
                raise ValueError("a partial quiz has at least 3 and fewer than the requested questions")
            if self.reason is not None or not self.notice:
                raise ValueError("a partial quiz needs a notice and no reason")
        else:
            if n != 0 or self.reason is None or not self.notice:
                raise ValueError("insufficient_material has no questions, and needs a reason and a notice")
        return self


