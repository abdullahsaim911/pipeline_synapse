"""
Quiz generation service (PROPOSED contract, docs/QUIZ_CONTRACT.md section 4).

Flow, with at most TWO model calls per request and no retry of provider errors:

  pre-model   no usable transcript / too short       -> insufficient_material (no model call)
              another generation for this video      -> QuizBusyError
  call 1      structured request for requested+2 candidates
  call 2      only when needed:
                * output unusable (not strict JSON / wrong shape / empty)  -> same request again
                * provider rejected the schema (invalid request)           -> same request, plain JSON
                * fewer accepted questions than requested                  -> top-up for shortfall+2
  outcome     complete | partial (>= 3 accepted) | insufficient_material (validation_failed)
              A failed top-up never discards questions that already passed screening.

Every candidate goes through quiz.screening, which only REJECTS; thresholds are never
relaxed to reach the requested count. Timestamps and quotes are server-owned (quiz.passages).
Logs carry counts and fixed reason codes only: never prompts, replies, transcript text,
candidate text, or keys.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from llm_interface.providers import (
    LLMEmptyResponseError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRequestError,
    TextGenerationProvider,
)
from quiz.limits import DEFAULT_LIMITS, MIN_USABLE_QUESTIONS, QuizLimits, candidate_target, max_output_tokens
from quiz.locks import VideoLocks
from quiz.passages import PassageSet, build_passages
from quiz.prompt import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_prompt, response_schema
from quiz.provider import QuizSettings, get_quiz_provider
from quiz.schemas import CONTRACT_VERSION, QuizOmitted, QuizQuestion, QuizResponse
from quiz.screening import ScreeningResult, screen_candidates

logger = logging.getLogger(__name__)

NO_MODEL = "none"  # `model` value when the response was decided without a model call

NOTICES = {
    "no_transcript": "This lecture doesn't have a usable transcript yet, so practice questions can't be made.",
    "too_short": "This lecture is too short to make reliable practice questions.",
    "validation_failed": "Questions were made, but too few passed our checks. Please try again.",
}


class QuizError(Exception):
    """Base for service-level failures that map to an HTTP status in the router."""


class QuizBusyError(QuizError):
    """Another generation for this video is already running."""


class QuizUnusableOutputError(QuizError):
    """The model's output was unusable on every allowed attempt."""


class QuizDeadlineError(QuizError):
    """The overall time limit ran out before enough questions were available."""


@dataclass
class _Call:
    status: str                          # "ok" | "unusable" | "error"
    candidates: List[Any]
    error: Optional[Exception] = None
    model: str = ""


def _reject_constant(_name: str) -> None:
    raise ValueError("non-finite number in model output")


def parse_candidates(text: str) -> Optional[List[Any]]:
    """
    Strict parse: the whole reply must be one JSON object {"questions": [...]}.
    No code-fence stripping, no extraction of JSON from surrounding prose, no NaN/Infinity.
    Returns None when the reply is unusable (including a truncated reply).
    """
    if not isinstance(text, str):
        return None
    try:
        data = json.loads(text.strip(), parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or set(data) != {"questions"} or not isinstance(data["questions"], list):
        return None
    return data["questions"]


class QuizService:
    def __init__(
        self,
        provider: Optional[TextGenerationProvider] = None,
        provider_factory: Callable[[], TextGenerationProvider] = get_quiz_provider,
        limits: QuizLimits = DEFAULT_LIMITS,
        locks: Optional[VideoLocks] = None,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], Any] = uuid.uuid4,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._provider = provider
        self._provider_factory = provider_factory
        self.limits = limits
        self._locks = locks or VideoLocks()
        self._clock = clock
        self._id_factory = id_factory
        self._now = now

    # ------------------------------------------------------------------ public

    def generate(self, video_id: str, requested_count: int, segments: Sequence[Any]) -> QuizResponse:
        """Raises QuizBusyError, QuizUnusableOutputError, QuizDeadlineError or LLMProviderError."""
        quiz_id = str(self._id_factory())
        passages = build_passages(segments, self.limits)

        if not passages.passages:
            return self._insufficient(quiz_id, video_id, requested_count, "no_transcript")
        words = sum(len(getattr(s, "text", "").split()) for s in segments)
        if words < self.limits.min_transcript_words:
            return self._insufficient(quiz_id, video_id, requested_count, "too_short")

        if not self._locks.try_acquire(video_id):
            raise QuizBusyError(video_id)
        try:
            return self._run(quiz_id, video_id, requested_count, passages)
        finally:
            self._locks.release(video_id)

    # ------------------------------------------------------------------ model phase

    def _provider_or_create(self) -> TextGenerationProvider:
        if self._provider is None:
            self._provider = self._provider_factory()
        return self._provider

    def _call(self, prompt: str, candidates_asked: int, structured: bool) -> _Call:
        settings = QuizSettings(
            temperature=self.limits.temperature,
            max_output_tokens=max_output_tokens(candidates_asked, self.limits),
            system_instruction=SYSTEM_INSTRUCTION,
            response_json_schema=response_schema() if structured else None,
        )
        provider = self._provider_or_create()  # configuration errors propagate to the router
        try:
            result = provider.generate(prompt, settings)
        except (LLMEmptyResponseError, LLMMalformedResponseError) as exc:
            logger.warning("Quiz model call unusable: %s", exc.code)
            return _Call("unusable", [], exc)
        except LLMProviderError as exc:
            logger.warning("Quiz model call failed: %s (upstream status %s)", exc.code, exc.status_code)
            return _Call("error", [], exc)

        model = getattr(result, "model", None) or getattr(provider, "model", "") or NO_MODEL
        usage = getattr(result, "usage", None)
        logger.info(
            "Quiz model call: structured=%s asked=%d prompt_tokens=%s output_tokens=%s",
            structured, candidates_asked,
            getattr(usage, "prompt_tokens", None), getattr(usage, "output_tokens", None),
        )
        candidates = parse_candidates(result.text)
        if candidates is None:
            logger.warning("Quiz model output was not a strict {\"questions\": [...]} object")
            return _Call("unusable", [], None, model)
        return _Call("ok", candidates[: self.limits.max_candidates], None, model)

    def _screen(self, candidates: List[Any], passages: PassageSet, quiz_id: str, requested: int, merged: bool) -> ScreeningResult:
        limits = replace(self.limits, max_candidates=self.limits.max_candidates * 2) if merged else self.limits
        return screen_candidates(candidates, passages, quiz_id=quiz_id, requested_count=requested, limits=limits)

    def _run(self, quiz_id: str, video_id: str, requested: int, passages: PassageSet) -> QuizResponse:
        started = self._clock()
        asked = candidate_target(requested, self.limits)
        base_prompt = build_prompt(passages.passages, asked)

        first = self._call(base_prompt, asked, structured=True)
        screened: Optional[ScreeningResult] = None
        second_prompt, second_asked, second_structured, topup = base_prompt, asked, True, False

        if first.status == "ok":
            screened = self._screen(first.candidates, passages, quiz_id, requested, merged=False)
            if len(screened.accepted) >= requested:
                return self._finish(quiz_id, video_id, requested, screened, calls=1, model=first.model)
            covered = [q.concept.label for q in screened.accepted]
            second_asked = min(requested - len(screened.accepted) + self.limits.candidate_extra, self.limits.max_candidates)
            second_prompt, topup = build_prompt(passages.passages, second_asked, covered), True
        elif first.status == "unusable":
            pass  # same request again
        elif isinstance(first.error, LLMRequestError):
            second_structured = False  # the provider rejected the schema: same request as plain JSON
        else:
            raise first.error  # provider errors are never retried

        accepted_so_far = len(screened.accepted) if screened else 0
        remaining = self.limits.deadline_seconds - (self._clock() - started)
        if remaining < self.limits.second_call_min_remaining:
            if accepted_so_far >= MIN_USABLE_QUESTIONS:
                return self._finish(quiz_id, video_id, requested, screened, calls=1, model=first.model)
            raise QuizDeadlineError()

        second = self._call(second_prompt, second_asked, second_structured)
        if second.status != "ok":
            if accepted_so_far >= MIN_USABLE_QUESTIONS:
                return self._finish(quiz_id, video_id, requested, screened, calls=2, model=first.model)  # keep what passed
            if second.status == "error":
                raise second.error
            raise QuizUnusableOutputError()

        candidates = (first.candidates + second.candidates) if topup else second.candidates
        final = self._screen(candidates, passages, quiz_id, requested, merged=topup)
        return self._finish(quiz_id, video_id, requested, final, calls=2, model=second.model or first.model)

    # ------------------------------------------------------------------ outcomes

    def _base(self, quiz_id: str, video_id: str, requested: int) -> Dict[str, Any]:
        return dict(
            contract_version=CONTRACT_VERSION, quiz_id=quiz_id, video_id=video_id,
            generated_at=self._now(), prompt_version=PROMPT_VERSION, requested_count=requested,
        )

    def _insufficient(self, quiz_id: str, video_id: str, requested: int, reason: str, omitted=(), model: str = NO_MODEL) -> QuizResponse:
        logger.info("Quiz %s for video %s", reason, video_id)
        return QuizResponse(
            **self._base(quiz_id, video_id, requested), model=model or NO_MODEL,
            returned_count=0, status="insufficient_material", reason=reason, notice=NOTICES[reason],
            omitted=list(omitted), questions=[],
        )

    def _finish(self, quiz_id: str, video_id: str, requested: int, result: ScreeningResult, calls: int, model: str) -> QuizResponse:
        chosen = self._select(result.accepted, requested)
        omitted = [QuizOmitted(reason=r, count=min(c, 100)) for r, c in sorted(result.rejection_counts.items())]
        surplus = len(result.accepted) - len(chosen)
        if surplus:
            omitted.append(QuizOmitted(reason="not_selected", count=min(surplus, 100)))

        logger.info(
            "Quiz outcome for video %s: calls=%d candidates=%d accepted=%d returned=%d rejected=%s dropped_hints=%d type_mix=%s",
            video_id, calls, result.candidates_seen, len(result.accepted), len(chosen),
            result.rejection_counts, len(result.dropped_hints), result.type_mix,
        )
        n = len(chosen)
        if n < MIN_USABLE_QUESTIONS:
            return self._insufficient(quiz_id, video_id, requested, "validation_failed", omitted, model)

        notice = None
        status = "complete"
        if n < requested:
            status = "partial"
            notice = f"Only {n} of the {requested} questions you asked for could be made and checked from this lecture."
        return QuizResponse(
            **self._base(quiz_id, video_id, requested), model=model or NO_MODEL,
            returned_count=n, status=status, reason=None, notice=notice, omitted=omitted, questions=chosen,
        )

    @staticmethod
    def _select(accepted: Sequence[QuizQuestion], requested: int) -> List[QuizQuestion]:
        """Lecture order; when there are surplus questions keep an even spread across the lecture."""
        ordered = [q for _, q in sorted(enumerate(accepted), key=lambda p: (p[1].evidence[0].start_seconds, p[0]))]
        n = len(ordered)
        if n > requested:
            picks = sorted({round(i * (n - 1) / (requested - 1)) for i in range(requested)})
            ordered = [ordered[i] for i in picks]
        return [q.model_copy(update={"id": f"q{i}"}) for i, q in enumerate(ordered, start=1)]
