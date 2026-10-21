"""
Fixtures for service and endpoint tests: a scripted fake provider (never touches the network)
and a synthetic transcript in which each fixture topic is exactly one evidence passage.
"""

import json
import logging
from typing import Any, List, Optional

from tests.quiz.helpers import PASSAGE_TEXTS, mcq, tf  # noqa: F401  (re-exported for tests)
from llm_interface.providers import (
    GenerationResult,
    GenerationSettings,
    TextGenerationProvider,
    TokenUsage,
)
from qa.transcript import normalize_segments
from quiz.limits import QuizLimits

# Small passage budget: every fixture topic (about 55 to 100 estimated tokens) becomes its own
# passage S1..S9, so mcq(n)/tf(n), which cite S(n+1), are valid against this transcript.
TEST_LIMITS = QuizLimits(passage_min_tokens=60, passage_max_tokens=100)

# Keep expected warnings out of the test output (assertLogs still captures them).
logging.getLogger("quiz").addHandler(logging.NullHandler())
logging.getLogger("quiz").propagate = False

VIDEO_ID = "AAAAAAAAAAA"
SECRET_PROMPT_WORD = "sk-SECRET-should-never-be-logged"


def transcript_rows(texts: Optional[List[str]] = None) -> List[dict]:
    return [{"start": i * 100.0, "end": i * 100.0 + 90.0, "text": t}
            for i, t in enumerate(texts if texts is not None else PASSAGE_TEXTS)]


def segments(texts: Optional[List[str]] = None):
    return normalize_segments(transcript_rows(texts))


def reply(candidates: List[Any]) -> str:
    return json.dumps({"questions": candidates})


def good_candidates(n: int = 8) -> List[dict]:
    """n distinct candidates that pass screening against the synthetic transcript (mcq 0..7, then tf 0..7)."""
    return ([mcq(i) for i in range(8)] + [tf(i) for i in range(8)])[:n]


class ScriptedProvider(TextGenerationProvider):
    """Returns/raises the scripted items in order; records every call. Never touches the network."""

    name = "fake"
    model = "fake-model"

    def __init__(self, *script, usage: Optional[TokenUsage] = None):
        self.script = list(script)
        self.calls: List[dict] = []
        self.usage = usage

    def generate(self, prompt: str, settings: Optional[GenerationSettings] = None) -> GenerationResult:
        self.calls.append({"prompt": prompt, "settings": settings})
        if not self.script:
            raise AssertionError("ScriptedProvider has no more scripted replies")
        item = self.script.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        return GenerationResult(text=item, provider=self.name, model=self.model, latency_ms=1, usage=self.usage)
