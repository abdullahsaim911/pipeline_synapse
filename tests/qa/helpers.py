"""Shared fixtures for Ask a Question tests: a small synthetic lecture and a fake provider."""

import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_interface.providers import (  # noqa: E402
    GenerationResult,
    GenerationSettings,
    TextGenerationProvider,
    TokenUsage,
)
from qa.transcript import normalize_segments  # noqa: E402
from tests.qa.liate_fixture import ORDER_FIRST_INDEX, ORDER_LAST_INDEX, ROWS as LIATE_ROWS  # noqa: E402

# One caption every 6 seconds; the topic changes every minute.
LECTURE_LINES = [
    # 0:00 - 0:59  introduction and formula
    "welcome to this lecture on integration by parts",
    "today we learn how to integrate a product of two functions",
    "the method comes from the product rule for derivatives",
    "start with the product rule and integrate both sides",
    "this gives the formula integral of u dv equals u v minus integral of v du",
    "remember this formula because we use it all the time",
    "the formula turns a hard integral into an easier one",
    "but only if we choose u and dv wisely",
    "choosing badly makes the new integral harder than the original",
    "so how do we choose the parts",
    # 1:00 - 1:59  choosing u
    "a helpful rule of thumb is the acronym LIATE",
    "L stands for logarithmic functions and I for inverse trigonometric functions",
    "A is for algebraic functions like x squared",
    "T is for trigonometric functions and E is for exponentials",
    "pick u as the function that comes first in LIATE",
    "the rest of the integrand becomes dv",
    "for example with x times e to the x pick u equals x",
    "because x is algebraic and e to the x is exponential",
    "then dv is e to the x dx",
    "now differentiate u and integrate dv",
    # 2:00 - 2:59  worked example
    "u equals x so du equals dx",
    "dv equals e to the x dx so v equals e to the x",
    "plug into the formula to get x e to the x minus the integral of e to the x dx",
    "the remaining integral is easy it is just e to the x",
    "so the answer is x e to the x minus e to the x plus C",
    "always check by differentiating the answer",
    "the derivative uses the product rule again and returns the original",
    "this confirms that integration by parts reverses the product rule",
    "let us look at the graph of this function",
    "notice how the curve grows quickly for large x",
    # 3:00 - 3:59  definite integrals and tabular method
    "for definite integrals evaluate the u v term at the limits",
    "then subtract the new integral over the same limits",
    "sometimes you must apply the method twice",
    "the tabular method organizes repeated integration by parts",
    "write derivatives of u in one column and integrals of dv in another",
    "multiply along the diagonals with alternating signs",
    "this is fast for polynomials times exponentials",
    "that is all for integration by parts",
    "in the next lecture we study partial fractions",
    "thank you for watching",
]

# A second, unrelated lecture used for cross-video isolation tests.
OTHER_LECTURE_LINES = [
    "mitochondria are the powerhouse of the cell",
    "they produce ATP through cellular respiration",
    "the zebrafish genome was sequenced in the nineteen nineties",
    "ribosomes build proteins from messenger RNA",
]


def raw_rows(lines: List[str], step: float = 6.0, overlap: float = 3.0) -> List[dict]:
    """Rows shaped like transcript.json entries, with overlapping `end` like real captions."""
    return [
        {"start": i * step, "end": i * step + step + overlap, "text": text}
        for i, text in enumerate(lines)
    ]


def make_segments(lines: Optional[List[str]] = None, step: float = 6.0):
    return normalize_segments(raw_rows(lines or LECTURE_LINES, step))


class FakeProvider(TextGenerationProvider):
    """Records calls; returns `reply` or raises `error`. Never touches the network."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "The instructor says to use the acronym LIATE.",
                 error: Optional[Exception] = None, usage: Optional[TokenUsage] = None):
        self.reply = reply
        self.error = error
        self.usage = usage
        self.calls: List[dict] = []

    def generate(self, prompt: str, settings: Optional[GenerationSettings] = None) -> GenerationResult:
        self.calls.append({"prompt": prompt, "settings": settings})
        if self.error is not None:
            raise self.error
        return GenerationResult(text=self.reply, provider=self.name, model=self.model, latency_ms=1,
                                usage=self.usage)


def forbidden_provider_factory():
    raise AssertionError("The LLM provider must not be created for this request")


# ---------------------------------------------------------------------------
# Real lecture: 3zLQfuMZMF4 (integration by parts), with its speech-recognition errors
# ---------------------------------------------------------------------------

LIATE_REGION = range(ORDER_FIRST_INDEX, ORDER_LAST_INDEX + 1)   # captions 45-56, 103.96-139.48 s


def liate_segments():
    """The real transcript (109 captions, about 3.9k characters)."""
    return normalize_segments({"start": a, "end": b, "text": t} for a, b, t in LIATE_ROWS)


def long_lecture_segments(before: int = 150, after: int = 180):
    """
    A transcript far larger than any default prompt budget: unrelated biology filler,
    the real LIATE explanation in the middle, more filler after. Returns
    (segments, region) where region is the index range of the ordering explanation.
    """
    rows = []
    clock = 0.0

    def filler(count):
        nonlocal clock
        for i in range(count):
            rows.append({"start": clock, "end": clock + 9.0,
                         "text": f"the cell membrane and ribosome number {len(rows)} regulate protein transport"})
            clock += 6.0

    filler(before)
    offset = clock
    for start, end, text in LIATE_ROWS:
        rows.append({"start": offset + start, "end": offset + end, "text": text})
    clock = offset + LIATE_ROWS[-1][0] + 6.0
    filler(after)
    segments = normalize_segments(rows)
    first = before + ORDER_FIRST_INDEX
    return segments, range(first, before + ORDER_LAST_INDEX + 1)


def covered_indices(blocks):
    """Segment indices contained in a list of evidence blocks."""
    covered = set()
    for block in blocks:
        covered |= set(range(block.first_index, block.last_index + 1))
    return covered
