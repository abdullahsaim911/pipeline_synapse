"""
Transcript evidence passages and server-side evidence -> time mapping.

The model may only cite passage ids ("S1", "S2", ...). Times come exclusively from
the transcript segments the server loaded, never from model output.

Passage ids are a pure function of the segments and the limits, so the same
transcript always yields the same ids (stable across calls).
"""

import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from qa.tokens import estimate_tokens
from qa.transcript import TranscriptSegment
from quiz.limits import DEFAULT_LIMITS, QuizLimits
from quiz.reasons import Reason

_SPACE_RE = re.compile(r"\s+")
_PASSAGE_ID_RE = re.compile(r"^S([1-9][0-9]{0,3})$")
_MOMENT_ID_RE = re.compile(r"^[Mm][0-9]+$")


@dataclass(frozen=True)
class Passage:
    id: str          # "S1", "S2", ...
    index: int       # 1-based position
    start: float     # seconds, from the first segment
    end: float       # seconds, from the last segment
    text: str
    tokens: int      # estimate


@dataclass(frozen=True)
class PassageSet:
    passages: Tuple[Passage, ...]
    lecture_end: float          # latest segment end, 0.0 when there is no usable transcript
    truncated: bool             # True when max_passages cut the transcript short
    skipped_segments: int       # malformed segments ignored

    def by_id(self, passage_id: str) -> Optional[Passage]:
        index = _passage_index(passage_id)
        if index is None or index > len(self.passages):
            return None
        return self.passages[index - 1]


@dataclass(frozen=True)
class EvidenceRange:
    start: float
    end: float
    quote: str
    passage_indexes: Tuple[int, ...] = ()   # the (adjacent) passages this range spans


def _passage_index(passage_id: Any) -> Optional[int]:
    if not isinstance(passage_id, str):
        return None
    match = _PASSAGE_ID_RE.fullmatch(passage_id)
    return int(match.group(1)) if match else None


def _is_time(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _usable(segment: Any) -> bool:
    return (
        isinstance(segment, TranscriptSegment)
        and _is_time(segment.start)
        and _is_time(segment.end)
        and segment.end >= segment.start
        and isinstance(segment.text, str)
        and bool(segment.text.strip())
    )


def _join(texts: List[str]) -> str:
    return _SPACE_RE.sub(" ", " ".join(texts)).strip()


def build_passages(
    segments: Sequence[Any], limits: QuizLimits = DEFAULT_LIMITS
) -> PassageSet:
    """
    Group consecutive segments into passages of about passage_min..passage_max
    estimated tokens. Malformed segments are skipped and counted. A single segment
    larger than passage_max becomes its own (oversized) passage: it is never split.
    A small trailing passage is merged into the previous one when the result still
    fits passage_max. At most max_passages are kept; `truncated` records a cut.
    """
    good = [s for s in (segments or []) if _usable(s)]
    skipped = len(segments or []) - len(good)
    good.sort(key=lambda s: (s.start, s.end))

    groups: List[List[TranscriptSegment]] = []
    current: List[TranscriptSegment] = []
    for segment in good:
        if current and estimate_tokens(_join([s.text for s in current] + [segment.text])) > limits.passage_max_tokens:
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)

    if len(groups) >= 2:
        tail = groups[-1]
        merged = groups[-2] + tail
        if (
            estimate_tokens(_join([s.text for s in tail])) < limits.passage_min_tokens // 2
            and estimate_tokens(_join([s.text for s in merged])) <= limits.passage_max_tokens
        ):
            groups[-2:] = [merged]

    truncated = len(groups) > limits.max_passages
    groups = groups[: limits.max_passages]

    passages = []
    for number, group in enumerate(groups, start=1):
        text = _join([s.text for s in group])
        passages.append(Passage(
            id=f"S{number}",
            index=number,
            start=group[0].start,
            end=max(s.end for s in group),
            text=text,
            tokens=estimate_tokens(text),
        ))
    lecture_end = max((p.end for p in passages), default=0.0)
    return PassageSet(tuple(passages), lecture_end, truncated, skipped)


def _truncate_quote(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut or text[:limit]


def map_evidence_ids(
    evidence_ids: Any,
    passage_set: PassageSet,
    limits: QuizLimits = DEFAULT_LIMITS,
) -> Tuple[List[EvidenceRange], List[Reason]]:
    """
    Map model-cited passage ids to server-owned time ranges.

    Returns (ranges, reasons). When `reasons` is non-empty `ranges` is empty:
    nothing partial is ever returned. Ranges are sorted by start, adjacent passages
    are merged, times are rounded to 2 decimals, and each quote is real transcript
    text taken from the first cited passage of the range.

    PROVISIONAL: the quote is the passage opening (<= quote_max chars). Choosing
    the best-overlapping sentence is deferred and is not implemented here.
    """
    if not isinstance(evidence_ids, (list, tuple)) or not evidence_ids:
        return [], [Reason.NO_EVIDENCE]
    if len(evidence_ids) > limits.max_evidence_ids:
        return [], [Reason.TOO_MANY_EVIDENCE_IDS]

    reasons: List[Reason] = []
    indexes: List[int] = []
    seen = set()
    for item in evidence_ids:
        if isinstance(item, str) and _MOMENT_ID_RE.fullmatch(item):
            reasons.append(Reason.UNSUPPORTED_MOMENT_EVIDENCE)
            continue
        index = _passage_index(item)
        if index is None:
            reasons.append(Reason.INVALID_EVIDENCE_ID)
            continue
        if item in seen:
            reasons.append(Reason.DUPLICATE_EVIDENCE_ID)
            continue
        seen.add(item)
        if index > len(passage_set.passages):
            reasons.append(Reason.UNKNOWN_EVIDENCE_ID)
            continue
        indexes.append(index)
    if reasons:
        return [], _unique(reasons)

    indexes.sort()
    groups: List[List[int]] = []
    for index in indexes:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    ranges = []
    for group in groups:
        first = passage_set.passages[group[0] - 1]
        last = passage_set.passages[group[-1] - 1]
        start, end = round(first.start, 2), round(last.end, 2)
        if not (0 <= start <= end <= round(passage_set.lecture_end, 2)):
            return [], [Reason.INVALID_EVIDENCE_ID]
        ranges.append(EvidenceRange(start, end, _truncate_quote(first.text, limits.quote_max), tuple(group)))
    return ranges, []


def truncate_quote(text: str, limit: int) -> str:
    return _truncate_quote(text, limit)


def verify_evidence_ranges(
    ranges: Sequence[EvidenceRange], passage_set: PassageSet, limits: QuizLimits = DEFAULT_LIMITS
) -> bool:
    """
    Internal-consistency check of finished evidence: every range is exactly the
    span of its recorded passages, lies inside the lecture, ranges are sorted and
    disjoint, and every quote is non-empty, within the length limit, and appears
    verbatim inside one of the range's own passages. Trusted-code self-check; it
    does not say the quote supports the question.
    """
    if not ranges:
        return False
    previous_end = -1.0
    for r in ranges:
        idx = r.passage_indexes
        if not idx or list(idx) != list(range(idx[0], idx[0] + len(idx))):
            return False
        if idx[0] < 1 or idx[-1] > len(passage_set.passages):
            return False
        first, last = passage_set.passages[idx[0] - 1], passage_set.passages[idx[-1] - 1]
        if (r.start, r.end) != (round(first.start, 2), round(last.end, 2)):
            return False
        if not (0 <= r.start <= r.end <= round(passage_set.lecture_end, 2)) or r.start < previous_end:
            return False
        previous_end = r.end
        if not r.quote or len(r.quote) > limits.quote_max:
            return False
        if not any(r.quote in passage_set.passages[i - 1].text for i in idx):
            return False
    return True


def _unique(reasons: List[Reason]) -> List[Reason]:
    out: List[Reason] = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return out
