"""
Current-moment (intervention) context for Ask a Question.

An intervention is only used if it belongs to the requested video. Generated
material is filtered for quality before it can reach the prompt.

Visual reliability
  A generated explanation is written by the explanation pipeline from the VLM
  snapshot, so it is only as trustworthy as that snapshot. The snapshot is
  therefore classified first:

    valid    - readable, no error marker, at least one reliable field
    failed   - explicitly marked failed (`_error`, `missing_elements`)
    invalid  - unreadable/corrupt file, wrong shape, or no reliable fields
    missing  - no snapshot recorded, or its file is absent / outside this video

  The explanation is passed on only when the snapshot is `valid`. For every
  other state the explanation is withheld: in the sample data a failed snapshot
  was followed by a 362-word "explanation" that described graphs and arrows the
  model never saw. The visual fields are likewise only used when valid.
  Transcript evidence is unaffected: the caption stays available and the
  retrieval layer still supplies transcript excerpts.

Explanation hygiene (applied before use)
  * Prompt scaffolding echoed by the model ("YOUR EXPLANATION:", numbered
    section headings from the synthesizer templates) and markdown artifacts are
    removed. Only those exact template headings are removed; other content,
    including math symbols and upper-case terms, is left alone.
  * An "explanation" that is just a verbatim piece of the transcript is
    discarded (the pipeline falls back to the caption when the VLM fails).

A visual is never described from the caption: if no reliable analysis survives,
`visual` is None and the prompt says so.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .transcript import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    TranscriptSegment,
    clean_text,
    is_valid_video_id,
    video_folder,
)

logger = logging.getLogger(__name__)

# Preferred first; explanations of any mode are acceptable.
_MODE_PREFERENCE = ("explanatory", "detailed", "brief")

# Top-level or under snapshot["visual_analysis"], in prompt order.
_VISUAL_FIELDS = (
    ("structural_description", "Structure"),
    ("layout", "Layout"),
    ("text_readout", "Text read from the screen"),
    ("conceptual_hints", "Concept hints"),
)

# Failure text the VLM wrapper writes into fields. Deliberately narrow: a field
# only counts as an error when it *starts* like one, so slide text such as
# "NullPointerException" or "Error analysis of the circuit" is kept.
_ERROR_TEXT_RE = re.compile(
    r"^\s*(?:error\s*[:\-]|\[?winerror\b)|\bwinerror\b|traceback \(most recent call last\)",
    re.IGNORECASE,
)
_EMPTY_MARKERS = {"", "null", "none", "n/a"}

# Snapshot states
SNAPSHOT_VALID = "valid"
SNAPSHOT_FAILED = "failed"
SNAPSHOT_INVALID = "invalid"
SNAPSHOT_MISSING = "missing"

MAX_EXPLANATION_CHARS = 1500
MAX_VISUAL_FIELD_CHARS = 700
_ECHO_MAX_WORDS = 40

# Exact headings from the synthesizer prompt templates (synthesizer/llm_synthesizer.py).
_SCAFFOLD_HEADINGS = (
    "QUICK VISUAL SNAPSHOT", "COMPLETE VISUAL DESCRIPTION", "EXHAUSTIVE VISUAL DESCRIPTION",
    "WHAT'S BEING SHOWN", "WHAT THE TEACHER IS EXPLAINING", "THE CONCEPT BEING EXPLAINED",
    "DEEP CONCEPTUAL FRAMEWORK", "KEY CONNECTION", "HOW THE VISUAL LINKS TO THE CONCEPT",
    "CROSS-DOMAIN CONNECTIONS", "RICH ACCESSIBILITY TECHNIQUES", "ACCESSIBILITY TECHNIQUES",
    "FINAL SCRIPT",
)
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d{1,2}[.)]\s*)?"
    r"(?:YOUR EXPLANATION(?:\s*-\s*FOCUS ON THESE ELEMENTS)?|"
    + "|".join(re.escape(h) for h in _SCAFFOLD_HEADINGS)
    + r")(?:\s*\([^)\n]{0,60}\))?\s*:?\s*(?:\*\*|__)?\s*$"
)
_INLINE_PREFIX_RE = re.compile(r"^\s*YOUR EXPLANATION(?:\s*-\s*FOCUS ON THESE ELEMENTS)?\s*:\s*")


class InterventionNotFoundError(LookupError):
    """The intervention does not exist for the requested video."""


@dataclass(frozen=True)
class MomentContext:
    intervention_id: str
    timestamp: float
    content_type: Optional[str]
    caption: str
    explanation: Optional[str]
    visual: Optional[Tuple[Tuple[str, str], ...]]  # (label, text) pairs
    visual_status: str = SNAPSHOT_MISSING

    @property
    def has_generated_content(self) -> bool:
        """True when an explanation or visual analysis will be shown to the model."""
        return bool(self.explanation or self.visual)


def _norm(text: str) -> str:
    return " ".join(clean_text(text).lower().split())


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def strip_explanation_scaffolding(text: str) -> str:
    """Remove echoed template headings and markdown artifacts from an explanation."""
    if not isinstance(text, str):
        return ""
    text = _INLINE_PREFIX_RE.sub("", text.replace("\r", ""), count=1)
    kept: List[str] = []
    for line in text.split("\n"):
        if _HEADING_LINE_RE.match(line):
            continue
        line = re.sub(r"^\s*#{1,6}\s+", "", line)        # markdown headings
        line = re.sub(r"^\s*[-*•]\s+", "", line)    # bullets
        line = re.sub(r"\*\*|__", "", line)              # bold / underline markers
        kept.append(line)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", "\n".join(kept))]
    return "\n\n".join(p for p in paragraphs if p)


def is_transcript_echo(text: str, segments: Sequence[TranscriptSegment], caption: str = "") -> bool:
    """True if a short text is just a verbatim excerpt of the transcript."""
    candidate = _norm(text)
    if not candidate or len(candidate.split()) > _ECHO_MAX_WORDS:
        return False
    haystacks = [_norm(caption)] + [" ".join(_norm(s.text) for s in segments)]
    # Captions are cut mid-sentence, so a prefix of a caption also counts.
    return any(candidate in h for h in haystacks if h)


def _clean_field(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    if not isinstance(value, str):
        return None
    text = clean_text(value)
    if text.lower() in _EMPTY_MARKERS or _ERROR_TEXT_RE.search(text):
        return None
    return _truncate(text, MAX_VISUAL_FIELD_CHARS)


def assess_snapshot(snapshot: Any) -> Tuple[str, Optional[Tuple[Tuple[str, str], ...]]]:
    """
    Classify a loaded snapshot and extract its reliable fields.

    Returns (state, pairs); pairs is non-None only for SNAPSHOT_VALID. `None`
    (nothing loaded) is SNAPSHOT_MISSING.
    """
    if snapshot is None:
        return SNAPSHOT_MISSING, None
    if not isinstance(snapshot, dict):
        return SNAPSHOT_INVALID, None
    if snapshot.get("_error"):
        return SNAPSHOT_FAILED, None
    missing = snapshot.get("missing_elements")
    if missing and not (isinstance(missing, str) and missing.strip().lower() in _EMPTY_MARKERS):
        return SNAPSHOT_FAILED, None

    nested = snapshot.get("visual_analysis")
    sources = [snapshot] + ([nested] if isinstance(nested, dict) else [])
    pairs: List[Tuple[str, str]] = []
    for key, label in _VISUAL_FIELDS:
        for source in sources:
            text = _clean_field(source.get(key))
            if text:
                pairs.append((label, text))
                break
    if not pairs:
        return SNAPSHOT_INVALID, None
    return SNAPSHOT_VALID, tuple(pairs)


def clean_visual_snapshot(snapshot: Any) -> Optional[Tuple[Tuple[str, str], ...]]:
    """Reliable (label, text) pairs from a snapshot, or None unless it is valid."""
    return assess_snapshot(snapshot)[1]


def _resolve_snapshot_path(path: str, video_id: str, data_dir: str) -> Optional[str]:
    """Resolve a stored snapshot path, confined to this video's exact-case folder."""
    allowed = video_folder(video_id, data_dir)
    if allowed is None:
        return None
    normalized = path.replace("\\", os.sep).replace("/", os.sep)
    candidates = [normalized] if os.path.isabs(normalized) else [
        os.path.join(PROJECT_ROOT, normalized),
        os.path.join(os.path.dirname(data_dir), normalized),
        os.path.join(data_dir, normalized),
        os.path.abspath(normalized),
    ]
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real.startswith(allowed + os.sep) and os.path.isfile(real):
            return real
    return None


def _load_snapshot(explanations: Sequence[Any], video_id: str, data_dir: str) -> Any:
    """
    Load the VLM snapshot recorded for an intervention.

    Returns the parsed object, None when no usable snapshot is recorded
    (SNAPSHOT_MISSING), or a sentinel non-dict when one exists but is corrupt.
    """
    corrupt = object()
    for explanation in explanations:
        if explanation.vlm_snapshot_path:
            resolved = _resolve_snapshot_path(explanation.vlm_snapshot_path, video_id, data_dir)
            if resolved:
                try:
                    with open(resolved, "r", encoding="utf-8") as handle:
                        return json.load(handle)
                except (OSError, ValueError):
                    logger.warning("Unreadable VLM snapshot for video %s", video_id)
                    return corrupt
        if explanation.vlm_snapshot:  # DB column; not populated by the current pipeline
            try:
                return json.loads(explanation.vlm_snapshot)
            except ValueError:
                return corrupt
    return None


def _ordered_explanations(explanations: Sequence[Any]) -> List[Any]:
    def rank(item: Any) -> int:
        mode = (item.output_mode or "").lower()
        return _MODE_PREFERENCE.index(mode) if mode in _MODE_PREFERENCE else len(_MODE_PREFERENCE)

    usable = [e for e in explanations if e.text_explanation and e.text_explanation.strip()]
    return sorted(usable, key=rank)


def load_moment(
    db: Any,
    video_id: str,
    intervention_id: str,
    segments: Sequence[TranscriptSegment],
    data_dir: str = DEFAULT_DATA_DIR,
) -> MomentContext:
    """
    Load and sanitize context for one intervention.

    Raises:
        InterventionNotFoundError: unknown ID, or it belongs to another video.
    """
    from api.models import Explanation, Intervention  # lazy: needs SQLAlchemy

    if not is_valid_video_id(video_id):
        raise InterventionNotFoundError(intervention_id)

    intervention = (
        db.query(Intervention)
        .filter(Intervention.id == intervention_id, Intervention.video_id == video_id)
        .first()
    )
    if intervention is None:
        raise InterventionNotFoundError(intervention_id)

    caption = clean_text(intervention.transcript_context)
    explanations = _ordered_explanations(
        db.query(Explanation).filter(Explanation.intervention_id == intervention.id).all()
    )

    status, visual = SNAPSHOT_MISSING, None
    if explanations:
        status, visual = assess_snapshot(_load_snapshot(explanations, video_id, data_dir))

    explanation_text = None
    if status == SNAPSHOT_VALID:
        for candidate in explanations:
            text = strip_explanation_scaffolding(candidate.text_explanation)
            if text and not is_transcript_echo(text, segments, caption):
                explanation_text = _truncate(text, MAX_EXPLANATION_CHARS)
                break
    elif explanations:
        logger.info("Withholding generated explanation: visual analysis is %s", status)

    return MomentContext(
        intervention_id=intervention.id,
        timestamp=float(intervention.timestamp or 0.0),
        content_type=intervention.content_type,
        caption=caption,
        explanation=explanation_text,
        visual=visual,
        visual_status=status,
    )
