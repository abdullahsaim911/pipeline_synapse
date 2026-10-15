"""
Conservative clean-up of speech-recognition (ASR) captions before indexing.

What it does (and deliberately does NOT do):

  * removes standalone filler sounds ("um", "uh", "er", "hmm"), word stutters ("the the")
    and cut-off word fragments ("th- the");
  * repairs a short, explicit list of split or mis-heard STEM terms ("anti derivative",
    "eigen value", "tan gent", "li8" -> "LIATE");
  * never touches digits, single-letter variables ("x x"), units ("mm"), or timestamps;
  * never rewrites a word into a different word unless the rewrite is in the table below;
  * keeps every caption's start and end time, so source time ranges stay exact. A caption
    whose text becomes empty is dropped and the remaining captions are re-indexed by
    `normalize_segments`, exactly as at load time.

It cannot fix a caption that is wrong but fluent: a mis-heard "plus" that became "minus" is
not detectable here. Domain terms can be added without code through a JSON file named by
ASK_ASR_TERMS_FILE ({"heard phrase": "intended phrase"}, matched as whole words, any case).

Off by default: enable with ASK_ASR_SANITIZE=1.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .transcript import TranscriptSegment, normalize_segments

logger = logging.getLogger(__name__)

TERMS_FILE_ENV = "ASK_ASR_TERMS_FILE"

# Standalone filler sounds. "mm" alone is excluded on purpose (millimetres).
_FILLER_RE = re.compile(
    r"(?<![\w'-])(?:u+m+|u+h+m*|e(?:r|rm|hm)|h+m{2,}|m{3,})(?![\w'-]),?",
    re.IGNORECASE,
)
# "th- the", "wh- what": a short fragment ending in a hyphen, followed by a word that starts with it.
_FRAGMENT_RE = re.compile(r"(?<![\w-])([A-Za-z]{1,6})-\s+(?=\1)", re.IGNORECASE)
# "the the", "no no no": the same alphabetic word (2+ letters) repeated immediately.
_REPEAT_RE = re.compile(r"\b([A-Za-z][A-Za-z']+)(?:\s+\1\b)+", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_PUNCT_RE = re.compile(r"([,;])\s*(?:[,;]\s*)+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

# (pattern, replacement). Replacements may use \1 / \2. Case of the first letter is preserved.
_TERM_FIXES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\banti[\s-]+(derivative)(s?)\b", re.IGNORECASE), r"antiderivative\2"),
    (re.compile(r"\beigen[\s-]+(value|vector|space|function|basis)(s?)\b", re.IGNORECASE), r"eigen\1\2"),
    (re.compile(r"\bco[\s-]+(efficient)(s?)\b", re.IGNORECASE), r"coefficient\2"),
    (re.compile(r"\btan[\s-]+(gent)(s?)\b", re.IGNORECASE), r"tangent\2"),
    (re.compile(r"\blog[\s-]+(arithm)(s?)\b", re.IGNORECASE), r"logarithm\2"),
    (re.compile(r"\bmito[\s-]+(chondri(?:a|on))\b", re.IGNORECASE), r"mitochondria"),
    # "cosign" is a real word elsewhere; only repair it when a trigonometric context follows.
    (re.compile(r"\bco[\s-]?signs?\b(?=\s+(?:of|x|theta|squared|function|law|rule|wave|waves|curve)\b)", re.IGNORECASE), "cosine"),
    # The mnemonic LIATE is heard as "li8" ("ate").
    (re.compile(r"\bli8\b", re.IGNORECASE), "LIATE"),
]


def _match_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() and not replacement.isupper() else replacement


def load_extra_terms(path: Optional[str]) -> List[Tuple[re.Pattern, str]]:
    """Whole-word, case-insensitive literal replacements from a JSON object. Bad files are ignored with a warning."""
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
        if not isinstance(table, dict):
            raise ValueError("expected a JSON object")
        pairs = []
        for heard, intended in table.items():
            if isinstance(heard, str) and isinstance(intended, str) and heard.strip() and len(heard) <= 60:
                literal = intended.replace(chr(92), chr(92) * 2)          # treat the replacement as plain text
                pairs.append((re.compile(r"\b" + re.escape(heard.strip()) + r"\b", re.IGNORECASE), literal))
        return pairs
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring %s: %s", TERMS_FILE_ENV, type(exc).__name__)
        return []


def _apply_terms(text: str, pairs: Sequence[Tuple[re.Pattern, str]]) -> str:
    for pattern, replacement in pairs:
        text = pattern.sub(lambda m, r=replacement, p=pattern: _match_case(m.group(0), m.expand(r)), text)
    return text


def clean_caption_text(
    text: str,
    *,
    remove_disfluencies: bool = True,
    normalize_terms: bool = True,
    extra_terms: Optional[Sequence[Tuple[re.Pattern, str]]] = None,
) -> str:
    """Return the cleaned caption text ('' if nothing meaningful is left)."""
    if not text:
        return ""
    out = text
    if remove_disfluencies:
        out = _FRAGMENT_RE.sub("", out)
        out = _FILLER_RE.sub(" ", out)
        out = _REPEAT_RE.sub(lambda m: m.group(1), out)
    if normalize_terms:
        out = _apply_terms(out, _TERM_FIXES)
        if extra_terms:
            out = _apply_terms(out, extra_terms)
    out = _MULTI_PUNCT_RE.sub(r"\1 ", out)
    out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", out)
    out = _MULTI_SPACE_RE.sub(" ", out).strip(" ,;")
    return out if re.search(r"\w", out) else ""


def sanitize_segments(
    segments: Sequence[TranscriptSegment],
    *,
    remove_disfluencies: bool = True,
    normalize_terms: bool = True,
    extra_terms: Optional[Sequence[Tuple[re.Pattern, str]]] = None,
) -> List[TranscriptSegment]:
    """
    Clean every caption and rebuild the segment list. Start times and the effective time
    spans are preserved; captions that become empty are dropped and indexes renumbered.
    Idempotent: sanitising twice gives the same result.
    """
    if extra_terms is None:
        extra_terms = load_extra_terms(os.environ.get(TERMS_FILE_ENV))
    rows: List[Dict] = []
    for seg in segments:
        rows.append({
            "start": seg.start,
            "end": seg.end,
            "text": clean_caption_text(seg.text, remove_disfluencies=remove_disfluencies,
                                       normalize_terms=normalize_terms, extra_terms=extra_terms),
        })
    return normalize_segments(rows)


def sanitize_stats(before: Sequence[TranscriptSegment], after: Sequence[TranscriptSegment]) -> Dict[str, int]:
    """Counts for logging (never the text itself)."""
    changed = sum(1 for a, b in zip(before, after) if a.text != b.text) if len(before) == len(after) else None
    return {"segments_before": len(before), "segments_after": len(after),
            "dropped": len(before) - len(after),
            "chars_removed": sum(len(s.text) for s in before) - sum(len(s.text) for s in after),
            **({"changed": changed} if changed is not None else {})}
