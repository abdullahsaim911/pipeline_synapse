"""
Read-only transcript access and the mechanical evidence checks.

These checks are AUTOMATIC facts, not human judgments. Each returns True, False, or None; None means
"could not be checked" (no transcript available) and must never be counted as a failure.
Transcript files are only read, never written (the STEM25 dataset under dataset/ is hash-pinned).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

_NORMALIZE_RE = re.compile(r"[^\w]+", re.UNICODE)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize(text: str) -> str:
    """Case-fold and reduce to words, so a quote matches despite punctuation, spacing or caption markers."""
    return " ".join(_NORMALIZE_RE.sub(" ", str(text).casefold().replace("_", " ")).split())


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir or PROJECT_ROOT, path)


def load_segments(path: str, lecture_id: str, base_dir: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Segments for one lecture from a JSON-lines file ({lecture_id, start_seconds, end_seconds, text}).
    Returns None when the file is missing or unreadable (transcript unavailable), [] when the lecture
    has no rows. Rows with unusable times or text are skipped.
    """
    full = resolve_path(path, base_dir)
    if not os.path.isfile(full):
        return None
    segments: List[Dict[str, Any]] = []
    try:
        with open(full, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or row.get("lecture_id") != lecture_id:
                    continue
                start = row.get("start_seconds", row.get("start"))
                end = row.get("end_seconds", row.get("end"))
                text = row.get("text")
                if (isinstance(start, (int, float)) and isinstance(end, (int, float)) and not isinstance(start, bool)
                        and isinstance(text, str) and text.strip() and end >= start >= 0):
                    segments.append({"start": float(start), "end": float(end), "text": text})
    except (OSError, UnicodeDecodeError):
        return None
    segments.sort(key=lambda s: s["start"])
    return segments


def word_count(segments: Optional[List[Dict[str, Any]]]) -> Optional[int]:
    """Whitespace word count of the transcript, or None when there is no transcript to count."""
    if not segments:
        return None
    return sum(len(s["text"].split()) for s in segments)


def lecture_end(segments: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    return max((s["end"] for s in segments), default=None) if segments else None


def check_evidence(
    quote: Any, start: Any, end: Any, segments: Optional[List[Dict[str, Any]]], tolerance: float = 2.0
) -> Dict[str, Optional[bool]]:
    """
    Mechanical checks of one evidence item against the transcript.

      range_well_formed       0 <= start <= end, both finite numbers (needs no transcript)
      range_inside_lecture    the range lies inside the lecture's time span (tolerance seconds)
      quote_in_range          the quote's words occur in the captions overlapping the range
      quote_elsewhere         the quote occurs in the transcript but NOT in that range
                              (a likely wrong timestamp, reported apart from a made-up quote)
      quote_in_transcript     the quote occurs somewhere in the transcript

    None = could not be checked. This says nothing about whether the quote SUPPORTS the question:
    that is a human judgment (evidence_relevant).
    """
    result: Dict[str, Optional[bool]] = {
        "range_well_formed": None, "range_inside_lecture": None,
        "quote_in_range": None, "quote_elsewhere": None, "quote_in_transcript": None,
    }
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
                  for v in (start, end))
    result["range_well_formed"] = bool(numbers and 0 <= start <= end)
    if not segments:
        return result
    if result["range_well_formed"]:
        result["range_inside_lecture"] = end <= (lecture_end(segments) or 0) + tolerance
    needle = normalize(quote) if isinstance(quote, str) else ""
    if not needle:
        result["quote_in_range"] = result["quote_in_transcript"] = False
        result["quote_elsewhere"] = False
        return result
    whole = normalize(" ".join(s["text"] for s in segments))
    result["quote_in_transcript"] = needle in whole
    if result["range_well_formed"]:
        window = normalize(" ".join(s["text"] for s in segments if s["end"] >= start - tolerance and s["start"] <= end + tolerance))
        result["quote_in_range"] = needle in window
        result["quote_elsewhere"] = bool(result["quote_in_transcript"] and not result["quote_in_range"])
    return result
