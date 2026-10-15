"""
Transcript loading for Ask a Question.

Segments come from the `transcripts` table when it holds usable rows for the
requested video, otherwise from `data/<video_id>/transcript.json`. Every lookup
is keyed by `video_id`, and the file path is confined to that video's folder.

Identity and errors:
  * A transcript file is used only if the folder and file names equal the
    requested ID exactly (case included); invalid IDs never reach the database
    or the filesystem.
  * Only "no usable database rows" falls back to the file. Database errors
    propagate.

Data-quality notes (verified against the sample lectures in data/):
  * Caption `end` values overlap the next caption's `start`, so they are not
    used as-is: the effective end is capped at the next segment's start.
  * The top-level `duration_seconds` in transcript.json is wrong (e.g. 4.9 s for
    a 273 s video) and is never read.
  * Failed transcriptions are stored as a valid JSON file with zero entries.
"""

import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Same character set the pipeline accepts in orchestrator._extract_video_id,
# restricted to ASCII so the value is always safe to use in a path.
VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
_VIDEO_ID_RE = re.compile(VIDEO_ID_PATTERN)

_FALLBACK_LAST_SEGMENT_SECONDS = 3.0
_SOUND_TAG_RE = re.compile(r"\[(?:music|applause|laughter|silence|noise|inaudible)\]", re.I)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TranscriptSegment:
    """One caption line with a reliable effective time span."""
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptResult:
    segments: List[TranscriptSegment]
    source: str          # "database", "file" or "none"
    file_exists: bool    # transcript.json exists for this video


def is_valid_video_id(video_id: Any) -> bool:
    # fullmatch: `$` alone would also accept a trailing newline.
    return isinstance(video_id, str) and bool(_VIDEO_ID_RE.fullmatch(video_id))


def clean_text(text: Any) -> str:
    """Unescape HTML entities, drop sound tags and speaker markers, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _SOUND_TAG_RE.sub(" ", text)
    text = text.replace(">>", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _as_time(value: Any) -> Optional[float]:
    """Return a finite, non-negative float or None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def normalize_segments(raw: Iterable[dict]) -> List[TranscriptSegment]:
    """
    Normalize raw rows ({"start", "end"?, "text"}) into ordered segments.

    Rows with no usable start time or no text are dropped. `end` is the raw end
    only when it is after `start` and does not run past the next segment's
    start; otherwise it is the next segment's start (or a short default for the
    last segment).
    """
    rows = []
    for item in raw:
        start = _as_time(item.get("start"))
        text = clean_text(item.get("text"))
        if start is None or not text:
            continue
        rows.append((start, _as_time(item.get("end")), text))

    rows.sort(key=lambda row: row[0])

    segments: List[TranscriptSegment] = []
    for i, (start, raw_end, text) in enumerate(rows):
        next_start = rows[i + 1][0] if i + 1 < len(rows) else None
        if raw_end is not None and raw_end <= start:
            raw_end = None  # zero/negative duration is unreliable

        if next_start is not None and next_start > start:
            end = next_start if raw_end is None else min(raw_end, next_start)
        elif raw_end is not None:
            end = raw_end
        else:
            end = start + _FALLBACK_LAST_SEGMENT_SECONDS
        segments.append(TranscriptSegment(index=i, start=start, end=end, text=text))
    return segments


TRANSCRIPT_FILENAME = "transcript.json"


def video_folder(video_id: str, data_dir: str = DEFAULT_DATA_DIR) -> Optional[str]:
    """
    Real path of this video's folder, or None.

    The folder name must equal the video ID exactly, including case. Case-
    insensitive filesystems (Windows, macOS) would otherwise resolve
    "abc..." to the folder of a different video "ABC...", so the name is
    compared against the directory listing instead of trusting the OS lookup.
    """
    if not is_valid_video_id(video_id):
        return None
    try:
        if video_id not in os.listdir(data_dir):
            return None
    except OSError:  # data directory missing or unreadable
        return None
    folder = os.path.realpath(os.path.join(data_dir, video_id))
    return folder if os.path.isdir(folder) else None


def transcript_file_path(video_id: str, data_dir: str = DEFAULT_DATA_DIR) -> Optional[str]:
    """Path of this video's transcript.json (exact-case, inside its folder), or None."""
    folder = video_folder(video_id, data_dir)
    if folder is None or TRANSCRIPT_FILENAME not in os.listdir(folder):
        return None
    path = os.path.realpath(os.path.join(folder, TRANSCRIPT_FILENAME))
    if os.path.dirname(path) != folder:  # e.g. a symlink pointing elsewhere
        return None
    return path


def _load_from_database(db: Any, video_id: str) -> List[TranscriptSegment]:
    from api.models import Transcript  # imported lazily: needs SQLAlchemy

    rows = (
        db.query(Transcript)
        .filter(Transcript.video_id == video_id)
        .order_by(Transcript.start_time)
        .all()
    )
    return normalize_segments(
        {"start": r.start_time, "end": r.end_time, "text": r.text} for r in rows
    )


def _load_from_file(path: str) -> List[TranscriptSegment]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    raw = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        end = entry.get("end")
        if end is None and _as_time(entry.get("start")) is not None:
            duration = _as_time(entry.get("duration"))
            end = None if duration is None else _as_time(entry.get("start")) + duration
        raw.append({"start": entry.get("start"), "end": end, "text": entry.get("text")})
    return normalize_segments(raw)


def load_transcript(
    video_id: str,
    db: Any = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> TranscriptResult:
    """
    Load timestamped segments for exactly one video.

    Prefers database rows. Falls back to the JSON file only when the database
    holds no usable rows for this video. Database errors (connection, missing
    table, programming or import errors) are NOT caught: they propagate so a
    broken database is never mistaken for "no transcript". Missing, empty and
    malformed transcript *data* yields an empty segment list.
    """
    if not is_valid_video_id(video_id):
        return TranscriptResult([], "none", False)

    path = transcript_file_path(video_id, data_dir)
    file_exists = path is not None and os.path.isfile(path)

    if db is not None:
        segments = _load_from_database(db, video_id)
        if segments:
            return TranscriptResult(segments, "database", file_exists)

    if file_exists:
        try:
            segments = _load_from_file(path)
        except ValueError as exc:  # malformed JSON or invalid UTF-8
            logger.warning("Transcript file for %s is malformed: %s", video_id, type(exc).__name__)
        else:
            if segments:
                return TranscriptResult(segments, "file", True)

    return TranscriptResult([], "none", file_exists)
