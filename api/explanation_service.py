"""Reusable explanation generation and cache persistence.

The HTTP endpoint and the background preparation worker both call this module.
Database sessions are deliberately short-lived: no transaction remains open
while VLM, synthesis, or TTS work is running.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy.orm import Session

from api.models import Explanation, Intervention, Video


class ExplanationTargetNotFoundError(LookupError):
    """The requested video or intervention does not exist."""


@dataclass(frozen=True)
class ExplanationArtifact:
    intervention_id: str
    text_explanation: str
    audio_file_path: Optional[str]
    text_file_path: Optional[str]
    vlm_snapshot_path: Optional[str]
    output_mode: str


@dataclass(frozen=True)
class _ExplanationTarget:
    intervention_id: str
    timestamp: float
    frame_path: str
    content_type: str
    trigger_reason: str


# The current VLM/Ollama/TTS stack owns process-wide model and GPU resources.
# Serialize manual and background generation until those implementations expose
# a stronger thread-safety contract.
MODEL_GENERATION_LOCK = threading.Lock()


def _artifact(row: Explanation) -> ExplanationArtifact:
    return ExplanationArtifact(
        intervention_id=row.intervention_id,
        text_explanation=row.text_explanation,
        audio_file_path=row.audio_file_path,
        text_file_path=row.text_file_path,
        vlm_snapshot_path=row.vlm_snapshot_path,
        output_mode=row.output_mode,
    )


def resolve_explanation_audio_path(
    data_dir: str, audio_file_path: Optional[str]
) -> Optional[Path]:
    """Resolve a servable MP3 strictly inside the configured data directory."""
    if not audio_file_path:
        return None
    storage_root = Path(data_dir).resolve()
    stored_path = Path(audio_file_path)
    if stored_path.is_absolute():
        audio_path = stored_path.resolve()
    else:
        parts = stored_path.parts
        if parts and parts[0].lower() == "data":
            stored_path = Path(*parts[1:])
        audio_path = (storage_root / stored_path).resolve()

    try:
        if os.path.commonpath((storage_root, audio_path)) != str(storage_root):
            return None
    except ValueError:
        return None
    if audio_path.suffix.lower() != ".mp3" or not audio_path.is_file():
        return None
    try:
        if audio_path.stat().st_size < 2:
            return None
        with audio_path.open("rb") as handle:
            header = handle.read(3)
    except OSError:
        return None
    is_id3 = header == b"ID3"
    is_mpeg_frame = len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
    return audio_path if is_id3 or is_mpeg_frame else None


def find_valid_cached_explanation(
    db: Session,
    intervention_id: str,
    output_mode: str,
    data_dir: str,
) -> Optional[ExplanationArtifact]:
    """Return the newest complete, servable cached explanation."""
    rows = db.query(Explanation).filter(
        Explanation.intervention_id == intervention_id,
        Explanation.output_mode == output_mode,
    ).order_by(Explanation.created_at.desc(), Explanation.id.desc()).all()
    for row in rows:
        if (
            row.text_explanation
            and row.text_explanation.strip()
            and resolve_explanation_audio_path(data_dir, row.audio_file_path) is not None
        ):
            return _artifact(row)
    return None


def _load_target(db: Session, video_id: str, intervention_id: str) -> _ExplanationTarget:
    if db.query(Video.id).filter(Video.id == video_id).first() is None:
        raise ExplanationTargetNotFoundError("Video not found")

    row = db.query(Intervention).filter(
        Intervention.id == intervention_id,
        Intervention.video_id == video_id,
    ).first()
    if row is None:
        raise ExplanationTargetNotFoundError("Intervention not found")

    content_type = row.content_type or "unknown"
    if row.clip_confidence is not None and row.clip_confidence < 0.40:
        content_type = "unknown"

    return _ExplanationTarget(
        intervention_id=row.id,
        timestamp=row.timestamp,
        frame_path=row.frame_path,
        content_type=content_type,
        trigger_reason=row.trigger_reason or "VLM Analysis",
    )


def _load_transcript(data_dir: str, video_id: str) -> list:
    transcript_path = os.path.join(data_dir, video_id, "transcript.json")
    if not os.path.isfile(transcript_path):
        return []
    with open(transcript_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    return entries if isinstance(entries, list) else []


def generate_or_get_explanation(
    *,
    session_factory: Callable[[], Session],
    orchestrator_factory: Callable[[], object],
    data_dir: str,
    video_id: str,
    intervention_id: str,
    output_mode: str,
) -> ExplanationArtifact:
    """Return a cached explanation or generate and atomically cache one."""
    with session_factory() as db:
        _load_target(db, video_id, intervention_id)
        cached = find_valid_cached_explanation(
            db, intervention_id, output_mode, data_dir
        )
        if cached is not None:
            return cached

    # Double-check after acquiring the process-wide model lock. This prevents a
    # manual request and a queued job from generating the same cache key twice.
    with MODEL_GENERATION_LOCK:
        with session_factory() as db:
            target = _load_target(db, video_id, intervention_id)
            cached = find_valid_cached_explanation(
                db, intervention_id, output_mode, data_dir
            )
            if cached is not None:
                return cached

        transcript_entries = _load_transcript(data_dir, video_id)
        result = orchestrator_factory().generate_intervention_explanation(
            video_id=video_id,
            intervention_id=target.intervention_id,
            intervention_timestamp=target.timestamp,
            frame_path=target.frame_path,
            content_type=target.content_type,
            trigger_reason=target.trigger_reason,
            transcript_entries=transcript_entries,
            output_mode=output_mode,
        )

        # Inference is complete before this new transaction begins.
        with session_factory() as db:
            cached = find_valid_cached_explanation(
                db, intervention_id, output_mode, data_dir
            )
            if cached is not None:
                return cached

            row = db.query(Explanation).filter(
                Explanation.intervention_id == intervention_id,
                Explanation.output_mode == output_mode,
            ).order_by(Explanation.created_at.desc(), Explanation.id.desc()).first()
            if row is None:
                row = Explanation(
                    intervention_id=intervention_id,
                    output_mode=output_mode,
                )
            row.text_explanation = result["text_explanation"]
            row.audio_file_path = result.get("audio_file_path")
            row.text_file_path = result.get("text_file_path")
            row.vlm_snapshot_path = result.get("vlm_snapshot_path")
            try:
                db.add(row)
                db.commit()
                db.refresh(row)
            except Exception:
                db.rollback()
                raise
            return _artifact(row)
