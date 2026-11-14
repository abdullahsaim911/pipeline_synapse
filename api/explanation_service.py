"""Reusable explanation generation and cache persistence.

The HTTP endpoint and the background preparation worker both call this module.
Database sessions are deliberately short-lived: no transaction remains open
while VLM, synthesis, or TTS work is running.
"""

from __future__ import annotations

import json
import inspect
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from api.models import Explanation, ExplanationPreparation, Intervention, Video
from category_normalization import prompt_category_for
from synthesizer import VisualEvidenceUnavailableError
from vlm_interface.vlm_types import PROVIDER_ENV_VAR, PROVIDER_MODAL
from stage_tracing import trace_stage


class EvidenceUnavailableError(VisualEvidenceUnavailableError):
    """Extraction finished without usable evidence."""

    def __init__(self, failure: Optional[dict]):
        super().__init__("Visual evidence is unavailable")
        self.failure = failure or {}


class ExplanationTargetNotFoundError(LookupError):
    """The requested video or intervention does not exist."""


class PreparationCancelledError(RuntimeError):
    """A cancelled or superseded attempt reached a guarded side effect."""


class InvalidPreparedArtifactError(RuntimeError):
    """Generation returned without creating a complete servable cache."""


@dataclass(frozen=True)
class ExplanationArtifact:
    intervention_id: str
    text_explanation: str
    audio_file_path: Optional[str]
    text_file_path: Optional[str]
    vlm_snapshot_path: Optional[str]
    output_mode: str
    explanation_id: Optional[int] = None


@dataclass(frozen=True)
class EvidenceArtifact:
    snapshot: Optional[dict]
    snapshot_path: Optional[str]
    reused: bool


@dataclass(frozen=True)
class _ExplanationTarget:
    intervention_id: str
    timestamp: float
    frame_path: str
    content_type: str
    trigger_reason: str
    classification_confidence: Optional[float]
    confidence_semantics: str
    raw_category_labels: tuple[str, ...]
    canonical_category_ids: tuple[str, ...]
    evidence_kind: str = "visual"


# One remote VLM request at a time, and one Mistral/TTS run at a time. They are
# separate so extraction of the next intervention can overlap explanation of the
# previous one. A local VLM shares the GPU with Mistral, so that case takes both.
VLM_EXTRACTION_LOCK = threading.Lock()
EXPLANATION_GENERATION_LOCK = threading.Lock()


@contextmanager
def _explanation_resources():
    provider = os.environ.get(PROVIDER_ENV_VAR, PROVIDER_MODAL).strip().lower()
    if provider == PROVIDER_MODAL:
        with EXPLANATION_GENERATION_LOCK:
            yield
    else:
        with VLM_EXTRACTION_LOCK, EXPLANATION_GENERATION_LOCK:
            yield


def _artifact(row: Explanation) -> ExplanationArtifact:
    return ExplanationArtifact(
        intervention_id=row.intervention_id,
        text_explanation=row.text_explanation,
        audio_file_path=row.audio_file_path,
        text_file_path=row.text_file_path,
        vlm_snapshot_path=row.vlm_snapshot_path,
        output_mode=row.output_mode,
        explanation_id=row.id,
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
    from explanation_cache import cache_is_current
    for row in rows:
        audio_path = resolve_explanation_audio_path(data_dir, row.audio_file_path)
        if (
            row.text_explanation
            and row.text_explanation.strip()
            and audio_path is not None
            and cache_is_current(row.text_explanation, audio_path, row.vlm_snapshot_path)
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

    raw_content_type = row.content_type or "unknown"
    content_type = prompt_category_for(raw_content_type)
    trigger_reason = row.trigger_reason or "VLM Analysis"

    # Only a newly written classifier score may be used for uncertainty.  The
    # historical ``clip_confidence`` column contains complexity values for old
    # rows and is intentionally ignored here.  Low confidence is disclosed to
    # downstream prompting without erasing the valid category hint.
    if (
        row.confidence_semantics == "classifier"
        and row.classification_confidence is not None
        and row.classification_confidence < 0.40
    ):
        trigger_reason = f"{trigger_reason}; low category confidence"

    return _ExplanationTarget(
        intervention_id=row.id,
        timestamp=row.timestamp,
        frame_path=row.frame_path,
        content_type=content_type,
        trigger_reason=trigger_reason,
        classification_confidence=row.classification_confidence,
        confidence_semantics=row.confidence_semantics or "unknown",
        raw_category_labels=tuple(row.raw_category_labels),
        canonical_category_ids=tuple(row.canonical_categories),
        evidence_kind=row.evidence_kind,
    )


def _load_transcript(data_dir: str, video_id: str) -> list:
    transcript_path = os.path.join(data_dir, video_id, "transcript.json")
    if not os.path.isfile(transcript_path):
        return []
    with open(transcript_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    return entries if isinstance(entries, list) else []


def prepare_visual_evidence(
    *,
    session_factory: Callable[[], Session],
    orchestrator_factory: Callable[[], object],
    data_dir: str,
    video_id: str,
    intervention_id: str,
    output_mode: str,
    before_inference: Optional[Callable[[], bool]] = None,
    ownership_check: Optional[Callable[[], None]] = None,
    checkpoint_commit: Optional[Callable[[Callable[[], None]], None]] = None,
) -> Optional[EvidenceArtifact]:
    """Ensure usable evidence is saved; ``None`` means the explanation exists.

    Raises:
        EvidenceUnavailableError: with the classified failure
    """
    if ownership_check is not None:
        ownership_check()
    with session_factory() as db:
        target = _load_target(db, video_id, intervention_id)
        if find_valid_cached_explanation(db, intervention_id, output_mode, data_dir):
            return None

    # The reuse check runs inside the lock, so a second caller for the same
    # intervention finds the first caller's snapshot instead of calling the VLM.
    with VLM_EXTRACTION_LOCK:
        if ownership_check is not None:
            ownership_check()
        if target.evidence_kind == "auto":
            _extract_requested_frame(session_factory, data_dir, video_id, intervention_id,
                ownership_check, checkpoint_commit)
            with session_factory() as db:
                target = _load_target(db, video_id, intervention_id)
        if target.evidence_kind == "transcript":
            with session_factory() as db:
                context = db.get(Intervention, intervention_id).transcript_context
            if not context or not context.strip():
                raise EvidenceUnavailableError({"code": "insufficient_context", "retryable": False})
            return EvidenceArtifact({"_transcript_only": True, "transcript_context": context}, None, True)
        extraction_started = time.monotonic()
        orchestrator = orchestrator_factory()
        extract = orchestrator.extract_visual_evidence
        kwargs = dict(
            video_id=video_id,
            intervention_id=target.intervention_id,
            intervention_timestamp=target.timestamp,
            frame_path=target.frame_path,
            content_type=target.content_type,
            classification_confidence=target.classification_confidence,
            confidence_semantics=target.confidence_semantics,
            raw_category_labels=list(target.raw_category_labels),
            canonical_category_ids=list(target.canonical_category_ids),
            before_inference=before_inference,
        )
        if "ownership_check" in inspect.signature(extract).parameters:
            kwargs["ownership_check"] = ownership_check
            kwargs["checkpoint_commit"] = checkpoint_commit
        evidence = extract(**kwargs)
        if ownership_check is not None:
            ownership_check()
        trace_stage(
            "evidence_preparation", "complete", started_at=extraction_started,
            usable=evidence.get("usable"), reused=evidence.get("reused"),
            failure=evidence.get("failure"),
        )
    if not evidence["usable"]:
        raise EvidenceUnavailableError(evidence.get("failure"))
    return EvidenceArtifact(
        snapshot=None if evidence["reused"] else evidence["snapshot"],
        snapshot_path=evidence["snapshot_path"],
        reused=evidence["reused"],
    )


def _extract_requested_frame(sessions, data_dir, video_id, point_id, ownership_check, checkpoint_commit):
    """Extract one timestamp in the evidence worker, with transcript fallback for unavailable media."""
    with sessions() as db:
        point = db.get(Intervention, point_id)
        timestamp = point.timestamp
        source = db.get(Video, video_id).source_url
    folder = Path(data_dir).resolve() / video_id / "interventions" / point_id
    folder.mkdir(parents=True, exist_ok=True)
    frame = folder / "requested.jpg"
    temporary = folder / "requested.tmp.jpg"
    try:
        if not frame.is_file():
            import imageio_ffmpeg
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                "format": "best[height<=720]/bestvideo[height<=720]", "socket_timeout": 15}) as downloader:
                info = downloader.extract_info(source, download=False)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error",
                "-ss", str(timestamp), "-i", info["url"], "-frames:v", "1", "-y", str(temporary)],
                check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Requested frame unavailable")
            if checkpoint_commit:
                checkpoint_commit(lambda: os.replace(temporary, frame))
            else:
                os.replace(temporary, frame)
        kind = "visual"
    except PreparationCancelledError:
        raise
    except Exception as error:
        trace_stage("requested_frame", "unavailable", error_type=type(error).__name__)
        kind = "transcript"
    finally:
        temporary.unlink(missing_ok=True)
    if ownership_check:
        ownership_check()
    with sessions() as db:
        point = db.get(Intervention, point_id)
        point.evidence_kind = kind
        point.frame_path = str(frame) if kind == "visual" else ""
        db.commit()


def generate_explanation_from_evidence(
    *,
    session_factory: Callable[[], Session],
    orchestrator_factory: Callable[[], object],
    data_dir: str,
    video_id: str,
    intervention_id: str,
    output_mode: str,
    vlm_snapshot: Optional[dict] = None,
    on_stage: Optional[Callable[[str], Any]] = None,
    ownership_check: Optional[Callable[[], None]] = None,
    checkpoint_commit: Optional[Callable[[Callable[[], None]], None]] = None,
    preparation_lease: Optional[tuple[int, str, int]] = None,
) -> ExplanationArtifact:
    """Return a cached explanation or generate and cache one from evidence."""
    if ownership_check is not None:
        ownership_check()
    with session_factory() as db:
        _load_target(db, video_id, intervention_id)
        cached = find_valid_cached_explanation(
            db, intervention_id, output_mode, data_dir
        )
        if cached is not None:
            return cached

    # Double-check after acquiring the lock. This prevents a manual request and
    # a queued job from generating the same cache key twice.
    with _explanation_resources():
        if ownership_check is not None:
            ownership_check()
        with session_factory() as db:
            target = _load_target(db, video_id, intervention_id)
            cached = find_valid_cached_explanation(
                db, intervention_id, output_mode, data_dir
            )
            if cached is not None:
                return cached

        transcript_entries = _load_transcript(data_dir, video_id)
        orchestrator = orchestrator_factory()
        synthesize = orchestrator.synthesize_intervention_explanation
        kwargs = dict(
            video_id=video_id,
            intervention_id=target.intervention_id,
            intervention_timestamp=target.timestamp,
            content_type=target.content_type,
            trigger_reason=target.trigger_reason,
            transcript_entries=transcript_entries,
            output_mode=output_mode,
            vlm_snapshot=vlm_snapshot,
            on_stage=on_stage,
        )
        if "ownership_check" in inspect.signature(synthesize).parameters:
            kwargs["ownership_check"] = ownership_check
            kwargs["checkpoint_commit"] = checkpoint_commit
        result = synthesize(**kwargs)
        if ownership_check is not None:
            ownership_check()

        # The API owns publication, including alternative synthesis providers.
        # Stamp only a freshly returned result, never an existing database row.
        from explanation_cache import cache_is_current, generation_manifest, write_manifest
        generated_audio = resolve_explanation_audio_path(data_dir, result.get("audio_file_path"))
        if generated_audio is None:
            raise InvalidPreparedArtifactError("Synthesis did not produce a servable MP3")
        if not cache_is_current(result["text_explanation"], generated_audio, result.get("vlm_snapshot_path")):
            provenance = generation_manifest(result["text_explanation"], generated_audio, result.get("vlm_snapshot_path"))
            publish_provenance = lambda: write_manifest(generated_audio, provenance)
            if checkpoint_commit is not None:
                checkpoint_commit(publish_provenance)
            else:
                publish_provenance()

        # Inference is complete before this new transaction begins.
        persistence_started = time.monotonic()
        trace_stage("explanation_persistence", "start")
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
                if preparation_lease is not None:
                    state_id, lease_owner, attempt_count = preparation_lease
                    owned = db.query(ExplanationPreparation).filter(
                        ExplanationPreparation.id == state_id,
                        ExplanationPreparation.status == "generating",
                        ExplanationPreparation.lease_owner == lease_owner,
                        ExplanationPreparation.attempt_count == attempt_count,
                    ).update({
                        ExplanationPreparation.updated_at:
                            datetime.now(timezone.utc).replace(tzinfo=None)
                    },
                             synchronize_session=False)
                    if owned != 1:
                        raise PreparationCancelledError(
                            "Preparation was cancelled before explanation commit"
                        )
                db.commit()
                db.refresh(row)
            except Exception:
                db.rollback()
                trace_stage(
                    "explanation_persistence", "failed",
                    started_at=persistence_started,
                )
                raise
            trace_stage(
                "explanation_persistence", "committed",
                started_at=persistence_started, explanation_id=row.id,
            )
            from api.personal_storage import capture_subscribers
            try:
                capture_subscribers(db, row)
                db.commit()
            except Exception as error:
                # A personal storage failure must not turn completed inference
                # into a failed preparation. Collection reads retry preservation
                # and report storage failures to the learner.
                db.rollback()
                trace_stage("personal_snapshot", "failed", error_type=type(error).__name__)
            return _artifact(row)


def generate_or_get_explanation(
    *,
    session_factory: Callable[[], Session],
    orchestrator_factory: Callable[[], object],
    data_dir: str,
    video_id: str,
    intervention_id: str,
    output_mode: str,
) -> ExplanationArtifact:
    """Run both stages in the calling thread (no background manager)."""
    evidence = prepare_visual_evidence(
        session_factory=session_factory,
        orchestrator_factory=orchestrator_factory,
        data_dir=data_dir,
        video_id=video_id,
        intervention_id=intervention_id,
        output_mode=output_mode,
    )
    return generate_explanation_from_evidence(
        session_factory=session_factory,
        orchestrator_factory=orchestrator_factory,
        data_dir=data_dir,
        video_id=video_id,
        intervention_id=intervention_id,
        output_mode=output_mode,
        vlm_snapshot=evidence.snapshot if evidence is not None else None,
    )
