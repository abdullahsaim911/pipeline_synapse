"""Durable coordination and cache validation for the existing video pipeline."""

from __future__ import annotations

import json
import inspect
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import exists, or_, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from api.database import PROJECT_ROOT
from api.models import (
    Explanation,
    ExplanationPreparation,
    Intervention,
    ProcessingJob,
    ProcessingJobRequestedMode,
    Transcript,
    Video,
)
from category_normalization import normalize_category_labels
from stage_tracing import TraceContext, bind_trace, trace_stage


logger = logging.getLogger(__name__)

SAFE_ERRORS = {
    "pipeline_failed": "Video processing failed. You can retry this job.",
    "persistence_failed": "Processing results could not be saved. You can retry this job.",
    "invalid_cache": "The saved video cache was incomplete and must be rebuilt.",
    "internal_error": "Video processing failed unexpectedly. You can retry this job.",
}
VALID_OUTPUT_MODES = {"brief", "explanatory", "detailed"}


class ProcessingPersistenceError(RuntimeError):
    """Detection completed, but required cache records were not saved."""


class ProcessingJobNotFoundError(LookupError):
    pass


class ProcessingJobTimeoutError(TimeoutError):
    pass


class ProcessingJobBusyError(RuntimeError):
    """A different video owns the single active processing session."""

    def __init__(self, video_id: str, job_id: str):
        super().__init__(f"Video {video_id} is already active")
        self.video_id = video_id
        self.job_id = job_id


class ProcessingCancelledError(RuntimeError):
    """The current processing generation no longer owns its durable job."""


@dataclass(frozen=True)
class CacheInspection:
    state: str
    reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.state == "valid"


def _utcnow() -> datetime:
    # SQLite returns naive UTC values even when DateTime(timezone=True) is used.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _frame_is_missing(frame_path: str) -> bool:
    path = Path(frame_path)
    if path.is_absolute():
        return not path.is_file()

    normalized = frame_path.replace("\\", "/")
    if not normalized.startswith("data/"):
        # Older/test producers may use a path relative to their own video root.
        return False
    return not (PROJECT_ROOT / path).is_file()


def inspect_video_cache(db: Session, video_id: str) -> CacheInspection:
    """Validate the database-backed cache and production media references."""
    video = db.get(Video, video_id)
    if video is None:
        return CacheInspection("missing")
    if not video.title or not video.source_url:
        return CacheInspection("invalid", "missing_video_metadata")
    if video.total_keyframes is None or video.total_keyframes < 0:
        return CacheInspection("invalid", "invalid_keyframe_count")

    transcripts = (
        db.query(Transcript)
        .filter(Transcript.video_id == video_id)
        .order_by(Transcript.start_time)
        .all()
    )
    if not transcripts:
        return CacheInspection("invalid", "missing_transcript")
    for segment in transcripts:
        if (
            segment.start_time is None
            or segment.end_time is None
            or segment.end_time < segment.start_time
            or not isinstance(segment.text, str)
            or not segment.text.strip()
        ):
            return CacheInspection("invalid", "invalid_transcript")

    interventions = db.query(Intervention).filter(Intervention.video_id == video_id).all()
    for intervention in interventions:
        if (
            not intervention.id
            or intervention.timestamp is None
            or intervention.timestamp < 0
            or not intervention.frame_path
        ):
            return CacheInspection("invalid", "invalid_intervention")
        if _frame_is_missing(intervention.frame_path):
            return CacheInspection("invalid", "missing_intervention_media")
    return CacheInspection("valid")


def _delete_invalid_cache(db: Session, video_id: str) -> None:
    intervention_ids = [
        row[0]
        for row in db.query(Intervention.id).filter(Intervention.video_id == video_id).all()
    ]
    db.query(ExplanationPreparation).filter(
        ExplanationPreparation.video_id == video_id
    ).delete(synchronize_session=False)
    if intervention_ids:
        db.query(Explanation).filter(
            Explanation.intervention_id.in_(intervention_ids)
        ).delete(synchronize_session=False)
    db.query(Intervention).filter(Intervention.video_id == video_id).delete(
        synchronize_session=False
    )
    db.query(Transcript).filter(Transcript.video_id == video_id).delete(
        synchronize_session=False
    )
    db.query(Video).filter(Video.id == video_id).delete(synchronize_session=False)
    db.flush()


def _transcript_end_time(entry: dict, start_time: float) -> float:
    if entry.get("end") is not None:
        return float(entry["end"])
    return start_time + float(entry.get("duration", 0.0))


def persist_processing_result(
    db: Session,
    video_id: str,
    source_url: str,
    result: dict,
    interventions: list[Intervention],
    *,
    job_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    lease_owner: Optional[str] = None,
) -> None:
    """Atomically replace an invalid cache with one complete processing result."""
    try:
        transcript_data = result.get("transcript")
        entries = transcript_data.get("entries") if isinstance(transcript_data, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("Detection result did not contain transcript entries")

        if db.get(Video, video_id) is not None:
            _delete_invalid_cache(db, video_id)

        db.add(Video(
            id=video_id,
            title=result.get("title") or f"Video {video_id}",
            source_url=source_url,
            duration_seconds=result.get("duration_seconds"),
            duration_formatted=result.get("duration_formatted"),
            total_keyframes=result.get("metadata", {}).get("total_keyframes", 0),
            subject=result.get("subject"),
        ))
        db.flush()

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Transcript entries must be objects")
            start_time = float(entry.get("start", 0.0))
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Transcript entries must contain text")
            db.add(Transcript(
                video_id=video_id,
                start_time=start_time,
                end_time=_transcript_end_time(entry, start_time),
                text=text,
            ))
        db.add_all(interventions)
        if job_id is not None:
            owned = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.status == "running",
                ProcessingJob.generation_id == generation_id,
                ProcessingJob.lease_owner == lease_owner,
            ).update({ProcessingJob.updated_at: _utcnow()}, synchronize_session=False)
            if owned != 1:
                raise ProcessingCancelledError(job_id)
        db.commit()
    except ProcessingCancelledError:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise ProcessingPersistenceError(
            f"Could not persist processing result for {video_id}"
        ) from exc


def _build_interventions(video_id: str, result: dict) -> list[Intervention]:
    records = []
    for point in result.get("intervention_points", []):
        intervention_id = str(uuid.uuid4())
        try:
            point.id = intervention_id
        except (AttributeError, TypeError):
            pass
        value = point.get if isinstance(point, dict) else lambda key, default=None: getattr(point, key, default)
        timestamp = float(value("timestamp", 0.0))
        raw_labels = value("raw_category_labels") or [value("content_type", "unknown")]
        normalization = normalize_category_labels(raw_labels)
        classification_confidence = value("classification_confidence")
        confidence_semantics = value("confidence_semantics")
        if classification_confidence is None:
            confidence_semantics = confidence_semantics or "legacy_complexity"
        else:
            confidence_semantics = confidence_semantics or "classifier"
        records.append(Intervention(
            id=intervention_id,
            video_id=video_id,
            timestamp=timestamp,
            timestamp_formatted=f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
            frame_path=value("frame_path", ""),
            content_type=value("content_type"),
            complexity_score=value("complexity_score"),
            clip_confidence=value("clip_confidence"),
            classification_confidence=classification_confidence,
            confidence_semantics=confidence_semantics,
            raw_category_labels_json=json.dumps(list(normalization.raw_labels), ensure_ascii=False),
            canonical_categories_json=json.dumps(list(normalization.canonical_categories), ensure_ascii=False),
            category_mapping_notes_json=json.dumps(list(normalization.notes), ensure_ascii=False),
            trigger_reason=value("trigger_reason"),
            confidence=value("confidence"),
            transcript_context=value("transcript_context"),
            is_bookmarked=False,
            created_at=_utcnow(),
        ))
    return records


def _write_interventions_json(video_id: str, result: dict, records: list[Intervention]) -> None:
    try:
        video_dir = Path(result.get("video_dir", PROJECT_ROOT / "data" / video_id))
        video_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_id": video_id,
            "total_interventions": len(records),
            "interventions": [
                {
                    "id": item.id,
                    "video_id": item.video_id,
                    "timestamp": item.timestamp,
                    "timestamp_formatted": item.timestamp_formatted,
                    "frame_path": item.frame_path,
                    "content_type": item.content_type,
                    "raw_category_labels": item.raw_category_labels,
                    "canonical_categories": item.canonical_categories,
                    "classification_confidence": item.classification_confidence,
                    "confidence_semantics": item.confidence_semantics,
                }
                for item in records
            ],
            "created_at": _utcnow().isoformat(),
        }
        (video_dir / "interventions.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        logger.warning("Could not write derived interventions JSON for %s", video_id, exc_info=True)


def serialize_job(
    job: ProcessingJob,
    *,
    requested_modes: Optional[list[str]] = None,
    joined_existing: bool = False,
    reused_cache: bool = False,
) -> dict:
    return {
        "job_id": str(job.id),
        "video_id": job.video_id,
        "status": job.status,
        "stage": job.stage,
        "attempt_count": job.attempt_count,
        "requested_modes": requested_modes or [job.requested_output_mode],
        "joined_existing": joined_existing,
        "reused_cache": reused_cache,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "cancel_requested_at": job.cancel_requested_at,
        "cancelled_at": job.cancelled_at,
    }


def _requested_modes(db: Session, job: ProcessingJob) -> list[str]:
    modes = [
        row[0]
        for row in db.query(ProcessingJobRequestedMode.output_mode)
        .filter(ProcessingJobRequestedMode.job_id == job.id)
        .order_by(
            ProcessingJobRequestedMode.requested_at,
            ProcessingJobRequestedMode.output_mode,
        )
        .all()
    ]
    # This fallback keeps ORM-created pre-migration fixtures and partially
    # upgraded development databases recoverable.
    return modes or [job.requested_output_mode]


def _record_requested_mode(db: Session, job_id: str, output_mode: str) -> None:
    exists = db.query(ProcessingJobRequestedMode.job_id).filter(
        ProcessingJobRequestedMode.job_id == job_id,
        ProcessingJobRequestedMode.output_mode == output_mode,
    ).first()
    if exists is not None:
        return
    try:
        with db.begin_nested():
            db.add(ProcessingJobRequestedMode(
                job_id=job_id,
                output_mode=output_mode,
            ))
            db.flush()
        db.commit()
    except IntegrityError:
        # A concurrent caller registered the same mode first.
        db.rollback()


def _integrity_constraint_name(exc: IntegrityError) -> Optional[str]:
    """Return a driver-reported constraint name without assuming PostgreSQL."""
    original = getattr(exc, "orig", None)
    diagnostic = getattr(original, "diag", None)
    return getattr(diagnostic, "constraint_name", None)


def _is_active_slot_conflict(exc: IntegrityError) -> bool:
    name = _integrity_constraint_name(exc)
    if name is not None:
        return name == "uq_processing_jobs_active_slot"
    return "processing_jobs.active_slot" in str(getattr(exc, "orig", exc)).lower()


def _is_video_job_conflict(exc: IntegrityError) -> bool:
    name = _integrity_constraint_name(exc)
    if name is not None:
        return name == "uq_processing_jobs_video_id"
    return "processing_jobs.video_id" in str(getattr(exc, "orig", exc)).lower()


def commit_active_slot(db: Session, video_id: str) -> bool:
    """Commit slot acquisition, translating only the durable slot constraint.

    ``False`` means a concurrent transaction acquired the same video's row, so
    the caller should reload and join it. Different-video winners are reported
    with the existing actionable busy exception.
    """
    try:
        db.commit()
        return True
    except IntegrityError as exc:
        db.rollback()
        if not _is_active_slot_conflict(exc):
            raise
        winner = db.query(ProcessingJob).filter(
            ProcessingJob.active_slot == 1
        ).one_or_none()
        if winner is None:
            raise
        if winner.video_id == video_id:
            return False
        raise ProcessingJobBusyError(winner.video_id, str(winner.id)) from exc


def release_terminal_video_slot(session_factory, video_id: str) -> bool:
    """Atomically release terminal ownership when no preparation is active."""
    with session_factory() as db:
        unfinished = exists().where(
            ExplanationPreparation.video_id == video_id,
            ExplanationPreparation.status.in_(("pending", "generating")),
        )
        released = db.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.video_id == video_id,
                ProcessingJob.active_slot == 1,
                ProcessingJob.status.in_(("succeeded", "failed", "cancelled")),
                ~unfinished,
            )
            .values(active_slot=None, updated_at=_utcnow())
        ).rowcount
        db.commit()
        return released == 1


def _normalize_job_id(job_id: str) -> str:
    try:
        return str(uuid.UUID(str(job_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProcessingJobNotFoundError(str(job_id)) from exc


class ProcessingJobManager:
    """Threaded runner backed by a database lease and one row per video."""

    def __init__(
        self,
        session_factory,
        orchestrator_factory: Callable,
        preparation_manager=None,
        *,
        video_id_extractor: Optional[Callable[[str], str]] = None,
        admission_guard: Optional[Callable[[], None]] = None,
        max_workers: int = 1,
        lease_seconds: float = 120.0,
        heartbeat_seconds: float = 30.0,
        recovery_interval_seconds: Optional[float] = None,
        heartbeat_retry_limit: int = 3,
        heartbeat_retry_backoff_seconds: float = 0.1,
        shutdown_grace_seconds: Optional[float] = None,
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        recovery_interval = (
            recovery_interval_seconds
            if recovery_interval_seconds is not None
            else min(30.0, lease_seconds / 3.0)
        )
        if recovery_interval <= 0 or recovery_interval >= lease_seconds:
            raise ValueError(
                "recovery_interval_seconds must be positive and shorter than the lease"
            )
        if heartbeat_retry_limit < 1:
            raise ValueError("heartbeat_retry_limit must be at least one")
        if heartbeat_retry_backoff_seconds <= 0:
            raise ValueError("heartbeat_retry_backoff_seconds must be positive")
        shutdown_grace = (
            float(os.getenv("PROCESSING_SHUTDOWN_GRACE_SECONDS", "5"))
            if shutdown_grace_seconds is None else shutdown_grace_seconds
        )
        if shutdown_grace < 0:
            raise ValueError("shutdown_grace_seconds cannot be negative")
        self.session_factory = session_factory
        self.orchestrator_factory = orchestrator_factory
        self.preparation_manager = preparation_manager
        self.video_id_extractor = video_id_extractor
        self.admission_guard = admission_guard
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = min(heartbeat_seconds, max(lease_seconds / 3, 0.05))
        self.recovery_interval_seconds = recovery_interval
        self.heartbeat_retry_limit = heartbeat_retry_limit
        self.heartbeat_retry_backoff_seconds = heartbeat_retry_backoff_seconds
        self.shutdown_grace_seconds = shutdown_grace
        self.owner = str(uuid.uuid4())
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="processing-job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._started = False
        self._shutdown_started = False
        self._stop_event = threading.Event()
        self._maintenance_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            if self._shutdown_started:
                raise RuntimeError("Processing job manager has been shut down")
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
        queued = []
        cached_to_prepare: list[tuple[str, list[str]]] = []
        now = _utcnow()
        with self.session_factory() as db:
            jobs = db.query(ProcessingJob).filter(
                ProcessingJob.status.in_(("queued", "running", "cancelling"))
            ).all()
            for job in jobs:
                if job.status == "cancelling":
                    if self.preparation_manager is not None:
                        self.preparation_manager.cancel_video(job.video_id)
                    self._mark_cancelled(job, now)
                    continue
                cache = inspect_video_cache(db, job.video_id)
                if cache.valid:
                    cached_to_prepare.append((job.video_id, _requested_modes(db, job)))
                    self._mark_succeeded(job, now)
                elif job.status == "queued" or self._lease_expired(job, now):
                    job.status = "queued"
                    job.stage = "queued"
                    job.lease_owner = None
                    job.lease_expires_at = None
                    job.updated_at = now
                    queued.append(str(job.id))
            db.commit()
        for video_id, modes in cached_to_prepare:
            self._schedule_modes(video_id, modes)
        for job_id in queued:
            self._submit(job_id)
        maintenance = threading.Thread(
            target=self._run_recovery_maintenance,
            name="processing-job-lease-recovery",
            daemon=True,
        )
        self._maintenance_thread = maintenance
        maintenance.start()

    def shutdown(
        self, wait: bool = True, timeout: Optional[float] = None
    ) -> bool:
        """Stop dispatch and wait only within the configured application grace.

        Returns whether every executor future stopped before the deadline. A
        false result is recoverable durable work, not permission to terminate a
        Python thread. Standard executor threads may still delay interpreter
        exit if a synchronous provider call is uninterruptible.
        """
        grace = self.shutdown_grace_seconds if timeout is None else timeout
        if grace < 0:
            raise ValueError("shutdown timeout cannot be negative")
        if not wait:
            grace = 0.0
        deadline = time.monotonic() + grace
        # Linearize shutdown against `_submit`: once the stop flag is visible,
        # no maintenance or completion callback can enqueue another future.
        with self._lock:
            self._shutdown_started = True
            self._stop_event.set()
            futures = list(self._futures.values())
        for future in futures:
            future.cancel()
        maintenance = self._maintenance_thread
        if maintenance is not None and maintenance.ident is not None:
            maintenance.join(timeout=max(0.0, deadline - time.monotonic()))
            if maintenance.is_alive():
                logger.warning(
                    "Processing lease recovery did not stop before shutdown deadline",
                    extra={
                        "stage": "shutdown",
                        "failure_classification": "maintenance_deadline_exceeded",
                    },
                )
            else:
                self._maintenance_thread = None
        # `wait=False` is essential here: waiting inside Executor.shutdown can
        # exceed the application deadline. We perform our own bounded wait.
        self._executor.shutdown(wait=False, cancel_futures=True)
        remaining = max(0.0, deadline - time.monotonic())
        _, unfinished = wait_futures(futures, timeout=remaining)
        # A Future becomes done before its callback returns. Give `_forget`
        # the remaining grace to close its database session as well.
        while not unfinished and time.monotonic() < deadline:
            with self._lock:
                callbacks_pending = any(
                    self._futures.get(job_id) is future
                    for job_id, future in list(self._futures.items())
                    if future in futures
                )
            if not callbacks_pending:
                break
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        with self._lock:
            callbacks_pending = any(
                self._futures.get(job_id) is future
                for job_id, future in list(self._futures.items())
                if future in futures
            )
        stopped = not unfinished and not callbacks_pending
        if stopped:
            # With no running work or callbacks left, this join cannot inherit
            # an uninterruptible provider call and lets idle executor threads
            # release thread-local database resources deterministically.
            self._executor.shutdown(wait=True, cancel_futures=True)
        if not stopped:
            logger.warning(
                "Processing workers exceeded the shutdown grace period",
                extra={
                    "stage": "shutdown",
                    "failure_classification": "worker_deadline_exceeded",
                    "unfinished_processing_workers": len(unfinished),
                },
            )
        return stopped

    def create_or_join(self, youtube_url: str, output_mode: str) -> dict:
        if self._stop_event.is_set():
            raise RuntimeError("Processing job manager is shutting down")
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError(f"Unsupported explanation mode: {output_mode}")
        if self.video_id_extractor is None:
            video_id = self.orchestrator_factory()._extract_video_id(youtube_url)
        else:
            video_id = self.video_id_extractor(youtube_url)
        now = _utcnow()
        should_submit = False
        joined = False
        reused = False

        self._release_completed_slots()

        with self.session_factory() as db:
            active = db.query(ProcessingJob).filter(
                ProcessingJob.video_id != video_id,
                ProcessingJob.active_slot == 1,
            ).order_by(ProcessingJob.updated_at.desc()).first()
            if active is None:
                active_preparation = db.query(ExplanationPreparation).filter(
                    ExplanationPreparation.video_id != video_id,
                    ExplanationPreparation.status.in_(("pending", "generating")),
                ).first()
                if active_preparation is not None:
                    active = db.query(ProcessingJob).filter(
                        ProcessingJob.video_id == active_preparation.video_id
                    ).first()
            if active is not None:
                raise ProcessingJobBusyError(active.video_id, str(active.id))
            cache = inspect_video_cache(db, video_id)
            if not cache.valid and self.admission_guard is not None:
                self.admission_guard()
            job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video_id).first()
            if job is not None and job.status in {"cancelling", "cancelled"}:
                return serialize_job(
                    job,
                    requested_modes=_requested_modes(db, job),
                    joined_existing=True,
                    reused_cache=cache.valid,
                )
            if cache.valid:
                reused = True
                if job is None:
                    job = ProcessingJob(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        source_url=youtube_url,
                        requested_output_mode=output_mode,
                        active_slot=1,
                    )
                    db.add(job)
                    try:
                        committed = commit_active_slot(db, video_id)
                    except IntegrityError as exc:
                        if not _is_video_job_conflict(exc):
                            raise
                        job = db.query(ProcessingJob).filter(
                            ProcessingJob.video_id == video_id
                        ).one_or_none()
                        if job is None:
                            raise
                        joined = True
                    else:
                        if not committed:
                            job = db.query(ProcessingJob).filter(
                                ProcessingJob.video_id == video_id
                            ).one()
                            joined = True
                self._mark_succeeded(job, now, keep_active=self.preparation_manager is not None)
                commit_active_slot(db, video_id)
                db.refresh(job)
                _record_requested_mode(db, str(job.id), output_mode)
                modes = _requested_modes(db, job)
                snapshot = serialize_job(
                    job,
                    requested_modes=modes,
                    joined_existing=joined,
                    reused_cache=True,
                )
                trace_stage(
                    "video_submission", "reused",
                    context=TraceContext(
                        trace_id=f"processing-{job.id}", video_id=video_id,
                        output_mode=output_mode, processing_job_id=str(job.id),
                        attempt=job.attempt_count,
                    ),
                    status="cached",
                )
                self._schedule_modes(video_id, modes)
                self._release_completed_slots()
                return snapshot

            if job is None:
                job = ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    source_url=youtube_url,
                    requested_output_mode=output_mode,
                    status="queued",
                    stage="queued",
                    active_slot=1,
                )
                db.add(job)
                try:
                    committed = commit_active_slot(db, video_id)
                except IntegrityError as exc:
                    if not _is_video_job_conflict(exc):
                        raise
                    job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video_id).one_or_none()
                    if job is None:
                        raise
                    joined = True
                else:
                    if committed:
                        db.refresh(job)
                        should_submit = True
                    else:
                        job = db.query(ProcessingJob).filter(
                            ProcessingJob.video_id == video_id
                        ).one()
                        joined = True
            else:
                joined = job.status in ("queued", "running") and not self._lease_expired(job, now)
                if not joined:
                    job.source_url = youtube_url
                    job.requested_output_mode = output_mode
                    job.status = "queued"
                    job.stage = "queued"
                    job.error_code = None
                    job.error_message = None
                    job.lease_owner = None
                    job.lease_expires_at = None
                    job.generation_id = None
                    job.cancel_requested_at = None
                    job.cancelled_at = None
                    job.active_slot = 1
                    job.completed_at = None
                    job.updated_at = now
                    commit_active_slot(db, video_id)
                    should_submit = True
                elif job.status == "queued":
                    should_submit = True
            _record_requested_mode(db, str(job.id), output_mode)
            db.refresh(job)
            job_id = str(job.id)
            snapshot = serialize_job(
                job,
                requested_modes=_requested_modes(db, job),
                joined_existing=joined,
                reused_cache=reused,
            )

        if should_submit:
            self._submit(job_id)
        trace_stage(
            "video_submission", "joined" if joined else "queued",
            context=TraceContext(
                trace_id=f"processing-{job_id}", video_id=video_id,
                output_mode=output_mode, processing_job_id=job_id,
                attempt=snapshot.get("attempt_count", 0),
            ),
            status=snapshot.get("status"),
        )
        return snapshot

    def retry(self, job_id: str) -> dict:
        job_id = _normalize_job_id(job_id)
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                raise ProcessingJobNotFoundError(job_id)
            if job.status in {"cancelling", "cancelled"}:
                return serialize_job(job, requested_modes=_requested_modes(db, job))
            source_url = job.source_url
            output_mode = job.requested_output_mode
        return self.create_or_join(source_url, output_mode)

    def cancel(self, job_id: str) -> dict:
        """Durably fence every unfinished attempt for this video's job."""
        job_id = _normalize_job_id(job_id)
        now = _utcnow()
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                raise ProcessingJobNotFoundError(job_id)
            video_id = job.video_id
            trace_stage("video_cancellation", "requested", video_id=video_id,
                        active_stage=job.stage, attempt=job.attempt_count)
            if job.status in {"cancelled", "failed"}:
                return serialize_job(job, requested_modes=_requested_modes(db, job))
            unfinished = db.query(ExplanationPreparation.id).filter(
                ExplanationPreparation.video_id == video_id,
                ExplanationPreparation.status.in_(("pending", "generating")),
            ).first() is not None
            if job.status == "succeeded" and not unfinished:
                return serialize_job(job, requested_modes=_requested_modes(db, job))
            if job.status in {"queued", "running", "succeeded", "failed", "cancelling"}:
                job.status = "cancelling"
                job.stage = "cancelling"
                job.cancel_requested_at = job.cancel_requested_at or now
                job.generation_id = None
                job.lease_owner = None
                job.lease_expires_at = None
                job.updated_at = now
                db.commit()
            modes = _requested_modes(db, job)

        if self.preparation_manager is not None:
            self.preparation_manager.cancel_video(video_id)
        with self._lock:
            future = self._futures.get(job_id)
            local_processing = future is not None and not future.done()
        local_abort_succeeded = False
        if local_processing:
            local_abort_succeeded = future.cancel()
        self._settle_cancelled(job_id)
        trace_stage("video_cancellation", "accepted", video_id=video_id,
                    local_abort=(
                        "succeeded" if local_abort_succeeded else
                        "unsupported_in_flight" if local_processing else "not_running"
                    ))
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            return serialize_job(job, requested_modes=modes)

    def resume(self, job_id: str) -> dict:
        """Create a fresh fenced attempt while preserving valid checkpoints."""
        if self._stop_event.is_set():
            raise RuntimeError("Processing job manager is shutting down")
        job_id = _normalize_job_id(job_id)
        now = _utcnow()
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                raise ProcessingJobNotFoundError(job_id)
            if job.status not in {"cancelling", "cancelled"}:
                return serialize_job(job, requested_modes=_requested_modes(db, job))
            active = db.query(ProcessingJob).filter(
                ProcessingJob.id != job.id,
                ProcessingJob.active_slot == 1,
            ).first()
            if active is not None:
                raise ProcessingJobBusyError(active.video_id, str(active.id))
            job.status = "queued"
            job.stage = "queued"
            job.error_code = None
            job.error_message = None
            job.generation_id = None
            job.lease_owner = None
            job.lease_expires_at = None
            job.cancel_requested_at = None
            job.cancelled_at = None
            job.active_slot = 1
            # A cancelled row can still contain stale terminal slot metadata.
            # Force this assignment into the UPDATE so a concurrent terminal
            # release cannot leave the newly queued attempt without ownership.
            flag_modified(job, "active_slot")
            job.completed_at = None
            job.updated_at = now
            modes = _requested_modes(db, job)
            commit_active_slot(db, job.video_id)
        if self.preparation_manager is not None:
            self.preparation_manager.resume_video(job.video_id, modes)
        trace_stage("video_resume", "started", video_id=job.video_id,
                    processing_job_id=job_id, next_attempt=job.attempt_count + 1)
        self._submit(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        job_id = _normalize_job_id(job_id)
        self._settle_cancelled(job_id)
        self._release_completed_slots()
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                raise ProcessingJobNotFoundError(job_id)
            return serialize_job(job, requested_modes=_requested_modes(db, job))

    def wait(self, job_id: str, timeout: Optional[float] = None) -> dict:
        timeout = timeout or float(os.getenv("PROCESSING_JOIN_TIMEOUT_SECONDS", "14400"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.get(job_id)
            if snapshot["status"] in ("succeeded", "failed", "cancelled"):
                return snapshot
            time.sleep(0.05)
        raise ProcessingJobTimeoutError(job_id)

    def _submit(self, job_id: str) -> bool:
        with self._lock:
            if self._stop_event.is_set():
                return False
            current = self._futures.get(job_id)
            if current is not None and not current.done():
                return False
            future = self._executor.submit(self._run, job_id)
            self._futures[job_id] = future
        # Register outside the lock: add_done_callback runs synchronously when
        # an exceptionally fast job has already completed.
        future.add_done_callback(lambda completed: self._forget(job_id, completed))
        return True

    def _forget(self, job_id: str, completed: Future) -> None:
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            should_resume = job is not None and job.status == "queued"
        with self._lock:
            if self._futures.get(job_id) is completed:
                self._futures.pop(job_id, None)
        if should_resume:
            self._submit(job_id)
        else:
            self._settle_cancelled(job_id)

    def _claim(self, job_id: str) -> Optional[str]:
        now = _utcnow()
        lease = now + timedelta(seconds=self.lease_seconds)
        generation_id = str(uuid.uuid4())
        with self.session_factory() as db:
            claimed = db.execute(
                update(ProcessingJob)
                .where(
                    ProcessingJob.id == job_id,
                    or_(
                        ProcessingJob.status == "queued",
                        (
                            (ProcessingJob.status == "running")
                            & or_(
                                ProcessingJob.lease_expires_at.is_(None),
                                ProcessingJob.lease_expires_at <= now,
                            )
                        ),
                    ),
                )
                .values(
                    status="running",
                    stage="processing",
                    attempt_count=ProcessingJob.attempt_count + 1,
                    error_code=None,
                    error_message=None,
                    lease_owner=self.owner,
                    lease_expires_at=lease,
                    generation_id=generation_id,
                    active_slot=1,
                    started_at=now,
                    completed_at=None,
                    updated_at=now,
                )
            ).rowcount
            db.commit()
            return generation_id if claimed == 1 else None

    def _run_recovery_maintenance(self) -> None:
        """Periodically dispatch expired running jobs; `_claim` picks one owner."""
        while not self._stop_event.wait(self.recovery_interval_seconds):
            try:
                self._schedule_expired_running_jobs()
            except Exception as exc:
                logger.warning(
                    "Could not scan expired processing leases (%s); will retry",
                    type(exc).__name__,
                )

    def _schedule_expired_running_jobs(self) -> int:
        if self._stop_event.is_set():
            return 0
        now = _utcnow()
        with self.session_factory() as db:
            job_ids = [
                str(row[0])
                for row in db.query(ProcessingJob.id).filter(
                    ProcessingJob.status == "running",
                    or_(
                        ProcessingJob.lease_expires_at.is_(None),
                        ProcessingJob.lease_expires_at <= now,
                    ),
                ).all()
            ]
        submitted = 0
        for job_id in job_ids:
            if self._stop_event.is_set():
                break
            with self._lock:
                current = self._futures.get(job_id)
                already_dispatched = current is not None and not current.done()
            if already_dispatched:
                continue
            if self._submit(job_id):
                submitted += 1
        return submitted

    def _run(self, job_id: str) -> None:
        if self._stop_event.is_set():
            return
        generation_id = self._claim(job_id)
        if generation_id is None:
            return
        if self._stop_event.is_set():
            return
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                return
            context = TraceContext(
                trace_id=f"processing-{job_id}",
                video_id=job.video_id,
                output_mode=job.requested_output_mode,
                processing_job_id=job_id,
                attempt=job.attempt_count,
            )
        with bind_trace(context):
            self._run_claimed(job_id, generation_id)

    def _run_claimed(self, job_id: str, generation_id: str) -> None:
        stop_heartbeat = threading.Event()
        heartbeat_lost = threading.Event()

        def should_continue() -> bool:
            if self._stop_event.is_set() or heartbeat_lost.is_set():
                return False
            try:
                return self._owns_generation(job_id, generation_id)
            except OperationalError as exc:
                heartbeat_lost.set()
                logger.warning(
                    "Processing ownership check could not reach the database",
                    extra={
                        "processing_job_id": job_id,
                        "generation_id": generation_id,
                        "stage": "processing",
                        "failure_classification": "database_unavailable",
                        "database_error_type": type(exc).__name__,
                    },
                )
                return False

        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, generation_id, stop_heartbeat, heartbeat_lost),
            name=f"processing-heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            cached_modes: Optional[list[str]] = None
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                if job is None:
                    return
                source_url = job.source_url
                video_id = job.video_id
                if inspect_video_cache(db, video_id).valid:
                    cached_modes = _requested_modes(db, job)
            if cached_modes is not None:
                self._require_generation(job_id, generation_id)
                self._schedule_modes(video_id, cached_modes)
                self._finish_success(job_id, generation_id)
                return

            processing_started = time.monotonic()
            trace_stage("video_processing", "start")
            orchestrator = self.orchestrator_factory()
            process = orchestrator.process_detection_only
            if "should_continue" in inspect.signature(process).parameters:
                result = process(
                    source_url,
                    should_continue=should_continue,
                )
            else:
                result = process(source_url)
            if heartbeat_lost.is_set():
                raise ProcessingCancelledError(job_id)
            self._require_generation(job_id, generation_id)
            trace_stage(
                "video_processing", "complete", started_at=processing_started,
                intervention_count=len(result.get("intervention_points", []))
                if isinstance(result, dict) else None,
            )
            if not isinstance(result, dict) or result.get("error"):
                trace_stage(
                    "video_processing", "failed", level=logging.ERROR,
                    status="failed", failure_type="pipeline_failed",
                )
                self._finish_failure(job_id, generation_id, "pipeline_failed")
                return

            if not self._set_stage(job_id, generation_id, "persisting"):
                logger.warning("Processing job %s lost its lease before persistence", job_id)
                return
            records = _build_interventions(video_id, result)
            persistence_started = time.monotonic()
            trace_stage(
                "intervention_persistence", "start",
                intervention_count=len(records),
            )
            with self.session_factory() as db:
                self._require_generation(job_id, generation_id)
                persist_processing_result(
                    db, video_id, source_url, result, records,
                    job_id=job_id, generation_id=generation_id,
                    lease_owner=self.owner,
                )
            trace_stage(
                "intervention_persistence", "committed",
                started_at=persistence_started,
                intervention_count=len(records),
            )
            _write_interventions_json(video_id, result, records)

            if not self._set_stage(job_id, generation_id, "preparing_explanations"):
                logger.warning("Processing job %s lost its lease after persistence", job_id)
                return
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                modes = _requested_modes(db, job) if job is not None else []
            self._schedule_modes(video_id, modes)
            self._finish_success(job_id, generation_id)
        except ProcessingCancelledError:
            trace_stage("video_cancellation", "late_result_discarded", status="cancelled")
            self._settle_cancelled(job_id)
        except OperationalError as exc:
            heartbeat_lost.set()
            logger.warning(
                "Processing stopped because ownership could not be verified",
                extra={
                    "processing_job_id": job_id,
                    "generation_id": generation_id,
                    "stage": "processing",
                    "failure_classification": "database_unavailable",
                    "database_error_type": type(exc).__name__,
                },
            )
        except ProcessingPersistenceError as exc:
            if self._stop_event.is_set() or heartbeat_lost.is_set():
                trace_stage(
                    "processing_shutdown", "late_result_discarded",
                    status="interrupted", failure_type=type(exc).__name__,
                )
                return
            trace_stage(
                "intervention_persistence", "failed", level=logging.ERROR,
                exc_info=(type(exc), exc, exc.__traceback__), status="failed",
                failure_type=type(exc).__name__,
            )
            self._finish_failure(job_id, generation_id, "persistence_failed")
        except Exception as exc:
            if self._stop_event.is_set() or heartbeat_lost.is_set():
                trace_stage(
                    "processing_shutdown", "late_result_discarded",
                    status="interrupted", failure_type=type(exc).__name__,
                )
                return
            trace_stage(
                "video_processing", "failed", level=logging.ERROR,
                exc_info=(type(exc), exc, exc.__traceback__), status="failed",
                failure_type=type(exc).__name__,
            )
            self._finish_failure(job_id, generation_id, "internal_error")
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=max(self.heartbeat_seconds * 2, 0.2))

    def _heartbeat(
        self,
        job_id: str,
        generation_id: str,
        stop: threading.Event,
        ownership_lost: threading.Event,
    ) -> None:
        while not stop.wait(self.heartbeat_seconds):
            if self._stop_event.is_set():
                return
            for attempt in range(1, self.heartbeat_retry_limit + 1):
                try:
                    now = _utcnow()
                    with self.session_factory() as db:
                        try:
                            renewed = db.query(ProcessingJob).filter(
                                ProcessingJob.id == job_id,
                                ProcessingJob.status == "running",
                                ProcessingJob.lease_owner == self.owner,
                                ProcessingJob.generation_id == generation_id,
                            ).update({
                                ProcessingJob.lease_expires_at: now + timedelta(
                                    seconds=self.lease_seconds
                                ),
                                ProcessingJob.updated_at: now,
                            }, synchronize_session=False)
                            db.commit()
                        except OperationalError:
                            db.rollback()
                            raise
                    if renewed != 1:
                        ownership_lost.set()
                        logger.warning(
                            "Processing heartbeat confirmed ownership loss",
                            extra={
                                "processing_job_id": job_id,
                                "generation_id": generation_id,
                                "stage": "heartbeat",
                                "failure_classification": "ownership_superseded",
                            },
                        )
                        return
                    break
                except OperationalError as exc:
                    logger.warning(
                        "Processing heartbeat database operation failed; retrying",
                        extra={
                            "processing_job_id": job_id,
                            "generation_id": generation_id,
                            "stage": "heartbeat",
                            "failure_classification": "transient_database",
                            "heartbeat_attempt": attempt,
                            "database_error_type": type(exc).__name__,
                        },
                    )
                    if attempt >= self.heartbeat_retry_limit:
                        ownership_lost.set()
                        logger.error(
                            "Processing heartbeat could not establish safe ownership",
                            extra={
                                "processing_job_id": job_id,
                                "generation_id": generation_id,
                                "stage": "heartbeat",
                                "failure_classification": "database_unavailable",
                            },
                        )
                        return
                    backoff = min(
                        self.heartbeat_retry_backoff_seconds * (2 ** (attempt - 1)),
                        max(self.heartbeat_seconds, self.heartbeat_retry_backoff_seconds),
                    )
                    if stop.wait(backoff) or self._stop_event.is_set():
                        return
                except Exception:
                    ownership_lost.set()
                    logger.exception(
                        "Unexpected processing heartbeat failure",
                        extra={
                            "processing_job_id": job_id,
                            "generation_id": generation_id,
                            "stage": "heartbeat",
                            "failure_classification": "unexpected",
                        },
                    )
                    raise

    def _schedule_modes(self, video_id: str, modes: list[str]) -> None:
        if self.preparation_manager is None or self._stop_event.is_set():
            return
        for output_mode in modes:
            try:
                self.preparation_manager.schedule_video(video_id, output_mode)
            except Exception:
                logger.exception(
                    "Could not schedule %s explanations for %s",
                    output_mode,
                    video_id,
                )

    def _set_stage(self, job_id: str, generation_id: str, stage: str) -> bool:
        with self.session_factory() as db:
            updated = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.status == "running",
                ProcessingJob.lease_owner == self.owner,
                ProcessingJob.generation_id == generation_id,
            ).update({ProcessingJob.stage: stage, ProcessingJob.updated_at: _utcnow()})
            db.commit()
            return updated == 1

    def _finish_success(self, job_id: str, generation_id: str) -> None:
        now = _utcnow()
        with self.session_factory() as db:
            job = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.lease_owner == self.owner,
                ProcessingJob.generation_id == generation_id,
                ProcessingJob.status == "running",
            ).one_or_none()
            if job is not None:
                unfinished = db.query(ExplanationPreparation.id).filter(
                    ExplanationPreparation.video_id == job.video_id,
                    ExplanationPreparation.status.in_(("pending", "generating")),
                ).first() is not None
                self._mark_succeeded(job, now, keep_active=unfinished)
                db.commit()

    def _finish_failure(self, job_id: str, generation_id: str, code: str) -> None:
        now = _utcnow()
        with self.session_factory() as db:
            job = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.lease_owner == self.owner,
                ProcessingJob.generation_id == generation_id,
                ProcessingJob.status == "running",
            ).one_or_none()
            if job is not None:
                job.status = "failed"
                job.stage = "failed"
                job.error_code = code
                job.error_message = SAFE_ERRORS[code]
                job.lease_owner = None
                job.lease_expires_at = None
                job.generation_id = None
                job.active_slot = None
                job.completed_at = now
                job.updated_at = now
                db.commit()

    @staticmethod
    def _mark_succeeded(
        job: ProcessingJob, now: datetime, *, keep_active: bool = False
    ) -> None:
        job.status = "succeeded"
        job.stage = "complete"
        job.error_code = None
        job.error_message = None
        job.lease_owner = None
        job.lease_expires_at = None
        job.generation_id = None
        job.active_slot = 1 if keep_active else None
        job.completed_at = now
        job.updated_at = now

    def _owns_generation(self, job_id: str, generation_id: str) -> bool:
        with self.session_factory() as db:
            return db.query(ProcessingJob.id).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.status == "running",
                ProcessingJob.lease_owner == self.owner,
                ProcessingJob.generation_id == generation_id,
            ).first() is not None

    def _require_generation(self, job_id: str, generation_id: str) -> None:
        if self._stop_event.is_set() or not self._owns_generation(job_id, generation_id):
            raise ProcessingCancelledError(job_id)

    def _release_completed_slots(self) -> None:
        """Release durable admission ownership after every preparation is terminal."""
        with self.session_factory() as db:
            jobs = db.query(ProcessingJob).filter(
                ProcessingJob.active_slot == 1,
                ProcessingJob.status == "succeeded",
            ).all()
            changed = False
            for job in jobs:
                unfinished = db.query(ExplanationPreparation.id).filter(
                    ExplanationPreparation.video_id == job.video_id,
                    ExplanationPreparation.status.in_(("pending", "generating")),
                ).first() is not None
                if not unfinished:
                    job.active_slot = None
                    job.updated_at = _utcnow()
                    changed = True
            if changed:
                db.commit()

    def _settle_cancelled(self, job_id: str) -> None:
        with self._lock:
            future = self._futures.get(job_id)
            processing_active = future is not None and not future.done()
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None or job.status != "cancelling":
                return
            preparation_active = (
                self.preparation_manager is not None
                and self.preparation_manager.has_inflight_video(job.video_id)
            )
            if processing_active or preparation_active:
                return
            self._mark_cancelled(job, _utcnow())
            db.commit()
            trace_stage("video_cancellation", "settled", video_id=job.video_id,
                        processing_job_id=job_id, status="cancelled")

    @staticmethod
    def _mark_cancelled(job: ProcessingJob, now: datetime) -> None:
        job.status = "cancelled"
        job.stage = "cancelled"
        job.error_code = None
        job.error_message = None
        job.lease_owner = None
        job.lease_expires_at = None
        job.generation_id = None
        job.active_slot = None
        job.cancelled_at = now
        job.completed_at = now
        job.updated_at = now

    @staticmethod
    def _lease_expired(job: ProcessingJob, now: datetime) -> bool:
        return job.status == "running" and (
            job.lease_expires_at is None or job.lease_expires_at <= now
        )
