"""Durable coordination and cache validation for the existing video pipeline."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
        db.commit()
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
        records.append(Intervention(
            id=intervention_id,
            video_id=video_id,
            timestamp=timestamp,
            timestamp_formatted=f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
            frame_path=value("frame_path", ""),
            content_type=value("content_type"),
            complexity_score=value("complexity_score"),
            clip_confidence=value("clip_confidence"),
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
        max_workers: int = 2,
        lease_seconds: float = 120.0,
        heartbeat_seconds: float = 30.0,
    ):
        self.session_factory = session_factory
        self.orchestrator_factory = orchestrator_factory
        self.preparation_manager = preparation_manager
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = min(heartbeat_seconds, max(lease_seconds / 3, 0.05))
        self.owner = str(uuid.uuid4())
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="processing-job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        queued = []
        cached_to_prepare: list[tuple[str, list[str]]] = []
        now = _utcnow()
        with self.session_factory() as db:
            jobs = db.query(ProcessingJob).filter(
                ProcessingJob.status.in_(("queued", "running"))
            ).all()
            for job in jobs:
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

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def create_or_join(self, youtube_url: str, output_mode: str) -> dict:
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError(f"Unsupported explanation mode: {output_mode}")
        orchestrator = self.orchestrator_factory()
        video_id = orchestrator._extract_video_id(youtube_url)
        now = _utcnow()
        should_submit = False
        joined = False
        reused = False

        with self.session_factory() as db:
            cache = inspect_video_cache(db, video_id)
            job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video_id).first()
            if cache.valid:
                reused = True
                if job is None:
                    job = ProcessingJob(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        source_url=youtube_url,
                        requested_output_mode=output_mode,
                    )
                    db.add(job)
                    try:
                        db.commit()
                    except IntegrityError:
                        db.rollback()
                        job = db.query(ProcessingJob).filter(
                            ProcessingJob.video_id == video_id
                        ).one()
                        joined = True
                self._mark_succeeded(job, now)
                db.commit()
                db.refresh(job)
                _record_requested_mode(db, str(job.id), output_mode)
                modes = _requested_modes(db, job)
                snapshot = serialize_job(
                    job,
                    requested_modes=modes,
                    joined_existing=joined,
                    reused_cache=True,
                )
                self._schedule_modes(video_id, modes)
                return snapshot

            if job is None:
                job = ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    source_url=youtube_url,
                    requested_output_mode=output_mode,
                    status="queued",
                    stage="queued",
                )
                db.add(job)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video_id).one()
                    joined = True
                else:
                    db.refresh(job)
                    should_submit = True
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
                    job.completed_at = None
                    job.updated_at = now
                    db.commit()
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
        return snapshot

    def retry(self, job_id: str) -> dict:
        job_id = _normalize_job_id(job_id)
        with self.session_factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None:
                raise ProcessingJobNotFoundError(job_id)
            source_url = job.source_url
            output_mode = job.requested_output_mode
        return self.create_or_join(source_url, output_mode)

    def get(self, job_id: str) -> dict:
        job_id = _normalize_job_id(job_id)
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
            if snapshot["status"] in ("succeeded", "failed"):
                return snapshot
            time.sleep(0.05)
        raise ProcessingJobTimeoutError(job_id)

    def _submit(self, job_id: str) -> None:
        with self._lock:
            current = self._futures.get(job_id)
            if current is not None and not current.done():
                return
            future = self._executor.submit(self._run, job_id)
            self._futures[job_id] = future
        # Register outside the lock: add_done_callback runs synchronously when
        # an exceptionally fast job has already completed.
        future.add_done_callback(lambda completed: self._forget(job_id, completed))

    def _forget(self, job_id: str, completed: Future) -> None:
        with self._lock:
            if self._futures.get(job_id) is completed:
                self._futures.pop(job_id, None)

    def _claim(self, job_id: str) -> bool:
        now = _utcnow()
        lease = now + timedelta(seconds=self.lease_seconds)
        with self.session_factory() as db:
            claimed = db.execute(
                update(ProcessingJob)
                .where(
                    ProcessingJob.id == job_id,
                    or_(
                        ProcessingJob.status == "queued",
                        (
                            (ProcessingJob.status == "running")
                            & (ProcessingJob.lease_expires_at < now)
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
                    started_at=now,
                    completed_at=None,
                    updated_at=now,
                )
            ).rowcount
            db.commit()
            return claimed == 1

    def _run(self, job_id: str) -> None:
        if not self._claim(job_id):
            return
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, stop_heartbeat),
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
                self._schedule_modes(video_id, cached_modes)
                self._finish_success(job_id)
                return

            result = self.orchestrator_factory().process_detection_only(source_url)
            if not isinstance(result, dict) or result.get("error"):
                logger.error("Video pipeline failed for %s: %s", video_id, result)
                self._finish_failure(job_id, "pipeline_failed")
                return

            if not self._set_stage(job_id, "persisting"):
                logger.warning("Processing job %s lost its lease before persistence", job_id)
                return
            records = _build_interventions(video_id, result)
            with self.session_factory() as db:
                persist_processing_result(db, video_id, source_url, result, records)
            _write_interventions_json(video_id, result, records)

            if not self._set_stage(job_id, "preparing_explanations"):
                logger.warning("Processing job %s lost its lease after persistence", job_id)
                return
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                modes = _requested_modes(db, job) if job is not None else []
            self._schedule_modes(video_id, modes)
            self._finish_success(job_id)
        except ProcessingPersistenceError:
            logger.exception("Could not persist processing result for job %s", job_id)
            self._finish_failure(job_id, "persistence_failed")
        except Exception:
            logger.exception("Unexpected processing failure for job %s", job_id)
            self._finish_failure(job_id, "internal_error")
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=max(self.heartbeat_seconds * 2, 0.2))

    def _heartbeat(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            now = _utcnow()
            with self.session_factory() as db:
                db.query(ProcessingJob).filter(
                    ProcessingJob.id == job_id,
                    ProcessingJob.status == "running",
                    ProcessingJob.lease_owner == self.owner,
                ).update({
                    ProcessingJob.lease_expires_at: now + timedelta(seconds=self.lease_seconds),
                    ProcessingJob.updated_at: now,
                }, synchronize_session=False)
                db.commit()

    def _schedule_modes(self, video_id: str, modes: list[str]) -> None:
        if self.preparation_manager is None:
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

    def _set_stage(self, job_id: str, stage: str) -> bool:
        with self.session_factory() as db:
            updated = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.status == "running",
                ProcessingJob.lease_owner == self.owner,
            ).update({ProcessingJob.stage: stage, ProcessingJob.updated_at: _utcnow()})
            db.commit()
            return updated == 1

    def _finish_success(self, job_id: str) -> None:
        now = _utcnow()
        with self.session_factory() as db:
            job = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.lease_owner == self.owner,
            ).one_or_none()
            if job is not None:
                self._mark_succeeded(job, now)
                db.commit()

    def _finish_failure(self, job_id: str, code: str) -> None:
        now = _utcnow()
        with self.session_factory() as db:
            job = db.query(ProcessingJob).filter(
                ProcessingJob.id == job_id,
                ProcessingJob.lease_owner == self.owner,
            ).one_or_none()
            if job is not None:
                job.status = "failed"
                job.stage = "failed"
                job.error_code = code
                job.error_message = SAFE_ERRORS[code]
                job.lease_owner = None
                job.lease_expires_at = None
                job.completed_at = now
                job.updated_at = now
                db.commit()

    @staticmethod
    def _mark_succeeded(job: ProcessingJob, now: datetime) -> None:
        job.status = "succeeded"
        job.stage = "complete"
        job.error_code = None
        job.error_message = None
        job.lease_owner = None
        job.lease_expires_at = None
        job.completed_at = now
        job.updated_at = now

    @staticmethod
    def _lease_expired(job: ProcessingJob, now: datetime) -> bool:
        return job.status == "running" and (
            job.lease_expires_at is None or job.lease_expires_at <= now
        )
