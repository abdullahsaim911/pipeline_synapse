"""Persistent two-stage explanation preparation for local Synapse.

The evidence worker runs VLM extraction one intervention at a time and hands
each validated result to the explanation worker, which runs reconciliation,
Mistral and TTS one at a time. Extraction of intervention N+1 therefore overlaps
explanation of intervention N, while a failure in one intervention leaves later
ones running. Durable status is unchanged; ``stage`` adds internal detail.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.explanation_service import (
    ExplanationArtifact,
    ExplanationTargetNotFoundError,
    find_valid_cached_explanation,
    generate_explanation_from_evidence,
    prepare_visual_evidence,
    PreparationCancelledError,
    InvalidPreparedArtifactError,
)
from api.models import (
    Explanation, ExplanationPreparation, Intervention, ProcessingJob, Video,
)
from api.processing_jobs import (
    _is_video_job_conflict,
    commit_active_slot,
    release_terminal_video_slot,
)
from stage_tracing import TraceContext, bind_trace, new_trace_id, trace_stage

logger = logging.getLogger(__name__)

VALID_OUTPUT_MODES = {"brief", "explanatory", "detailed"}
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_VLM_RETRY_SECONDS = 30.0
DEFAULT_VLM_MAX_WAIT_SECONDS = 900.0
# Evidence waiting for Mistral. A full queue pauses further VLM work.
EXPLANATION_QUEUE_CAPACITY = 4
PERMANENT_FAILURE_REASONS = {
    "vlm_authentication_failed",
    "vlm_remote_rejected",
    "evidence_reported_unavailable",
}

STAGE_PENDING = "pending_extraction"
STAGE_WAITING_FOR_VLM = "waiting_for_vlm"
STAGE_EXTRACTING = "extracting_evidence"
STAGE_EVIDENCE_READY = "evidence_ready"
STAGE_EXPLAINING = "generating_explanation"
STAGE_TTS = "generating_tts"
STAGE_READY = "ready"
STAGE_FAILED = "failed"


def _utcnow() -> datetime:
    """Naive UTC for compatibility with existing SQLite DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _positive_seconds(variable: str, default: float) -> float:
    raw = os.getenv(variable, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return value if value > 0 else default


class PreparationNotRetryableError(RuntimeError):
    """The requested key is not currently in a failed state."""


class PreparationRetryExhaustedError(RuntimeError):
    """The persistent retry limit has been reached."""

    def __init__(self, attempt_count: int):
        super().__init__("Explanation preparation retry limit reached")
        self.attempt_count = attempt_count


@dataclass(frozen=True)
class PreparationJob:
    video_id: str
    intervention_id: str
    output_mode: str
    trace_id: str = field(default_factory=new_trace_id, compare=False)
    enqueued_at: float = field(default=0.0, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.video_id, self.intervention_id, self.output_mode)


class _PreparationQueue(queue.Queue):
    """Stable priority at dispatch, including promotion of already queued work."""

    def __init__(self, priorities, maxsize=0, wait_for_priority=False, priority_order=None, deferred=None):
        self.priorities = priorities
        self.priority_order = priority_order if priority_order is not None else {}
        self.deferred = deferred if deferred is not None else set()
        self.wait_for_priority = wait_for_priority
        super().__init__(maxsize)

    def get(self, block=True, timeout=None):
        # Keep synthesis available for an explicit request whose evidence is
        # still preparing, rather than starting another background generation.
        if not self.wait_for_priority or not block:
            return super().get(block, timeout)
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self.not_empty:
            while True:
                foreground = any(item is not None and getattr(item, "job", item).key in self.priorities for item in self.queue)
                stopping = any(item is None for item in self.queue)
                runnable_priority = self.priorities - self.deferred
                if self._qsize() and (not runnable_priority or foreground or stopping):
                    item = self._get()
                    self.not_full.notify()
                    return item
                remaining = deadline - time.monotonic() if deadline is not None else 0.1
                if deadline is not None and remaining <= 0:
                    raise queue.Empty
                self.not_empty.wait(min(remaining, 0.1))

    def displace_background(self):
        """Move one ready background checkpoint out to make room for a request."""
        with self.mutex:
            for item in reversed(self.queue):
                if item is not None and item.job.key not in self.priorities:
                    self.queue.remove(item)
                    self.unfinished_tasks -= 1
                    if not self.unfinished_tasks:
                        self.all_tasks_done.notify_all()
                    self.not_full.notify()
                    return item
        return None

    def _get(self):
        candidates = [item for item in self.queue if item is not None
            and getattr(item, "job", item).key in self.priorities]
        if candidates:
            item = min(candidates, key=lambda entry: self.priority_order.get(
                getattr(entry, "job", entry).key, datetime.max))
            self.queue.remove(item)
            return item
        return super()._get()


@dataclass(frozen=True)
class _ExplanationWork:
    job: PreparationJob
    lease: "_Lease"
    snapshot: Optional[dict]


@dataclass(frozen=True)
class _Lease:
    owner: str
    preparation_job_id: str
    attempt_count: int


@dataclass
class _TrackedLease:
    job: PreparationJob
    lease: _Lease
    last_renewed: float
    lost: bool = False


@dataclass(frozen=True)
class PreparationSnapshot:
    intervention_id: str
    timestamp: float
    status: str
    error_code: Optional[str]
    attempt_count: int
    requested: bool = True
    retryable: bool = False


@dataclass(frozen=True)
class RetryResult:
    status: str
    error_code: Optional[str]
    attempt_count: int
    queued: bool = False


class ExplanationPreparationManager:
    """A durable SQLite-backed queue with one consumer per model stage."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        orchestrator_factory: Callable[[], object],
        data_dir: str,
        generator: Optional[Callable] = None,
        evidence_preparer: Optional[Callable] = None,
        before_inference: Optional[Callable[[], bool]] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lease_timeout_seconds: float = 1800.0,
        maintenance_interval_seconds: Optional[float] = None,
        shutdown_grace_seconds: Optional[float] = None,
        vlm_retry_seconds: Optional[float] = None,
        vlm_max_wait_seconds: Optional[float] = None,
    ) -> None:
        self._session_factory = session_factory
        self._orchestrator_factory = orchestrator_factory
        self._data_dir = data_dir
        # A caller-supplied generator alone replaces the whole explanation job.
        if evidence_preparer is None and generator is None:
            evidence_preparer = prepare_visual_evidence
        self._evidence_preparer = evidence_preparer
        self._generator = generator or generate_explanation_from_evidence
        self._before_inference = before_inference
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be positive")
        default_maintenance_interval = min(30.0, lease_timeout_seconds / 3.0)
        maintenance_interval = (
            maintenance_interval_seconds
            if maintenance_interval_seconds is not None
            else default_maintenance_interval
        )
        if maintenance_interval <= 0 or maintenance_interval >= lease_timeout_seconds:
            raise ValueError(
                "maintenance_interval_seconds must be positive and shorter than the lease timeout"
            )
        self._max_attempts = max_attempts
        self._lease_timeout = timedelta(seconds=lease_timeout_seconds)
        self._lease_timeout_seconds = lease_timeout_seconds
        self._maintenance_interval = maintenance_interval
        self._vlm_retry_seconds = (
            _positive_seconds(
                "SYNAPSE_VLM_DEFER_RETRY_SECONDS", DEFAULT_VLM_RETRY_SECONDS
            )
            if vlm_retry_seconds is None else vlm_retry_seconds
        )
        self._vlm_max_wait_seconds = (
            _positive_seconds(
                "SYNAPSE_VLM_MAX_WAIT_SECONDS", DEFAULT_VLM_MAX_WAIT_SECONDS
            )
            if vlm_max_wait_seconds is None else vlm_max_wait_seconds
        )
        if self._vlm_retry_seconds <= 0:
            raise ValueError("vlm_retry_seconds must be positive")
        if self._vlm_max_wait_seconds < self._vlm_retry_seconds:
            raise ValueError("vlm_max_wait_seconds must not be shorter than retry delay")
        shutdown_grace = (
            float(os.getenv("PREPARATION_SHUTDOWN_GRACE_SECONDS", "5"))
            if shutdown_grace_seconds is None else shutdown_grace_seconds
        )
        if shutdown_grace < 0:
            raise ValueError("shutdown_grace_seconds cannot be negative")
        self._shutdown_grace_seconds = shutdown_grace
        self._priority_keys: set[tuple[str, str, str]] = set()
        self._priority_order = {}
        self._deferred_keys = set()
        self._queue = _PreparationQueue(self._priority_keys, priority_order=self._priority_order)
        self._explanation_queue = _PreparationQueue(
            self._priority_keys, maxsize=EXPLANATION_QUEUE_CAPACITY, wait_for_priority=True,
            priority_order=self._priority_order, deferred=self._deferred_keys,
        )
        self._scheduled: set[tuple[str, str, str]] = set()
        self._active_leases: set[tuple[str, str]] = set()
        self._tracked_leases: dict[str, _TrackedLease] = {}
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._dispatch_ready = threading.Event()
        self._shutdown_started = False
        self._thread: Optional[threading.Thread] = None
        self._explanation_thread: Optional[threading.Thread] = None
        self._maintenance_thread: Optional[threading.Thread] = None
        self._worker_id = ""
        self._provider_retry_after = 0.0
        self._provider_probe_key: Optional[tuple[str, str, str]] = None
        # Recent stage transitions: (monotonic time, intervention, mode, stage).
        self.timeline: deque[tuple[float, str, str, str]] = deque(maxlen=1000)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def start(self, before_dispatch=None) -> None:
        with self._state_lock:
            if self._shutdown_started:
                raise RuntimeError("Explanation preparation manager has been shut down")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._dispatch_ready.clear()
            self._worker_id = str(uuid.uuid4())
            thread = threading.Thread(
                target=self._run_evidence_worker,
                name="synapse-evidence-preparation",
                daemon=True,
            )
            explanation_thread = threading.Thread(
                target=self._run_explanation_worker,
                name="synapse-explanation-preparation",
                daemon=True,
            )
            maintenance_thread = threading.Thread(
                target=self._run_lease_maintenance,
                name="synapse-explanation-lease-maintenance",
                daemon=True,
            )
            self._thread = thread
            self._explanation_thread = explanation_thread
            self._maintenance_thread = maintenance_thread
        try:
            self._recover_tracked_jobs()
            thread.start()
            explanation_thread.start()
            if before_dispatch is not None:
                before_dispatch()
            self._dispatch_ready.set()
            maintenance_thread.start()
        except Exception:
            self._stop_event.set()
            self._dispatch_ready.set()
            with self._state_lock:
                self._thread = None
                self._explanation_thread = None
                self._maintenance_thread = None
            raise

    def schedule_video(self, video_id: str, output_mode: str) -> int:
        """Persist and queue missing work in educational timestamp order."""
        self._require_running()
        self._validate_mode(output_mode)
        to_queue: list[PreparationJob] = []
        now = _utcnow()

        with self._session_factory() as db:
            video_job = db.query(ProcessingJob.status).filter(
                ProcessingJob.video_id == video_id
            ).first()
            if video_job is not None and video_job[0] in {"cancelling", "cancelled"}:
                return 0
            intervention_rows = db.query(Intervention).filter(
                Intervention.video_id == video_id,
                Intervention.origin == "detected",
            ).order_by(
                Intervention.timestamp.asc(),
                Intervention.id.asc(),
            ).all()
            for intervention in intervention_rows:
                job = PreparationJob(video_id, intervention.id, output_mode)
                state = self._get_state(db, job)
                cached = find_valid_cached_explanation(
                    db, intervention.id, output_mode, self._data_dir
                )
                if cached is not None:
                    if state is None or state.status != "generating":
                        self._set_ready_state(db, job, state, now)
                    continue

                if state is None:
                    state = self._new_state(job, status="pending")
                    db.add(state)
                elif state.status == "ready":
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="missing_audio"
                    )
                if state.status == "pending" and state.stage != STAGE_WAITING_FOR_VLM:
                    to_queue.append(job)
            db.commit()

        queued = sum(1 for job in to_queue if self._enqueue(job))
        self._recover_stale_preparations(now)
        return queued

    def cancel_video(self, video_id: str) -> int:
        """Durably fence all unfinished preparation attempts for a video."""
        now = _utcnow()
        was_inflight = self.has_inflight_video(video_id)
        with self._session_factory() as db:
            changed = db.query(ExplanationPreparation).filter(
                ExplanationPreparation.video_id == video_id,
                ExplanationPreparation.status.in_(("pending", "generating")),
            ).update({
                ExplanationPreparation.status: "cancelled",
                ExplanationPreparation.priority_at: None,
                ExplanationPreparation.stage: "cancelled",
                ExplanationPreparation.error_code: None,
                ExplanationPreparation.failure_reason: "video_cancelled",
                ExplanationPreparation.lease_owner: None,
                ExplanationPreparation.lease_started_at: None,
                ExplanationPreparation.provider_wait_started_at: None,
                ExplanationPreparation.deferred_until: None,
                ExplanationPreparation.updated_at: now,
            }, synchronize_session=False)
            db.commit()
        release_terminal_video_slot(self._session_factory, video_id)
        self._invalidate_video_leases(video_id)
        with self._state_lock:
            for key in list(self._priority_keys):
                if key[0] == video_id:
                    self._forget_priority(key)
        trace_stage("video_cancellation", "preparations_fenced", video_id=video_id,
                    preparation_count=changed,
                    local_abort="unsupported_in_flight" if was_inflight else "not_running")
        return changed

    def resume_video(self, video_id: str, output_modes: list[str]) -> int:
        """Make only cancelled, incomplete checkpoints eligible for a new lease."""
        now = _utcnow()
        with self._session_factory() as db:
            changed = db.query(ExplanationPreparation).filter(
                ExplanationPreparation.video_id == video_id,
                ExplanationPreparation.output_mode.in_(output_modes),
                ExplanationPreparation.status == "cancelled",
            ).update({
                ExplanationPreparation.status: "pending",
                ExplanationPreparation.stage: STAGE_PENDING,
                ExplanationPreparation.error_code: None,
                ExplanationPreparation.failure_reason: None,
                ExplanationPreparation.lease_owner: None,
                ExplanationPreparation.lease_started_at: None,
                ExplanationPreparation.provider_wait_started_at: None,
                ExplanationPreparation.deferred_until: None,
                ExplanationPreparation.updated_at: now,
            }, synchronize_session=False)
            db.commit()
        trace_stage("video_resume", "checkpoints_reused", video_id=video_id,
                    resumed_preparation_count=changed)
        return changed

    def has_inflight_video(self, video_id: str) -> bool:
        with self._state_lock:
            return any(active_video == video_id for active_video, _ in self._active_leases)

    def get_status(self, video_id: str, output_mode: str) -> list[PreparationSnapshot]:
        """Return effective ordered status without mutating persistent state."""
        self._validate_mode(output_mode)
        with self._session_factory() as db:
            if db.query(Video.id).filter(Video.id == video_id).first() is None:
                raise ExplanationTargetNotFoundError("Video not found")
            interventions = db.query(Intervention).filter(
                Intervention.video_id == video_id,
                Intervention.origin == "detected",
            ).order_by(
                Intervention.timestamp.asc(),
                Intervention.id.asc(),
            ).all()
            snapshots = []
            for intervention in interventions:
                job = PreparationJob(video_id, intervention.id, output_mode)
                state = self._get_state(db, job)
                has_cache_row = False
                cached = find_valid_cached_explanation(
                    db, intervention.id, output_mode, self._data_dir
                )
                if cached is not None:
                    status, error_code = "ready", None
                elif state is not None:
                    status, error_code = state.status, state.error_code
                    if status == "ready":
                        status, error_code = "failed", "missing_audio"
                else:
                    has_cache_row = db.query(Explanation.id).filter(
                        Explanation.intervention_id == intervention.id,
                        Explanation.output_mode == output_mode,
                    ).first() is not None
                    status = "failed" if has_cache_row else "pending"
                    error_code = "missing_audio" if has_cache_row else None
                snapshots.append(PreparationSnapshot(
                    intervention_id=intervention.id,
                    timestamp=intervention.timestamp,
                    status=status,
                    error_code=error_code,
                    attempt_count=state.attempt_count if state is not None else 0,
                    requested=state is not None or cached is not None or has_cache_row,
                    retryable=status == "failed" and (state is None or (
                        state.attempt_count < self._max_attempts and state.failure_reason not in PERMANENT_FAILURE_REASONS
                    )),
                ))
            return snapshots

    def request(self, video_id: str, intervention_id: str, output_mode: str, *, requested_at=None) -> RetryResult:
        """Admit one explicit request through the existing durable worker pipeline."""
        self._require_running()
        self._validate_mode(output_mode)
        job = PreparationJob(video_id, intervention_id, output_mode)
        with self._session_factory() as db:
            self._load_intervention(db, job)
            cached = find_valid_cached_explanation(
                db, intervention_id, output_mode, self._data_dir
            )
            state = self._get_state(db, job)
            if cached is not None:
                return RetryResult("ready", None, state.attempt_count if state else 0)
            if state is not None and state.status == "failed":
                return RetryResult(state.status, state.error_code, state.attempt_count)
            if state is not None and state.status == "ready":
                return RetryResult("failed", "missing_audio", state.attempt_count)
        admitted = self._admit_manual(job, restart_cancelled=True)
        if not admitted:
            return RetryResult("ready", None, 0)
        self._promote(job, requested_at)
        queued = self._enqueue(job)
        with self._session_factory() as db:
            state = self._get_state(db, job)
            return RetryResult(state.status, state.error_code, state.attempt_count, queued=queued)

    def retry(self, video_id: str, intervention_id: str, output_mode: str) -> RetryResult:
        result = self._retry(video_id, intervention_id, output_mode)
        if result.status in {"pending", "generating"}:
            job = PreparationJob(video_id, intervention_id, output_mode)
            self._promote(job)
            self._enqueue(job)
        return result

    def _forget_priority(self, key):
        """Forget terminal dispatch metadata while the state lock is held."""
        self._priority_keys.discard(key)
        self._priority_order.pop(key, None)
        self._deferred_keys.discard(key)

    def _promote(self, job, requested_at=None):
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if state is None or state.status not in {"pending", "generating"}:
                return
            state.priority_at = state.priority_at or requested_at or _utcnow()
            requested_at = state.priority_at
            deferred = state.stage == STAGE_WAITING_FOR_VLM
            db.commit()
        with self._state_lock:
            self._priority_order[job.key] = requested_at
            self._priority_keys.add(job.key)
            if deferred:
                self._deferred_keys.add(job.key)

    def _retry(self, video_id: str, intervention_id: str, output_mode: str) -> RetryResult:
        """Requeue exactly one failed key without waiting for generation."""
        self._require_running()
        self._validate_mode(output_mode)
        job = PreparationJob(video_id, intervention_id, output_mode)
        now = _utcnow()
        with self._session_factory() as db:
            intervention = self._load_intervention(db, job)
            cached = find_valid_cached_explanation(
                db, intervention.id, output_mode, self._data_dir
            )
            state = self._get_state(db, job)
            if cached is not None:
                state = self._set_ready_state(db, job, state, now)
                db.commit()
                return RetryResult("ready", None, state.attempt_count)

            has_cache_row = db.query(Explanation.id).filter(
                Explanation.intervention_id == intervention_id,
                Explanation.output_mode == output_mode,
            ).first() is not None
            if state is None:
                if not has_cache_row:
                    raise PreparationNotRetryableError(
                        "Explanation preparation has not failed"
                    )
                state = self._new_state(
                    job, status="failed", error_code="missing_audio"
                )
                db.add(state)
            elif state.status == "ready":
                state.status = "failed"
                state.stage = STAGE_FAILED
                state.error_code = "missing_audio"
                state.failed_at = now

            if state.status in {"pending", "generating"}:
                result = RetryResult(
                    state.status, state.error_code, state.attempt_count
                )
                db.commit()
                return result
            if state.status != "failed":
                raise PreparationNotRetryableError(
                    "Explanation preparation has not failed"
                )
            if state.failure_reason in PERMANENT_FAILURE_REASONS:
                raise PreparationNotRetryableError(
                    "Explanation preparation failed permanently"
                )
            if state.attempt_count >= self._max_attempts:
                state.error_code = "retry_exhausted"
                state.updated_at = now
                db.commit()
                raise PreparationRetryExhaustedError(state.attempt_count)

            video_job = db.query(ProcessingJob).filter(
                ProcessingJob.video_id == video_id
            ).one_or_none()
            if video_job is not None and video_job.status in {
                "cancelling", "cancelled"
            }:
                raise PreparationCancelledError("Video preparation is cancelled")
            if video_job is None:
                video = db.get(Video, video_id)
                video_job = ProcessingJob(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    source_url=video.source_url,
                    requested_output_mode=output_mode,
                    status="succeeded",
                    stage="complete",
                    active_slot=1,
                    completed_at=now,
                    updated_at=now,
                )
                db.add(video_job)
            else:
                video_job.active_slot = 1
                video_job.updated_at = now

            state.status = "pending"
            state.stage = STAGE_PENDING
            state.error_code = None
            state.failure_reason = None
            state.lease_owner = None
            state.lease_started_at = None
            state.provider_wait_started_at = None
            state.deferred_until = None
            state.failed_at = None
            state.updated_at = now
            try:
                committed = commit_active_slot(db, video_id)
            except IntegrityError as exc:
                if not _is_video_job_conflict(exc):
                    raise
                state = self._get_state(db, job)
                if state is None or state.status not in {"pending", "generating"}:
                    raise
                return RetryResult(
                    state.status, state.error_code, state.attempt_count
                )
            if not committed:
                # A same-video request inserted the legacy coordination row.
                # The failed transaction rolled back our state transition, so
                # reapply it while joining that durable owner.
                state = self._get_state(db, job)
                if state is None:
                    raise RuntimeError("Explanation retry state disappeared")
                if state.status in {"pending", "generating"}:
                    return RetryResult(
                        state.status, state.error_code, state.attempt_count
                    )
                if state.status != "failed":
                    raise PreparationNotRetryableError(
                        "Explanation preparation has not failed"
                    )
                video_job = db.query(ProcessingJob).filter(
                    ProcessingJob.video_id == video_id
                ).one()
                video_job.active_slot = 1
                video_job.updated_at = now
                state.status = "pending"
                state.stage = STAGE_PENDING
                state.error_code = None
                state.failure_reason = None
                state.lease_owner = None
                state.lease_started_at = None
                state.provider_wait_started_at = None
                state.deferred_until = None
                state.failed_at = None
                state.updated_at = now
                commit_active_slot(db, video_id)
            attempts = state.attempt_count

        queued = self._enqueue(job)
        return RetryResult("pending", None, attempts, queued=queued)

    def generate_manual(
        self, video_id: str, intervention_id: str, output_mode: str
    ) -> ExplanationArtifact:
        """Coordinate the compatible manual endpoint with persistent state."""
        self._require_running()
        self._validate_mode(output_mode)
        job = PreparationJob(video_id, intervention_id, output_mode)
        owns_slot = self._admit_manual(job)
        try:
            lease = self._begin_manual(job)
            if lease is None:
                return self._wait_for_existing(job)
            self._track_lease(job, lease)
            try:
                with bind_trace(self._trace_context(job, lease)):
                    self._activate(job, lease)
                    try:
                        snapshot = self._prepare_evidence(job, lease)
                        result = self._generate(job, lease, snapshot)
                        self._mark_ready(job, lease)
                    finally:
                        self._deactivate(job, lease)
                    return result
            except Exception as exc:
                if not (
                    self._is_retryable_vlm_unavailable(exc)
                    and self._defer_for_vlm(job, lease.owner)
                ):
                    self._mark_failed(job, exc, lease.owner)
                raise
            finally:
                self._untrack_lease(lease)
        finally:
            if owns_slot:
                release_terminal_video_slot(self._session_factory, video_id)

    def _admit_manual(self, job: PreparationJob, *, restart_cancelled: bool = False) -> bool:
        """Create durable preparation state and acquire the video slot atomically."""
        now = _utcnow()
        for _attempt in range(3):
            with self._session_factory() as db:
                self._load_intervention(db, job)
                state = self._get_state(db, job)
                cached = find_valid_cached_explanation(
                    db, job.intervention_id, job.output_mode, self._data_dir
                )
                if cached is not None:
                    if state is None or state.status != "generating":
                        self._set_ready_state(db, job, state, now)
                    db.commit()
                    return False

                video_job = db.query(ProcessingJob).filter(
                    ProcessingJob.video_id == job.video_id
                ).one_or_none()
                if video_job is not None and video_job.status in {
                    "cancelling", "cancelled"
                }:
                    raise PreparationCancelledError("Video preparation is cancelled")
                video = db.get(Video, job.video_id) if video_job is None else None
                if state is None:
                    db.add(self._new_state(job, status="pending"))
                elif restart_cancelled and state.status == "cancelled":
                    # Explicit request after reopening; the old attempt remains fenced.
                    state.status = "pending"
                    state.stage = STAGE_PENDING
                    state.error_code = None
                    state.failure_reason = None
                    state.updated_at = now

                if video_job is None:
                    video_job = ProcessingJob(
                        id=str(uuid.uuid4()),
                        video_id=job.video_id,
                        source_url=video.source_url,
                        requested_output_mode=job.output_mode,
                        status="succeeded",
                        stage="complete",
                        active_slot=1,
                        completed_at=now,
                        updated_at=now,
                    )
                    db.add(video_job)
                else:
                    video_job.active_slot = 1
                    video_job.updated_at = now

                try:
                    committed = commit_active_slot(db, job.video_id)
                except IntegrityError as exc:
                    message = str(getattr(exc, "orig", exc)).lower()
                    constraint = getattr(
                        getattr(getattr(exc, "orig", None), "diag", None),
                        "constraint_name",
                        None,
                    )
                    same_key_race = constraint in {
                        "uq_processing_jobs_video_id",
                        "uq_explanation_preparation_key",
                    } or any(column in message for column in (
                        "processing_jobs.video_id",
                        "explanation_preparations.video_id, "
                        "explanation_preparations.intervention_id, "
                        "explanation_preparations.output_mode",
                    ))
                    if not same_key_race:
                        raise
                    continue
                if committed:
                    return True
                # Another request inserted the same video's durable job first.
        raise RuntimeError("Could not join concurrent manual explanation admission")

    def begin_shutdown(self) -> bool:
        """Fence leases and stop dispatch once; joining is handled separately."""
        with self._state_lock:
            if self._shutdown_started:
                return False
            self._shutdown_started = True
            self._stop_event.set()
            for tracked in self._tracked_leases.values():
                tracked.lost = True
        for pending_queue in (self._queue, self._explanation_queue):
            while True:
                try:
                    queued = pending_queue.get_nowait()
                except queue.Empty:
                    break
                if queued is not None:
                    job = queued.job if isinstance(queued, _ExplanationWork) else queued
                    if isinstance(queued, _ExplanationWork):
                        self._untrack_lease(queued.lease)
                    with self._state_lock:
                        self._scheduled.discard(job.key)
                pending_queue.task_done()
        self._queue.put(None)
        self._explanation_queue.put(None)
        return True

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        """Stop accepting work and join workers only within the configured grace."""
        self.begin_shutdown()
        grace = self._shutdown_grace_seconds if timeout is None else timeout
        if grace < 0:
            raise ValueError("shutdown timeout cannot be negative")
        deadline = time.monotonic() + grace
        alive = []
        for thread in (
            self._maintenance_thread,
            self._thread,
            self._explanation_thread,
        ):
            if thread is not None and thread.ident is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    alive.append(thread.name)
                    logger.warning(
                        "%s did not stop within the preparation shutdown grace",
                        thread.name,
                    )
        with self._state_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if (
                self._explanation_thread is not None
                and not self._explanation_thread.is_alive()
            ):
                self._explanation_thread = None
            if (
                self._maintenance_thread is not None
                and not self._maintenance_thread.is_alive()
            ):
                self._maintenance_thread = None
        return not alive

    def _run_evidence_worker(self) -> None:
        self._dispatch_ready.wait()
        if self._stop_event.is_set():
            return
        while True:
            entry = self._queue.get()
            if entry is None:
                self._queue.task_done()
                break
            parked = entry if isinstance(entry, _ExplanationWork) else None
            job = parked.job if parked is not None else entry
            lease = None
            handed_off = False
            try:
                if parked is not None:
                    lease = parked.lease
                    if self._holds_lease(job, lease.owner):
                        handed_off = self._hand_off(parked)
                    continue
                lease = self._claim(job)
                if lease is not None:
                    self._track_lease(job, lease)
                    with bind_trace(self._trace_context(job, lease)):
                        trace_stage(
                            "vlm_queue", "dequeued",
                            started_at=job.enqueued_at or None,
                        )
                        self._activate(job, lease)
                        try:
                            snapshot = self._prepare_evidence(job, lease)
                        finally:
                            self._deactivate(job, lease)
                        handed_off = self._hand_off(
                            _ExplanationWork(job, lease, snapshot)
                        )
            except Exception as exc:
                self._finish_failed(job, lease, exc)
            finally:
                # A handed-off key stays scheduled until explanation finishes.
                if not handed_off:
                    if lease is not None:
                        self._untrack_lease(lease)
                    with self._state_lock:
                        self._scheduled.discard(job.key)
                    self._reschedule_if_pending(job)
                self._queue.task_done()

    def _run_explanation_worker(self) -> None:
        self._dispatch_ready.wait()
        if self._stop_event.is_set():
            return
        while True:
            work = self._explanation_queue.get()
            if work is None:
                self._explanation_queue.task_done()
                break
            job = work.job
            try:
                with bind_trace(self._trace_context(job, work.lease)):
                    trace_stage("explanation_queue", "dequeued")
                    if self._holds_lease(job, work.lease.owner):
                        self._activate(job, work.lease)
                        try:
                            self._generate(job, work.lease, work.snapshot)
                            self._mark_ready(job, work.lease)
                        finally:
                            self._deactivate(job, work.lease)
            except Exception as exc:
                self._finish_failed(job, work.lease, exc)
            finally:
                self._untrack_lease(work.lease)
                with self._state_lock:
                    self._scheduled.discard(job.key)
                self._reschedule_if_pending(job)
                self._explanation_queue.task_done()

    def _finish_failed(
        self, job: PreparationJob, lease: Optional[_Lease], exc: Exception
    ) -> None:
        # Publish a failed state only after this active queue key is no longer
        # considered scheduled. An immediate explicit retry can then enqueue
        # replacement work instead of being stranded by the old key during this
        # worker's final cleanup.
        with self._state_lock:
            self._scheduled.discard(job.key)
        context = (
            self._trace_context(job, lease) if lease is not None else
            TraceContext(
                trace_id=job.trace_id, video_id=job.video_id,
                intervention_id=job.intervention_id, output_mode=job.output_mode,
            )
        )
        if isinstance(exc, PreparationCancelledError):
            trace_stage(
                "video_cancellation", "late_stage_result_discarded",
                context=context, status="cancelled",
                failure_type=type(exc).__name__,
            )
            return
        if lease is not None and not self._local_lease_is_healthy(lease.owner):
            trace_stage(
                "preparation_state", "stale_result_discarded",
                context=context, status="superseded",
                failure_type=type(exc).__name__,
            )
            return
        if (
            lease is not None
            and self._is_retryable_vlm_unavailable(exc)
            and self._defer_for_vlm(job, lease.owner)
        ):
            trace_stage(
                "preparation_state", "deferred", context=context,
                status="pending", failure_reason="vlm_remote_unavailable",
            )
            return
        if lease is not None:
            self._mark_failed(job, exc, lease.owner)
        trace_stage(
            "preparation_state", "failed", context=context, level=logging.ERROR,
            exc_info=(type(exc), exc, exc.__traceback__),
            status="failed", failure_type=type(exc).__name__,
        )

    def _hand_off(self, work: _ExplanationWork) -> bool:
        """Queue evidence for explanation, pausing while the queue is full."""
        while not self._stop_event.is_set():
            try:
                self._explanation_queue.put(work, timeout=0.1)
                trace_stage("explanation_queue", "entered")
                return True
            except queue.Full:
                if work.job.key in self._priority_keys:
                    displaced = self._explanation_queue.displace_background()
                    if displaced is not None:
                        self._queue.put(displaced)
                else:
                    with self._queue.mutex:
                        foreground_waiting = any(entry is not None and getattr(entry, "job", entry).key in self._priority_keys for entry in self._queue.queue)
                    if foreground_waiting:
                        # Retain the prepared checkpoint and lease; let the
                        # evidence worker dispatch the explicit request next.
                        self._queue.put(work)
                        return True
                continue
        return False

    def _prepare_evidence(
        self, job: PreparationJob, lease: _Lease
    ) -> Optional[dict]:
        """Return in-memory evidence once it is saved and valid."""
        if self._evidence_preparer is None:
            return None
        self._require_lease(job, lease.owner)
        self._set_stage(job, lease.owner, STAGE_EXTRACTING)
        readiness_check = (
            (lambda: self._check_inference_readiness(job))
            if self._before_inference is not None else None
        )
        kwargs = dict(
            session_factory=self._session_factory,
            orchestrator_factory=self._orchestrator_factory,
            data_dir=self._data_dir,
            video_id=job.video_id,
            intervention_id=job.intervention_id,
            output_mode=job.output_mode,
            before_inference=readiness_check,
        )
        if self._evidence_preparer is prepare_visual_evidence:
            kwargs["ownership_check"] = lambda: self._require_lease(job, lease.owner)
            kwargs["checkpoint_commit"] = lambda action: self._commit_checkpoint(
                job, lease, action
            )
        evidence = self._evidence_preparer(**kwargs)
        self._require_lease(job, lease.owner)
        self._set_stage(job, lease.owner, STAGE_EVIDENCE_READY)
        return evidence.snapshot if evidence is not None else None

    def _check_inference_readiness(self, job: PreparationJob) -> bool:
        with self._state_lock:
            retry_after = self._provider_retry_after
            probe_key = self._provider_probe_key
            if job.key in self._priority_keys and probe_key not in self._priority_keys:
                self._provider_probe_key = job.key
                probe_key = job.key
        if probe_key is not None and probe_key != job.key:
            return False
        if time.monotonic() < retry_after:
            return False
        ready = bool(self._before_inference and self._before_inference())
        if ready:
            with self._state_lock:
                self._provider_retry_after = 0.0
                self._provider_probe_key = None
        return ready

    def _generate(
        self,
        job: PreparationJob,
        lease: _Lease,
        snapshot: Optional[dict],
    ) -> ExplanationArtifact:
        self._require_lease(job, lease.owner)
        self._set_stage(job, lease.owner, STAGE_EXPLAINING)
        kwargs = dict(
            session_factory=self._session_factory,
            orchestrator_factory=self._orchestrator_factory,
            data_dir=self._data_dir,
            video_id=job.video_id,
            intervention_id=job.intervention_id,
            output_mode=job.output_mode,
            vlm_snapshot=snapshot,
            on_stage=lambda stage: self._advance_stage(job, lease.owner, stage),
        )
        if self._generator is generate_explanation_from_evidence:
            kwargs["ownership_check"] = lambda: self._require_lease(job, lease.owner)
            kwargs["checkpoint_commit"] = lambda action: self._commit_checkpoint(
                job, lease, action
            )
            kwargs["preparation_lease"] = (
                int(lease.preparation_job_id), lease.owner, lease.attempt_count
            )
        result = self._generator(**kwargs)
        self._require_lease(job, lease.owner)
        return result

    def _set_stage(
        self, job: PreparationJob, lease_owner: Optional[str], stage: str
    ) -> None:
        """Record progress and renew the lease. Progress never fails a job."""
        self.timeline.append(
            (time.monotonic(), job.intervention_id, job.output_mode, stage)
        )
        logger.debug(
            "Preparation stage: video=%s intervention=%s mode=%s stage=%s",
            job.video_id, job.intervention_id, job.output_mode, stage,
        )
        trace_stage("preparation_state", "transition", preparation_stage=stage)
        if lease_owner is None:
            return
        now = _utcnow()
        try:
            with self._session_factory() as db:
                state = self._get_state(db, job)
                if (
                    state is not None and state.status == "generating"
                    and state.lease_owner == lease_owner
                ):
                    state.stage = stage
                    if stage == STAGE_EVIDENCE_READY:
                        state.provider_wait_started_at = None
                        state.deferred_until = None
                    state.lease_started_at = now
                    state.updated_at = now
                    db.commit()
        except Exception:
            logger.exception("Could not record preparation stage %s", stage)

    def _holds_lease(self, job: PreparationJob, lease_owner: str) -> bool:
        if not self._local_lease_is_healthy(lease_owner):
            return False
        with self._session_factory() as db:
            state = self._get_state(db, job)
            return (
                state is not None
                and state.status == "generating"
                and state.lease_owner == lease_owner
            )

    def _require_lease(self, job: PreparationJob, lease_owner: str) -> None:
        if not self._local_lease_is_healthy(lease_owner):
            raise PreparationCancelledError(
                "Preparation lease heartbeat was lost or stopped"
            )
        if not self._holds_lease(job, lease_owner):
            raise PreparationCancelledError("Preparation lease was cancelled or superseded")

    def _advance_stage(self, job: PreparationJob, lease_owner: str, stage: str) -> None:
        self._require_lease(job, lease_owner)
        self._set_stage(job, lease_owner, stage)

    def _activate(self, job: PreparationJob, lease: _Lease) -> None:
        with self._state_lock:
            self._deferred_keys.discard(job.key)
            self._active_leases.add((job.video_id, lease.owner))

    def _deactivate(self, job: PreparationJob, lease: _Lease) -> None:
        with self._state_lock:
            self._active_leases.discard((job.video_id, lease.owner))

    def _commit_checkpoint(
        self, job: PreparationJob, lease: _Lease, action: Callable[[], None]
    ) -> None:
        """Order a file checkpoint atomically against durable cancellation."""
        self._require_lease(job, lease.owner)
        with self._session_factory() as db:
            owned = db.query(ExplanationPreparation).filter(
                ExplanationPreparation.id == int(lease.preparation_job_id),
                ExplanationPreparation.status == "generating",
                ExplanationPreparation.lease_owner == lease.owner,
                ExplanationPreparation.attempt_count == lease.attempt_count,
            ).update({
                ExplanationPreparation.lease_started_at: _utcnow(),
                ExplanationPreparation.updated_at: _utcnow(),
            }, synchronize_session=False)
            if owned != 1:
                db.rollback()
                raise PreparationCancelledError(
                    "Preparation was cancelled before checkpoint commit"
                )
            try:
                action()
                db.commit()
            except Exception:
                db.rollback()
                raise

    def _claim(self, job: PreparationJob) -> Optional[_Lease]:
        now = _utcnow()
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if state is None or state.status != "pending":
                if state is None or state.status in {"ready", "failed", "cancelled"}:
                    with self._state_lock:
                        self._forget_priority(job.key)
                return None
            if (
                state.stage == STAGE_WAITING_FOR_VLM
                and state.deferred_until is not None
                and state.deferred_until > now
            ):
                return None
            cached = find_valid_cached_explanation(
                db, job.intervention_id, job.output_mode, self._data_dir
            )
            if cached is not None:
                self._set_ready_state(db, job, state, now)
                db.commit()
                with self._state_lock:
                    self._forget_priority(job.key)
                return None
            if state.attempt_count >= self._max_attempts:
                self._set_failed_state(state, now, "retry_exhausted")
                db.commit()
                with self._state_lock:
                    self._forget_priority(job.key)
                return None
            lease_owner = f"{self._worker_id}:worker:{uuid.uuid4()}"
            previous_attempts = state.attempt_count
            state_id = state.id
            claimed = db.execute(
                update(ExplanationPreparation)
                .where(
                    ExplanationPreparation.id == state_id,
                    ExplanationPreparation.status == "pending",
                    ExplanationPreparation.attempt_count == previous_attempts,
                )
                .values(
                    status="generating",
                    stage=STAGE_EXTRACTING,
                    attempt_count=previous_attempts + 1,
                    error_code=None,
                    failure_reason=None,
                    lease_owner=lease_owner,
                    lease_started_at=now,
                    deferred_until=None,
                    updated_at=now,
                    failed_at=None,
                )
            ).rowcount
            db.commit()
            if claimed != 1:
                return None
            return _Lease(lease_owner, str(state_id), previous_attempts + 1)

    def _begin_manual(self, job: PreparationJob) -> Optional[_Lease]:
        now = _utcnow()
        with self._session_factory() as db:
            self._load_intervention(db, job)
            video_job = db.query(ProcessingJob.status).filter(
                ProcessingJob.video_id == job.video_id
            ).first()
            if video_job is not None and video_job[0] in {"cancelling", "cancelled"}:
                raise PreparationCancelledError("Video preparation is cancelled")
            state = self._get_state(db, job)
            cached = find_valid_cached_explanation(
                db, job.intervention_id, job.output_mode, self._data_dir
            )
            if cached is not None:
                if state is None or state.status != "generating":
                    self._set_ready_state(db, job, state, now)
                db.commit()
                return None
            if state is not None and state.status == "generating":
                return None
            if state is not None and state.status == "cancelled":
                raise PreparationCancelledError("Video preparation is cancelled")
            if state is not None and state.status != "ready" and state.attempt_count >= self._max_attempts:
                raise PreparationRetryExhaustedError(state.attempt_count)
            if state is None:
                state = self._new_state(job, status="pending")
                db.add(state)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    return None
            lease_owner = f"{self._worker_id}:manual:{uuid.uuid4()}"
            previous_attempts = state.attempt_count
            next_attempt = 1 if state.status == "ready" else previous_attempts + 1
            state_id = state.id
            claimed = db.execute(
                update(ExplanationPreparation)
                .where(
                    ExplanationPreparation.id == state_id,
                    ExplanationPreparation.status.in_(("pending", "failed", "ready")),
                    ExplanationPreparation.attempt_count == previous_attempts,
                )
                .values(
                    status="generating",
                    stage=STAGE_EXTRACTING,
                    attempt_count=next_attempt,
                    error_code=None,
                    failure_reason=None,
                    lease_owner=lease_owner,
                    lease_started_at=now,
                    deferred_until=None,
                    updated_at=now,
                    failed_at=None,
                )
            ).rowcount
            db.commit()
            if claimed != 1:
                return None
            return _Lease(lease_owner, str(state_id), next_attempt)

    def _wait_for_existing(self, job: PreparationJob) -> ExplanationArtifact:
        """Join the durable owner instead of duplicating work in another process."""
        deadline = time.monotonic() + self._lease_timeout.total_seconds()
        while time.monotonic() < deadline:
            with self._session_factory() as db:
                cached = find_valid_cached_explanation(
                    db, job.intervention_id, job.output_mode, self._data_dir
                )
                if cached is not None:
                    return cached
                state = self._get_state(db, job)
                if state is None or state.status == "failed":
                    raise InvalidPreparedArtifactError(
                        "The joined explanation preparation did not complete"
                    )
                if state.status == "cancelled":
                    raise PreparationCancelledError("Video preparation is cancelled")
                stale = state.status == "generating" and self._lease_is_stale(
                    state, _utcnow()
                )
            if stale:
                lease = self._begin_manual(job)
                if lease is not None:
                    with bind_trace(self._trace_context(job, lease)):
                        self._activate(job, lease)
                        try:
                            snapshot = self._prepare_evidence(job, lease)
                            result = self._generate(job, lease, snapshot)
                            self._mark_ready(job, lease)
                        finally:
                            self._deactivate(job, lease)
                        return result
            if self._stop_event.wait(0.1):
                break
        raise InvalidPreparedArtifactError("Timed out joining explanation preparation")

    def _mark_ready(self, job: PreparationJob, lease: _Lease) -> None:
        self._require_lease(job, lease.owner)
        now = _utcnow()
        with self._session_factory() as db:
            cached = find_valid_cached_explanation(
                db, job.intervention_id, job.output_mode, self._data_dir
            )
            if cached is None:
                raise InvalidPreparedArtifactError(
                    "Generated explanation cache is incomplete"
                )
            state = self._get_state(db, job)
            if (
                state is None or state.status != "generating"
                or state.lease_owner != lease.owner
                or state.attempt_count != lease.attempt_count
            ):
                raise PreparationCancelledError("Preparation lease was cancelled or superseded")
            self._set_ready_state(db, job, state, now)
            db.commit()
        with self._state_lock:
            self._forget_priority(job.key)
        release_terminal_video_slot(self._session_factory, job.video_id)
        self.timeline.append(
            (time.monotonic(), job.intervention_id, job.output_mode, STAGE_READY)
        )
        trace_stage("frontend_ready", "complete")

    def _mark_failed(
        self, job: PreparationJob, exc: Exception, lease_owner: str
    ) -> None:
        now = _utcnow()
        failure = getattr(exc, "failure", None)
        reason = failure.get("code") if isinstance(failure, dict) else None
        reason = reason or type(exc).__name__
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if state is None or state.status in {"ready", "cancelled"}:
                return
            if state.status != "generating" or state.lease_owner != lease_owner:
                return
            self._set_failed_state(state, now, self._safe_error_code(exc), reason)
            db.commit()
        with self._state_lock:
            self._forget_priority(job.key)
        release_terminal_video_slot(self._session_factory, job.video_id)
        self.timeline.append(
            (time.monotonic(), job.intervention_id, job.output_mode, STAGE_FAILED)
        )
        trace_stage(
            "preparation_state", "failed",
            error_code=self._safe_error_code(exc), failure_reason=reason,
        )

    @staticmethod
    def _is_retryable_vlm_unavailable(exc: Exception) -> bool:
        failure = getattr(exc, "failure", None)
        return bool(
            isinstance(failure, dict)
            and failure.get("code") == "vlm_remote_unavailable"
            and failure.get("retryable") is True
        )

    def _defer_for_vlm(self, job: PreparationJob, lease_owner: str) -> bool:
        """Return a provider failure to durable pending state without an attempt."""
        now = _utcnow()
        retry_at = now + timedelta(seconds=self._vlm_retry_seconds)
        terminal = False
        changed = False
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if (
                state is None
                or state.status != "generating"
                or state.lease_owner != lease_owner
            ):
                return False
            wait_started = state.provider_wait_started_at or now
            terminal = (
                now - wait_started
                >= timedelta(seconds=self._vlm_max_wait_seconds)
            )
            state.attempt_count = max(0, state.attempt_count - 1)
            state.error_code = "vlm_remote_unavailable"
            state.failure_reason = "vlm_remote_unavailable"
            state.lease_owner = None
            state.lease_started_at = None
            state.provider_wait_started_at = wait_started
            state.updated_at = now
            if terminal:
                state.status = "failed"
                state.priority_at = None
                state.stage = STAGE_FAILED
                state.deferred_until = None
                state.failed_at = now
            else:
                state.status = "pending"
                state.stage = STAGE_WAITING_FOR_VLM
                state.deferred_until = retry_at
                state.failed_at = None
            db.commit()
            changed = True
        with self._state_lock:
            self._provider_retry_after = max(
                self._provider_retry_after,
                time.monotonic() + self._vlm_retry_seconds,
            )
            self._deferred_keys.add(job.key)
            if self._provider_probe_key is None:
                self._provider_probe_key = job.key
            if terminal and self._provider_probe_key == job.key:
                self._provider_probe_key = None
            if terminal:
                self._forget_priority(job.key)
        if terminal:
            release_terminal_video_slot(self._session_factory, job.video_id)
        return changed

    def _recover_tracked_jobs(self) -> None:
        now = _utcnow()
        to_queue: list[PreparationJob] = []
        videos_seen: set[str] = set()
        with self._session_factory() as db:
            rows = db.query(ExplanationPreparation, Intervention.timestamp).join(
                Intervention,
                Intervention.id == ExplanationPreparation.intervention_id,
            ).order_by(
                Intervention.timestamp.asc(),
                ExplanationPreparation.intervention_id.asc(),
            ).all()
            for state, _timestamp in rows:
                videos_seen.add(state.video_id)
                if state.priority_at and state.status in {"pending", "generating"}:
                    key = (state.video_id, state.intervention_id, state.output_mode)
                    self._priority_keys.add(key)
                    self._priority_order[key] = state.priority_at
                    if state.stage == STAGE_WAITING_FOR_VLM:
                        self._deferred_keys.add(key)
                if state.status == "generating":
                    continue
                if state.status == "cancelled":
                    continue
                job = PreparationJob(
                    state.video_id, state.intervention_id, state.output_mode
                )
                cached = find_valid_cached_explanation(
                    db, state.intervention_id, state.output_mode, self._data_dir
                )
                if cached is not None:
                    self._set_ready_state(db, job, state, now)
                    continue
                if state.status == "ready":
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="missing_audio"
                    )
                if state.status == "pending" and state.stage != STAGE_WAITING_FOR_VLM:
                    to_queue.append(job)
            db.commit()
        for video_id in videos_seen:
            release_terminal_video_slot(self._session_factory, video_id)
        for job in to_queue:
            self._enqueue(job)
        self._recover_stale_preparations(now)
        self._recover_deferred_preparations(now)

    def _recover_deferred_preparations(
        self, now: Optional[datetime] = None
    ) -> int:
        """Queue due provider waits or end waits that exceeded their deadline."""
        now = now or _utcnow()
        to_queue: list[PreparationJob] = []
        terminal_videos: set[str] = set()
        changed = 0
        with self._session_factory() as db:
            rows = db.query(ExplanationPreparation, Intervention.timestamp).join(
                Intervention,
                Intervention.id == ExplanationPreparation.intervention_id,
            ).filter(
                ExplanationPreparation.status == "pending",
                ExplanationPreparation.stage == STAGE_WAITING_FOR_VLM,
            ).order_by(
                Intervention.timestamp.asc(),
                ExplanationPreparation.intervention_id.asc(),
            ).all()
            for state, _timestamp in rows:
                job = PreparationJob(
                    state.video_id, state.intervention_id, state.output_mode
                )
                cached = find_valid_cached_explanation(
                    db, state.intervention_id, state.output_mode, self._data_dir
                )
                if cached is not None:
                    self._set_ready_state(db, job, state, now)
                    terminal_videos.add(job.video_id)
                    changed += 1
                    continue
                wait_started = state.provider_wait_started_at or now
                expired = (
                    now - wait_started
                    >= timedelta(seconds=self._vlm_max_wait_seconds)
                )
                if expired:
                    self._set_failed_state(
                        state, now, "vlm_remote_unavailable",
                        failure_reason="vlm_remote_unavailable",
                    )
                    terminal_videos.add(job.video_id)
                    changed += 1
                elif state.deferred_until is None or state.deferred_until <= now:
                    to_queue.append(job)
            db.commit()
        for video_id in terminal_videos:
            release_terminal_video_slot(self._session_factory, video_id)
        if to_queue:
            with self._state_lock:
                to_queue.sort(key=lambda item: self._priority_order.get(item.key, datetime.max))
                if self._provider_probe_key is None:
                    self._provider_probe_key = to_queue[0].key
        for job in to_queue:
            self._enqueue(job)
        return changed

    def _run_lease_maintenance(self) -> None:
        """Renew live owners and recover only leases that remain stale."""
        while not self._stop_event.wait(self._maintenance_interval):
            self._renew_tracked_leases()
            if self._stop_event.is_set():
                break
            try:
                self._recover_stale_preparations()
                self._recover_deferred_preparations()
            except Exception:
                logger.exception("Could not scan for stale explanation preparations")

    def _renew_tracked_leases(self) -> None:
        monotonic_now = time.monotonic()
        with self._state_lock:
            tracked = list(self._tracked_leases.values())
        for item in tracked:
            if item.lost:
                continue
            if monotonic_now - item.last_renewed >= self._lease_timeout_seconds:
                self._mark_tracked_lease_lost(item.lease)
                logger.error(
                    "Explanation preparation lease renewal exceeded its timeout: id=%s attempt=%s",
                    item.lease.preparation_job_id,
                    item.lease.attempt_count,
                )
                continue
            now = _utcnow()
            try:
                with self._session_factory() as db:
                    renewed = db.execute(
                        update(ExplanationPreparation)
                        .where(
                            ExplanationPreparation.id
                            == int(item.lease.preparation_job_id),
                            ExplanationPreparation.status == "generating",
                            ExplanationPreparation.lease_owner == item.lease.owner,
                            ExplanationPreparation.attempt_count
                            == item.lease.attempt_count,
                        )
                        .values(lease_started_at=now, updated_at=now)
                    ).rowcount
                    db.commit()
                if renewed != 1:
                    self._mark_tracked_lease_lost(item.lease)
                    continue
                with self._state_lock:
                    current = self._tracked_leases.get(item.lease.owner)
                    if current is item and not current.lost:
                        current.last_renewed = monotonic_now
            except Exception:
                logger.warning(
                    "Could not renew explanation preparation lease: id=%s attempt=%s",
                    item.lease.preparation_job_id,
                    item.lease.attempt_count,
                    exc_info=True,
                )

    def _recover_stale_preparations(
        self, now: Optional[datetime] = None
    ) -> int:
        """Atomically make stale generating rows pending, failed, or ready."""
        now = now or _utcnow()
        stale_before = now - self._lease_timeout
        recovered_jobs: list[PreparationJob] = []
        terminal_videos: set[str] = set()
        transition_count = 0
        with self._state_lock:
            healthy_local_owners = {
                owner for owner, tracked in self._tracked_leases.items()
                if not tracked.lost
            }
        with self._session_factory() as db:
            stale_filter = or_(
                ExplanationPreparation.lease_started_at.is_(None),
                ExplanationPreparation.lease_started_at <= stale_before,
            )
            if healthy_local_owners:
                stale_filter = stale_filter & (
                    ExplanationPreparation.lease_owner.notin_(healthy_local_owners)
                )
            rows = db.query(
                ExplanationPreparation,
                Intervention.timestamp,
                ProcessingJob.status,
            ).join(
                Intervention,
                Intervention.id == ExplanationPreparation.intervention_id,
            ).outerjoin(
                ProcessingJob,
                ProcessingJob.video_id == ExplanationPreparation.video_id,
            ).filter(
                ExplanationPreparation.status == "generating",
                stale_filter,
            ).order_by(
                Intervention.timestamp.asc(),
                ExplanationPreparation.intervention_id.asc(),
            ).all()
            for state, _timestamp, processing_status in rows:
                job = PreparationJob(
                    state.video_id, state.intervention_id, state.output_mode
                )
                cached = find_valid_cached_explanation(
                    db, state.intervention_id, state.output_mode, self._data_dir
                )
                conditions = [
                    ExplanationPreparation.id == state.id,
                    ExplanationPreparation.status == "generating",
                    ExplanationPreparation.lease_owner == state.lease_owner,
                    ExplanationPreparation.attempt_count == state.attempt_count,
                ]
                if state.lease_started_at is None:
                    conditions.append(ExplanationPreparation.lease_started_at.is_(None))
                else:
                    conditions.append(
                        ExplanationPreparation.lease_started_at
                        == state.lease_started_at
                    )
                if processing_status in {"cancelling", "cancelled"}:
                    values = dict(
                        status="cancelled",
                        stage="cancelled",
                        error_code=None,
                        failure_reason="video_cancelled",
                        lease_owner=None,
                        lease_started_at=None,
                        provider_wait_started_at=None,
                        deferred_until=None,
                        ready_at=None,
                        failed_at=None,
                        updated_at=now,
                    )
                    enqueue = False
                elif cached is not None:
                    values = dict(
                        status="ready",
                        stage=STAGE_READY,
                        error_code=None,
                        failure_reason=None,
                        lease_owner=None,
                        lease_started_at=None,
                        provider_wait_started_at=None,
                        deferred_until=None,
                        ready_at=now,
                        failed_at=None,
                        updated_at=now,
                    )
                    enqueue = False
                elif state.attempt_count >= self._max_attempts:
                    values = dict(
                        status="failed",
                        stage=STAGE_FAILED,
                        error_code="retry_exhausted",
                        failure_reason="retry_exhausted",
                        lease_owner=None,
                        lease_started_at=None,
                        provider_wait_started_at=None,
                        deferred_until=None,
                        ready_at=None,
                        failed_at=now,
                        updated_at=now,
                    )
                    enqueue = False
                else:
                    values = dict(
                        status="pending",
                        stage=STAGE_PENDING,
                        error_code="interrupted",
                        failure_reason=None,
                        lease_owner=None,
                        lease_started_at=None,
                        provider_wait_started_at=None,
                        deferred_until=None,
                        ready_at=None,
                        failed_at=None,
                        updated_at=now,
                    )
                    enqueue = True
                changed = db.execute(
                    update(ExplanationPreparation).where(*conditions).values(**values)
                ).rowcount
                if changed == 1:
                    transition_count += 1
                    if enqueue:
                        recovered_jobs.append(job)
                    else:
                        terminal_videos.add(job.video_id)
            db.commit()
        for video_id in terminal_videos:
            release_terminal_video_slot(self._session_factory, video_id)
        for job in recovered_jobs:
            self._enqueue(job)
        return transition_count

    def _track_lease(self, job: PreparationJob, lease: _Lease) -> None:
        with self._state_lock:
            self._tracked_leases[lease.owner] = _TrackedLease(
                job=job,
                lease=lease,
                last_renewed=time.monotonic(),
            )

    def _untrack_lease(self, lease: _Lease) -> None:
        with self._state_lock:
            current = self._tracked_leases.get(lease.owner)
            if current is not None and current.lease == lease:
                self._tracked_leases.pop(lease.owner, None)

    def _mark_tracked_lease_lost(self, lease: _Lease) -> None:
        with self._state_lock:
            current = self._tracked_leases.get(lease.owner)
            if current is not None and current.lease == lease:
                current.lost = True

    def _local_lease_is_healthy(self, lease_owner: str) -> bool:
        with self._state_lock:
            tracked = self._tracked_leases.get(lease_owner)
            return tracked is None or not tracked.lost

    def _invalidate_video_leases(self, video_id: str) -> None:
        with self._state_lock:
            for tracked in self._tracked_leases.values():
                if tracked.job.video_id == video_id:
                    tracked.lost = True

    def _invalidate_all_tracked_leases(self) -> None:
        with self._state_lock:
            for tracked in self._tracked_leases.values():
                tracked.lost = True

    def _set_recoverable_or_exhausted(
        self, state: ExplanationPreparation, now: datetime, error_code: str
    ) -> None:
        if state.status == "ready":
            # A completed artifact that is missing/stale starts a new generation
            # cycle; successful historical attempts are not failed retries.
            state.attempt_count = 0
        if state.attempt_count >= self._max_attempts:
            self._set_failed_state(state, now, "retry_exhausted")
            return
        state.status = "pending"
        state.stage = STAGE_PENDING
        state.error_code = error_code
        state.lease_owner = None
        state.lease_started_at = None
        state.provider_wait_started_at = None
        state.deferred_until = None
        state.ready_at = None
        state.failed_at = None
        state.updated_at = now

    @staticmethod
    def _set_failed_state(
        state: ExplanationPreparation,
        now: datetime,
        error_code: str,
        failure_reason: Optional[str] = None,
    ) -> None:
        state.status = "failed"
        state.priority_at = None
        state.stage = STAGE_FAILED
        state.error_code = error_code
        state.failure_reason = failure_reason or error_code
        state.lease_owner = None
        state.lease_started_at = None
        state.provider_wait_started_at = None
        state.deferred_until = None
        state.ready_at = None
        state.failed_at = now
        state.updated_at = now

    def _set_ready_state(
        self,
        db: Session,
        job: PreparationJob,
        state: Optional[ExplanationPreparation],
        now: datetime,
    ) -> ExplanationPreparation:
        if state is None:
            state = self._new_state(job, status="ready")
            db.add(state)
        state.status = "ready"
        state.priority_at = None
        state.stage = STAGE_READY
        state.error_code = None
        state.failure_reason = None
        state.lease_owner = None
        state.lease_started_at = None
        state.provider_wait_started_at = None
        state.deferred_until = None
        state.ready_at = now
        state.failed_at = None
        state.updated_at = now
        return state

    @staticmethod
    def _new_state(
        job: PreparationJob,
        *,
        status: str,
        error_code: Optional[str] = None,
    ) -> ExplanationPreparation:
        return ExplanationPreparation(
            video_id=job.video_id,
            intervention_id=job.intervention_id,
            output_mode=job.output_mode,
            status=status,
            stage=STAGE_PENDING if status == "pending" else None,
            attempt_count=0,
            error_code=error_code,
        )

    @staticmethod
    def _get_state(
        db: Session, job: PreparationJob
    ) -> Optional[ExplanationPreparation]:
        return db.query(ExplanationPreparation).filter(
            ExplanationPreparation.video_id == job.video_id,
            ExplanationPreparation.intervention_id == job.intervention_id,
            ExplanationPreparation.output_mode == job.output_mode,
        ).one_or_none()

    @staticmethod
    def _load_intervention(db: Session, job: PreparationJob) -> Intervention:
        if db.query(Video.id).filter(Video.id == job.video_id).first() is None:
            raise ExplanationTargetNotFoundError("Video not found")
        intervention = db.query(Intervention).filter(
            Intervention.id == job.intervention_id,
            Intervention.video_id == job.video_id,
        ).first()
        if intervention is None:
            raise ExplanationTargetNotFoundError("Intervention not found")
        return intervention

    def _lease_is_stale(
        self, state: ExplanationPreparation, now: datetime
    ) -> bool:
        return (
            state.lease_started_at is None
            or now - state.lease_started_at >= self._lease_timeout
        )

    def _enqueue(self, job: PreparationJob) -> bool:
        with self._state_lock:
            if job.key in self._scheduled:
                return False
            self._scheduled.add(job.key)
            queued_job = replace(job, enqueued_at=time.monotonic())
            self._queue.put(queued_job)
            trace_stage(
                "vlm_queue",
                "entered",
                context=TraceContext(
                    trace_id=queued_job.trace_id,
                    video_id=queued_job.video_id,
                    intervention_id=queued_job.intervention_id,
                    output_mode=queued_job.output_mode,
                ),
            )
            return True

    def _reschedule_if_pending(self, job: PreparationJob) -> None:
        """Close the cancel/resume race after stale queued work drains."""
        if self._stop_event.is_set():
            return
        with self._session_factory() as db:
            state = self._get_state(db, job)
            now = _utcnow()
            pending = (
                state is not None
                and state.status == "pending"
                and (
                    state.stage != STAGE_WAITING_FOR_VLM
                    or state.deferred_until is None
                    or state.deferred_until <= now
                )
            )
        if pending:
            self._enqueue(replace(job, trace_id=new_trace_id(), enqueued_at=0.0))

    @staticmethod
    def _trace_context(job: PreparationJob, lease: _Lease) -> TraceContext:
        return TraceContext(
            trace_id=job.trace_id,
            video_id=job.video_id,
            intervention_id=job.intervention_id,
            output_mode=job.output_mode,
            preparation_job_id=lease.preparation_job_id,
            attempt=lease.attempt_count,
        )

    def _require_running(self) -> None:
        if self._stop_event.is_set() or not self.is_running:
            raise RuntimeError("Explanation preparation worker is not running")

    @staticmethod
    def _validate_mode(output_mode: str) -> None:
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError("Invalid explanation output mode")

    @staticmethod
    def _safe_error_code(exc: Exception) -> str:
        if isinstance(exc, InvalidPreparedArtifactError):
            return "invalid_cache"
        if isinstance(exc, ExplanationTargetNotFoundError):
            return "target_not_found"
        if isinstance(exc, PreparationRetryExhaustedError):
            return "retry_exhausted"
        synthesis_code = getattr(exc, "error_code", None)
        if synthesis_code in {
            "visual_evidence_unavailable",
            "text_synthesis_unavailable",
        }:
            return synthesis_code
        return "generation_failed"
