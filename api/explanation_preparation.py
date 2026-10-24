"""Persistent single-consumer explanation preparation for local Synapse."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from api.explanation_service import (
    ExplanationArtifact,
    ExplanationTargetNotFoundError,
    find_valid_cached_explanation,
    generate_or_get_explanation,
)
from api.models import Explanation, ExplanationPreparation, Intervention, Video

logger = logging.getLogger(__name__)

VALID_OUTPUT_MODES = {"brief", "explanatory", "detailed"}
DEFAULT_MAX_ATTEMPTS = 3


def _utcnow() -> datetime:
    """Naive UTC for compatibility with existing SQLite DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PreparationNotRetryableError(RuntimeError):
    """The requested key is not currently in a failed state."""


class PreparationRetryExhaustedError(RuntimeError):
    """The persistent retry limit has been reached."""

    def __init__(self, attempt_count: int):
        super().__init__("Explanation preparation retry limit reached")
        self.attempt_count = attempt_count


class InvalidPreparedArtifactError(RuntimeError):
    """Generation returned without creating a complete servable cache."""


@dataclass(frozen=True)
class PreparationJob:
    video_id: str
    intervention_id: str
    output_mode: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.video_id, self.intervention_id, self.output_mode)


@dataclass(frozen=True)
class PreparationSnapshot:
    intervention_id: str
    timestamp: float
    status: str
    error_code: Optional[str]
    attempt_count: int


@dataclass(frozen=True)
class RetryResult:
    status: str
    error_code: Optional[str]
    attempt_count: int
    queued: bool = False


class ExplanationPreparationManager:
    """A durable SQLite-backed queue with one inference consumer."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        orchestrator_factory: Callable[[], object],
        data_dir: str,
        generator: Callable = generate_or_get_explanation,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lease_timeout_seconds: float = 1800.0,
    ) -> None:
        self._session_factory = session_factory
        self._orchestrator_factory = orchestrator_factory
        self._data_dir = data_dir
        self._generator = generator
        self._max_attempts = max_attempts
        self._lease_timeout = timedelta(seconds=lease_timeout_seconds)
        self._queue: queue.Queue[Optional[PreparationJob]] = queue.Queue()
        self._scheduled: set[tuple[str, str, str]] = set()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._worker_id = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_id = str(uuid.uuid4())
            thread = threading.Thread(
                target=self._run,
                name="synapse-explanation-preparation",
                daemon=True,
            )
            self._thread = thread
        try:
            self._recover_tracked_jobs()
            thread.start()
        except Exception:
            with self._state_lock:
                self._thread = None
            raise

    def schedule_video(self, video_id: str, output_mode: str) -> int:
        """Persist and queue missing work in educational timestamp order."""
        self._require_running()
        self._validate_mode(output_mode)
        to_queue: list[PreparationJob] = []
        now = _utcnow()

        with self._session_factory() as db:
            intervention_rows = db.query(Intervention).filter(
                Intervention.video_id == video_id,
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
                    self._set_ready_state(db, job, state, now)
                    continue

                if state is None:
                    state = self._new_state(job, status="pending")
                    db.add(state)
                elif state.status == "ready":
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="missing_audio"
                    )
                elif state.status == "generating" and self._lease_is_stale(state, now):
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="interrupted"
                    )
                if state.status == "pending":
                    to_queue.append(job)
            db.commit()

        return sum(1 for job in to_queue if self._enqueue(job))

    def get_status(self, video_id: str, output_mode: str) -> list[PreparationSnapshot]:
        """Return effective ordered status without mutating persistent state."""
        self._validate_mode(output_mode)
        with self._session_factory() as db:
            if db.query(Video.id).filter(Video.id == video_id).first() is None:
                raise ExplanationTargetNotFoundError("Video not found")
            interventions = db.query(Intervention).filter(
                Intervention.video_id == video_id,
            ).order_by(
                Intervention.timestamp.asc(),
                Intervention.id.asc(),
            ).all()
            snapshots = []
            for intervention in interventions:
                job = PreparationJob(video_id, intervention.id, output_mode)
                state = self._get_state(db, job)
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
                ))
            return snapshots

    def retry(self, video_id: str, intervention_id: str, output_mode: str) -> RetryResult:
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
            if state.attempt_count >= self._max_attempts:
                state.error_code = "retry_exhausted"
                state.updated_at = now
                db.commit()
                raise PreparationRetryExhaustedError(state.attempt_count)

            state.status = "pending"
            state.error_code = None
            state.lease_owner = None
            state.lease_started_at = None
            state.failed_at = None
            state.updated_at = now
            db.commit()
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
        lease_owner = self._begin_manual(job)
        try:
            result = self._generate(job)
            self._mark_ready(job)
            return result
        except Exception as exc:
            if lease_owner is not None:
                self._mark_failed(job, self._safe_error_code(exc), lease_owner)
            raise

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop accepting work, preserve pending state, and bound shutdown wait."""
        self._stop_event.set()
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued is not None:
                with self._state_lock:
                    self._scheduled.discard(queued.key)
            self._queue.task_done()
        self._queue.put(None)

        thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Explanation preparation worker did not stop within the timeout")
        with self._state_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            lease_owner = None
            try:
                lease_owner = self._claim(job)
                if lease_owner is not None:
                    self._generate(job)
                    self._mark_ready(job)
            except Exception as exc:
                # Publish a failed state only after this active queue key is no
                # longer considered scheduled. An immediate explicit retry can
                # then enqueue replacement work instead of being stranded by
                # the old key during this worker's final cleanup.
                with self._state_lock:
                    self._scheduled.discard(job.key)
                if lease_owner is not None:
                    self._mark_failed(job, self._safe_error_code(exc), lease_owner)
                logger.exception(
                    "Background explanation preparation failed for video %s, intervention %s, mode %s",
                    job.video_id,
                    job.intervention_id,
                    job.output_mode,
                )
            finally:
                with self._state_lock:
                    self._scheduled.discard(job.key)
                self._queue.task_done()

    def _generate(self, job: PreparationJob) -> ExplanationArtifact:
        return self._generator(
            session_factory=self._session_factory,
            orchestrator_factory=self._orchestrator_factory,
            data_dir=self._data_dir,
            video_id=job.video_id,
            intervention_id=job.intervention_id,
            output_mode=job.output_mode,
        )

    def _claim(self, job: PreparationJob) -> Optional[str]:
        now = _utcnow()
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if state is None or state.status != "pending":
                return None
            cached = find_valid_cached_explanation(
                db, job.intervention_id, job.output_mode, self._data_dir
            )
            if cached is not None:
                self._set_ready_state(db, job, state, now)
                db.commit()
                return None
            if state.attempt_count >= self._max_attempts:
                self._set_failed_state(state, now, "retry_exhausted")
                db.commit()
                return None
            lease_owner = f"{self._worker_id}:worker"
            state.status = "generating"
            state.attempt_count += 1
            state.error_code = None
            state.lease_owner = lease_owner
            state.lease_started_at = now
            state.updated_at = now
            state.failed_at = None
            db.commit()
            return lease_owner

    def _begin_manual(self, job: PreparationJob) -> Optional[str]:
        now = _utcnow()
        with self._session_factory() as db:
            self._load_intervention(db, job)
            state = self._get_state(db, job)
            cached = find_valid_cached_explanation(
                db, job.intervention_id, job.output_mode, self._data_dir
            )
            if cached is not None:
                self._set_ready_state(db, job, state, now)
                db.commit()
                return None
            if state is not None and state.status == "generating":
                return None
            if state is not None and state.attempt_count >= self._max_attempts:
                raise PreparationRetryExhaustedError(state.attempt_count)
            if state is None:
                state = self._new_state(job, status="generating")
                db.add(state)
            else:
                state.status = "generating"
            lease_owner = f"{self._worker_id}:manual:{uuid.uuid4()}"
            state.attempt_count += 1
            state.error_code = None
            state.lease_owner = lease_owner
            state.lease_started_at = now
            state.updated_at = now
            state.failed_at = None
            db.commit()
            return lease_owner

    def _mark_ready(self, job: PreparationJob) -> None:
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
            self._set_ready_state(db, job, state, now)
            db.commit()

    def _mark_failed(
        self, job: PreparationJob, error_code: str, lease_owner: str
    ) -> None:
        now = _utcnow()
        with self._session_factory() as db:
            state = self._get_state(db, job)
            if state is None or state.status == "ready":
                return
            if state.lease_owner not in {None, lease_owner}:
                return
            self._set_failed_state(state, now, error_code)
            db.commit()

    def _recover_tracked_jobs(self) -> None:
        now = _utcnow()
        to_queue: list[PreparationJob] = []
        with self._session_factory() as db:
            rows = db.query(ExplanationPreparation, Intervention.timestamp).join(
                Intervention,
                Intervention.id == ExplanationPreparation.intervention_id,
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
                    continue
                if state.status == "generating":
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="interrupted"
                    )
                elif state.status == "ready":
                    self._set_recoverable_or_exhausted(
                        state, now, error_code="missing_audio"
                    )
                if state.status == "pending":
                    to_queue.append(job)
            db.commit()
        for job in to_queue:
            self._enqueue(job)

    def _set_recoverable_or_exhausted(
        self, state: ExplanationPreparation, now: datetime, error_code: str
    ) -> None:
        if state.attempt_count >= self._max_attempts:
            self._set_failed_state(state, now, "retry_exhausted")
            return
        state.status = "pending"
        state.error_code = error_code
        state.lease_owner = None
        state.lease_started_at = None
        state.ready_at = None
        state.failed_at = None
        state.updated_at = now

    @staticmethod
    def _set_failed_state(
        state: ExplanationPreparation, now: datetime, error_code: str
    ) -> None:
        state.status = "failed"
        state.error_code = error_code
        state.lease_owner = None
        state.lease_started_at = None
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
        state.error_code = None
        state.lease_owner = None
        state.lease_started_at = None
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
            self._queue.put(job)
            return True

    def _require_running(self) -> None:
        if not self.is_running:
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
        return "generation_failed"
