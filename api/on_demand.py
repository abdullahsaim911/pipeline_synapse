"""Owned request/session lifecycle around the existing shared preparation workers."""
from datetime import datetime, timedelta, timezone
from contextvars import ContextVar
import hashlib
import json
import logging
import threading
import uuid

from fastapi import HTTPException
from sqlalchemy import or_

from api.models import (Explanation, ExplanationPreparation, ExplanationRequest, Intervention,
                        LectureSessionInterest, ProcessingJob, SavedExplanation, Transcript, Video)
from api.personal_storage import snapshot
from api.processing_jobs import ProcessingJobBusyError

logger = logging.getLogger(__name__)


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class OnDemandExplanations:
    """One lifecycle owner per local backend, with durable jobs and request identity."""

    def __init__(self, sessions, preparation, processing):
        self._sessions = sessions
        self.authorization = ContextVar("explanation_request_account", default=None)
        self.preparation = preparation
        self.processing = processing
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    def sessions(self):
        db = self._sessions()
        credential = self.authorization.get()
        if credential:
            db.info["account_session_digest"] = credential
        return db

    def start(self):
        self.maintain()
        self._thread = threading.Thread(target=self._run, name="synapse-request-lifecycle", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(5):
            try:
                self.maintain()
            except Exception:
                logger.exception("Could not reconcile owned explanation requests")

    def _owned_session(self, db, identity, owner):
        row = db.get(LectureSessionInterest, identity)
        if row is None or row.user_id != owner:
            raise HTTPException(404, "Lecture session not found.")
        return row

    def open(self, owner, video):
        with self._lock, self.sessions() as db:
            now = utcnow()
            row = LectureSessionInterest(id=str(uuid.uuid4()), user_id=owner, video_id=video,
                state="active", account_digest=self.authorization.get(),
                expires_at=now + timedelta(seconds=90), updated_at=now)
            db.add(row)
            db.query(ExplanationRequest).filter_by(user_id=owner, video_id=video,
                saved_explanation_id=None).update({ExplanationRequest.session_id: row.id}, synchronize_session=False)
            db.commit()
            return {"id": row.id, "video_id": video}

    def heartbeat(self, owner, identity):
        with self._lock, self.sessions() as db:
            row = self._owned_session(db, identity, owner)
            if row.state != "active":
                raise HTTPException(409, "This lecture session has ended. Reopen the lecture.")
            row.expires_at = utcnow() + timedelta(seconds=90)
            row.updated_at = utcnow()
            db.commit()
            return {"id": row.id}

    def release(self, owner, identity, reason):
        with self._lock:
            with self.sessions() as db:
                row = self._owned_session(db, identity, owner)
                if row.state != "active":
                    return
                row.state = "continuing" if reason == "signout" else "released"
                row.updated_at = utcnow()
                video = row.video_id
                db.commit()
            self._cancel_uninterested(video)

    def _cancel_uninterested(self, video):
        with self.sessions() as db:
            now = utcnow()
            interested = db.query(LectureSessionInterest.id).filter(
                LectureSessionInterest.video_id == video,
                or_(LectureSessionInterest.state == "continuing",
                    (LectureSessionInterest.state == "active") & (LectureSessionInterest.expires_at > now))).first()
            if interested:
                return
            job = db.query(ProcessingJob).filter_by(video_id=video).one_or_none()
            job_id = str(job.id) if job else None
        if job_id:
            self.processing.cancel(job_id)

    def request(self, owner, session_id, operation_id, video, timestamp, depth, point_id=None, retry=False):
        intent_hash = hashlib.sha256(json.dumps([video, timestamp, depth, point_id]).encode()).hexdigest()
        with self._lock:
            with self.sessions() as db:
                session = self._owned_session(db, session_id, owner)
                if session.video_id != video or session.state != "active" or session.expires_at <= utcnow():
                    raise HTTPException(409, "Reopen this lecture before requesting an explanation.")
                existing = db.query(ExplanationRequest).filter_by(user_id=owner, operation_id=operation_id).one_or_none()
                if existing:
                    if existing.intent_hash != intent_hash:
                        raise HTTPException(409, "This request identifier already has different content.")
                    identity = existing.id
                else:
                    if db.get(Video, video) is None and db.query(ProcessingJob.id).filter_by(video_id=video).first() is None:
                        raise HTTPException(404, "Start processing this lecture before requesting an explanation.")
                    point = db.get(Intervention, point_id) if point_id else None
                    if point_id and (point is None or point.video_id != video):
                        raise HTTPException(404, "Explanation moment not found.")
                    if point is not None and point.origin == "manual":
                        owned = db.query(ExplanationRequest.id).filter_by(user_id=owner, intervention_id=point.id).first()
                        if not owned:
                            raise HTTPException(404, "Explanation moment not found.")
                    row = ExplanationRequest(id=str(uuid.uuid4()), user_id=owner, session_id=session_id,
                        operation_id=operation_id, intent_hash=intent_hash, video_id=video, timestamp=point.timestamp if point else timestamp,
                        intervention_id=point_id, output_mode=depth, created_at=utcnow())
                    db.add(row)
                    db.commit()
                    identity = row.id
            with self.sessions() as db:
                job = db.query(ProcessingJob).filter_by(video_id=video).one_or_none()
                resume_job = str(job.id) if job and job.status in {"cancelled", "cancelling"} else None
            try:
                if resume_job:
                    self.processing.resume(resume_job)
                self._advance(identity, retry=retry)
            except ProcessingJobBusyError:
                self._admission_failed(identity, "video_processing_busy")
            return self.get(owner, identity)

    def _admission_failed(self, identity, code):
        with self.sessions() as db:
            row = db.get(ExplanationRequest, identity)
            if row:
                row.resolution_error = code
                db.commit()

    def _context(self, db, video, timestamp):
        segments = db.query(Transcript).filter(Transcript.video_id == video,
            Transcript.start_time <= timestamp + 1, Transcript.end_time >= timestamp - 15,
        ).order_by(Transcript.start_time).all()
        text = " ".join(segment.text for segment in segments)[-12000:]
        key = hashlib.sha256(json.dumps(["on-demand-v1", video,
            round(timestamp, 2), text], ensure_ascii=False).encode()).hexdigest()
        return text, key

    def _resolve(self, db, row):
        if row.intervention_id and db.get(Intervention, row.intervention_id):
            point = db.get(Intervention, row.intervention_id)
            _, key = self._context(db, row.video_id, row.timestamp)
            row.context_key = row.context_key or point.context_key or key
            return point
        video = db.get(Video, row.video_id)
        if video is None:
            job = db.query(ProcessingJob).filter_by(video_id=row.video_id).one_or_none()
            if job and job.status in {"failed", "cancelled"}:
                row.resolution_error = "source_unavailable"
            return None
        if video.duration_seconds and row.timestamp > video.duration_seconds:
            row.resolution_error = "timestamp_out_of_range"
            return None
        text, source_key = self._context(db, row.video_id, row.timestamp)
        if not text.strip():
            row.resolution_error = "insufficient_context"
            return None
        if row.context_key and row.context_key != source_key:
            row.resolution_error = "source_changed"
            return None
        point = db.query(Intervention).filter_by(context_key=source_key).one_or_none()
        if point is None:
            point = db.query(Intervention).filter_by(video_id=row.video_id,
                timestamp=row.timestamp, origin="detected").order_by(Intervention.id).first()
        if point is None:
            # Resolve/extract this frame in the existing evidence worker, outside SQL transactions.
            point = Intervention(id=str(uuid.uuid4()), video_id=row.video_id, timestamp=row.timestamp,
                timestamp_formatted=f"{int(row.timestamp // 60)}:{int(row.timestamp % 60):02d}",
                frame_path="", origin="manual", evidence_kind="auto", context_key=source_key,
                content_type="unknown", transcript_context=text, trigger_reason="User requested explanation")
            db.add(point)
            db.flush()
        row.context_key = source_key
        row.intervention_id = point.id
        row.resolution_error = None
        return point

    def _advance(self, identity, retry=False):
        with self.sessions() as db:
            row = db.get(ExplanationRequest, identity)
            if row is None or row.saved_explanation_id:
                return
            if row.resolution_error and not retry:
                return
            session = db.get(LectureSessionInterest, row.session_id) if row.session_id else None
            if not session or session.state not in {"active", "continuing"}:
                return
            if session.state == "active" and session.expires_at <= utcnow():
                return
            point = self._resolve(db, row)
            db.commit()
            if point is None:
                return
            video, point_id, depth = row.video_id, point.id, row.output_mode
            state = db.query(ExplanationPreparation).filter_by(video_id=video,
                intervention_id=point_id, output_mode=depth).one_or_none()
            if state is None or state.status == "ready":
                explanation = db.query(Explanation).filter_by(intervention_id=point_id, output_mode=depth).order_by(Explanation.id.desc()).first()
                if explanation:
                    try:
                        row.saved_explanation_id = snapshot(db, explanation, row.user_id).id
                        db.commit()
                    except HTTPException as error:
                        if error.status_code != 409:
                            raise
                        row.resolution_error = "saved_content_unavailable"
                        db.commit()
                    return
            if state and state.status == "failed" and not retry:
                return
            if state and state.status == "cancelled":
                video_job = db.query(ProcessingJob).filter_by(video_id=video).one_or_none()
                if video_job and video_job.status in {"cancelling", "cancelled"}:
                    return
            if state and state.status in {"pending", "generating"} and state.priority_at:
                return
        # Admission owns its own short transactions and durable video capacity.
        try:
            if retry and state and state.status == "failed":
                self.preparation.retry(video, point_id, depth)
            else:
                self.preparation.request(video, point_id, depth, requested_at=row.created_at)
        except ProcessingJobBusyError:
            self._admission_failed(identity, "video_processing_busy")
        except (RuntimeError, LookupError) as error:
            self._admission_failed(identity, "preparation_unavailable")
            logger.info("Owned explanation admission deferred: %s", type(error).__name__)

    def retry(self, owner, identity, session_id):
        with self._lock, self.sessions() as db:
            row = db.get(ExplanationRequest, identity)
            if row is None or row.user_id != owner:
                raise HTTPException(404, "Explanation request not found.")
            session = self._owned_session(db, session_id, owner)
            if session.video_id != row.video_id or session.state != "active" or session.expires_at <= utcnow():
                raise HTTPException(409, "Reopen this lecture before retrying.")
            row.session_id = session.id
            row.resolution_error = None
            db.commit()
        with self._lock:
            self._advance(identity, retry=True)
        return self.get(owner, identity)

    def source_ready(self, video):
        if self._stop.is_set():
            return
        with self._lock, self.sessions() as db:
            identities = [row[0] for row in db.query(ExplanationRequest.id).filter_by(video_id=video,
                saved_explanation_id=None, resolution_error=None).join(LectureSessionInterest,
                    ExplanationRequest.session_id == LectureSessionInterest.id).filter(
                    LectureSessionInterest.state.in_(("active", "continuing")))
                    .order_by(ExplanationRequest.created_at, ExplanationRequest.id).all()]
        for identity in identities:
            with self._lock:
                try:
                    self._advance(identity)
                except Exception:
                    logger.exception("Could not reconcile explanation request %s", identity)

    def _response(self, db, row):
        from api.explanation_preparation import PERMANENT_FAILURE_REASONS
        state = db.query(ExplanationPreparation).filter_by(intervention_id=row.intervention_id,
            output_mode=row.output_mode).one_or_none() if row.intervention_id else None
        point = db.get(Intervention, row.intervention_id) if row.intervention_id else None
        saved = db.get(SavedExplanation, row.saved_explanation_id) if row.saved_explanation_id else None
        payload = json.loads(saved.payload) if saved else {}
        status = "ready" if saved else "failed" if row.resolution_error else state.status if state else "waiting_for_context"
        point_id = row.intervention_id or payload.get("pointId")
        return {"id": row.id, "video_id": row.video_id, "intervention_id": point_id,
            "output_mode": row.output_mode, "timestamp_seconds": row.timestamp,
            "status": status, "error_code": row.resolution_error or (state.error_code if state else None),
            "retryable": row.resolution_error in {"saved_content_unavailable", "video_processing_busy", "preparation_unavailable"} or bool(state and state.status == "failed"
                and state.attempt_count < self.preparation.max_attempts and state.failure_reason not in PERMANENT_FAILURE_REASONS),
            "saved_explanation_id": row.saved_explanation_id, "heard": row.heard_at is not None,
            "dismissed": row.dismissed, "partial_seconds": row.partial_seconds,
            "satisfaction": row.satisfaction,
            "point": {"id": point_id, "timestamp": f"{int(row.timestamp // 60)}:{int(row.timestamp % 60):02d}",
                "title": "Requested explanation", "category": point.content_type if point else payload.get("category", "unknown"),
                "origin": point.origin if point else "manual"}}

    def get(self, owner, identity):
        with self.sessions() as db:
            row = db.get(ExplanationRequest, identity)
            if row is None or row.user_id != owner:
                raise HTTPException(404, "Explanation request not found.")
            return self._response(db, row)

    def list(self, owner, video, before=None):
        with self.sessions() as db:
            query = db.query(ExplanationRequest).filter_by(user_id=owner, video_id=video)
            if before:
                cursor = query.filter_by(id=before).one_or_none()
                if cursor is None:
                    raise HTTPException(404, "Explanation request page not found.")
                query = query.filter(or_(ExplanationRequest.created_at < cursor.created_at,
                    (ExplanationRequest.created_at == cursor.created_at) & (ExplanationRequest.id < cursor.id)))
            rows = query.order_by(ExplanationRequest.created_at.desc(), ExplanationRequest.id.desc()).limit(201).all()
            page = rows[:200]
            return {"items": [self._response(db, row) for row in reversed(page)],
                "next_cursor": page[-1].id if len(rows) > 200 else None}

    def progress(self, owner, identity, values):
        with self._lock, self.sessions() as db:
            row = db.get(ExplanationRequest, identity)
            if row is None or row.user_id != owner:
                raise HTTPException(404, "Explanation request not found.")
            if values.get("heard"):
                row.heard_at = row.heard_at or utcnow()
            if "partial_seconds" in values:
                row.partial_seconds = max(row.partial_seconds, values["partial_seconds"])
            if "satisfaction" in values:
                row.satisfaction = values["satisfaction"]
            if "dismissed" in values:
                row.dismissed = values["dismissed"]
            db.commit()

    def maintain(self):
        with self._lock:
            with self.sessions() as db:
                expired = db.query(LectureSessionInterest).filter_by(state="active").filter(
                    LectureSessionInterest.expires_at <= utcnow()).all()
                videos = {row.video_id for row in expired}
                for row in expired:
                    row.state = "expired"
                db.commit()
                requested_videos = {row[0] for row in db.query(ExplanationRequest.video_id).join(LectureSessionInterest,
                    ExplanationRequest.session_id == LectureSessionInterest.id).filter(
                    ExplanationRequest.saved_explanation_id.is_(None), ExplanationRequest.resolution_error.is_(None),
                    LectureSessionInterest.state.in_(("active", "continuing"))).distinct().all()}
            for video in requested_videos:
                self.source_ready(video)
            for video in videos:
                self._cancel_uninterested(video)
            with self.sessions() as db:
                continuing = db.query(LectureSessionInterest).filter_by(state="continuing").all()
                for row in continuing:
                    job = db.query(ProcessingJob).filter_by(video_id=row.video_id).one_or_none()
                    unfinished = db.query(ExplanationPreparation.id).filter(
                        ExplanationPreparation.video_id == row.video_id,
                        ExplanationPreparation.status.in_(("pending", "generating"))).first()
                    if not unfinished and (job is None or job.status in {"succeeded", "failed", "cancelled"}):
                        row.state = "released"
                db.commit()
