"""Local account personal storage. No client-supplied ownership or file paths."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import uuid
from types import SimpleNamespace

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import (Explanation, Intervention, LibraryItem, MediaDeletion,
                        SavedExplanation, User, Video, VoiceNote, VoiceNoteExplanation)
from api.personal_storage import (MAX_RECORDING_BYTES, capture_lecture, drain_deletions,
    explanation_response, media_path, publish, snapshot, validate_recording)
from api.routers.learning_history import get_local_development_user
from api.security import enforce_allowed_origin


def local_transport(request: Request):
    # TestClient uses testclient; a deployed local API must not accept remote peers.
    if request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "Personal storage is available only on this computer.")


router = APIRouter(prefix="/me", tags=["personal-storage"],
                   dependencies=[Depends(enforce_allowed_origin), Depends(local_transport)])


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation_id: int


class NoteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=128)
    videoId: str = Field(min_length=1, max_length=128)
    lectureSeconds: float = Field(ge=0, le=172800)
    label: str = Field(default="", max_length=200)
    createdAt: int = Field(gt=0)
    silentWarning: bool = False


def owned(db, user, identity):
    row = db.query(SavedExplanation).filter_by(id=identity, user_id=user.id).one_or_none()
    if row is None:
        raise HTTPException(404, "Saved explanation not found.")
    return row


@router.put("/saved-library/{video_id}")
def subscribe(video_id: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Processed lecture not found.")
    existing = db.query(LibraryItem).filter_by(user_id=user.id, video_id=video_id).one_or_none()
    if not existing:
        try:
            with db.begin_nested():
                db.add(LibraryItem(user_id=user.id, video_id=video_id, preferred_mode="explanatory",
                    lecture_snapshot=json.dumps({"videoId": video.id, "videoTitle": video.title,
                        "videoUrl": video.source_url, "videoThumbnail": video.thumbnail_url})))
                db.flush()
        except IntegrityError:
            pass
    capture_lecture(db, user.id, video_id)
    db.commit()
    return {"videoId": video_id}


@router.get("/saved-library")
def library(db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    items = db.query(LibraryItem, Video).outerjoin(Video, Video.id == LibraryItem.video_id).filter(
        LibraryItem.user_id == user.id).order_by(LibraryItem.last_accessed_at.desc()).all()
    result = []
    for item, video in items:
        if video:
            capture_lecture(db, user.id, video.id)
        versions = db.query(SavedExplanation).filter_by(user_id=user.id, video_id=item.video_id).order_by(
            SavedExplanation.created_at.desc(), SavedExplanation.id).all()
        metadata = json.loads(item.lecture_snapshot) if item.lecture_snapshot else {}
        if video:
            metadata = {"videoId": video.id, "videoTitle": video.title, "videoUrl": video.source_url, "videoThumbnail": video.thumbnail_url}
            item.lecture_snapshot = json.dumps(metadata)
        result.append({**metadata, "videoId": item.video_id,
                       "savedAt": int(item.added_at.replace(tzinfo=timezone.utc).timestamp() * 1000),
                       "explanations": [explanation_response(row) for row in versions]})
    db.commit()
    return result


@router.post("/saved-explanations")
def save(body: SaveRequest, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    explanation = db.get(Explanation, body.explanation_id)
    if explanation is None:
        raise HTTPException(404, "Explanation not found.")
    row = snapshot(db, explanation, user.id)
    db.commit()
    db.refresh(row)
    return explanation_response(row)


@router.get("/saved-explanations")
def list_saved(kind: str = Query("recent", pattern="^(recent|bookmarks)$"),
               db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    query = db.query(SavedExplanation).filter_by(user_id=user.id)
    if kind == "bookmarks":
        query = query.filter(SavedExplanation.bookmarked_at.isnot(None)).order_by(SavedExplanation.bookmarked_at.desc())
    else:
        query = query.filter(SavedExplanation.viewed_at >= utcnow() - timedelta(days=30)).order_by(SavedExplanation.viewed_at.desc())
    return [explanation_response(row) for row in query.order_by(SavedExplanation.id).all()]


@router.get("/saved-explanations/{identity}")
def get_saved(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    return explanation_response(owned(db, user, identity))


@router.get("/saved-lectures/{video_id}")
def saved_lecture(video_id: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    return [explanation_response(row) for row in db.query(SavedExplanation).filter_by(
        user_id=user.id, video_id=video_id).order_by(SavedExplanation.created_at.desc(), SavedExplanation.id).all()]


@router.put("/saved-explanations/{identity}/viewed")
def viewed(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    row = owned(db, user, identity)
    row.viewed_at = utcnow()
    db.commit()
    return explanation_response(row)


@router.put("/saved-explanations/{identity}/bookmark")
def bookmark(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    row = owned(db, user, identity)
    row.bookmarked_at = row.bookmarked_at or utcnow()
    db.commit()
    return explanation_response(row)


@router.delete("/saved-explanations/{identity}/bookmark", status_code=204)
def unbookmark(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    owned(db, user, identity).bookmarked_at = None
    db.commit()
    return Response(status_code=204)


@router.get("/saved-explanations/{identity}/{asset}")
def asset(identity: str, asset: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    body = json.loads(owned(db, user, identity).payload)
    key = body.get({"audio": "audioKey", "frame": "frameKey"}.get(asset, ""))
    if not key or not media_path(key).is_file():
        raise HTTPException(404, "Saved media is unavailable.")
    return FileResponse(media_path(key), media_type="audio/mpeg" if asset == "audio" else None,
                        headers={"Cache-Control": "private, no-store"})


def note_response(db, row):
    body = json.loads(row.payload)
    links = db.query(SavedExplanation).join(VoiceNoteExplanation,
        VoiceNoteExplanation.explanation_id == SavedExplanation.id).filter(
            VoiceNoteExplanation.note_id == row.id, SavedExplanation.user_id == row.user_id).all()
    return {**body, "id": row.id, "operationId": row.operation_id, "durationMs": row.duration_ms,
            "mimeType": row.mime_type, "sizeBytes": row.size_bytes,
            "audioUrl": f"/me/voice-notes/{row.id}/audio",
            "explanations": [explanation_response(item) for item in links]}


@router.post("/voice-notes")
def upload(metadata: str = Form(..., max_length=4096), file: UploadFile = File(...),
           db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    try:
        body = NoteMetadata.model_validate_json(metadata)
    except ValueError as exc:
        raise HTTPException(422, "Invalid note metadata.") from exc
    video = db.get(Video, body.videoId)
    if video is None:
        preserved = db.query(SavedExplanation).filter_by(user_id=user.id, video_id=body.videoId).first()
        if preserved:
            details = json.loads(preserved.payload)
            video = SimpleNamespace(id=body.videoId, title=details["videoTitle"],
                source_url=details["videoUrl"], duration_seconds=details.get("durationSeconds"))
    if not video:
        raise HTTPException(404, "Processed lecture not found. Your local recording is retained.")
    if video.duration_seconds and body.lectureSeconds > video.duration_seconds:
        raise HTTPException(422, "Note timestamp is beyond this lecture.")
    data = file.file.read(MAX_RECORDING_BYTES + 1)
    fingerprint = hashlib.sha256(body.model_dump_json().encode() + data).hexdigest()
    existing = db.query(VoiceNote).filter_by(user_id=user.id, operation_id=body.id).one_or_none()
    if existing:
        if existing.fingerprint != fingerprint:
            raise HTTPException(409, "This recording ID already belongs to different content.")
        return note_response(db, existing)
    mime, suffix, duration = validate_recording(data)
    identity = str(uuid.uuid4())
    key = publish(data, f"users/{user.id}/voice_notes/{identity}", suffix)
    payload = {**body.model_dump(), "videoTitle": video.title, "videoUrl": video.source_url,
               "timestampedVideoUrl": f"https://www.youtube.com/watch?v={video.id}&t={int(body.lectureSeconds)}s"}
    row = VoiceNote(id=identity, user_id=user.id, operation_id=body.id, fingerprint=fingerprint,
        video_id=video.id, lecture_seconds=body.lectureSeconds, duration_ms=duration, media_key=key,
        mime_type=mime, size_bytes=len(data), payload=json.dumps(payload))
    try:
        db.add(row)
        db.flush()
        for explanation in db.query(Explanation).join(Intervention).filter(
            Intervention.video_id == video.id,
            Intervention.timestamp >= max(0, body.lectureSeconds - 60),
            Intervention.timestamp <= body.lectureSeconds + 60).all():
            try:
                saved = snapshot(db, explanation, user.id)
                db.merge(VoiceNoteExplanation(note_id=identity, explanation_id=saved.id))
            except HTTPException as error:
                if error.status_code != 409:
                    raise
        for saved in db.query(SavedExplanation).filter_by(user_id=user.id, video_id=video.id).all():
            if abs(json.loads(saved.payload)["lectureSeconds"] - body.lectureSeconds) <= 60:
                db.merge(VoiceNoteExplanation(note_id=identity, explanation_id=saved.id))
        db.commit()
    except IntegrityError:
        db.rollback()
        media_path(key).unlink(missing_ok=True)
        existing = db.query(VoiceNote).filter_by(user_id=user.id, operation_id=body.id).one_or_none()
        if existing and existing.fingerprint == fingerprint:
            return note_response(db, existing)
        raise HTTPException(409, "This recording ID already belongs to different content.")
    return note_response(db, row)


@router.get("/voice-notes")
def notes(video_id: str | None = None, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    drain_deletions(db)
    query = db.query(VoiceNote).filter_by(user_id=user.id)
    if video_id:
        query = query.filter_by(video_id=video_id)
    return [note_response(db, row) for row in query.order_by(VoiceNote.created_at.desc(), VoiceNote.id).all()]


@router.get("/voice-notes/{identity}/audio")
def recording(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    row = db.query(VoiceNote).filter_by(id=identity, user_id=user.id).one_or_none()
    if row is None or not media_path(row.media_key).is_file():
        raise HTTPException(404, "Recording not found.")
    return FileResponse(media_path(row.media_key), media_type=row.mime_type, headers={"Cache-Control": "private, no-store"})


@router.delete("/voice-notes/{identity}", status_code=204)
def delete_recording(identity: str, db: Session = Depends(get_db), user: User = Depends(get_local_development_user)):
    row = db.query(VoiceNote).filter_by(id=identity, user_id=user.id).one_or_none()
    if row:
        db.add(MediaDeletion(media_key=row.media_key))
        db.delete(row)
        db.commit()
        drain_deletions(db)
    return Response(status_code=204)
