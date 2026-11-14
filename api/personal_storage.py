"""Durable snapshots and managed media for the local learning profile."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from api.database import PROJECT_ROOT
from api.models import (Explanation, ExplanationPreparation, Intervention, LibraryItem, MediaDeletion,
                        SavedExplanation, Video)

MAX_RECORDING_BYTES = 20 * 1024 * 1024
# MediaRecorder can finish an already-encoded final frame after its 5-minute stop.
MAX_RECORDING_DURATION_MS = 300250


def data_root():
    return (PROJECT_ROOT / "data").resolve()


def media_path(key):
    root = data_root()
    target = (root / key).resolve()
    if not target.is_relative_to(root) or target == root:
        raise HTTPException(400, "Invalid media key.")
    return target


def publish(data, prefix, suffix):
    digest = hashlib.sha256(data).hexdigest()
    key = f"{prefix}/{digest}{suffix}"
    target = media_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        return key
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return key


def preserve_path(stored, suffix):
    if not stored:
        return None
    root = data_root()
    source = Path(stored)
    candidates = [source] if source.is_absolute() else [PROJECT_ROOT / source, root / source]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return publish(resolved.read_bytes(), "saved-assets", suffix)
    return None


def snapshot(db, explanation, user_id, *, committed_generation=False):
    def check_preparation():
        state = db.query(ExplanationPreparation.status).filter_by(
            intervention_id=explanation.intervention_id, output_mode=explanation.output_mode).scalar()
        if state is not None and state != "ready" and not committed_generation:
            raise HTTPException(409, "Explanation preparation has not committed ready content yet.")

    check_preparation()
    source_text = explanation.text_explanation
    source_audio = explanation.audio_file_path
    point = db.get(Intervention, explanation.intervention_id)
    video = db.get(Video, point.video_id) if point else None
    if video is None:
        raise HTTPException(404, "Explanation lecture is unavailable.")
    audio = preserve_path(explanation.audio_file_path, ".mp3")
    # Only successfully prepared text + audio is a durable normal explanation.
    if not audio or not explanation.text_explanation.strip():
        raise HTTPException(409, "Explanation audio is not ready for saving.")
    frame = preserve_path(point.frame_path, Path(point.frame_path).suffix or ".jpg")
    title = " ".join((point.transcript_context or "").split()[:6]) or point.content_type or "Explanation"
    title = title[:1].upper() + title[1:]
    body = {
        "sourceExplanationId": explanation.id, "videoId": video.id,
        "pointId": point.id, "jobId": video.id, "mode": explanation.output_mode,
        "videoTitle": video.title, "videoUrl": video.source_url,
        "videoThumbnail": video.thumbnail_url, "videoDuration": video.duration_formatted,
        "durationSeconds": video.duration_seconds, "lectureSeconds": point.timestamp,
        "timestamp": point.timestamp_formatted, "category": "diagram" if point.content_type == "slide" else point.content_type,
        "title": title,
        "text": explanation.text_explanation, "transcriptContext": point.transcript_context,
        "audioKey": audio, "frameKey": frame,
        "snapshotKey": preserve_path(explanation.vlm_snapshot_path, ".json"),
        "snapshot": explanation.vlm_snapshot,
        "generatedAt": explanation.created_at.replace(tzinfo=timezone.utc).isoformat() if explanation.created_at else None,
    }
    # The cache publishes mode-specific files before its metadata transaction.
    # Recheck after copying so a concurrent preparation cannot pair new audio
    # with old cache text. The worker's post-commit hook is already fenced.
    check_preparation()
    current = db.query(Explanation.text_explanation, Explanation.audio_file_path).filter_by(id=explanation.id).one_or_none()
    if current is None or current != (source_text, source_audio):
        raise HTTPException(409, "Explanation changed during saving. Please retry.")
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    existing = db.query(SavedExplanation).filter_by(user_id=user_id, fingerprint=fingerprint).one_or_none()
    if existing:
        return existing
    row = SavedExplanation(user_id=user_id, video_id=video.id, intervention_id=point.id,
                           output_mode=explanation.output_mode, fingerprint=fingerprint,
                           payload=json.dumps(body))
    # A concurrent list/save can capture the same immutable result.
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        return db.query(SavedExplanation).filter_by(user_id=user_id, fingerprint=fingerprint).one()
    return row


def capture_lecture(db, user_id, video_id):
    rows = db.query(Explanation).join(Intervention).filter(
        Intervention.video_id == video_id, Intervention.origin == "detected").all()
    for explanation in rows:
        try:
            snapshot(db, explanation, user_id)
        except HTTPException as error:
            if error.status_code != 409:
                raise


def capture_subscribers(db, explanation):
    point = db.get(Intervention, explanation.intervention_id)
    if point and point.origin != "manual":
        for item in db.query(LibraryItem).filter_by(video_id=point.video_id).all():
            snapshot(db, explanation, item.user_id, committed_generation=True)


def explanation_response(row):
    body = json.loads(row.payload)
    # Internal inference evidence stays on the backend.
    body.pop("snapshot", None)
    body.pop("snapshotKey", None)
    body.pop("audioKey", None)
    frame = body.pop("frameKey", None)
    body.update(id=row.id, savedAt=int(row.created_at.replace(tzinfo=timezone.utc).timestamp() * 1000),
                viewedAt=int(row.viewed_at.replace(tzinfo=timezone.utc).timestamp() * 1000) if row.viewed_at else None,
                bookmarked=row.bookmarked_at is not None,
                audioUrl=f"/me/saved-explanations/{row.id}/audio",
                frameUrl=f"/me/saved-explanations/{row.id}/frame" if frame else None,
                timestampedVideoUrl=f"https://www.youtube.com/watch?v={row.video_id}&t={int(body['lectureSeconds'])}s")
    return body


def drain_deletions(db):
    for entry in db.query(MediaDeletion).all():
        try:
            media_path(entry.media_key).unlink(missing_ok=True)
        except OSError:
            continue  # Durable outbox remains for the next request/startup.
        db.delete(entry)
    db.commit()


def reconcile_media(db, *, grace_seconds=86400):
    """Remove old interrupted publications only inside the managed private areas."""
    import time
    from api.models import VoiceNote
    referenced = {row.media_key for row in db.query(VoiceNote).all()}
    for row in db.query(SavedExplanation).all():
        payload = json.loads(row.payload)
        referenced.update(key for key in [payload.get("audioKey"), payload.get("frameKey"), payload.get("snapshotKey")] if key)
    cutoff = time.time() - grace_seconds
    directories = [media_path("saved-assets")]
    users = media_path("users")
    if users.is_dir():
        directories.extend(path / "voice_notes" for path in users.iterdir() if path.is_dir() and not path.is_symlink())
    for directory in directories:
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_mtime >= cutoff:
                continue
            key = candidate.relative_to(data_root()).as_posix()
            if key not in referenced:
                try:
                    media_path(key).unlink(missing_ok=True)
                except OSError:
                    pass
    drain_deletions(db)


def validate_recording(data):
    """Decode bounded input with the already-installed FFmpeg; retain original bytes."""
    if not data or len(data) > MAX_RECORDING_BYTES:
        raise HTTPException(413, "Recording is empty or exceeds 20 MB.")
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        mime, suffix = "audio/wav", ".wav"
    elif data.startswith(b"\x1aE\xdf\xa3"):
        mime, suffix = "audio/webm", ".webm"
    elif data.startswith(b"OggS"):
        mime, suffix = "audio/ogg", ".ogg"
    elif data[4:8] == b"ftyp":
        mime, suffix = "audio/mp4", ".m4a"
    else:
        raise HTTPException(422, "Unsupported recording format.")
    import imageio_ffmpeg
    fd, temporary = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin",
                "-i", temporary, "-t", "301", "-vn", "-ac", "1", "-ar", "16000",
                "-f", "s16le", "pipe:1"], capture_output=True, timeout=30, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(503, "Recording validation is unavailable. Your draft can be retried.") from exc
        duration = round(len(result.stdout) / 32)
        if result.returncode or not 1 <= duration <= MAX_RECORDING_DURATION_MS:
            raise HTTPException(422, "Recording must be valid audio no longer than five minutes.")
        return mime, suffix, duration
    finally:
        Path(temporary).unlink(missing_ok=True)
