"""Local-development APIs for personal lecture and explanation history."""

from datetime import datetime, timedelta, timezone
import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath, Query, Response
from sqlalchemy.orm import Session

from api.database import get_db
from api.history_schemas import (
    LectureActivityResponse,
    LectureOpenedRequest,
    LibraryItemResponse,
    LibraryUpsertRequest,
    PlaybackProgressResponse,
    PlaybackProgressUpsertRequest,
    RecentExplanationResponse,
)
from api.models import (
    Explanation,
    Intervention,
    LectureActivity,
    LibraryItem,
    LOCAL_DEVELOPMENT_PROFILE_KEY,
    LOCAL_DEVELOPMENT_USER_ID,
    PlaybackProgress,
    RecentExplanation,
    User,
    Video,
)
from api.security import enforce_allowed_origin


router = APIRouter(prefix="/me", tags=["learning-history"])
RETENTION_DAYS = 30
LOCAL_ENVIRONMENTS = {"", "dev", "development", "local", "test"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_local_development_user(db: Session = Depends(get_db)) -> User:
    """Resolve the sole backend-controlled profile; never trust a client user ID."""
    environment = os.getenv("SYNAPSE_ENV", "development").strip().lower()
    if environment not in LOCAL_ENVIRONMENTS:
        raise HTTPException(
            status_code=404,
            detail="Personal learning history is available only in local development.",
        )
    user = db.query(User).filter(
        User.id == LOCAL_DEVELOPMENT_USER_ID,
        User.profile_key == LOCAL_DEVELOPMENT_PROFILE_KEY,
        User.is_local.is_(True),
    ).one_or_none()
    if user is None:
        raise HTTPException(status_code=503, detail="Local development profile is unavailable.")
    return user


def _video(db: Session, video_id: str) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return video


def _intervention(db: Session, video_id: str, intervention_id: str | None) -> Intervention | None:
    if intervention_id is None:
        return None
    intervention = db.get(Intervention, intervention_id)
    if intervention is None or intervention.video_id != video_id:
        raise HTTPException(status_code=404, detail="Intervention not found for this video.")
    return intervention


def _purge_expired_history(db: Session, user_id: str) -> None:
    cutoff = _utcnow() - timedelta(days=RETENTION_DAYS)
    activity_count = db.query(LectureActivity).filter(
        LectureActivity.user_id == user_id,
        LectureActivity.last_opened_at < cutoff,
    ).delete(synchronize_session=False)
    explanation_count = db.query(RecentExplanation).filter(
        RecentExplanation.user_id == user_id,
        RecentExplanation.viewed_at < cutoff,
    ).delete(synchronize_session=False)
    if activity_count or explanation_count:
        db.commit()


def _touch_activity(
    db: Session,
    user_id: str,
    video_id: str,
    *,
    intervention_id: str | None = None,
    count_open: bool = False,
) -> LectureActivity:
    now = _utcnow()
    activity = db.query(LectureActivity).filter(
        LectureActivity.user_id == user_id,
        LectureActivity.video_id == video_id,
    ).one_or_none()
    if activity is None:
        activity = LectureActivity(
            user_id=user_id,
            video_id=video_id,
            first_opened_at=now,
            last_opened_at=now,
            open_count=1,
            last_intervention_id=intervention_id,
        )
        db.add(activity)
    else:
        activity.last_opened_at = now
        if count_open:
            activity.open_count += 1
        if intervention_id is not None:
            activity.last_intervention_id = intervention_id
    return activity


def _lecture_response(activity: LectureActivity, video: Video) -> LectureActivityResponse:
    return LectureActivityResponse(
        id=str(activity.id),
        video_id=video.id,
        title=video.title,
        youtube_url=video.source_url,
        thumbnail_url=video.thumbnail_url,
        duration_seconds=video.duration_seconds,
        processed_at=video.date_processed,
        first_opened_at=activity.first_opened_at,
        last_opened_at=activity.last_opened_at,
        open_count=activity.open_count,
        last_intervention_id=activity.last_intervention_id,
    )


@router.get("/lectures/recent", response_model=list[LectureActivityResponse])
def list_recent_lectures(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    _purge_expired_history(db, str(user.id))
    rows = db.query(LectureActivity, Video).join(
        Video, Video.id == LectureActivity.video_id
    ).filter(
        LectureActivity.user_id == user.id
    ).order_by(LectureActivity.last_opened_at.desc()).limit(limit).all()
    return [_lecture_response(activity, video) for activity, video in rows]


@router.put(
    "/lectures/{video_id}/opened",
    response_model=LectureActivityResponse,
    dependencies=[Depends(enforce_allowed_origin)],
)
def record_lecture_opened(
    body: LectureOpenedRequest,
    video_id: str = ApiPath(..., min_length=1, max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    video = _video(db, video_id)
    intervention = _intervention(db, video_id, body.last_intervention_id)
    activity = _touch_activity(
        db,
        str(user.id),
        video_id,
        intervention_id=intervention.id if intervention else None,
        count_open=True,
    )
    db.commit()
    db.refresh(activity)
    return _lecture_response(activity, video)


@router.delete(
    "/lectures/{video_id}/recent",
    status_code=204,
    dependencies=[Depends(enforce_allowed_origin)],
)
def delete_recent_lecture(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    db.query(LectureActivity).filter(
        LectureActivity.user_id == user.id,
        LectureActivity.video_id == video_id,
    ).delete(synchronize_session=False)
    db.commit()
    return Response(status_code=204)


def _library_response(item: LibraryItem, video: Video) -> LibraryItemResponse:
    return LibraryItemResponse(
        id=str(item.id),
        video_id=video.id,
        title=video.title,
        youtube_url=video.source_url,
        thumbnail_url=video.thumbnail_url,
        duration_seconds=video.duration_seconds,
        added_at=item.added_at,
        last_accessed_at=item.last_accessed_at,
        preferred_mode=item.preferred_mode,
    )


@router.get("/library", response_model=list[LibraryItemResponse])
def list_library(
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    rows = db.query(LibraryItem, Video).join(Video, Video.id == LibraryItem.video_id).filter(
        LibraryItem.user_id == user.id
    ).order_by(LibraryItem.last_accessed_at.desc()).all()
    return [_library_response(item, video) for item, video in rows]


@router.put(
    "/library/{video_id}",
    response_model=LibraryItemResponse,
    dependencies=[Depends(enforce_allowed_origin)],
)
def save_library_item(
    body: LibraryUpsertRequest,
    video_id: str = ApiPath(..., min_length=1, max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    video = _video(db, video_id)
    now = _utcnow()
    item = db.query(LibraryItem).filter(
        LibraryItem.user_id == user.id,
        LibraryItem.video_id == video_id,
    ).one_or_none()
    if item is None:
        item = LibraryItem(
            user_id=user.id,
            video_id=video_id,
            added_at=now,
            last_accessed_at=now,
            preferred_mode=body.preferred_mode,
        )
        db.add(item)
    else:
        item.last_accessed_at = now
        item.preferred_mode = body.preferred_mode
    db.commit()
    db.refresh(item)
    return _library_response(item, video)


@router.delete(
    "/library/{video_id}",
    status_code=204,
    dependencies=[Depends(enforce_allowed_origin)],
)
def delete_library_item(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    db.query(LibraryItem).filter(
        LibraryItem.user_id == user.id,
        LibraryItem.video_id == video_id,
    ).delete(synchronize_session=False)
    db.commit()
    return Response(status_code=204)


def _progress_response(
    progress: PlaybackProgress,
    activity: LectureActivity | None,
) -> PlaybackProgressResponse:
    return PlaybackProgressResponse(
        id=str(progress.id),
        video_id=progress.video_id,
        position_seconds=progress.position_seconds,
        duration_seconds=progress.duration_seconds,
        completed=progress.completed,
        last_intervention_id=activity.last_intervention_id if activity else None,
        updated_at=progress.updated_at,
    )


@router.get("/playback/{video_id}", response_model=PlaybackProgressResponse)
def get_playback_progress(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    progress = db.query(PlaybackProgress).filter(
        PlaybackProgress.user_id == user.id,
        PlaybackProgress.video_id == video_id,
    ).one_or_none()
    if progress is None:
        raise HTTPException(status_code=404, detail="Playback progress not found.")
    activity = db.query(LectureActivity).filter(
        LectureActivity.user_id == user.id,
        LectureActivity.video_id == video_id,
    ).one_or_none()
    return _progress_response(progress, activity)


@router.put(
    "/playback/{video_id}",
    response_model=PlaybackProgressResponse,
    dependencies=[Depends(enforce_allowed_origin)],
)
def update_playback_progress(
    body: PlaybackProgressUpsertRequest,
    video_id: str = ApiPath(..., min_length=1, max_length=128),
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    _video(db, video_id)
    intervention = _intervention(db, video_id, body.last_intervention_id)
    now = _utcnow()
    progress = db.query(PlaybackProgress).filter(
        PlaybackProgress.user_id == user.id,
        PlaybackProgress.video_id == video_id,
    ).one_or_none()
    if progress is None:
        progress = PlaybackProgress(user_id=user.id, video_id=video_id)
        db.add(progress)
    progress.position_seconds = body.position_seconds
    progress.duration_seconds = body.duration_seconds
    progress.completed = body.completed
    progress.updated_at = now
    activity = _touch_activity(
        db,
        str(user.id),
        video_id,
        intervention_id=intervention.id if intervention else None,
    )
    db.commit()
    db.refresh(progress)
    db.refresh(activity)
    return _progress_response(progress, activity)


@router.delete(
    "/playback/{video_id}",
    status_code=204,
    dependencies=[Depends(enforce_allowed_origin)],
)
def delete_playback_progress(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    db.query(PlaybackProgress).filter(
        PlaybackProgress.user_id == user.id,
        PlaybackProgress.video_id == video_id,
    ).delete(synchronize_session=False)
    db.commit()
    return Response(status_code=204)


def _recent_explanation_response(
    row: RecentExplanation,
    video: Video,
    intervention: Intervention,
    explanation: Explanation | None,
) -> RecentExplanationResponse:
    audio_url = None
    if explanation is not None and explanation.audio_file_path:
        audio_url = (
            f"/video/intervention/{quote(intervention.id, safe='')}/audio"
            f"?output_mode={quote(row.output_mode, safe='')}"
        )
    return RecentExplanationResponse(
        id=str(row.id),
        explanation_id=row.explanation_id,
        video_id=row.video_id,
        video_title=video.title,
        youtube_url=video.source_url,
        intervention_id=row.intervention_id,
        intervention_timestamp=intervention.timestamp,
        output_mode=row.output_mode,
        viewed_at=row.viewed_at,
        text_explanation=explanation.text_explanation if explanation else None,
        audio_url=audio_url,
    )


@router.get("/explanations/recent", response_model=list[RecentExplanationResponse])
def list_recent_explanations(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    _purge_expired_history(db, str(user.id))
    rows = db.query(RecentExplanation, Video, Intervention, Explanation).join(
        Video, Video.id == RecentExplanation.video_id
    ).join(
        Intervention, Intervention.id == RecentExplanation.intervention_id
    ).outerjoin(
        Explanation, Explanation.id == RecentExplanation.explanation_id
    ).filter(
        RecentExplanation.user_id == user.id
    ).order_by(RecentExplanation.viewed_at.desc()).limit(limit).all()
    return [
        _recent_explanation_response(history, video, intervention, explanation)
        for history, video, intervention, explanation in rows
    ]


@router.put(
    "/explanations/{explanation_id}/viewed",
    response_model=RecentExplanationResponse,
    dependencies=[Depends(enforce_allowed_origin)],
)
def record_explanation_viewed(
    explanation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    explanation = db.get(Explanation, explanation_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Explanation not found.")
    intervention = db.get(Intervention, explanation.intervention_id)
    if intervention is None:
        raise HTTPException(status_code=404, detail="Explanation intervention not found.")
    video = _video(db, intervention.video_id)
    now = _utcnow()
    history = db.query(RecentExplanation).filter(
        RecentExplanation.user_id == user.id,
        RecentExplanation.explanation_id == explanation.id,
    ).one_or_none()
    if history is None:
        history = RecentExplanation(
            user_id=user.id,
            video_id=video.id,
            intervention_id=intervention.id,
            explanation_id=explanation.id,
            output_mode=explanation.output_mode,
            viewed_at=now,
        )
        db.add(history)
    else:
        history.viewed_at = now
    _touch_activity(
        db,
        str(user.id),
        video.id,
        intervention_id=intervention.id,
    )
    db.commit()
    db.refresh(history)
    return _recent_explanation_response(history, video, intervention, explanation)


@router.delete(
    "/explanations/recent/{history_id}",
    status_code=204,
    dependencies=[Depends(enforce_allowed_origin)],
)
def delete_recent_explanation(
    history_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_local_development_user),
):
    db.query(RecentExplanation).filter(
        RecentExplanation.id == history_id,
        RecentExplanation.user_id == user.id,
    ).delete(synchronize_session=False)
    db.commit()
    return Response(status_code=204)
