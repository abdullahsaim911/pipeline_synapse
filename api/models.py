"""
SQLAlchemy models for Synapse database.
"""

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


LOCAL_DEVELOPMENT_USER_ID = "00000000-0000-4000-8000-000000000001"
LOCAL_DEVELOPMENT_PROFILE_KEY = "local-development"


class User(Base):
    """Application identity; authentication is intentionally deferred."""
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("profile_key", name="uq_users_profile_key"),)

    id = Column(
        Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    profile_key = Column(String(64), nullable=False)
    display_name = Column(String(120), nullable=False)
    is_local = Column(Boolean, nullable=False, default=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    preferences = relationship(
        "UserPreference",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    lecture_activity = relationship("LectureActivity", cascade="all, delete-orphan")
    library_items = relationship("LibraryItem", cascade="all, delete-orphan")
    playback_progress = relationship("PlaybackProgress", cascade="all, delete-orphan")
    recent_explanations = relationship("RecentExplanation", cascade="all, delete-orphan")


class UserPreference(Base):
    """One preference record per user."""
    __tablename__ = "user_preferences"
    __table_args__ = (
        CheckConstraint("theme IN ('paper', 'dark')", name="ck_user_preferences_theme"),
        CheckConstraint(
            "presentation_mode IN ('accessibility', 'study')",
            name="ck_user_preferences_presentation_mode",
        ),
        CheckConstraint(
            "explanation_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_user_preferences_explanation_mode",
        ),
        CheckConstraint(
            "playback_rate >= 0.5 AND playback_rate <= 3.0",
            name="ck_user_preferences_playback_rate",
        ),
        CheckConstraint(
            "text_scale >= 0.75 AND text_scale <= 2.0",
            name="ck_user_preferences_text_scale",
        ),
    )

    user_id = Column(
        Uuid(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    theme = Column(String(32), nullable=False, default="paper", server_default="paper")
    presentation_mode = Column(
        String(32), nullable=False, default="accessibility", server_default="accessibility"
    )
    explanation_mode = Column(
        String(32), nullable=False, default="explanatory", server_default="explanatory"
    )
    playback_rate = Column(Float, nullable=False, default=1.0, server_default="1.0")
    text_scale = Column(Float, nullable=False, default=1.0, server_default="1.0")
    high_contrast = Column(Boolean, nullable=False, default=False, server_default="0")
    reduced_motion = Column(Boolean, nullable=True)
    locale = Column(String(16), nullable=False, default="en", server_default="en")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="preferences")


class Video(Base):
    """Video metadata table."""
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)  # YouTube video ID
    title = Column(String, nullable=False)
    duration_seconds = Column(Integer)
    duration_formatted = Column(String)
    total_keyframes = Column(Integer)
    subject = Column(String)
    source_url = Column(String, nullable=False)
    date_processed = Column(DateTime, server_default=func.now())
    thumbnail_url = Column(String)  # YouTube thumbnail URL

    # Relationships
    interventions = relationship("Intervention", back_populates="video")
    transcripts = relationship("Transcript", back_populates="video")


class ProcessingJob(Base):
    """Durable, shared coordination state for one YouTube video."""
    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_processing_jobs_video_id"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_processing_jobs_status",
        ),
        CheckConstraint(
            "stage IN ('queued', 'processing', 'persisting', "
            "'preparing_explanations', 'complete', 'failed')",
            name="ck_processing_jobs_stage",
        ),
        CheckConstraint(
            "requested_output_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_processing_jobs_output_mode",
        ),
    )

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, nullable=False)
    source_url = Column(String, nullable=False)
    requested_output_mode = Column(
        String(32), nullable=False, default="explanatory", server_default="explanatory"
    )
    status = Column(String(16), nullable=False, default="queued", server_default="queued", index=True)
    stage = Column(String(32), nullable=False, default="queued", server_default="queued", index=True)
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    error_code = Column(String(64))
    error_message = Column(String(255))
    lease_owner = Column(String(64))
    lease_expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))


class ProcessingJobRequestedMode(Base):
    """Every explanation mode requested by callers sharing a processing job."""
    __tablename__ = "processing_job_requested_modes"
    __table_args__ = (
        CheckConstraint(
            "output_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_processing_job_requested_modes_output_mode",
        ),
    )

    job_id = Column(
        Uuid(as_uuid=False),
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    output_mode = Column(String(32), primary_key=True)
    requested_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Intervention(Base):
    """Intervention points table."""
    __tablename__ = "interventions"

    id = Column(String, primary_key=True, index=True)  # UUID
    video_id = Column(String, ForeignKey("videos.id"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False, index=True)  # Float timestamp (seconds)
    timestamp_formatted = Column(String)  # e.g., "02:35"
    frame_path = Column(String, nullable=False)  # Relative path to keyframe
    content_type = Column(String)  # "graph", "chemistry", "equation", etc.
    complexity_score = Column(Float)
    clip_confidence = Column(Float)
    trigger_reason = Column(String)  # "Deictic + High Complexity"
    confidence = Column(String)  # "high", "medium", "low"
    transcript_context = Column(Text)
    is_bookmarked = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    video = relationship("Video", back_populates="interventions")
    explanations = relationship("Explanation", back_populates="intervention")


class Explanation(Base):
    """Generated explanations table."""
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    intervention_id = Column(String, ForeignKey("interventions.id"), nullable=False, index=True)
    text_explanation = Column(Text, nullable=False)
    audio_file_path = Column(String)  # Relative path to generated MP3
    text_file_path = Column(String)  # Relative path to generated TXT file
    vlm_snapshot_path = Column(String)  # Relative path to VLM snapshot JSON file
    output_mode = Column(String)  # "brief", "explanatory", "detailed"
    vlm_snapshot = Column(Text)  # JSON string (SQLite doesn't have JSONB, use TEXT)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    intervention = relationship("Intervention", back_populates="explanations")


class ExplanationPreparation(Base):
    """Durable preparation state for one intervention and output mode."""
    __tablename__ = "explanation_preparations"
    __table_args__ = (
        UniqueConstraint(
            "video_id", "intervention_id", "output_mode",
            name="uq_explanation_preparation_key",
        ),
        CheckConstraint(
            "output_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_explanation_preparation_mode",
        ),
        CheckConstraint(
            "status IN ('pending', 'generating', 'ready', 'failed')",
            name="ck_explanation_preparation_status",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False, index=True)
    intervention_id = Column(
        String, ForeignKey("interventions.id"), nullable=False, index=True
    )
    output_mode = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="pending", index=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    error_code = Column(String)
    lease_owner = Column(String)
    lease_started_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    ready_at = Column(DateTime)
    failed_at = Column(DateTime)


class LectureActivity(Base):
    """One recent-lecture activity row per user and shared video."""
    __tablename__ = "lecture_activity"
    __table_args__ = (
        UniqueConstraint("user_id", "video_id", name="uq_lecture_activity_user_video"),
        Index("ix_lecture_activity_user_last_opened", "user_id", "last_opened_at"),
    )

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    video_id = Column(
        String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    first_opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    open_count = Column(Integer, nullable=False, default=1, server_default="1")
    last_intervention_id = Column(
        String, ForeignKey("interventions.id", ondelete="SET NULL"), nullable=True
    )


class LibraryItem(Base):
    """A durable user-owned reference to a shared processed lecture."""
    __tablename__ = "library_items"
    __table_args__ = (
        UniqueConstraint("user_id", "video_id", name="uq_library_items_user_video"),
        CheckConstraint(
            "preferred_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_library_items_preferred_mode",
        ),
        Index("ix_library_items_user_last_accessed", "user_id", "last_accessed_at"),
    )

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    video_id = Column(
        String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    added_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    preferred_mode = Column(
        String(32), nullable=False, default="explanatory", server_default="explanatory"
    )


class PlaybackProgress(Base):
    """Latest playback position for one user and shared video."""
    __tablename__ = "playback_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "video_id", name="uq_playback_progress_user_video"),
        CheckConstraint(
            "position_seconds >= 0", name="ck_playback_progress_nonnegative_position"
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_playback_progress_nonnegative_duration",
        ),
        Index("ix_playback_progress_user_updated", "user_id", "updated_at"),
    )

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    video_id = Column(
        String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position_seconds = Column(Float, nullable=False, default=0.0, server_default="0")
    duration_seconds = Column(Float)
    completed = Column(Boolean, nullable=False, default=False, server_default="0")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RecentExplanation(Base):
    """Rolling history of shared explanations viewed by one user."""
    __tablename__ = "recent_explanations"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "explanation_id", name="uq_recent_explanations_user_explanation"
        ),
        CheckConstraint(
            "output_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_recent_explanations_output_mode",
        ),
        Index("ix_recent_explanations_user_viewed", "user_id", "viewed_at"),
        Index("ix_recent_explanations_user_video", "user_id", "video_id"),
    )

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    video_id = Column(
        String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    intervention_id = Column(
        String, ForeignKey("interventions.id", ondelete="CASCADE"), nullable=False
    )
    explanation_id = Column(
        Integer, ForeignKey("explanations.id", ondelete="SET NULL"), nullable=True
    )
    output_mode = Column(String(32), nullable=False)
    viewed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Transcript(Base):
    """Transcript segments table."""
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False, index=True)
    start_time = Column(Float, nullable=False, index=True)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)

    # Relationships
    video = relationship("Video", back_populates="transcripts")
