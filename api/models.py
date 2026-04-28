"""
SQLAlchemy models for Synapse database.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


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
