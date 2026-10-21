"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


# ============================================================================
# Video Schemas
# ============================================================================

class VideoBase(BaseModel):
    """Base video schema."""
    title: str
    source_url: str
    duration_seconds: Optional[int] = None
    duration_formatted: Optional[str] = None
    total_keyframes: Optional[int] = None
    subject: Optional[str] = None
    thumbnail_url: Optional[str] = None


class VideoCreate(VideoBase):
    """Schema for creating a video."""
    pass


class VideoResponse(VideoBase):
    """Schema for video response."""
    id: str
    date_processed: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Transcript Schemas
# ============================================================================

class TranscriptBase(BaseModel):
    """Base transcript schema."""
    start_time: float
    end_time: float
    text: str


class TranscriptCreate(TranscriptBase):
    """Schema for creating a transcript segment."""
    video_id: str


class TranscriptResponse(TranscriptBase):
    """Schema for transcript response."""
    id: int
    video_id: str

    class Config:
        from_attributes = True


# ============================================================================
# Intervention Schemas
# ============================================================================

class InterventionBase(BaseModel):
    """Base intervention schema."""
    timestamp: float
    timestamp_formatted: Optional[str] = None
    frame_path: str
    content_type: Optional[str] = None
    complexity_score: Optional[float] = None
    clip_confidence: Optional[float] = None
    trigger_reason: Optional[str] = None
    confidence: Optional[str] = None
    transcript_context: Optional[str] = None
    is_bookmarked: bool = False


class InterventionCreate(InterventionBase):
    """Schema for creating an intervention."""
    video_id: str


class InterventionResponse(InterventionBase):
    """Schema for intervention response."""
    id: str
    video_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# API Request/Response Schemas
# ============================================================================

# Explanation schemas (for future Endpoint 3 implementation)
class ExplanationBase(BaseModel):
    """Base explanation schema."""
    text_explanation: str
    audio_file_path: Optional[str] = None
    output_mode: str = "explanatory"


class ExplanationCreate(ExplanationBase):
    """Schema for creating an explanation."""
    intervention_id: str
    vlm_snapshot: Optional[Dict[str, Any]] = None


class ExplanationResponse(ExplanationBase):
    """Schema for explanation response."""
    id: int
    intervention_id: str
    vlm_snapshot: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class GenerateExplanationRequest(BaseModel):
    """Request schema for generating explanation for intervention."""
    video_id: str = Field(..., description="YouTube video ID (for validation and path construction)")
    intervention_id: str
    output_mode: str = Field(default="explanatory", description="Output mode: brief, explanatory, detailed")


class GenerateExplanationResponse(BaseModel):
    """Response schema for generated explanation."""
    intervention_id: str
    text_explanation: str
    audio_file_path: Optional[str] = None
    text_file_path: Optional[str] = None
    vlm_snapshot_path: Optional[str] = None
    output_mode: str
    # vlm_snapshot removed - internal data only, not sent to frontend


class ProcessVideoRequest(BaseModel):
    """Request schema for processing a video."""
    youtube_url: str = Field(..., description="YouTube video URL")


class CategoryCounts(BaseModel):
    """Category-wise count of interventions."""
    text: int = 0
    equation: int = 0
    graph: int = 0
    circuit: int = 0
    diagram: int = 0
    code: int = 0
    handwritten_notes: int = 0
    biology: int = 0
    chemistry: int = 0
    physics: int = 0
    slide: int = 0
    table: int = 0
    unknown: int = 0


class ProcessVideoResponse(BaseModel):
    """Response schema for processed video (detection-only)."""
    video_id: str
    title: Optional[str] = None
    duration_seconds: Optional[int] = None
    duration_formatted: Optional[str] = None
    total_keyframes: Optional[int] = None
    total_interventions: int
    interventions: List[InterventionResponse]
    category_counts: CategoryCounts


class VideoMetaResponse(BaseModel):
    """Response schema for video metadata endpoint."""
    video_id: str
    title: Optional[str] = None
    duration_seconds: Optional[int] = None
    duration_formatted: Optional[str] = None
    thumbnail_url: Optional[str] = None
    exists: bool
    processed: bool


class BookmarkRequest(BaseModel):
    """Request schema for bookmarking an intervention."""
    bookmarked: bool


class BookmarkResponse(BaseModel):
    """Response schema for bookmark."""
    intervention_id: str
    is_bookmarked: bool


# ============================================================================
# Ask a Question
# ============================================================================

class AskRequest(BaseModel):
    """Request schema for asking a question about a lecture."""
    video_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{11}$", description="YouTube video ID")
    question: str = Field(..., min_length=1, max_length=500)
    current_time_seconds: Optional[float] = Field(
        default=None, ge=0, le=172800, allow_inf_nan=False,
        description="Current playback position in seconds"
    )
    intervention_id: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9-]{1,64}$",
        description="Intervention the user is currently viewing"
    )

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class AskSource(BaseModel):
    """Lecture material, located by backend-computed times, that was given to the model."""
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., ge=0)
    intervention_id: Optional[str] = None

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must not be before start_seconds")
        return self


class AskResponse(BaseModel):
    """Response schema for Ask a Question."""
    status: Literal["answered", "not_in_lecture", "insufficient_context"]
    scope: Literal["lecture", "lecture_with_moment"]
    answer_text: str = Field(..., min_length=1)
    sources: List[AskSource]

    @model_validator(mode="after")
    def _sources_match_status(self):
        if self.status == "answered" and not self.sources:
            raise ValueError("an answered response must include at least one source")
        if self.status != "answered" and self.sources:
            raise ValueError("a non-answered response must not include sources")
        return self
