"""Strict request and response contracts for local learning history."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ExplanationMode = Literal["brief", "explanatory", "detailed"]


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LectureOpenedRequest(StrictRequest):
    last_intervention_id: Optional[str] = None


class LectureActivityResponse(BaseModel):
    id: str
    video_id: str
    title: str
    youtube_url: str
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    processed_at: Optional[datetime] = None
    first_opened_at: datetime
    last_opened_at: datetime
    open_count: int
    last_intervention_id: Optional[str] = None


class LibraryUpsertRequest(StrictRequest):
    preferred_mode: ExplanationMode = "explanatory"


class LibraryItemResponse(BaseModel):
    id: str
    video_id: str
    title: str
    youtube_url: str
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    added_at: datetime
    last_accessed_at: datetime
    preferred_mode: ExplanationMode


class PlaybackProgressUpsertRequest(StrictRequest):
    position_seconds: float = Field(ge=0)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    completed: bool = False
    last_intervention_id: Optional[str] = None


class PlaybackProgressResponse(BaseModel):
    id: str
    video_id: str
    position_seconds: float
    duration_seconds: Optional[float] = None
    completed: bool
    last_intervention_id: Optional[str] = None
    updated_at: datetime


class RecentExplanationResponse(BaseModel):
    id: str
    explanation_id: Optional[int] = None
    video_id: str
    video_title: str
    youtube_url: str
    intervention_id: str
    intervention_timestamp: float
    output_mode: ExplanationMode
    viewed_at: datetime
    text_explanation: Optional[str] = None
    audio_url: Optional[str] = None
