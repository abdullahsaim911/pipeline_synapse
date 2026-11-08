"""
Request and response contract for the voice endpoints.

These models are the only thing the frontend and backend share. They know
nothing about the action catalog; ``voice.catalog`` validates action names and
arguments, so the two stay independently testable.
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Screen = Literal[
    "home", "subject", "timeline", "explanation",
    "bookmarks", "library", "notes", "quiz",
]

# What is audibly playing right now. "unknown" when the frontend cannot tell.
Playing = Literal["lecture", "explanation", "none", "unknown"]

ItemKind = Literal["result", "moment", "bookmark", "library_item", "note", "subject"]

# What the frontend should do with an interpreted utterance.
Decision = Literal[
    "execute",       # confident: run the action
    "confirm",       # plausible or risky: ask a yes/no question first
    "clarify",       # two candidates are close: ask which one
    "ask_question",  # not a command: send to the lecture Q&A
    "unsupported",   # understood speech, but no supported action fits
    "no_speech",     # nothing usable was heard
]

MAX_UTTERANCE_CHARS = 300
MAX_CONTEXT_ITEMS = 60
MAX_LABEL_CHARS = 160

_ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


class ContextItem(BaseModel):
    """One thing on screen the user may refer to ("the second one", "the graph")."""
    ref: str = Field(..., min_length=1, max_length=64)
    kind: ItemKind
    label: str = Field(..., min_length=1, max_length=MAX_LABEL_CHARS)
    position: int = Field(..., ge=1, le=500, description="1-based position in its list")


class VoiceContext(BaseModel):
    """Snapshot of the app that the frontend sends with every utterance."""
    screen: Screen
    video_id: Optional[str] = None
    current_moment_ref: Optional[str] = Field(default=None, max_length=64)
    playing: Playing = "unknown"
    presentation_mode: Optional[Literal["accessibility", "study"]] = None
    lecture_seconds: Optional[float] = Field(default=None, ge=0, le=172800, allow_inf_nan=False)
    items: List[ContextItem] = Field(default_factory=list, max_length=MAX_CONTEXT_ITEMS)
    # Action ids the frontend can execute on this screen right now.
    available_actions: List[str] = Field(default_factory=list, max_length=80)
    # Set while a yes/no question is open: the action waiting for an answer.
    pending_confirmation: Optional[str] = None
    last_action: Optional[str] = None
    language_hint: Optional[str] = None

    @field_validator("video_id")
    @classmethod
    def _video_id(cls, value):
        if value is not None and not _VIDEO_ID_RE.match(value):
            raise ValueError("video_id must be an 11-character YouTube id")
        return value

    @field_validator("available_actions")
    @classmethod
    def _action_ids(cls, value):
        for action_id in value:
            if not _ACTION_ID_RE.match(action_id):
                raise ValueError(f"invalid action id: {action_id!r}")
        return value

    @field_validator("pending_confirmation", "last_action")
    @classmethod
    def _optional_action_id(cls, value):
        if value is not None and not _ACTION_ID_RE.match(value):
            raise ValueError(f"invalid action id: {value!r}")
        return value

    @field_validator("language_hint")
    @classmethod
    def _language(cls, value):
        if value is not None and not _LANG_RE.match(value):
            raise ValueError("language_hint must look like 'en' or 'pt-BR'")
        return value

    @model_validator(mode="after")
    def _unique_item_refs(self):
        refs = [item.ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("context item refs must be unique")
        return self


class TranscribeResponse(BaseModel):
    """Result of speech recognition. ``no_speech`` is true when nothing usable was heard."""
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = Field(default=None, ge=0, le=1)
    duration_seconds: float = Field(..., ge=0)
    no_speech: bool = False


class InterpretRequest(BaseModel):
    """Text of one utterance plus the app context it was spoken in."""
    text: str = Field(..., min_length=1, max_length=MAX_UTTERANCE_CHARS)
    language: Optional[str] = None
    context: VoiceContext

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("language")
    @classmethod
    def _language(cls, value):
        if value is not None and not _LANG_RE.match(value):
            raise ValueError("language must look like 'en' or 'pt-BR'")
        return value


class ActionCandidate(BaseModel):
    action: str
    score: float = Field(..., ge=0, le=1)
    args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def _action_id(cls, value):
        if not _ACTION_ID_RE.match(value):
            raise ValueError(f"invalid action id: {value!r}")
        return value


class InterpretResponse(BaseModel):
    """What the frontend should do. Only ``execute`` may be run without asking."""
    decision: Decision
    action: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=1)
    candidates: List[ActionCandidate] = Field(default_factory=list, max_length=3)
    # Key into the per-language feedback catalog (module 6).
    message_key: str = Field(..., min_length=1, max_length=64)
    language: Optional[str] = None

    @field_validator("action")
    @classmethod
    def _action_id(cls, value):
        if value is not None and not _ACTION_ID_RE.match(value):
            raise ValueError(f"invalid action id: {value!r}")
        return value

    @model_validator(mode="after")
    def _decision_matches_fields(self):
        needs_action = self.decision in ("execute", "confirm")
        if needs_action and not self.action:
            raise ValueError(f"decision '{self.decision}' requires an action")
        if self.decision in ("unsupported", "no_speech") and self.action:
            raise ValueError(f"decision '{self.decision}' must not carry an action")
        if self.decision == "clarify" and len(self.candidates) < 2:
            raise ValueError("decision 'clarify' requires at least two candidates")
        if self.decision == "ask_question":
            question = self.args.get("question")
            if self.action != "ask_question" or not isinstance(question, str) or not question.strip():
                raise ValueError("decision 'ask_question' requires action 'ask_question' with a question")
        return self
