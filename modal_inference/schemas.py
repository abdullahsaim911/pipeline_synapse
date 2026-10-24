"""
SYNAPSE — Modal Inference Request Schemas & Validation
Stage M1A: Bounded multimodal schemas, image payload verification,
and OpenAI-compatible request structures for Qwen3-VL.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

from modal_inference.config import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_PROMPT,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    SUPPORTED_IMAGE_MIMES,
)

# ---------------------------------------------------------------------------
# Magic Byte Signatures for Image Format Detection
# ---------------------------------------------------------------------------
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_MARKER = b"WEBP"


def detect_image_mime(image_bytes: bytes) -> str | None:
    """
    Detect the MIME type of an image from its magic byte header.
    Returns MIME string (e.g. 'image/jpeg') or None if unrecognized.
    """
    if len(image_bytes) < 12:
        return None

    if image_bytes.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if image_bytes.startswith(_PNG_MAGIC):
        return "image/png"
    if image_bytes.startswith(_WEBP_RIFF) and image_bytes[8:12] == _WEBP_MARKER:
        return "image/webp"

    return None


def validate_image_payload(
    raw_bytes: bytes,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[bool, str, str | None]:
    """
    Validate that image bytes do not exceed size bounds and have a valid format.
    Returns (is_valid, message, detected_mime).
    """
    byte_count = len(raw_bytes)
    if byte_count == 0:
        return False, "Image payload is empty (0 bytes).", None

    if byte_count > max_bytes:
        mb = byte_count / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        return (
            False,
            f"Image payload size ({mb:.2f} MB) exceeds maximum allowed size ({max_mb:.2f} MB).",
            None,
        )

    mime = detect_image_mime(raw_bytes)
    if mime is None:
        return (
            False,
            "Unsupported or corrupted image header. Must be valid JPEG, PNG, or WEBP.",
            None,
        )

    if mime not in SUPPORTED_IMAGE_MIMES:
        return (
            False,
            f"Image MIME '{mime}' is not in supported types: {SUPPORTED_IMAGE_MIMES}.",
            mime,
        )

    return True, "Valid image payload.", mime


# Regex for parsing RFC 2397 Data URIs
_DATA_URI_PATTERN = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9\+\-\.]+);base64,(?P<data>[A-Za-z0-9+/=]+)$",
    re.DOTALL,
)


def validate_and_parse_data_uri(
    data_uri: str,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[bool, str, bytes | None, str | None]:
    """
    Validate and decode a base64 Data URI string.
    Returns (is_valid, message, decoded_bytes, detected_mime).
    """
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        return False, "String is not a valid Data URI (must start with 'data:').", None, None

    match = _DATA_URI_PATTERN.match(data_uri.strip())
    if not match:
        return False, "Malformed Data URI format. Expected 'data:<mime>;base64,<data>'.", None, None

    declared_mime = match.group("mime").lower()
    b64_string = match.group("data")

    try:
        raw_bytes = base64.b64decode(b64_string, validate=True)
    except Exception as exc:
        return False, f"Failed to base64-decode image payload: {exc}", None, None

    is_valid, msg, detected_mime = validate_image_payload(raw_bytes, max_bytes=max_bytes)
    if not is_valid:
        return False, msg, None, None

    # Verify declared mime matches detected magic bytes
    if detected_mime and declared_mime not in (detected_mime, "image/jpg" if detected_mime == "image/jpeg" else ""):
        return (
            False,
            f"Declared MIME '{declared_mime}' contradicts detected image format '{detected_mime}'.",
            None,
            None,
        )

    return True, "Valid image Data URI.", raw_bytes, detected_mime


def clamp_generation_tokens(
    requested_tokens: int | None,
    max_allowed: int = MAX_OUTPUT_TOKENS,
    default_tokens: int = 1024,
) -> int:
    """
    Ensure requested output tokens stay strictly within bounds [1, max_allowed].
    """
    if requested_tokens is None:
        return min(default_tokens, max_allowed)
    if requested_tokens < 1:
        return 1
    return min(requested_tokens, max_allowed)


def validate_auth_header(
    auth_header: str | None,
    expected_token: str | None,
) -> tuple[bool, str]:
    """
    Validate an incoming HTTP Authorization header against the expected Bearer token.
    Returns (is_authorized, reason_message).
    """
    if not expected_token:
        return False, "Server authentication token is not configured."

    if not auth_header:
        return False, "Missing 'Authorization' header in request."

    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False, "Malformed Authorization header. Format must be 'Bearer <token>'."

    presented_token = parts[1].strip()
    if presented_token != expected_token.strip():
        return False, "Invalid authentication token."

    return True, "Authorized."


# ---------------------------------------------------------------------------
# OpenAI-Compatible Multimodal Request Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ImageUrlPart:
    url: str
    detail: str = "auto"

    def to_dict(self) -> dict[str, str]:
        return {"url": self.url, "detail": self.detail}


@dataclass
class ContentPart:
    type: str  # "text" | "image_url"
    text: str | None = None
    image_url: ImageUrlPart | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        elif self.type == "image_url" and self.image_url is not None:
            return {"type": "image_url", "image_url": self.image_url.to_dict()}
        raise ValueError(f"Invalid ContentPart configuration: {self}")


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str | list[ContentPart]

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            serialized_content: Any = self.content
        else:
            serialized_content = [part.to_dict() for part in self.content]
        return {"role": self.role, "content": serialized_content}


@dataclass
class ChatCompletionRequest:
    """
    Standard OpenAI-compatible chat completion payload with multimodal content.
    """
    model: str = MODEL_ID
    messages: list[ChatMessage] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False

    def validate(self) -> tuple[bool, str]:
        """
        Validate all messages, bounding token limits and inspecting image parts.
        """
        if not self.messages:
            return False, "Request messages list cannot be empty."

        image_count = 0
        for msg_idx, msg in enumerate(self.messages):
            if msg.role not in ("system", "user", "assistant"):
                return False, f"Message {msg_idx} has invalid role '{msg.role}'."

            if isinstance(msg.content, list):
                for part_idx, part in enumerate(msg.content):
                    if part.type == "image_url":
                        image_count += 1
                        if not part.image_url or not part.image_url.url:
                            return False, f"Message {msg_idx} part {part_idx} has empty image_url."
                        url = part.image_url.url
                        if url.startswith("data:"):
                            is_valid, err, _, _ = validate_and_parse_data_uri(url)
                            if not is_valid:
                                return False, f"Message {msg_idx} image {image_count} invalid: {err}"
                    elif part.type != "text":
                        return False, f"Message {msg_idx} part {part_idx} unknown type '{part.type}'."

        if image_count > MAX_IMAGES_PER_PROMPT:
            return False, (
                f"Request contains {image_count} images; maximum is "
                f"{MAX_IMAGES_PER_PROMPT}."
            )

        self.max_tokens = clamp_generation_tokens(self.max_tokens)
        return True, "Valid request."

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
        }

    @classmethod
    def from_image_and_prompt(
        cls,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        model: str = MODEL_ID,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> ChatCompletionRequest:
        """
        Convenience builder to construct a validated multimodal request from raw bytes and prompt.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

        content_parts = [
            ContentPart(type="text", text=prompt),
            ContentPart(type="image_url", image_url=ImageUrlPart(url=data_uri)),
        ]

        msg = ChatMessage(role="user", content=content_parts)
        req = cls(
            model=model,
            messages=[msg],
            max_tokens=clamp_generation_tokens(max_tokens),
            temperature=temperature,
        )
        return req


# ---------------------------------------------------------------------------
# Optional Pydantic Integration (available when pydantic is installed)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field

    class PydanticImageUrl(BaseModel):
        url: str
        detail: str = "auto"

    class PydanticContentPart(BaseModel):
        type: str
        text: str | None = None
        image_url: PydanticImageUrl | None = None

    class PydanticChatMessage(BaseModel):
        role: str
        content: str | list[PydanticContentPart]

    class PydanticChatCompletionRequest(BaseModel):
        model: str = MODEL_ID
        messages: list[PydanticChatMessage]
        max_tokens: int = Field(default=1024, ge=1, le=MAX_OUTPUT_TOKENS)
        temperature: float = Field(default=0.2, ge=0.0, le=2.0)
        top_p: float = Field(default=0.95, ge=0.0, le=1.0)
        stream: bool = False

except ImportError:
    PydanticChatCompletionRequest = None  # type: ignore

