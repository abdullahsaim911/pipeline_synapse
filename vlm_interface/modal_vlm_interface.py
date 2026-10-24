"""
SYNAPSE — Modal VLM Interface Adapter (M2B-1)

A lightweight, reusable client that sends images + prompts to a remote
Modal-hosted vLLM endpoint (OpenAI-compatible /v1/chat/completions) and
returns a VLMResponse.

Design principles:
  - Zero local GPU / CUDA / Transformers / GGUF dependencies.
  - Drop-in compatible with the analyze() / cleanup() contract used by
    SnapshotEngine.
  - All educational prompts remain in SnapshotEngine; this adapter is
    prompt-agnostic.
  - Credentials and authorization tokens are never logged or included in
    exception messages.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from vlm_interface.vlm_types import (
    VLMAuthenticationError,
    VLMInterfaceError,
    VLMRemoteError,
    VLMResponse,
    VLMTimeoutError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by environment variables)
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT_SECONDS: float = 120.0
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_MAX_TOKENS: int = 1024
_DEFAULT_MODEL_ID: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"

# Image conversion limits
_MAX_IMAGE_DIMENSION: int = 1344
_JPEG_QUALITY: int = 85
_MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# HTTP status codes that must NOT be retried
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 422})

# HTTP status codes that SHOULD be retried (beyond timeouts)
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Base backoff multiplier (seconds).  Actual delay = _BACKOFF_BASE * 2^attempt
_BACKOFF_BASE: float = 1.0
_BACKOFF_MAX: float = 30.0


def _redact(value: str | None, keep: int = 4) -> str:
    """Return a redacted version of *value* safe for logging."""
    if not value:
        return "<not set>"
    if len(value) <= keep:
        return "****"
    return value[:keep] + "****"


class ModalVLMInterface:
    """
    Remote VLM adapter that proxies ``analyze()`` calls to a Modal-hosted
    vLLM OpenAI-compatible endpoint.

    Required environment variables (or constructor args):
        SYNAPSE_MODAL_URL        — Base HTTPS URL of the Modal deployment.
        SYNAPSE_MODAL_API_KEY    — Bearer token for authentication.

    Optional environment variables:
        SYNAPSE_MODAL_TIMEOUT_SECONDS  — Per-request timeout (default 120).
        SYNAPSE_MODAL_MAX_RETRIES      — Retry ceiling for transient errors (default 3).
        SYNAPSE_MODAL_MAX_TOKENS       — Maximum generation tokens (default 1024).
    """

    SUPPORTED_FORMATS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> None:
        # ----- endpoint -----
        self._endpoint_url = (
            endpoint_url or os.environ.get("SYNAPSE_MODAL_URL", "")
        ).rstrip("/")
        if not self._endpoint_url:
            raise VLMInterfaceError(
                "Modal endpoint URL is required.  Set SYNAPSE_MODAL_URL or "
                "pass endpoint_url= to ModalVLMInterface."
            )
        if not self._endpoint_url.startswith("https://"):
            raise VLMInterfaceError(
                "Modal endpoint URL must use HTTPS.  "
                f"Received scheme: {self._endpoint_url.split('://')[0] if '://' in self._endpoint_url else 'none'}"
            )

        # ----- authentication -----
        self._api_key = api_key or os.environ.get("SYNAPSE_MODAL_API_KEY", "")
        if not self._api_key:
            raise VLMAuthenticationError(
                "Modal API key is required.  Set SYNAPSE_MODAL_API_KEY or "
                "pass api_key= to ModalVLMInterface."
            )

        # ----- tuning knobs -----
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.environ.get("SYNAPSE_MODAL_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
        )
        self._max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("SYNAPSE_MODAL_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
        )
        self._max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.environ.get("SYNAPSE_MODAL_MAX_TOKENS", _DEFAULT_MAX_TOKENS))
        )
        self.model_name: str = model_id or _DEFAULT_MODEL_ID

        # Re-usable SSL context — default CA bundle, no custom certs needed.
        self._ssl_ctx = ssl.create_default_context()

        logger.info(
            "[ModalVLM] Adapter initialised — endpoint=%s  model=%s  "
            "timeout=%ss  max_retries=%d  max_tokens=%d  api_key=%s",
            self._endpoint_url,
            self.model_name,
            self._timeout,
            self._max_retries,
            self._max_tokens,
            _redact(self._api_key),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_path_or_pil: Union[str, Path, Image.Image],
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> VLMResponse:
        """
        Send *image_path_or_pil* and *prompt* to the remote Modal endpoint.

        Args:
            image_path_or_pil: Local file path (str/Path) or PIL Image.
            prompt: Text prompt to pass to the model.
            generation_params: Optional overrides (max_new_tokens, temperature, do_sample).

        Returns:
            VLMResponse with generated text and available metadata.
        """
        # 1.  Prepare image payload
        jpeg_bytes = self._prepare_image(image_path_or_pil)

        # 2.  Build OpenAI-compatible request body
        gen = generation_params or {}
        max_tokens = min(gen.get("max_new_tokens", self._max_tokens), 2048)
        temperature = gen.get("temperature", 0.2)

        payload = self._build_payload(jpeg_bytes, prompt, max_tokens, temperature)

        # 3.  Send with retry
        start_ms = time.perf_counter()
        response_data = self._send_with_retry(payload)
        total_duration_ms = int((time.perf_counter() - start_ms) * 1000)

        # 4.  Extract result
        text = self._extract_text(response_data)
        usage = response_data.get("usage", {})

        return VLMResponse(
            text=text,
            model=self.model_name,
            total_duration_ms=total_duration_ms,
            prompt_eval_count=usage.get("prompt_tokens", 0),
            eval_count=usage.get("completion_tokens", 0),
        )

    def cleanup(self) -> None:
        """No-op.  The remote endpoint manages its own lifecycle."""
        logger.debug("[ModalVLM] cleanup() called — no-op for remote adapter.")

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _prepare_image(
        self, source: Union[str, Path, Image.Image]
    ) -> bytes:
        """Load, validate, resize and convert the image to bounded RGB JPEG bytes."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            suffix = path.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format '{suffix}'.  "
                    f"Supported: {self.SUPPORTED_FORMATS}"
                )
            image = Image.open(path)
        elif isinstance(source, Image.Image):
            image = source
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        # Convert to RGB (drops alpha, handles palette / grayscale)
        image = image.convert("RGB")

        # Resize if any dimension exceeds the cap
        if max(image.size) > _MAX_IMAGE_DIMENSION:
            image.thumbnail((_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION))

        # Encode as JPEG
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        jpeg_bytes = buf.getvalue()

        if len(jpeg_bytes) > _MAX_IMAGE_BYTES:
            raise VLMInterfaceError(
                f"Encoded image exceeds {_MAX_IMAGE_BYTES // (1024*1024)} MB limit "
                f"({len(jpeg_bytes) / (1024*1024):.2f} MB)."
            )

        return jpeg_bytes

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> bytes:
        """Build and serialize the OpenAI-compatible multimodal JSON payload."""
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        return json.dumps(body).encode("utf-8")

    # ------------------------------------------------------------------
    # HTTP transport with bounded retry / backoff
    # ------------------------------------------------------------------

    def _send_with_retry(self, payload: bytes) -> dict[str, Any]:
        """
        POST *payload* to ``/v1/chat/completions`` with bounded
        exponential-backoff retry for transient failures.
        """
        url = f"{self._endpoint_url}/v1/chat/completions"
        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                return self._http_post(url, payload)

            except VLMAuthenticationError:
                raise  # never retry auth failures

            except VLMRemoteError:
                raise  # never retry non-retryable server errors

            except (VLMTimeoutError, VLMInterfaceError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                    logger.warning(
                        "[ModalVLM] Attempt %d/%d failed (%s).  "
                        "Retrying in %.1fs …",
                        attempt + 1,
                        1 + self._max_retries,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[ModalVLM] All %d attempts exhausted.",
                        1 + self._max_retries,
                    )

        # All attempts failed
        raise VLMInterfaceError(
            f"Remote inference failed after {1 + self._max_retries} attempt(s)."
        ) from last_exc

    def _http_post(self, url: str, payload: bytes) -> dict[str, Any]:
        """
        Low-level synchronous HTTPS POST.

        Raises:
            VLMAuthenticationError  — 401 / 403
            VLMRemoteError          — other non-retryable HTTP errors
            VLMTimeoutError         — socket or urllib timeout
            VLMInterfaceError       — retryable server errors (429, 5xx)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "Synapse-ModalVLM/1.0",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout, context=self._ssl_ctx
            ) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as exc:
            status = exc.code
            # Read error body but redact anything that might echo credentials
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            safe_msg = self._sanitize_error(err_body, status)

            if status in (401, 403):
                raise VLMAuthenticationError(
                    f"Authentication failed (HTTP {status}).  "
                    "Check SYNAPSE_MODAL_API_KEY."
                ) from exc

            if status in _NON_RETRYABLE_STATUS_CODES:
                raise VLMRemoteError(
                    f"Non-retryable remote error (HTTP {status}): {safe_msg}"
                ) from exc

            if status in _RETRYABLE_STATUS_CODES:
                raise VLMInterfaceError(
                    f"Retryable remote error (HTTP {status}): {safe_msg}"
                ) from exc

            # Unknown status — treat as non-retryable
            raise VLMRemoteError(
                f"Unexpected HTTP {status}: {safe_msg}"
            ) from exc

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise VLMTimeoutError(
                f"Connection/timeout error: {type(exc).__name__}"
            ) from exc

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """
        Extract ``choices[0].message.content`` from an OpenAI-compatible
        response, raising on missing/empty structures.
        """
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise VLMRemoteError(
                "Remote response missing 'choices' array or array is empty."
            )

        first = choices[0]
        message = first.get("message")
        if not message or not isinstance(message, dict):
            raise VLMRemoteError(
                "Remote response choices[0] missing 'message' object."
            )

        content = message.get("content")
        if content is None:
            raise VLMRemoteError(
                "Remote response choices[0].message.content is null/missing."
            )

        return str(content).strip()

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    def _sanitize_error(self, body: str, status: int) -> str:
        """Strip any token / credential echoes from an error body."""
        safe = body
        if self._api_key and self._api_key in safe:
            safe = safe.replace(self._api_key, "[REDACTED]")
        # Truncate very long error bodies
        if len(safe) > 500:
            safe = safe[:500] + " …[truncated]"
        return safe

