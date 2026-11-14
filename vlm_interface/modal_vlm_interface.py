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

from modal_inference.config import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_MODEL_LEN,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
)
from stage_tracing import trace_stage
from visual_evidence.compact_schema import COMPACT_OBSERVATION_JSON_SCHEMA

from vlm_interface.vlm_types import (
    VLMAuthenticationError,
    VLMContextBudgetError,
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
_DEFAULT_MAX_TOKENS: int = DEFAULT_STRUCTURED_MAX_TOKENS
_DEFAULT_MODEL_ID: str = MODEL_ID

# Image conversion limits
_MAX_IMAGE_DIMENSION: int = MAX_IMAGE_DIMENSION
_JPEG_QUALITY: int = 85
_MAX_IMAGE_BYTES: int = MAX_IMAGE_BYTES

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
        SYNAPSE_MODAL_MAX_TOKENS       — Maximum generation tokens (default 8192).
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
        context_budget_estimator: Any | None = None,
        require_exact_context_budget: bool | None = None,
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
        configured_max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.environ.get("SYNAPSE_MODAL_MAX_TOKENS", _DEFAULT_MAX_TOKENS))
        )
        self._max_tokens = self._validate_output_tokens(
            configured_max_tokens, "configured max_tokens"
        )
        self.model_name: str = model_id or _DEFAULT_MODEL_ID

        self._context_budget_estimator = context_budget_estimator
        if require_exact_context_budget is None:
            raw_required = os.environ.get(
                "SYNAPSE_VLM_REQUIRE_EXACT_CONTEXT_BUDGET", "1"
            )
            require_exact_context_budget = raw_required.strip().lower() in {
                "1", "true", "yes"
            }
        self._require_exact_context_budget = require_exact_context_budget
        self.last_context_budget: dict[str, Any] | None = None

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
        max_tokens = self._validate_output_tokens(
            gen.get("max_new_tokens", self._max_tokens), "requested max_new_tokens"
        )
        temperature = gen.get("temperature", DEFAULT_TEMPERATURE)

        # Use the exact image sent to vLLM, not the pre-resize source image.
        with Image.open(io.BytesIO(jpeg_bytes)) as encoded_image:
            budget_image = encoded_image.convert("RGB")
        budget_parts = gen.get("_context_budget_parts") or {}
        requested_max_tokens = max_tokens
        self.last_context_budget = None
        try:
            if self._context_budget_estimator is None:
                budget_report = self._native_context_budget(jpeg_bytes, prompt, max_tokens, budget_image.size)
            else:
                budget_report = self._context_budget_estimator.estimate(
                    image=budget_image,
                    prompt=prompt,
                    requested_output_tokens=max_tokens,
                    common_prompt=str(budget_parts.get("common_prompt") or ""),
                    category_directives=str(budget_parts.get("category_directives") or ""),
                )
            self.last_context_budget = budget_report.to_dict()
        except VLMContextBudgetError:
            report = getattr(self._context_budget_estimator, "last_report", None)
            if report is not None:
                self.last_context_budget = report.to_dict()
            # Exact accounting can safely reduce only the impossible unused
            # tail of a request. The configured capability remains available
            # whenever prompt + image tokens leave enough context for it.
            input_tokens = (
                self.last_context_budget.get("input_tokens")
                if self.last_context_budget else None
            )
            if input_tokens is None:
                raise
            max_tokens = MAX_MODEL_LEN - int(input_tokens) - 32
            if max_tokens < 512:
                raise
        if (
            self._require_exact_context_budget
            and not self.last_context_budget.get("exact", False)
        ):
            raise VLMContextBudgetError(
                "Exact Qwen3-VL context accounting is required but the official "
                "processor is unavailable."
            )

        self.last_context_budget["configured_output_tokens"] = requested_max_tokens
        self.last_context_budget["effective_output_tokens"] = max_tokens
        self.last_context_budget["safety_reserve_tokens"] = 32
        self.last_context_budget["remaining_context_tokens"] = (
            MAX_MODEL_LEN
            - int(self.last_context_budget.get("input_tokens") or 0)
            - max_tokens
        )
        self.last_context_budget["fits"] = True

        payload = self._build_payload(jpeg_bytes, prompt, max_tokens, temperature,
                                      response_schema=gen.get("_response_schema"))

        # 3.  Send with retry
        start_ms = time.perf_counter()
        trace_stage(
            "modal_request", "dispatch", request_bytes=len(payload),
            configured_output_token_limit=requested_max_tokens,
            output_token_limit=max_tokens,
        )
        response_data = self._send_with_retry(payload)
        total_duration_ms = int((time.perf_counter() - start_ms) * 1000)

        # 4.  Extract result
        text = self._extract_text(response_data)
        finish_reason = self._extract_finish_reason(response_data)
        usage = response_data.get("usage", {})
        trace_stage(
            "modal_request",
            "response",
            started_at=start_ms,
            finish_reason=finish_reason if finish_reason is not None else "missing",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model_name,
        )

        return VLMResponse(
            text=text,
            model=self.model_name,
            total_duration_ms=total_duration_ms,
            prompt_eval_count=usage.get("prompt_tokens", 0),
            eval_count=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
        )

    def cleanup(self) -> None:
        """No-op.  The remote endpoint manages its own lifecycle."""
        logger.debug("[ModalVLM] cleanup() called — no-op for remote adapter.")

    @staticmethod
    def _validate_output_tokens(value: Any, field_name: str) -> int:
        """Reject invalid limits instead of silently clamping requests."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer.")
        if not 1 <= value <= MAX_OUTPUT_TOKENS:
            raise ValueError(
                f"{field_name} must be between 1 and {MAX_OUTPUT_TOKENS}; "
                f"received {value}."
            )
        return value

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

        if max(image.size) > _MAX_IMAGE_DIMENSION:
            image.thumbnail((_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

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

    @staticmethod
    def _chat_messages(jpeg_bytes: bytes, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii"), "detail": "auto",
            }}, {"type": "text", "text": prompt},
        ]}]

    def _native_context_budget(self, jpeg_bytes: bytes, prompt: str, output_tokens: int, image_size: tuple[int, int]):
        """Use the serving processor; never load Transformers/PyTorch locally."""
        from vlm_interface.qwen_context_budget import ContextBudgetReport

        payload = json.dumps({"model": self.model_name, "messages": self._chat_messages(jpeg_bytes, prompt),
                              "add_generation_prompt": True, "add_special_tokens": False}).encode("utf-8")
        response = self._send_with_retry(payload, route="/tokenize")
        count = response.get("count")
        if type(count) is not int or count < 1:
            raise VLMContextBudgetError("Serving processor returned an invalid multimodal token count")
        if response.get("max_model_len") != MAX_MODEL_LEN:
            raise VLMContextBudgetError("Serving model context does not match the configured context contract")
        remaining = MAX_MODEL_LEN - count - output_tokens
        report = ContextBudgetReport(
            model_id=self.model_name, max_model_len=MAX_MODEL_LEN, requested_output_tokens=output_tokens,
            exact=True, processor_available=True, input_tokens=count,
            remaining_context_tokens=remaining, fits=remaining >= 32, image_size=image_size,
            image_token_method="Serving vLLM /tokenize multimodal processor (total input count)",
        )
        self.last_context_budget = report.to_dict()
        if not report.fits:
            raise VLMContextBudgetError("Multimodal input and requested output exceed the context budget")
        return report

    def _build_payload(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
        response_schema: Optional[dict] = None,
    ) -> bytes:
        """Build and serialize the OpenAI-compatible multimodal JSON payload."""
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._chat_messages(jpeg_bytes, prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Prevent overview record loops without penalizing repeated exact
            # characters in focused mathematical/code transcriptions.
            "repetition_penalty": 1.05 if response_schema is None else 1.0,
            "stream": False,
            # vLLM 0.19 structured outputs constrain decoding to this compact
            # contract instead of merely asking the model to produce JSON.
            "structured_outputs": {"json": response_schema or COMPACT_OBSERVATION_JSON_SCHEMA},
        }

        return json.dumps(body).encode("utf-8")

    # ------------------------------------------------------------------
    # HTTP transport with bounded retry / backoff
    # ------------------------------------------------------------------

    def _send_with_retry(self, payload: bytes, *, route: str = "/v1/chat/completions") -> dict[str, Any]:
        """
        POST *payload* to ``/v1/chat/completions`` with bounded
        exponential-backoff retry for transient failures.
        """
        url = f"{self._endpoint_url}{route}"
        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            attempt_started = time.monotonic()
            try:
                trace_stage(
                    "modal_attempt", "dispatch", request_attempt=attempt + 1,
                    max_attempts=1 + self._max_retries,
                )
                result = self._http_post(url, payload)
                trace_stage(
                    "modal_attempt", "response", started_at=attempt_started,
                    request_attempt=attempt + 1, status="succeeded",
                )
                return result

            except VLMAuthenticationError:
                trace_stage(
                    "modal_attempt", "response", started_at=attempt_started,
                    request_attempt=attempt + 1, status="authentication_failed",
                )
                raise  # never retry auth failures

            except VLMRemoteError:
                trace_stage(
                    "modal_attempt", "response", started_at=attempt_started,
                    request_attempt=attempt + 1, status="permanent_failure",
                )
                raise  # never retry non-retryable server errors

            except (VLMTimeoutError, VLMInterfaceError) as exc:
                trace_stage(
                    "modal_attempt", "response", started_at=attempt_started,
                    request_attempt=attempt + 1, status="transient_failure",
                    error_type=type(exc).__name__, level=logging.WARNING,
                )
                last_exc = exc
                if attempt < self._max_retries:
                    delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                    logger.debug(
                        "[ModalVLM] Attempt %d/%d failed (%s).  "
                        "Retrying in %.1fs …",
                        attempt + 1,
                        1 + self._max_retries,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.debug(
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

    @staticmethod
    def _extract_finish_reason(data: dict[str, Any]) -> str | None:
        """Return validated OpenAI-compatible completion metadata.

        Older compatible endpoints may omit the field or return null. A
        present value must be a non-empty string; malformed metadata is a
        remote protocol error rather than an implicit successful completion.
        """

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise VLMRemoteError(
                "Remote response missing 'choices' array or array is empty."
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise VLMRemoteError("Remote response choices[0] must be an object.")
        finish_reason = first.get("finish_reason")
        if finish_reason is None:
            return None
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise VLMRemoteError(
                "Remote response choices[0].finish_reason must be a non-empty string or null."
            )
        return finish_reason.strip()

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

