"""
Gemini text provider (Google Gen AI SDK, `google-genai`).

Configuration (environment variables):
    GEMINI_API_KEY          required, never logged
    GEMINI_MODEL            default: gemini-3.5-flash-lite
    GEMINI_TIMEOUT_SECONDS  default: 60

The SDK is imported lazily so the rest of the backend (including the Ollama
pipeline) works without it installed.
"""

import logging
import os
import time
from typing import Any, Optional

from .providers import (
    GenerationResult,
    GenerationSettings,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    TextGenerationProvider,
    TokenUsage,
)

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_ENV_VAR = "GEMINI_MODEL"
TIMEOUT_ENV_VAR = "GEMINI_TIMEOUT_SECONDS"

# Stable, low-cost model that Google's pricing page lists with a free tier.
# Free-tier availability and quotas can change; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 60.0


class GeminiProvider(TextGenerationProvider):
    """Text generation through the Gemini Developer API."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any = None,
    ):
        """
        Args:
            api_key: Gemini API key.
            model: Gemini model identifier.
            timeout_seconds: Per-request timeout.
            client: Pre-built SDK client (used by tests); skips SDK import.

        Raises:
            LLMConfigurationError: key or model missing, or bad timeout.
        """
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                f"{API_KEY_ENV_VAR} is not set. Create a key in Google AI Studio "
                f"and set the {API_KEY_ENV_VAR} environment variable.",
                provider=self.name,
            )
        if not model or not model.strip():
            raise LLMConfigurationError(
                f"{MODEL_ENV_VAR} must not be empty.", provider=self.name
            )
        if timeout_seconds <= 0:
            raise LLMConfigurationError(
                f"{TIMEOUT_ENV_VAR} must be greater than 0.", provider=self.name
            )

        self._api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_env(cls) -> "GeminiProvider":
        """Build a provider from environment variables."""
        raw_timeout = os.environ.get(TIMEOUT_ENV_VAR)
        if raw_timeout:
            try:
                timeout = float(raw_timeout)
            except ValueError:
                raise LLMConfigurationError(
                    f"{TIMEOUT_ENV_VAR} must be a number of seconds.",
                    provider=cls.name,
                )
        else:
            timeout = DEFAULT_TIMEOUT_SECONDS

        return cls(
            api_key=os.environ.get(API_KEY_ENV_VAR, ""),
            model=os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL,
            timeout_seconds=timeout,
        )

    def __repr__(self) -> str:
        # Deliberately excludes the API key.
        return f"GeminiProvider(model={self.model!r}, timeout_seconds={self.timeout_seconds})"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        settings: Optional[GenerationSettings] = None,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        settings = settings or GenerationSettings()
        client = self._get_client()
        config = self._build_config(settings)

        # Log sizes only: prompts may contain lecture content or user questions.
        logger.info(
            "Gemini request: model=%s prompt_chars=%d max_output_tokens=%d",
            self.model, len(prompt), settings.max_output_tokens,
        )

        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            error = self._map_exception(exc)
            logger.warning("Gemini request failed: %s (%s)", error.code, type(exc).__name__)
            raise error from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = self._extract_text(response)
        if not text:
            logger.warning("Gemini returned no text (model=%s)", self.model)
            raise LLMEmptyResponseError(
                "Gemini returned an empty response. The prompt may have been "
                "blocked by safety filters or the output limit was exhausted.",
                provider=self.name,
            )

        logger.info("Gemini response: model=%s latency_ms=%d chars=%d",
                    self.model, latency_ms, len(text))
        return GenerationResult(
            text=text, provider=self.name, model=self.model, latency_ms=latency_ms,
            usage=self._extract_usage(response),
        )

    # ------------------------------------------------------------------
    # SDK plumbing
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai",
                provider=self.name,
            ) from exc

        # HttpOptions.timeout is in milliseconds.
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )
        return self._client

    def _build_config(self, settings: GenerationSettings) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai",
                provider=self.name,
            ) from exc

        kwargs = {
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
        }
        if settings.system_instruction:
            kwargs["system_instruction"] = settings.system_instruction
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _extract_usage(response: Any) -> Optional[TokenUsage]:
        """Token counts from `usage_metadata`; never raises, never invents numbers."""
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return None

        def count(name: str) -> Optional[int]:
            value = getattr(metadata, name, None)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        usage = TokenUsage(
            prompt_tokens=count("prompt_token_count"),
            output_tokens=count("candidates_token_count"),
            thoughts_tokens=count("thoughts_token_count"),
            total_tokens=count("total_token_count"),
            cached_tokens=count("cached_content_token_count"),
        )
        return None if usage == TokenUsage() else usage

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            text = response.text
        except Exception:
            return ""
        return text.strip() if isinstance(text, str) else ""

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_exception(self, exc: Exception) -> LLMProviderError:
        """Translate an SDK or transport exception into an LLMProviderError."""
        if isinstance(exc, LLMProviderError):
            return exc

        # google.genai.errors.APIError (ClientError / ServerError) expose an
        # integer `code` (HTTP status) and a `message`.
        code = getattr(exc, "code", None)
        if isinstance(code, int) and type(exc).__module__.startswith("google.genai"):
            return self._map_api_error(code, str(getattr(exc, "message", "") or ""))

        names = {cls.__name__ for cls in type(exc).__mro__}
        if isinstance(exc, TimeoutError) or any("Timeout" in n for n in names):
            return LLMTimeoutError(
                f"Gemini did not respond within {self.timeout_seconds:g} seconds.",
                provider=self.name,
            )
        if isinstance(exc, ConnectionError) or names & {
            "ConnectError", "NetworkError", "TransportError", "RequestError",
        }:
            return LLMConnectionError(
                "Could not reach the Gemini API. Check the network connection.",
                provider=self.name,
            )

        return LLMProviderError(
            f"Unexpected Gemini error ({type(exc).__name__}).", provider=self.name
        )

    def _map_api_error(self, status: int, message: str) -> LLMProviderError:
        lowered = message.lower()
        kwargs = {"provider": self.name, "status_code": status}

        # Gemini reports an invalid key as HTTP 400 "API key not valid".
        if status in (401, 403) or "api key" in lowered:
            return LLMAuthenticationError(
                f"Gemini rejected the API key or permissions (HTTP {status}). "
                f"Check {API_KEY_ENV_VAR}.", **kwargs
            )
        if status == 429:
            return LLMRateLimitError(
                "Gemini quota or rate limit exceeded (HTTP 429). "
                "Free-tier limits are low; wait and retry.", **kwargs
            )
        if status in (408, 504):
            return LLMTimeoutError(
                f"Gemini timed out (HTTP {status}).", **kwargs
            )
        if status >= 500:
            return LLMUnavailableError(
                f"Gemini is temporarily unavailable (HTTP {status}).", **kwargs
            )
        return LLMRequestError(
            f"Gemini rejected the request (HTTP {status}).", **kwargs
        )
