"""
Quiz-owned structured-output provider.

The shared provider files (llm_interface/*) are hash-pinned by archival experiments
and must not change, so structured output is added here by SUBCLASSING:

  * QuizSettings extends GenerationSettings with an optional JSON schema.
  * QuizGeminiProvider extends GeminiProvider and overrides only _build_config to add
    response_mime_type / response_json_schema. Key handling, timeout, error mapping,
    usage reporting and logging are inherited unchanged.

Whether the configured model actually honours the schema is UNVERIFIED (no real call
has been made). The reply is parsed strictly and screened regardless of how it was
produced.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_interface.gemini_provider import GeminiProvider
from llm_interface.providers import (
    PROVIDER_ENV_VAR,
    GenerationSettings,
    LLMConfigurationError,
    TextGenerationProvider,
)


@dataclass(frozen=True)
class QuizSettings(GenerationSettings):
    """GenerationSettings plus an optional JSON schema for structured output."""

    response_json_schema: Optional[Dict[str, Any]] = None


class QuizGeminiProvider(GeminiProvider):
    """GeminiProvider that can request schema-constrained JSON."""

    def _build_config(self, settings: GenerationSettings) -> Any:
        schema = getattr(settings, "response_json_schema", None)
        if not schema:
            return super()._build_config(settings)
        try:
            from google.genai import types
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-genai' package is not installed. Run: pip install google-genai",
                provider=self.name,
            ) from exc

        kwargs = {
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }
        if settings.system_instruction:
            kwargs["system_instruction"] = settings.system_instruction
        return types.GenerateContentConfig(**kwargs)


def get_quiz_provider() -> TextGenerationProvider:
    """
    Build the quiz provider from the environment (GEMINI_API_KEY, GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS). Only Gemini is supported. Never logs the key.

    Raises:
        LLMConfigurationError: unsupported provider name, or incomplete configuration.
    """
    name = (os.environ.get(PROVIDER_ENV_VAR) or "gemini").strip().lower()
    if name != "gemini":
        raise LLMConfigurationError(
            f"Quiz generation supports only the gemini provider (got '{name}')."
        )
    return QuizGeminiProvider.from_env()
