"""
Provider-neutral text generation interface.

Defines the contract that future services (e.g. Ask a Question) depend on, the
error hierarchy providers raise, and a small factory that selects a provider by
name. Nothing here knows about a specific vendor SDK.

The existing OllamaClient is intentionally NOT wrapped here: its generate()
currently ignores per-call options, so an adapter could not honour
GenerationSettings. Existing explanation code keeps using OllamaClient directly.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

PROVIDER_ENV_VAR = "LLM_PROVIDER"
DEFAULT_PROVIDER = "gemini"


# ============================================================================
# Errors
# ============================================================================

class LLMProviderError(Exception):
    """
    Base class for text-provider failures.

    Messages are safe to show to callers: they never contain API keys or
    prompt text. `code` is a stable machine-readable identifier and
    `retryable` tells the caller whether trying again later may succeed.
    """

    code = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code  # upstream HTTP status, if any


class LLMConfigurationError(LLMProviderError):
    """Missing or invalid configuration (API key, model, SDK, provider name)."""
    code = "configuration_error"


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the credentials or permissions."""
    code = "authentication_error"


class LLMRateLimitError(LLMProviderError):
    """Quota or rate limit exceeded (typical on free tiers)."""
    code = "rate_limited"
    retryable = True


class LLMTimeoutError(LLMProviderError):
    """The provider did not answer within the configured timeout."""
    code = "timeout"
    retryable = True


class LLMConnectionError(LLMProviderError):
    """The provider could not be reached (DNS, network, TLS)."""
    code = "connection_error"
    retryable = True


class LLMUnavailableError(LLMProviderError):
    """The provider reported a server-side failure (5xx)."""
    code = "provider_unavailable"
    retryable = True


class LLMRequestError(LLMProviderError):
    """The provider rejected the request itself (bad parameters, too large)."""
    code = "invalid_request"


class LLMEmptyResponseError(LLMProviderError):
    """The provider answered but returned no usable text (e.g. blocked)."""
    code = "empty_response"


class LLMMalformedResponseError(LLMProviderError):
    """The model's reply broke the output contract (e.g. a malformed status marker)."""
    code = "malformed_response"
    retryable = True


# ============================================================================
# Interface
# ============================================================================

@dataclass(frozen=True)
class GenerationSettings:
    """Provider-neutral generation parameters."""

    temperature: float = 0.2
    max_output_tokens: int = 1024
    system_instruction: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the provider for one call (None when not reported)."""

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thoughts_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


@dataclass(frozen=True)
class GenerationResult:
    """Generated text plus minimal metadata (no prompt content)."""

    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Optional[TokenUsage] = None


class TextGenerationProvider(ABC):
    """A backend that turns a prompt into text."""

    name: str = "unknown"
    model: str = ""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        settings: Optional[GenerationSettings] = None,
    ) -> GenerationResult:
        """
        Generate text for `prompt`.

        Raises:
            ValueError: empty prompt.
            LLMProviderError: any provider-side failure (see subclasses).
        """


# ============================================================================
# Factory
# ============================================================================

def get_text_provider(name: Optional[str] = None) -> TextGenerationProvider:
    """
    Build the configured text provider.

    Args:
        name: Provider name. Defaults to $LLM_PROVIDER, then "gemini".

    Raises:
        LLMConfigurationError: unknown provider or incomplete configuration.
    """
    selected = (name or os.environ.get(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER)
    selected = selected.strip().lower()

    if selected == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider.from_env()

    raise LLMConfigurationError(
        f"Unknown LLM provider '{selected}'. Supported providers: gemini."
    )
