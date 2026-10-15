"""
LLM Interface Module (M5b)
"""

from .ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaResponse,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaServerError,
    OllamaClientError,
    FUSION_AVAILABLE
)
from .providers import (
    TextGenerationProvider,
    GenerationSettings,
    GenerationResult,
    TokenUsage,
    LLMProviderError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError,
    LLMUnavailableError,
    LLMRequestError,
    LLMEmptyResponseError,
    LLMMalformedResponseError,
    get_text_provider,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaResponse",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaServerError",
    "OllamaClientError",
    "FUSION_AVAILABLE",
    "TextGenerationProvider",
    "GenerationSettings",
    "GenerationResult",
    "TokenUsage",
    "LLMProviderError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMUnavailableError",
    "LLMRequestError",
    "LLMEmptyResponseError",
    "LLMMalformedResponseError",
    "get_text_provider"
]
