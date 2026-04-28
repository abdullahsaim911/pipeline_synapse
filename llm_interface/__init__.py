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

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaResponse",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaServerError",
    "OllamaClientError",
    "FUSION_AVAILABLE"
]
