"""
VLM Types — Lightweight, provider-neutral response and exception types.

This module deliberately avoids importing PyTorch, Transformers, or any
GPU-bound dependency so that it can be used by remote/Modal adapters
without pulling the local inference stack into the process.

Backward compatibility:
    from vlm_interface import VLMResponse          # still works
    from vlm_interface.vlm_types import VLMResponse  # preferred for lightweight use
"""

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Provider Selection
# ---------------------------------------------------------------------------
PROVIDER_ENV_VAR: Final[str] = "SYNAPSE_VLM_PROVIDER"
"""Environment variable that controls which VLM backend Synapse uses."""

PROVIDER_LOCAL: Final[str] = "local"
"""Use the local GGUF / Transformers backend (default)."""

PROVIDER_MODAL: Final[str] = "modal"
"""Use the remote Modal-hosted vLLM endpoint."""

VALID_PROVIDERS: Final[frozenset[str]] = frozenset({PROVIDER_LOCAL, PROVIDER_MODAL})
"""All recognised values for SYNAPSE_VLM_PROVIDER."""


@dataclass
class VLMResponse:
    """Response from VLM generation (local or remote)."""
    text: str
    model: str
    total_duration_ms: int
    prompt_eval_count: int
    eval_count: int


class VLMInterfaceError(Exception):
    """Base exception for VLM Interface errors."""
    pass


class VLMTimeoutError(VLMInterfaceError):
    """Timeout during VLM generation."""
    pass


class VLMAuthenticationError(VLMInterfaceError):
    """Authentication failure (invalid/missing credentials)."""
    pass


class VLMRemoteError(VLMInterfaceError):
    """Non-retryable error from a remote VLM endpoint."""
    pass

