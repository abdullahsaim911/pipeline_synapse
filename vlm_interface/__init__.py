"""
VLM Interface Module (M5a)

VLMResponse and exception types are re-exported from the lightweight
vlm_types sub-module so that consumers that only need the data contract
(e.g. ModalVLMInterface) never pull in PyTorch/Transformers.
"""

# Lightweight, provider-neutral types — no GPU dependencies
from .vlm_types import (
    VLMResponse,
    VLMInterfaceError,
    VLMTimeoutError,
    VLMAuthenticationError,
    VLMRemoteError,
    PROVIDER_ENV_VAR,
    PROVIDER_LOCAL,
    PROVIDER_MODAL,
    VALID_PROVIDERS,
)

# The heavy local-inference class is imported lazily only when someone
# explicitly asks for it.  This keeps  `from vlm_interface import VLMResponse`
# fast and GPU-free.
def __getattr__(name):
    if name == "Qwen2VLInterface":
        from .vlm_interface import Qwen2VLInterface
        return Qwen2VLInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Qwen2VLInterface",
    "VLMResponse",
    "VLMInterfaceError",
    "VLMTimeoutError",
    "VLMAuthenticationError",
    "VLMRemoteError",
    "PROVIDER_ENV_VAR",
    "PROVIDER_LOCAL",
    "PROVIDER_MODAL",
    "VALID_PROVIDERS",
]
