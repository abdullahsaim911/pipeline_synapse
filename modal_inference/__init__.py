"""
SYNAPSE — Modal Remote Inference Package
Stage M1A: Standalone Qwen3-VL-8B-Instruct inference deployment on Modal.
"""

from modal_inference.config import (
    APP_NAME,
    AUTH_ENV_VAR,
    DEFAULT_GPU,
    MAX_IMAGE_BYTES,
    MAX_MODEL_LEN,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODAL_SECRET_NAME,
    VLLM_VERSION,
)

__all__ = [
    "APP_NAME",
    "AUTH_ENV_VAR",
    "DEFAULT_GPU",
    "MAX_IMAGE_BYTES",
    "MAX_MODEL_LEN",
    "MAX_OUTPUT_TOKENS",
    "MODEL_ID",
    "MODAL_SECRET_NAME",
    "VLLM_VERSION",
]

