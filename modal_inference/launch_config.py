"""Pure vLLM command construction for the Modal deployment.

Keeping command assembly independent of the Modal SDK makes the complete
server profile deterministic and offline-testable without importing Modal.
"""

from __future__ import annotations

from modal_inference.config import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_QUANTIZATION,
    MULTIMODAL_LIMITS_JSON,
    VLLM_GENERATION_CONFIG_JSON,
    VLLM_MM_PROCESSOR_KWARGS_JSON,
    VLLM_PORT,
)


def redact_sensitive_text(text: str, *secrets: str) -> str:
    """Remove exact runtime credentials from third-party diagnostics."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def build_vllm_serve_command(api_key: str) -> list[str]:
    """Return the pinned command without placing credentials in argv.

    vLLM reads ``VLLM_API_KEY`` from its environment. Keeping the secret out
    of argv prevents vLLM's startup argument diagnostics from exposing it.
    """

    if not api_key or not api_key.strip():
        raise ValueError("A non-empty API key is required to build the vLLM command.")
    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        MODEL_DTYPE,
        "--quantization",
        MODEL_QUANTIZATION,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--limit-mm-per-prompt",
        MULTIMODAL_LIMITS_JSON,
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--generation-config",
        "vllm",
        "--override-generation-config",
        VLLM_GENERATION_CONFIG_JSON,
        "--mm-processor-kwargs",
        VLLM_MM_PROCESSOR_KWARGS_JSON,
        "--trust-remote-code",
    ]
