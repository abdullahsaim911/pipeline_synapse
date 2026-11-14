"""Pinned, text-only L4 serving profile; independent of the Qwen deployment."""

MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"
MODEL_REVISION = "29439f81c2be264d8d393273f99e7db9c0961120"
MAX_MODEL_LEN = 12288
MAX_OUTPUT_TOKENS = 4096
VLLM_VERSION = "0.19.1"


def build_serve_command():
    return [
        "vllm", "serve", MODEL_ID, "--revision", MODEL_REVISION,
        "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16",
        "--tokenizer-mode", "mistral", "--config-format", "mistral",
        "--load-format", "mistral", "--quantization", "fp8",
        "--language-model-only", "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "1", "--gpu-memory-utilization", "0.90",
        "--enforce-eager", "--generation-config", "vllm",
    ]
