"""
SYNAPSE — Modal Remote Inference Configuration for Qwen3-VL
Stage M1A: Configuration constants and runtime boundaries.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Application & Model Metadata
# ---------------------------------------------------------------------------
APP_NAME: Final[str] = "synapse-qwen3-vl"
MODEL_ID: Final[str] = "Qwen/Qwen3-VL-8B-Instruct-FP8"
MODEL_PRECISION: Final[str] = "FP8 W8A8 (E4M3 weights, dynamic activations)"
MODEL_DTYPE: Final[str] = "bfloat16"
MODEL_QUANTIZATION: Final[str] = "fp8"
VLLM_VERSION: Final[str] = "0.19.1"
VLLM_PORT: Final[int] = 8000

# ---------------------------------------------------------------------------
# Persistent Cache Volumes
# ---------------------------------------------------------------------------
# Mount points inside the remote Modal container
HF_CACHE_DIR: Final[str] = "/root/.cache/huggingface"
VLLM_CACHE_DIR: Final[str] = "/root/.cache/vllm"

HF_CACHE_VOL_NAME: Final[str] = "synapse-hf-cache"
VLLM_CACHE_VOL_NAME: Final[str] = "synapse-vllm-cache"

# ---------------------------------------------------------------------------
# Secret & Authentication
# ---------------------------------------------------------------------------
MODAL_SECRET_NAME: Final[str] = "synapse-modal-secret"
AUTH_ENV_VAR: Final[str] = "SYNAPSE_MODAL_API_KEY"

# ---------------------------------------------------------------------------
# Hardware & Scale-to-Zero Configuration
# ---------------------------------------------------------------------------
# NVIDIA L4: 24 GB GDDR6, Ada compute capability 8.9 with native FP8 support.
DEFAULT_GPU: Final[str] = "L4"
DEFAULT_GPU_COUNT: Final[int] = 1

# Worker limits: exactly 1 worker initially, scaling down to 0 when idle
MIN_CONTAINERS: Final[int] = 0
MAX_CONTAINERS: Final[int] = 1

# Keep GPU warm for 5 minutes of idle time across pipeline stages or frames.
SCALEDOWN_WINDOW_SECONDS: Final[int] = 300

# Container timeout in seconds (10 minutes max per connection/execution)
CONTAINER_TIMEOUT_SECONDS: Final[int] = 600

# Startup timeout in seconds (10 minutes for cold start / initial model weight pull)
STARTUP_TIMEOUT_SECONDS: Final[int] = 600

# ---------------------------------------------------------------------------
# Runtime & Inference Boundaries
# ---------------------------------------------------------------------------
# Initial L4 profile. These values are configuration targets and are not proof
# that the model has successfully initialized within the available VRAM.
# Maximum total model context length (prompt tokens + image tokens + completion tokens)
MAX_MODEL_LEN: Final[int] = 12288

# Maximum generation output tokens. The total request still cannot exceed
# MAX_MODEL_LEN after prompt and image tokens are accounted for.
MAX_OUTPUT_TOKENS: Final[int] = 8192

# Structured extraction generation default. Kept separate from the ceiling so
# future callers may explicitly request smaller completions.
DEFAULT_STRUCTURED_MAX_TOKENS: Final[int] = 8192
DEFAULT_TEMPERATURE: Final[float] = 0.1

# Maximum number of images allowed per prompt
MAX_IMAGES_PER_PROMPT: Final[int] = 1

# Explicitly reject video input at the vLLM boundary.
MAX_VIDEOS_PER_PROMPT: Final[int] = 0

# vLLM 0.19.1 expects --limit-mm-per-prompt as a JSON object string.
MULTIMODAL_LIMITS_JSON: Final[str] = '{"image":1,"video":0}'

# Maximum concurrency (concurrent sequences scheduled by vLLM engine)
MAX_NUM_SEQS: Final[int] = 1

# Maximum concurrent requests routed into the container by Modal proxy
MAX_CONCURRENT_INPUTS: Final[int] = 1

# vLLM-managed GPU memory fraction for the initial L4 profile. The remaining
# 10% is outside the vLLM allocator; live startup validation is still required.
GPU_MEMORY_UTILIZATION: Final[float] = 0.90

# Synapse performs dimension-preserving thumbnailing before remote dispatch.
MAX_IMAGE_DIMENSION: Final[int] = 1344
MAX_IMAGE_PIXELS: Final[int] = MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION

# vLLM reads these JSON values as server defaults. In vLLM 0.19.1,
# max_new_tokens in generation config is a server-wide output-token limit.
VLLM_GENERATION_CONFIG_JSON: Final[str] = (
    '{"max_new_tokens":8192,"temperature":0.1}'
)
VLLM_MM_PROCESSOR_KWARGS_JSON: Final[str] = '{"max_pixels":1806336}'

# Image payload size bounds (bytes)
# 10 MB maximum per image payload to protect against memory exhaustion
MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024

# Supported image MIME types
SUPPORTED_IMAGE_MIMES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/png",
    "image/webp",
)

# Inference HTTP request timeout (seconds) for local clients
CLIENT_REQUEST_TIMEOUT_SECONDS: Final[float] = 120.0

