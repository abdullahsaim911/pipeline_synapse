"""
SYNAPSE — Modal Remote Inference Application for Qwen3-VL
Stage M1B: Cost-bounded Modal deployment for the official Qwen3-VL FP8 checkpoint.

Runs exclusively on one remote Modal NVIDIA L4 GPU.
Exposes a secure OpenAI-compatible multimodal endpoint over HTTPS.
Enforces persistent model caching, scale-to-zero, and strict runtime boundaries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Final

# Ensure parent directory is on sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import modal

from modal_inference.config import (
    APP_NAME,
    AUTH_ENV_VAR,
    CONTAINER_TIMEOUT_SECONDS,
    DEFAULT_GPU,
    GPU_MEMORY_UTILIZATION,
    HF_CACHE_DIR,
    HF_CACHE_VOL_NAME,
    MAX_CONCURRENT_INPUTS,
    MAX_CONTAINERS,
    MAX_IMAGES_PER_PROMPT,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MIN_CONTAINERS,
    MODAL_SECRET_NAME,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_QUANTIZATION,
    MULTIMODAL_LIMITS_JSON,
    SCALEDOWN_WINDOW_SECONDS,
    STARTUP_TIMEOUT_SECONDS,
    VLLM_CACHE_DIR,
    VLLM_CACHE_VOL_NAME,
    VLLM_PORT,
    VLLM_VERSION,
)

# ---------------------------------------------------------------------------
# 1. Container Image Definition
# ---------------------------------------------------------------------------
# Debian Slim base with pinned vLLM release, Hugging Face transfer, and Qwen utilities
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]>=0.28.0",
        "qwen-vl-utils>=0.0.14",
        "torchvision",
        "pillow",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_MODELSCOPE": "False",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        }
    )
    # Modal mounts this package at /root/modal_inference. /root is on the
    # Function container's PYTHONPATH, so imports also work when the entrypoint
    # itself is materialized as /root/app.py.
    .add_local_python_source("modal_inference")
)

# ---------------------------------------------------------------------------
# 2. Persistent Model Cache Volumes
# ---------------------------------------------------------------------------
# Durable cloud storage to persist downloaded HuggingFace model weights
# and vLLM kernel/compilation artifacts across container restarts.
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(VLLM_CACHE_VOL_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# 3. Application Definition
# ---------------------------------------------------------------------------
app = modal.App(name=APP_NAME)

# Secret reference holding SYNAPSE_MODAL_API_KEY
modal_secret = modal.Secret.from_name(MODAL_SECRET_NAME)


# ---------------------------------------------------------------------------
# 4. Helper: Background Log Streamer
# ---------------------------------------------------------------------------
def _stream_process_output(pipe, prefix: str = "[vLLM]") -> None:
    """Stream process stdout/stderr line-by-line to container stdout."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sys.stdout.write(f"{prefix} {line}")
            sys.stdout.flush()
    except Exception as exc:
        sys.stderr.write(f"[LogStreamer] Error streaming output: {exc}\n")
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# 5. Remote Web Server Function
# ---------------------------------------------------------------------------
@app.function(
    image=inference_image,
    gpu=DEFAULT_GPU,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=CONTAINER_TIMEOUT_SECONDS,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    secrets=[modal_secret],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve():
    """
    Launches vLLM in an OpenAI-compatible API serving configuration
    loading the explicitly selected official FP8 checkpoint on remote GPU.
    """
    print("=" * 72)
    print("SYNAPSE REMOTE VLM INFERENCE WORKER INITIALIZING")
    print("=" * 72)

    # 1. Startup Diagnostics: GPU & Environment
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Hardware] GPU Detected: {device_name} ({vram_gb:.2f} GB VRAM)")
        else:
            print("[Hardware WARNING] CUDA is not available to PyTorch!")
    except Exception as e:
        print(f"[Hardware] Error inspecting GPU: {e}")

    # Inspect persistent cache volume
    try:
        disk_usage = shutil.disk_usage(HF_CACHE_DIR)
        free_gb = disk_usage.free / (1024**3)
        print(f"[Storage] Persistent HF Volume Mount: {HF_CACHE_DIR} ({free_gb:.2f} GB free)")
    except Exception as e:
        print(f"[Storage] Persistent cache check: {e}")

    # 2. Strict Authentication Verification
    api_key = os.environ.get(AUTH_ENV_VAR)
    if not api_key or not api_key.strip():
        err_msg = (
            f"FATAL: Environment variable '{AUTH_ENV_VAR}' is missing or empty! "
            f"The server refuses to start in an unauthenticated state. "
            f"Ensure Modal Secret '{MODAL_SECRET_NAME}' contains '{AUTH_ENV_VAR}'."
        )
        print(f"[Security FATAL] {err_msg}", file=sys.stderr)
        raise RuntimeError(err_msg)

    print(f"[Security] Authentication configured successfully via {AUTH_ENV_VAR}.")

    # 3. Build vLLM Launch Command with Strict Boundaries
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        api_key.strip(),
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
        "--trust-remote-code",
    ]

    print("[Launch] Invoking vLLM server:")
    # Print command with obfuscated API key for security
    safe_cmd = [
        arg if arg != api_key.strip() else "[REDACTED_API_KEY]"
        for arg in cmd
    ]
    print(f"[Launch] Command: {' '.join(safe_cmd)}")

    # 4. Spawn vLLM Subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Launch background thread to stream vLLM logs to stdout
    log_thread = threading.Thread(
        target=_stream_process_output,
        args=(process.stdout,),
        daemon=True,
    )
    log_thread.start()

    # 5. Startup Health Check & Explicit Error Reporting
    import urllib.request
    import urllib.error

    health_url = f"http://127.0.0.1:{VLLM_PORT}/health"
    start_time = time.time()
    max_wait = STARTUP_TIMEOUT_SECONDS - 30  # Safety buffer before Modal timeout

    print(f"[Health] Awaiting vLLM health check on {health_url} (timeout: {max_wait}s)...")

    healthy = False
    while time.time() - start_time < max_wait:
        # Check if subprocess exited unexpectedly
        exit_code = process.poll()
        if exit_code is not None:
            err_msg = (
                f"vLLM server process crashed during startup with exit code {exit_code}! "
                f"Check container logs above for CUDA OOM, architecture mismatch, or weight download errors."
            )
            print(f"[Health FATAL] {err_msg}", file=sys.stderr)
            raise RuntimeError(err_msg)

        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError):
            # Server still loading weights or capturing CUDA graphs
            pass

        time.sleep(2.0)

    if not healthy:
        process.terminate()
        err_msg = f"vLLM failed to respond on {health_url} within {max_wait} seconds."
        print(f"[Health TIMEOUT] {err_msg}", file=sys.stderr)
        raise TimeoutError(err_msg)

    elapsed = time.time() - start_time
    print("=" * 72)
    print(f"SYNAPSE REMOTE VLM SERVER HEALTHY & READY in {elapsed:.1f}s")
    print(f"Model: {MODEL_ID}")
    print(f"Port: {VLLM_PORT}")
    print(f"Max Model Length: {MAX_MODEL_LEN} tokens")
    print(f"Max Images / Prompt: {MAX_IMAGES_PER_PROMPT}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 6. One-Shot Weight Download Helper (Optional Pre-warming)
# ---------------------------------------------------------------------------
@app.function(
    image=inference_image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=900,
    secrets=[modal_secret],
)
def download_weights():
    """
    One-shot utility to pre-populate the persistent HuggingFace cache volume
    with Qwen3-VL-8B-Instruct weights before starting the web server.
    Run via: modal run modal_inference/app.py::download_weights
    """
    import os
    from huggingface_hub import snapshot_download

    print(f"[Pre-warm] Pre-downloading model weights for '{MODEL_ID}'...")
    print(f"[Pre-warm] Target cache directory: {HF_CACHE_DIR}")

    hf_token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=MODEL_ID,
        token=hf_token,
        cache_dir=HF_CACHE_DIR,
    )

    hf_cache_vol.commit()
    print(f"[Pre-warm] Weights successfully downloaded and committed to volume '{HF_CACHE_VOL_NAME}'.")
