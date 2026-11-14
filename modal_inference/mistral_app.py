"""Separate authenticated Mistral explanation worker; does not redeploy Qwen."""

import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

import modal

from modal_inference.mistral_config import VLLM_VERSION, build_serve_command
from modal_inference.launch_config import redact_sensitive_text


app = modal.App("synapse-ministral-reasoning")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "mistral-common>=1.8.6")
    # Newer Transformers removed PixtralRotaryEmbedding used by vLLM 0.19.
    .pip_install("transformers==5.5.3")
    .add_local_python_source("modal_inference")
)
weights = modal.Volume.from_name("synapse-hf-cache", create_if_missing=True)


@app.function(
    image=image, gpu=["L4", "A10G"], min_containers=0, max_containers=1,
    scaledown_window=300, timeout=900,
    volumes={"/root/.cache/huggingface": weights},
    secrets=[modal.Secret.from_name("synapse-modal-secret")],
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    key = os.environ.get("SYNAPSE_MODAL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SYNAPSE_MODAL_API_KEY is required; refusing unauthenticated serving")
    environment = dict(os.environ, VLLM_API_KEY=key)
    process = subprocess.Popen(
        build_serve_command(), env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def stream_logs():
        for line in process.stdout:
            print(redact_sensitive_text(line.rstrip(), key), flush=True)

    threading.Thread(target=stream_logs, daemon=True).start()
    deadline = time.monotonic() + 860
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Mistral worker exited during startup; inspect deployment logs")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    process.terminate()
    raise TimeoutError("Mistral worker did not become healthy")
