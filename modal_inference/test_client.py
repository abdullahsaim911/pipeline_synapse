"""
SYNAPSE — Local Test Client for Modal Remote VLM Inference
Stage M1A: Sends a local JPEG/PNG image and text prompt over HTTPS to the
deployed Qwen3-VL Modal endpoint and displays the inference response.

Usage:
    python modal_inference/test_client.py \\
        --url https://<workspace>--synapse-qwen3-vl-serve.modal.run \\
        --image path/to/sample.png \\
        --prompt "Describe the educational content in this slide." \\
        --api-key "$SYNAPSE_MODAL_API_KEY"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Ensure parent directory is on sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from modal_inference.config import (
    AUTH_ENV_VAR,
    CLIENT_REQUEST_TIMEOUT_SECONDS,
    MAX_IMAGE_BYTES,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
)
from modal_inference.schemas import detect_image_mime, validate_image_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synapse Qwen3-VL Modal Remote Inference Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("SYNAPSE_MODAL_URL", ""),
        help="Base URL of deployed Modal app (or via SYNAPSE_MODAL_URL env var).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to local JPEG or PNG image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what is visible in this image in detail.",
        help="Text prompt / instruction for the vision-language model.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(AUTH_ENV_VAR, ""),
        help=f"Bearer authentication token (or via {AUTH_ENV_VAR} env var).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help=f"Maximum generation tokens to output (clamped to {MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (lower = more deterministic).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=CLIENT_REQUEST_TIMEOUT_SECONDS,
        help="HTTPS request timeout in seconds.",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Check the /health endpoint without performing inference.",
    )
    return parser.parse_args()


def check_health(base_url: str, timeout: float = 10.0) -> bool:
    """Check if the remote Modal web server is healthy."""
    health_url = f"{base_url.rstrip('/')}/health"
    print(f"[Client] Probing health endpoint: {health_url}")
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                print(f"[Client] Health OK (status {resp.status}).")
                return True
            print(f"[Client] Health check returned non-200 status: {resp.status}")
            return False
    except urllib.error.HTTPError as exc:
        print(f"[Client] Health check HTTP error {exc.code}: {exc.reason}")
        return False
    except Exception as exc:
        print(f"[Client] Health check connection error: {exc}")
        return False


def send_inference_request(
    base_url: str,
    api_key: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any] | None:
    """Send image + text multimodal request to OpenAI-compatible endpoint."""
    # 1. Local Image Validation
    if not image_path.exists() or not image_path.is_file():
        print(f"[Error] Image file does not exist: {image_path}", file=sys.stderr)
        return None

    raw_bytes = image_path.read_bytes()
    is_valid, msg, detected_mime = validate_image_payload(raw_bytes, max_bytes=MAX_IMAGE_BYTES)
    if not is_valid:
        print(f"[Error] Local image validation failed: {msg}", file=sys.stderr)
        return None

    file_size_kb = len(raw_bytes) / 1024
    print(f"[Client] Image validated: {image_path.name} ({detected_mime}, {file_size_kb:.1f} KB)")

    # 2. Base64 Data URI Construction
    b64_data = base64.b64encode(raw_bytes).decode("ascii")
    data_uri = f"data:{detected_mime};base64,{b64_data}"

    # 3. Construct OpenAI-Compatible Multimodal Payload
    bounded_tokens = min(max(1, max_tokens), MAX_OUTPUT_TOKENS)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
        "max_tokens": bounded_tokens,
        "temperature": temperature,
        "stream": False,
    }

    payload_json = json.dumps(payload).encode("utf-8")
    endpoint_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    print(f"[Client] Target endpoint: {endpoint_url}")
    print(f"[Client] Prompt: {prompt!r}")
    print(f"[Client] Generation bounds: max_tokens={bounded_tokens}, temp={temperature}")
    print(f"[Client] Sending request (timeout={timeout}s)...")

    # 4. HTTP Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Synapse-Modal-Client/1.0",
    }

    req = urllib.request.Request(
        endpoint_url,
        data=payload_json,
        headers=headers,
        method="POST",
    )

    # 5. Execute Request
    start_time = time.perf_counter()
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            latency_s = time.perf_counter() - start_time
            status_code = response.status
            body = response.read().decode("utf-8")
            data = json.loads(body)

            print("=" * 72)
            print(f"INFERENCE SUCCESSFUL (Status: {status_code}, Latency: {latency_s:.2f}s)")
            print("=" * 72)

            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "")
                print("\n[Model Response]:")
                print("-" * 72)
                print(content.strip())
                print("-" * 72)
            else:
                print(f"[Warning] Unexpected response structure:\n{json.dumps(data, indent=2)}")

            usage = data.get("usage", {})
            if usage:
                prompt_toks = usage.get("prompt_tokens", "N/A")
                comp_toks = usage.get("completion_tokens", "N/A")
                tot_toks = usage.get("total_tokens", "N/A")
                print(f"\n[Token Usage] Prompt: {prompt_toks} | Output: {comp_toks} | Total: {tot_toks}")

            return data

    except urllib.error.HTTPError as exc:
        latency_s = time.perf_counter() - start_time
        err_body = exc.read().decode("utf-8", errors="replace")
        print(f"\n[HTTP Error {exc.code}] {exc.reason} (after {latency_s:.2f}s):", file=sys.stderr)
        try:
            parsed_err = json.loads(err_body)
            print(json.dumps(parsed_err, indent=2), file=sys.stderr)
        except Exception:
            print(err_body, file=sys.stderr)

        if exc.code == 401:
            print("\n[Diagnosis] 401 Unauthorized: The Bearer API key was rejected or missing.", file=sys.stderr)
            print(f"Check your '{AUTH_ENV_VAR}' token configuration.", file=sys.stderr)
        elif exc.code == 404:
            print("\n[Diagnosis] 404 Not Found: The endpoint path was not recognized.", file=sys.stderr)
            print(f"Verify base URL ({base_url}) and that /v1/chat/completions is served.", file=sys.stderr)
        elif exc.code == 413:
            print("\n[Diagnosis] 413 Payload Too Large: The image or request exceeded size bounds.", file=sys.stderr)

        return None

    except urllib.error.URLError as exc:
        latency_s = time.perf_counter() - start_time
        print(f"\n[Connection Error] Failed to reach {base_url} ({exc.reason}) after {latency_s:.2f}s", file=sys.stderr)
        print("\n[Diagnosis]:", file=sys.stderr)
        print("- Is the Modal application deployed and running?", file=sys.stderr)
        print("- If the container was cold, the initial spin-up may take ~1-2 minutes.", file=sys.stderr)
        return None

    except TimeoutError:
        print(f"\n[Timeout] Request timed out after {timeout}s.", file=sys.stderr)
        print("Model may still be warming up or processing a heavy request.", file=sys.stderr)
        return None


def main() -> int:
    args = parse_args()

    if not args.url:
        print("[Error] Missing Modal application URL! Pass via --url or SYNAPSE_MODAL_URL env var.", file=sys.stderr)
        return 1

    base_url = args.url.strip()

    if args.health_only:
        ok = check_health(base_url, timeout=args.timeout)
        return 0 if ok else 1

    if not args.api_key:
        print(f"[Error] Missing API key! Pass via --api-key or {AUTH_ENV_VAR} env var.", file=sys.stderr)
        return 1

    if not args.image:
        print("[Error] Missing image path! Pass via --image <path/to/image.jpg|png>.", file=sys.stderr)
        return 1

    image_path = Path(args.image).resolve()

    result = send_inference_request(
        base_url=base_url,
        api_key=args.api_key,
        image_path=image_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
