import argparse
import os
import sys

def download_qwen_awq(repo_id: str, destination: str, revision: str | None, force: bool, token: str | None) -> str:
    """Download AWQ-quantized model from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with:\n"
            "pip install huggingface-hub"
        ) from exc

    destination = os.path.abspath(destination)
    os.makedirs(destination, exist_ok=True)

    print(f"Downloading AWQ-quantized model '{repo_id}' into '{destination}'...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        local_dir_use_symlinks=False,
        revision=revision,
        resume_download=True,
        force_download=force,
        token=token,
    )

    print(f"Download complete: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AWQ-quantized Qwen2-VL-7B-Instruct model."
    )
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        help="Hugging Face model repo id (default: Qwen/Qwen2-VL-7B-Instruct-AWQ).",
    )
    parser.add_argument(
        "--output-dir",
        default="models/Qwen2-VL-7B-Instruct-AWQ",
        help="Local directory where the model should be downloaded.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, branch, or tag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face auth token for private models.",
    )

    args = parser.parse_args()

    try:
        download_qwen_awq(
            repo_id=args.repo_id,
            destination=args.output_dir,
            revision=args.revision,
            force=args.force,
            token=args.token,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)