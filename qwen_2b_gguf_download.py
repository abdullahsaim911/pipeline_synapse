import argparse
import os
import sys

def download_qwen_gguf(repo_id: str, destination: str, revision: str | None, force: bool, token: str | None) -> str:
    """Download GGUF 4-bit model from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with:\n"
            "pip install huggingface-hub"
        ) from exc

    destination = os.path.abspath(destination)
    os.makedirs(destination, exist_ok=True)

    print(f"Downloading GGUF model '{repo_id}' into '{destination}'...")
    print("Note: Only downloading the 4-bit (Q4_K_M) weights and mmproj vision file to save space...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        revision=revision,
        force_download=force,
        token=token,
        # CRITICAL: This stops the script from downloading 100GB+ of other quantizations
        allow_patterns=["*Q4_K_M.gguf", "*mmproj*"] 
    )

    print(f"Download complete: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GGUF 4-bit Qwen2-VL-2B-Instruct model."
    )
    parser.add_argument(
        "--repo-id",
        default="bartowski/Qwen2-VL-2B-Instruct-GGUF",
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/Qwen2-VL-2B-Instruct-GGUF",
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
        download_qwen_gguf(
            repo_id=args.repo_id,
            destination=args.output_dir,
            revision=args.revision,
            force=args.force,
            token=args.token,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)