"""One-time local model cache preparation used by the PowerShell setup command."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=("api", "voice", "pipeline"))
    args = parser.parse_args()

    if args.runtime == "api":
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tts_engine.assets import prepare_assets
        prepare_assets()
    elif args.runtime == "voice":
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
        )
    else:
        from faster_whisper import WhisperModel

        WhisperModel("base", device="cpu", compute_type="int8")


if __name__ == "__main__":
    main()
