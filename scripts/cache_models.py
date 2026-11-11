"""One-time local model cache preparation used by the PowerShell setup command."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=("api", "voice", "pipeline"))
    args = parser.parse_args()

    if args.runtime == "api":
        # Cache the offline TTS fallback used when Edge-TTS is unavailable.
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="Xenova/transformers.js-docs",
            repo_type="dataset",
            filename="speaker_embeddings.bin",
        )
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
