#!/usr/bin/env python3
"""
Transcription Module (Synapse) — Main Entry Point

Command-line interface for video transcription.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcriber import TranscriptionEngine


def print_transcript(result, max_entries: int = 20):
    """Print transcript entries to console."""
    print(f"\n{'=' * 60}")
    print(f"Transcription Results: {result.video_id}")
    print(f"{'=' * 60}")
    print(f"Method: {result.method}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Entries: {result.entry_count}")
    print(f"Language: {result.language or 'Auto-detected'}")
    print(f"{'=' * 60}\n")

    if result.error:
        print(f"ERROR: {result.error}")
        return

    # Print transcript entries
    for i, entry in enumerate(result.entries[:max_entries], 1):
        print(f"[{entry._format_time(entry.start):>6s}] {entry.text}")

    if result.entry_count > max_entries:
        print(f"\n... and {result.entry_count - max_entries} more entries")
        print(f"(Full transcript: {len(result.transcript_text):,} characters)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos or audio files using API + Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a YouTube video (tries API first, then Whisper)
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

  # Save transcript to JSON file
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o transcript.json

  # Use larger Whisper model for better accuracy
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -m medium

  # Use streaming mode (faster, no temp file)
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --stream

  # Transcribe local audio file
  python main.py audio.mp3

  # Skip API check, go directly to Whisper
  python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --skip-api
        """
    )

    parser.add_argument(
        "input",
        help="YouTube URL or local audio file path"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path"
    )

    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        default="base",
        help="Whisper model size (default: base)"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode (faster, no temp file download)"
    )

    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip YouTube API check, go directly to Whisper"
    )

    parser.add_argument(
        "--video-id",
        help="YouTube video ID (auto-extracted if not provided)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Progress callback
    def progress(msg: str):
        if not args.quiet:
            print(msg)

    # Initialize engine
    engine = TranscriptionEngine(
        whisper_model=args.model,
        prefer_streaming=args.stream,
        progress_callback=progress
    )

    # Run transcription
    if args.output:
        result = engine.transcribe_to_json(args.input, args.output, args.video_id)
    else:
        result = engine.transcribe(args.input, args.video_id, skip_api=args.skip_api)

    # Print results
    if not args.quiet:
        print_transcript(result)

    # Exit with error code if failed
    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
