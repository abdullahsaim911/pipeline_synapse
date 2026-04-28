"""
Transcription Engine

Main orchestration layer for video transcription.
Tries YouTube API first (fastest), falls back to Whisper AI (slower but universal).
"""

import time
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass

from .api_fetcher import APIFetcher, TranscriptEntry
from .whisper_runner import WhisperRunner


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    video_id: str
    entries: List[TranscriptEntry]
    method: str  # "api" or "whisper"
    language: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def transcript_text(self) -> str:
        """Get full transcript as plain text."""
        return " ".join(e.text for e in self.entries)

    @property
    def entry_count(self) -> int:
        """Number of transcript segments."""
        return len(self.entries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "video_id": self.video_id,
            "method": self.method,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "entry_count": self.entry_count,
            "error": self.error,
            "entries": [e.to_dict() for e in self.entries]
        }


class TranscriptionEngine:
    """
    Main transcription engine with hybrid API + Whisper support.

    Workflow:
        1. Try YouTube Transcript API (instant if available)
        2. Fall back to Whisper AI transcription
    """

    def __init__(
        self,
        whisper_model: str = "base",
        prefer_streaming: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize transcription engine.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo)
            prefer_streaming: If True, use streaming mode for Whisper (faster, no temp file)
            progress_callback: Optional callback for progress updates
        """
        self.api_fetcher = APIFetcher()
        self.whisper_runner = WhisperRunner(model_size=whisper_model)
        self.prefer_streaming = prefer_streaming
        self.progress_callback = progress_callback or (lambda x: None)

    def _log(self, message: str):
        """Log progress via callback."""
        self.progress_callback(message)

    def transcribe(
        self,
        input_source: str,
        video_id: Optional[str] = None,
        skip_api: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe a video or audio file.

        Args:
            input_source: YouTube URL or local audio file path
            video_id: YouTube video ID (for API lookup). Auto-extracted if None.
            skip_api: If True, skip API check and go directly to Whisper

        Returns:
            TranscriptionResult with transcript entries and metadata
        """
        start_time = time.perf_counter()

        # Extract video ID if not provided
        if video_id is None and input_source.startswith(("http://", "https://")):
            from utils.url_parser import extract_video_id
            video_id = extract_video_id(input_source)

        if not video_id:
            video_id = "unknown"

        # Try YouTube API first (fastest)
        if not skip_api and video_id != "unknown":
            self._log(f"🔍 Checking YouTube API for {video_id}...")
            try:
                entries = self.api_fetcher.fetch(video_id)
                duration = time.perf_counter() - start_time
                self._log(f"✅ API Success! Found in {duration:.2f}s")
                return TranscriptionResult(
                    video_id=video_id,
                    entries=entries,
                    method="api",
                    duration_seconds=duration
                )
            except Exception as e:
                self._log(f"❌ API unavailable: {e}. Switching to Whisper...")

        # Fall back to Whisper AI
        self._log("🧠 Starting Whisper AI transcription...")

        # Choose mode
        mode = "stream" if self.prefer_streaming else "auto"

        try:
            entries = self.whisper_runner.transcribe(input_source, mode=mode)
            duration = time.perf_counter() - start_time
            self._log(f"✅ Whisper completed in {duration:.2f}s")

            # Extract language from first entry if available
            language = None
            if hasattr(self.whisper_runner, 'model'):
                # Language would need to be captured during transcription
                pass

            return TranscriptionResult(
                video_id=video_id,
                entries=entries,
                method="whisper",
                language=language,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            self._log(f"❌ Transcription failed: {e}")
            return TranscriptionResult(
                video_id=video_id,
                entries=[],
                method="failed",
                error=str(e),
                duration_seconds=duration
            )

    def transcribe_to_json(
        self,
        input_source: str,
        output_file: str,
        video_id: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe and save result to JSON file.

        Args:
            input_source: YouTube URL or local audio file path
            output_file: Path to output JSON file
            video_id: YouTube video ID

        Returns:
            TranscriptionResult
        """
        import json

        result = self.transcribe(input_source, video_id)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self._log(f"💾 Saved transcript to {output_file}")
        return result
