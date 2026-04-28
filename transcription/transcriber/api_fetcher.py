"""
YouTube Transcript API Fetcher

Fetches available transcripts from YouTube's API.
This is the fastest method when transcripts are available.
"""

from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


class TranscriptEntry:
    """Represents a single transcript segment with timing information."""

    def __init__(self, start: float, duration: float, text: str):
        self.start = start
        self.duration = duration
        self.text = text

    @property
    def end(self) -> float:
        """Calculate end timestamp."""
        return self.start + self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "time": self._format_time(self.start),
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS string."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"


class APIFetcher:
    """Fetches transcripts using YouTube Transcript API."""

    def __init__(self):
        self.api = YouTubeTranscriptApi()

    def fetch(self, video_id: str) -> List[TranscriptEntry]:
        """
        Fetch transcript for a given video ID.

        Args:
            video_id: 11-character YouTube video ID

        Returns:
            List of TranscriptEntry objects

        Raises:
            TranscriptsDisabled: If transcripts are disabled for this video
            NoTranscriptFound: If no transcript is available
        """
        transcript = self.api.fetch(video_id)

        entries = []
        for entry in transcript:
            transcript_entry = TranscriptEntry(
                start=entry.start,
                duration=entry.duration,
                text=entry.text
            )
            entries.append(transcript_entry)

        return entries

    def is_available(self, video_id: str) -> bool:
        """
        Check if a transcript is available for the video.

        Args:
            video_id: 11-character YouTube video ID

        Returns:
            True if transcript is available, False otherwise
        """
        try:
            self.api.fetch(video_id)
            return True
        except (TranscriptsDisabled, NoTranscriptFound):
            return False
