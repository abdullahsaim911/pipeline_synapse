"""
Whisper/Faster-Whisper Transcription Engine

Provides AI-based speech-to-text transcription using Whisper models.
Supports both download-then-transcribe and streaming modes.
"""

import time
import os
import torch
import numpy as np
import subprocess
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from .api_fetcher import TranscriptEntry


@dataclass
class WhisperSegment:
    """Represents a transcribed segment from Whisper."""
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        minutes, seconds = divmod(int(self.start), 60)
        return {
            "time": f"{minutes:02d}:{seconds:02d}",
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


class WhisperRunner:
    """
    Handles Whisper-based AI transcription with hardware acceleration.
    """

    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """
        Initialize Whisper runner.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3, large-v3-turbo)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            compute_type: Compute precision ('float16', 'float32', 'int8'). Auto-detected if None.
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model_size. Must be one of: {self.MODEL_SIZES}")

        # Auto-detect hardware
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-detect compute type
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

    def transcribe_file(
        self,
        audio_file: str,
        beam_size: int = 5,
        language: Optional[str] = None
    ) -> List[TranscriptEntry]:
        """
        Transcribe an audio file.

        Args:
            audio_file: Path to audio file (m4a, mp3, wav, etc.)
            beam_size: Beam search size (1 = fastest, 5 = balanced, higher = more accurate)
            language: Language code (e.g., 'en', 'es'). Auto-detected if None.

        Returns:
            List of TranscriptEntry objects
        """
        self.load_model()

        segments, info = self.model.transcribe(
            audio_file,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )

        entries = []
        for segment in segments:
            entry = TranscriptEntry(
                start=segment.start,
                duration=segment.end - segment.start,
                text=segment.text
            )
            entries.append(entry)

        return entries

    def transcribe_stream(
        self,
        video_url: str,
        beam_size: int = 1,
        language: Optional[str] = None
    ) -> List[TranscriptEntry]:
        """
        Transcribe a YouTube video by streaming audio directly (no download).

        This pipes audio from yt-dlp through ffmpeg to Whisper in memory.

        Args:
            video_url: YouTube video URL
            beam_size: Beam search size (1 = fastest for streaming)
            language: Language code. Auto-detected if None.

        Returns:
            List of TranscriptEntry objects
        """
        self.load_model()

        # Build streaming command: yt-dlp | ffmpeg -> raw PCM
        cmd = [
            "yt-dlp", "-q", "--no-warnings",
            "-f", "ba*[ext=m4a]",
            "-o", "-", video_url,
            "|", "ffmpeg",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1"
        ]

        # Run command with shell for pipe support
        process = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        try:
            # Read entire stream into memory
            raw_audio = process.stdout.read()

            if not raw_audio:
                raise RuntimeError("No audio data received from stream")

            # Convert to normalized float32 numpy array
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=beam_size,
                language=language,
                task="transcribe"
            )

            entries = []
            for segment in segments:
                entry = TranscriptEntry(
                    start=segment.start,
                    duration=segment.end - segment.start,
                    text=segment.text
                )
                entries.append(entry)

            return entries

        finally:
            process.terminate()

    def transcribe(
        self,
        input_source: str,
        mode: str = "auto",
        beam_size: int = 5,
        language: Optional[str] = None,
        temp_audio_file: str = "temp_audio.m4a"
    ) -> List[TranscriptEntry]:
        """
        Main transcription method with flexible input handling.

        Args:
            input_source: YouTube URL or local audio file path
            mode: 'auto', 'stream', or 'file'
                - 'auto': Stream if URL, file otherwise
                - 'stream': Always use streaming (URL only)
                - 'file': Download then transcribe (URL only)
            beam_size: Beam search size
            language: Language code
            temp_audio_file: Temporary file path for download mode

        Returns:
            List of TranscriptEntry objects
        """
        if mode == "stream" or (mode == "auto" and input_source.startswith(("http://", "https://"))):
            return self.transcribe_stream(input_source, beam_size, language)

        # Download mode or local file
        if input_source.startswith(("http://", "https://")):
            self._download_audio(input_source, temp_audio_file)
            audio_file = temp_audio_file
            cleanup = True
        else:
            audio_file = input_source
            cleanup = False

        try:
            return self.transcribe_file(audio_file, beam_size, language)
        finally:
            if cleanup and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)

    @staticmethod
    def _download_audio(video_url: str, output_file: str):
        """Download audio from YouTube using yt-dlp."""
        import yt_dlp

        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "outtmpl": output_file.replace(".m4a", ""),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
            "quiet": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
