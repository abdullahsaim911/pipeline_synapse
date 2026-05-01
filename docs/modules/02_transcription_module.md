# Transcription Module (M0b) Documentation

## Overview

The Transcription Module converts spoken content from educational videos into text transcripts with precise timing information. It uses a hybrid approach that prioritizes YouTube's built-in transcripts for speed while falling back to Whisper AI for comprehensive coverage.

## Purpose

- Extract spoken content from video lectures
- Generate accurate transcripts with word-level timing
- Support multiple languages and dialects
- Handle both YouTube videos and local audio files
- Provide metadata about the transcription process

## Tools & Technologies

### Core Dependencies
- **YouTube Transcript API**: Fast access to YouTube's built-in transcripts
- **Whisper AI (OpenAI)**: Local speech recognition for fallback
  - `faster-whisper`: Optimized implementation for faster processing
  - Models: tiny, base, small, medium, large-v3, large-v3-turbo
- **yt-dlp**: Audio extraction from YouTube videos
- **FFmpeg**: Audio processing and format conversion
- **Python 3.9+**: Primary programming language

### Whisper Models

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|-----------|------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Lower |
| base | 74M | ~1GB | Fast | Good |
| small | 244M | ~2GB | Medium | Better |
| medium | 769M | ~5GB | Slower | High |
| large-v3 | 1550M | ~10GB | Slowest | Best |
| large-v3-turbo | 809M | ~5GB | Fast | High |

## Implementation Details

### Module Structure

```
transcription/
├── __init__.py
├── main.py                    # CLI entry point
├── transcriber/
│   ├── __init__.py
│   ├── transcription_engine.py # Main orchestration
│   ├── api_fetcher.py          # YouTube API integration
│   └── whisper_runner.py       # Whisper AI integration
└── utils/
    ├── __init__.py
    └── url_parser.py           # URL parsing utilities
```

### Core Components

#### 1. Transcription Engine

**Hybrid Approach**: Tries API first, falls back to Whisper

```python
class TranscriptionEngine:
    """Main transcription engine with hybrid API + Whisper support."""

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
            prefer_streaming: If True, use streaming mode (faster, no temp file)
            progress_callback: Optional callback for progress updates
        """
        self.api_fetcher = APIFetcher()
        self.whisper_runner = WhisperRunner(model_size=whisper_model)
        self.prefer_streaming = prefer_streaming
        self.progress_callback = progress_callback or (lambda x: None)
```

**Workflow**:
1. Try YouTube Transcript API (instant if available)
2. Fall back to Whisper AI transcription if API fails
3. Return structured result with timing information

#### 2. API Fetcher

**YouTube Transcript API Integration**: Fast access to existing transcripts

```python
class APIFetcher:
    """Fetches transcripts from YouTube's Transcript API."""

    def fetch(self, video_id: str) -> List[TranscriptEntry]:
        """
        Fetch transcript from YouTube API.

        Args:
            video_id: YouTube video ID (11 characters)

        Returns:
            List of TranscriptEntry objects with timing and text

        Raises:
            TranscriptionError: If API is unavailable or transcript doesn't exist
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', 'en-US', 'en-GB']
            )

            return [
                TranscriptEntry(
                    start=item['start'],
                    end=item['start'] + item['duration'],
                    text=item['text']
                )
                for item in transcript_list
            ]
        except Exception as e:
            raise TranscriptionError(f"YouTube API unavailable: {e}")
```

**Advantages**:
- Instant retrieval (no audio processing)
- High accuracy (human-reviewed transcripts)
- Multiple language support
- No computational cost

**Limitations**:
- Not available for all videos
- No custom language models
- Dependent on YouTube's availability

#### 3. Whisper Runner

**Local Speech Recognition**: Full-featured fallback for any audio

```python
class WhisperRunner:
    """Runs Whisper AI transcription with multiple modes."""

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper runner.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3, large-v3-turbo)
        """
        self.model_size = model_size
        self.model = None  # Lazy-loaded on first use

    def transcribe(
        self,
        input_source: str,
        mode: str = "auto"
    ) -> List[TranscriptEntry]:
        """
        Transcribe audio using Whisper.

        Args:
            input_source: YouTube URL or local audio file path
            mode: "auto" (choose best), "stream" (faster), "download" (more compatible)

        Returns:
            List of TranscriptEntry objects
        """
        if mode == "stream":
            return self._transcribe_streaming(input_source)
        else:
            return self._transcribe_with_download(input_source)
```

**Streaming Mode**: Faster, no temporary file
```python
def _transcribe_streaming(self, input_source: str) -> List[TranscriptEntry]:
    """Transcribe using streaming audio (faster, no temp file)."""
    # Extract audio stream directly from YouTube
    # Pipe audio to Whisper without saving to disk
    # Process in chunks for memory efficiency
```

**Download Mode**: More compatible, higher quality
```python
def _transcribe_with_download(self, input_source: str) -> List[TranscriptEntry]:
    """Transcribe by downloading audio first (more compatible)."""
    # Download audio file to temporary location
    # Run Whisper on the file
    # Clean up temporary file after processing
```

#### 4. URL Parser

**Video ID Extraction**: Robust YouTube URL parsing

```python
def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.

    Supports:
    - youtube.com/watch?v=ID
    - youtu.be/ID
    - youtube.com/embed/ID
    - youtube.com/shorts/ID

    Args:
        url: YouTube URL

    Returns:
        Video ID (11 characters) or None if invalid
    """
    pattern = r'(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([\w-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None
```

### Data Structures

#### TranscriptEntry

```python
@dataclass
class TranscriptEntry:
    """Single transcript segment with timing information."""
    start: float      # Start time in seconds
    end: float        # End time in seconds
    text: str         # Spoken text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.end - self.start,
            "text": self.text
        }
```

#### TranscriptionResult

```python
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
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "method": self.method,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "entry_count": self.entry_count,
            "error": self.error,
            "entries": [e.to_dict() for e in self.entries]
        }
```

## Configuration Parameters

### TranscriptionEngine Parameters

```python
TranscriptionEngine(
    whisper_model: str = "base",        # Whisper model size
    prefer_streaming: bool = False,      # Use streaming mode for Whisper
    progress_callback: Optional[Callable[[str], None]] = None  # Progress updates
)
```

### Whisper Runner Parameters

```python
WhisperRunner(
    model_size: str = "base"            # Model size selection
)
```

### API Fetcher Parameters

```python
# Language priority list (tried in order)
languages = ['en', 'en-US', 'en-GB']
```

## Output Format

### Transcript JSON Structure

```json
{
  "video_id": "dQw4w9WgXcQ",
  "method": "whisper",
  "language": "en",
  "duration_seconds": 212.0,
  "entry_count": 45,
  "error": null,
  "entries": [
    {
      "start": 0.0,
      "end": 2.5,
      "duration": 2.5,
      "text": "Welcome to today's lecture on calculus."
    },
    {
      "start": 2.5,
      "end": 5.0,
      "duration": 2.5,
      "text": "We'll be covering derivatives and their applications."
    }
  ]
}
```

## Usage Examples

### Basic Usage

```python
from transcriber import TranscriptionEngine

# Initialize engine
engine = TranscriptionEngine(whisper_model="base")

# Transcribe YouTube video
result = engine.transcribe(
    input_source="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)

print(f"Method: {result.method}")
print(f"Entries: {result.entry_count}")
print(f"Duration: {result.duration_seconds:.2f}s")

# Print first few entries
for entry in result.entries[:5]:
    print(f"[{entry.start:.1f}s] {entry.text}")
```

### Save to JSON

```python
result = engine.transcribe_to_json(
    input_source="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_file="transcript.json"
)
```

### With Progress Callback

```python
def progress_handler(message: str):
    print(f"Progress: {message}")

engine = TranscriptionEngine(
    whisper_model="base",
    progress_callback=progress_handler
)

result = engine.transcribe("video_url")
```

### Command Line Interface

```bash
# Basic transcription
python transcription/main.py https://www.youtube.com/watch?v=VIDEO_ID

# Save to file
python transcription/main.py https://www.youtube.com/watch?v=VIDEO_ID -o transcript.json

# Use larger model for better accuracy
python transcription/main.py https://www.youtube.com/watch?v=VIDEO_ID -m medium

# Use streaming mode (faster)
python transcription/main.py https://www.youtube.com/watch?v=VIDEO_ID --stream

# Skip API check, go directly to Whisper
python transcription/main.py https://www.youtube.com/watch?v=VIDEO_ID --skip-api

# Transcribe local audio file
python transcription/main.py audio.mp3
```

### Custom Model Selection

```python
# Use larger model for better accuracy
engine = TranscriptionEngine(whisper_model="large-v3-turbo")
result = engine.transcribe("video_url")
```

## Performance Characteristics

### Processing Time (10-minute video)

| Method | Time | VRAM | Notes |
|--------|------|------|-------|
| YouTube API | < 1s | 0 | Instant retrieval |
| Whisper (tiny) | 30-60s | ~1GB | Fastest local |
| Whisper (base) | 60-120s | ~1GB | Good balance |
| Whisper (small) | 120-240s | ~2GB | Better accuracy |
| Whisper (medium) | 300-600s | ~5GB | High accuracy |
| Whisper (large-v3) | 600-1200s | ~10GB | Best accuracy |

### Memory Usage

| Model | RAM | VRAM |
|-------|-----|------|
| tiny | 500MB | ~1GB |
| base | 1GB | ~1GB |
| small | 2GB | ~2GB |
| medium | 4GB | ~5GB |
| large-v3 | 8GB | ~10GB |

### Accuracy Comparison

| Model | Word Error Rate (WER) | Use Case |
|-------|---------------------|----------|
| YouTube API | 1-2% | Best (human-reviewed) |
| large-v3 | 2-3% | Professional production |
| large-v3-turbo | 3-4% | High-quality production |
| medium | 4-5% | Good quality |
| small | 5-7% | Moderate quality |
| base | 7-10% | Quick drafts |
| tiny | 10-15% | Rough drafts |

## Troubleshooting

### Common Issues

**Issue**: "YouTube API unavailable"
- **Solution**: This is normal; system will fall back to Whisper automatically

**Issue**: "CUDA out of memory"
- **Solution**: Use smaller model (base instead of large) or CPU mode

**Issue**: "Audio download failed"
- **Solution**: Check internet connection and video availability

**Issue**: "Transcription is in wrong language"
- **Solution**: Specify language in WhisperRunner or use API with language parameter

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check API availability first
from transcriber.api_fetcher import APIFetcher
fetcher = APIFetcher()
try:
    entries = fetcher.fetch("video_id")
    print(f"API available: {len(entries)} entries")
except Exception as e:
    print(f"API not available: {e}")
```

## Best Practices

### Model Selection

```python
# For quick prototyping
engine = TranscriptionEngine(whisper_model="base")

# For production quality
engine = TranscriptionEngine(whisper_model="large-v3-turbo")

# For maximum accuracy (slowest)
engine = TranscriptionEngine(whisper_model="large-v3")
```

### Error Handling

```python
try:
    result = engine.transcribe(video_url)
    if result.error:
        print(f"Transcription failed: {result.error}")
    else:
        print(f"Success: {result.entry_count} entries")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Progress Tracking

```python
def track_progress(message: str):
    """Custom progress handler with timestamps."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

engine = TranscriptionEngine(progress_callback=track_progress)
```

## Future Enhancements

1. **Multi-Language Support**: Extended language detection and transcription
2. **Custom Vocabularies**: Domain-specific terminology for STEM subjects
3. **Speaker Diarization**: Identify and separate different speakers
4. **Real-Time Transcription**: Live captioning support
5. **Post-Processing**: Punctuation, capitalization, and formatting improvements
6. **Batch Processing**: Efficient processing of multiple videos
7. **Confidence Scores**: Per-word confidence metrics
8. **Language Detection**: Automatic language identification
