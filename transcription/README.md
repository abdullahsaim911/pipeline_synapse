# Transcription Module (Synapse)

Hybrid speech-to-text pipeline for YouTube videos and audio files.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Transcribe a YouTube video
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Save to JSON
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o transcript.json
```

## Features

- **YouTube Transcript API** — Instant transcripts when available
- **Whisper AI** — Universal speech-to-text fallback
- **Hardware Acceleration** — Auto-detects CUDA GPU
- **Streaming Mode** — No download required
- **Multiple Models** — tiny to large-v3-turbo

## Usage

### Python API

```python
from transcriber import TranscriptionEngine

# Initialize engine
engine = TranscriptionEngine(whisper_model="base", prefer_streaming=True)

# Transcribe a video
result = engine.transcribe("https://www.youtube.com/watch?v=VIDEO_ID")

# Access results
print(f"Method: {result.method}")  # "api" or "whisper"
print(f"Entries: {result.entry_count}")
print(f"Text: {result.transcript_text}")

# Save to JSON
engine.transcribe_to_json(url, "output.json")
```

### Command Line

```bash
# Basic usage
python main.py <URL or audio file>

# Options
-o, --output      Save to JSON file
-m, --model       Whisper model (tiny/base/small/medium/large-v3/large-v3-turbo)
--stream          Use streaming mode (faster)
--skip-api        Skip YouTube API, go directly to Whisper
-q, --quiet       Suppress progress output
```

## Module Structure

```
transcription_module/
├── transcriber/           # Core transcription logic
│   ├── api_fetcher.py    # YouTube API
│   ├── whisper_runner.py # Whisper AI
│   └── transcription_engine.py # Main engine
├── utils/                # Utilities
│   └── url_parser.py     # URL parsing
└── main.py              # CLI entry point
```

## Output Format

```json
{
  "video_id": "dQw4w9WgXcQ",
  "method": "whisper",
  "language": "en",
  "duration_seconds": 12.34,
  "entry_count": 42,
  "entries": [
    {
      "time": "00:05",
      "start": 5.0,
      "end": 7.5,
      "duration": 2.5,
      "text": "Never gonna give you up"
    }
  ]
}
```

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)

See [requirements.txt](requirements.txt) for Python dependencies.

## Documentation

Full documentation: [transcription_module_documentation.md](transcription_module_documentation.md)
