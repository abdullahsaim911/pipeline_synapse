# Transcription Module (Synapse) — Complete Project Analysis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - [transcriber/api_fetcher.py](#31-transcriberapi_fetcherpy)
   - [transcriber/whisper_runner.py](#32-transcriberwhisper_runnerpy)
   - [transcriber/transcription_engine.py](#33-transcribertranscription_enginepy)
   - [utils/url_parser.py](#34-utilsurl_parserpy)
   - [main.py](#35-mainpy)
4. [Implemented Features — Detailed Explanation](#4-implemented-features--detailed-explanation)
   - [Feature 1: YouTube Transcript API Integration](#feature-1-youtube-transcript-api-integration)
   - [Feature 2: Whisper/Faster-Whisper AI Transcription](#feature-2-whisperfaster-whisper-ai-transcription)
   - [Feature 3: Hybrid API + Whisper Fallback](#feature-3-hybrid-api--whisper-fallback)
   - [Feature 4: Hardware Acceleration (CUDA/CPU)](#feature-4-hardware-acceleration-cudacpu)
   - [Feature 5: Streaming Transcription Mode](#feature-5-streaming-transcription-mode)
   - [Feature 6: Flexible Input Handling](#feature-6-flexible-input-handling)
5. [Complete End-to-End Workflow](#5-complete-end-to-end-workflow)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [Technology Stack](#7-technology-stack)
8. [Design Decisions & Notes](#8-design-decisions--notes)

---

## 1. Project Overview

**Transcription Module (Synapse)** is a hybrid speech-to-text pipeline designed to extract spoken content from YouTube videos and audio files.

The module operates on a **two-tier strategy**:

1. **First, try the YouTube Transcript API** — If the video has an existing transcript (either manually uploaded or auto-generated), this returns instantly (~0.1–0.5 seconds) with no compute cost.

2. **Fall back to Whisper AI** — If no transcript exists, use OpenAI's Whisper model (via the optimized `faster-whisper` implementation) to transcribe the audio from scratch. This is slower (~5–30 seconds depending on video length and hardware) but works on any audio.

This module is part of the broader **Memoona** system, providing the **transcript stream** that feeds into the Fusion Module alongside visual keyframes. Together, they create a complete multimodal accessibility pipeline: spoken words (transcript) + visual content (keyframes) → fused explanation.

---

## 2. Project Structure

```
transcription_module/
│
├── transcriber/
│   ├── __init__.py              # Package exports
│   ├── api_fetcher.py           # YouTube Transcript API wrapper
│   ├── whisper_runner.py        # Whisper/Faster-Whisper AI engine
│   └── transcription_engine.py  # Main orchestration layer
│
├── utils/
│   ├── __init__.py              # Package exports
│   └── url_parser.py            # YouTube URL parsing utilities
│
├── data/                        # Output directory (optional, for JSON exports)
│
└── main.py                      # CLI entry point
```

---

## 3. File-by-File Breakdown

### 3.1 `transcriber/api_fetcher.py`

**Type:** Data Source — Fast Path

**Purpose:** Fetches pre-existing transcripts from YouTube's API without any AI processing.

**Key Classes:**

#### `TranscriptEntry`
Represents a single transcript segment.

```python
@dataclass
class TranscriptEntry:
    start: float           # Start timestamp in seconds
    duration: float        # Segment duration
    text: str              # Spoken text content

    @property
    def end(self) -> float:
        """Calculate end timestamp."""
        return self.start + self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

#### `APIFetcher`
Main class for YouTube Transcript API operations.

```python
class APIFetcher:
    def __init__(self):
        self.api = YouTubeTranscriptApi()

    def fetch(self, video_id: str) -> List[TranscriptEntry]:
        """Fetch transcript for a given video ID."""

    def is_available(self, video_id: str) -> bool:
        """Check if a transcript is available."""
```

**Role in the system:** This is the "fast path" — if a transcript exists, we get it instantly with no model loading, no GPU inference, no network download. It's called first in every transcription request before considering the slower Whisper fallback.

---

### 3.2 `transcriber/whisper_runner.py`

**Type:** AI Model — Slow Path (Universal Fallback)

**Purpose:** Provides AI-based speech-to-text transcription using Whisper models when the YouTube API has no transcript.

**Key Classes:**

#### `WhisperSegment`
Represents a transcribed segment from Whisper.

```python
@dataclass
class WhisperSegment:
    start: float    # Start timestamp
    end: float      # End timestamp
    text: str       # Transcribed text
```

#### `WhisperRunner`
Main Whisper transcription engine with hardware acceleration.

```python
class WhisperRunner:
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

    def __init__(self, model_size: str = "base", device: Optional[str] = None, ...):
        """Initialize with auto-detected hardware."""

    def transcribe_file(self, audio_file: str, beam_size: int = 5, ...):
        """Transcribe a local audio file."""

    def transcribe_stream(self, video_url: str, beam_size: int = 1, ...):
        """Stream audio directly from YouTube to Whisper (no download)."""

    def transcribe(self, input_source: str, mode: str = "auto", ...):
        """Main method with flexible input handling."""
```

**Supported Models:**
| Model | Size | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| `tiny` | ~39 MB | Fastest | Lower | Quick drafts |
| `base` | ~74 MB | Fast | Good | Default choice |
| `small` | ~244 MB | Moderate | Better | Balanced |
| `medium` | ~769 MB | Slow | Very Good | High accuracy |
| `large-v3` | ~1.5 GB | Slowest | Excellent | Best quality |
| `large-v3-turbo` | ~1.5 GB | Moderate | Excellent | Best for streaming |

**Role in the system:** This is the "slow path" fallback. When the YouTube API has no transcript (most educational videos don't), Whisper runs full speech-to-text. The streaming mode (`transcribe_stream`) pipes audio directly from `yt-dlp` → `ffmpeg` → Whisper without writing to disk, saving time and disk space.

---

### 3.3 `transcriber/transcription_engine.py`

**Type:** Orchestration — Main Entry Point

**Purpose:** Combines API and Whisper into a single intelligent transcription service.

**Key Classes:**

#### `TranscriptionResult`
Result object containing all transcription metadata.

```python
@dataclass
class TranscriptionResult:
    video_id: str
    entries: List[TranscriptEntry]
    method: str              # "api" or "whisper"
    language: Optional[str]
    duration_seconds: float
    error: Optional[str]

    @property
    def transcript_text(self) -> str:
        """Get full transcript as plain text."""

    @property
    def entry_count(self) -> int:
        """Number of transcript segments."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

#### `TranscriptionEngine`
Main orchestration class implementing the hybrid strategy.

```python
class TranscriptionEngine:
    def __init__(self, whisper_model: str = "base", prefer_streaming: bool = False, ...):
        """Initialize with API fetcher and Whisper runner."""

    def transcribe(self, input_source: str, video_id: Optional[str] = None, skip_api: bool = False):
        """
        Main transcription method.

        Workflow:
        1. Extract video ID from URL if needed
        2. Try YouTube API (unless skip_api=True)
        3. Fall back to Whisper if API fails
        4. Return TranscriptionResult
        """

    def transcribe_to_json(self, input_source: str, output_file: str, ...):
        """Transcribe and save result to JSON file."""
```

**Role in the system:** This is the primary interface. External code calls `TranscriptionEngine.transcribe()` and gets a transcript regardless of whether it came from the API or Whisper. The engine abstracts away the complexity of trying multiple sources.

---

### 3.4 `utils/url_parser.py`

**Type:** Utility

**Purpose:** Extract video IDs and handle YouTube URL parsing.

**Functions:**

```python
def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from any URL format."""

def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS timestamp."""

def format_timestamp_with_ms(seconds: float) -> str:
    """Format seconds to MM:SS.mmm timestamp."""
```

**Supported URL Formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/v/VIDEO_ID`

**Role in the system:** Centralizes URL parsing logic so it's consistent across the module. Used by the transcription engine to extract video IDs for API lookup.

---

### 3.5 `main.py`

**Type:** CLI Entry Point

**Purpose:** Command-line interface for the transcription module.

**Usage:**
```bash
# Basic transcription
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Save to JSON
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o transcript.json

# Use larger model
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -m medium

# Streaming mode (faster)
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --stream

# Skip API, go directly to Whisper
python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --skip-api
```

**Role in the system:** Provides an easy way to run transcription from the command line for testing and batch processing. Internally, it uses the same `TranscriptionEngine` class that would be used programmatically.

---

## 4. Implemented Features — Detailed Explanation

---

### Feature 1: YouTube Transcript API Integration

**Files involved:** `transcriber/api_fetcher.py`

**What it does:**
Attempts to fetch pre-existing transcripts from YouTube's API. This is the fastest possible method because no AI processing or audio download is required.

**Workflow:**
1. Create `APIFetcher` instance (loads `YouTubeTranscriptApi`)
2. Call `fetch(video_id)` with an 11-character video ID
3. API returns list of transcript entries with timestamps
4. Entries are wrapped in `TranscriptEntry` objects for consistency

**Why this matters:**
Many educational videos have transcripts available (either auto-generated by YouTube or manually uploaded). Fetching these is instantaneous (0.1–0.5s) compared to 5–30s for Whisper. Trying the API first dramatically improves average performance across a batch of videos.

**Limitations:**
- Not all videos have transcripts
- Auto-generated transcripts may have timing inaccuracies
- Manual transcripts are rare on educational content

---

### Feature 2: Whisper/Faster-Whisper AI Transcription

**Files involved:** `transcriber/whisper_runner.py`

**What it does:**
Uses OpenAI's Whisper speech-to-text model to transcribe audio from scratch when no transcript is available. The module uses `faster-whisper`, an optimized reimplementation that's 4–5× faster than the original.

**Workflow:**
```python
# Load model (auto-detect hardware)
model = WhisperModel("base", device="cuda", compute_type="float16")

# Transcribe
segments, info = model.transcribe(
    audio_data,
    beam_size=5,           # Search width
    vad_filter=True,       # Skip silences
    language="en"          # Optional, auto-detected if None
)
```

**Key Parameters:**
| Parameter | Effect |
|---|---|
| `model_size` | Model size (tiny → large-v3-turbo) |
| `device` | "cuda" for GPU, "cpu" for CPU |
| `compute_type` | "float16" (GPU fast), "int8" (CPU efficient) |
| `beam_size` | 1 = fastest, 5 = balanced, 10+ = most accurate |
| `vad_filter` | Voice Activity Detection skips silence |

**Why this matters:**
Whisper is the most accurate open-source speech-to-text model available. It supports 99 languages, handles accented speech well, and works on any audio source. This provides a universal fallback when the YouTube API fails.

---

### Feature 3: Hybrid API + Whisper Fallback

**Files involved:** `transcriber/transcription_engine.py`

**What it does:**
Implements a two-tier strategy: try the fast API first, fall back to Whisper if it fails. This combines the best of both worlds — instant results when available, universal coverage otherwise.

**Workflow:**
```python
def transcribe(input_source, video_id=None, skip_api=False):
    # 1. Extract video ID if not provided
    if not video_id:
        video_id = extract_video_id(input_source)

    # 2. Try YouTube API (unless skipped)
    if not skip_api and video_id != "unknown":
        try:
            entries = api_fetcher.fetch(video_id)
            return TranscriptionResult(..., method="api")
        except Exception:
            pass  # Fall through to Whisper

    # 3. Fall back to Whisper
    entries = whisper_runner.transcribe(input_source)
    return TranscriptionResult(..., method="whisper")
```

**Why this matters:**
This hybrid approach provides the best average performance across a diverse video collection. Popular videos often have transcripts (fast), while niche educational content doesn't (Whisper). The user gets consistent results regardless of the source.

---

### Feature 4: Hardware Acceleration (CUDA/CPU)

**Files involved:** `transcriber/whisper_runner.py`

**What it does:**
Automatically detects and uses available hardware (NVIDIA GPU with CUDA) for maximum performance, with CPU fallback.

**Auto-detection Logic:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
```

**Performance Comparison:**
| Hardware | Model | 10-min video |
|---|---|---|
| CPU (8-core) | base | ~25s |
| GPU (RTX 3060) | base | ~5s |
| GPU (RTX 3060) | large-v3-turbo | ~8s |

**Why this matters:**
Hardware acceleration makes Whisper practical for batch processing. A 10-minute video that takes 25s on CPU becomes 5s on GPU — a 5× speedup. The auto-detection means the same code works on any machine.

---

### Feature 5: Streaming Transcription Mode

**Files involved:** `transcriber/whisper_runner.py`

**What it does:**
Pipes audio directly from `yt-dlp` → `ffmpeg` → Whisper without writing to disk. This eliminates the download step and saves disk space.

**Pipeline:**
```bash
yt-dlp -o - URL | ffmpeg -i pipe:0 -f s16le -ar 16000 pipe:1 | Whisper
```

**Python Implementation:**
```python
# Build command
cmd = ['yt-dlp', '-f', 'ba*[ext=m4a]', '-o', '-', url,
       '|', 'ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '16000', 'pipe:1']

# Run process
process = subprocess.Popen(" ".join(cmd), shell=True, stdout=subprocess.PIPE)

# Read stream directly into numpy array
raw_audio = process.stdout.read()
audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

# Transcribe from memory
segments, info = model.transcribe(audio_np)
```

**Why this matters:**
Streaming mode is typically 20–30% faster than download-then-transcribe because:
1. No disk I/O for the audio file
2. Transcription can start before the full stream is buffered
3. No temporary file cleanup required

This is especially useful for long videos where downloading a 100MB+ file adds significant overhead.

---

### Feature 6: Flexible Input Handling

**Files involved:** `transcriber/whisper_runner.py`, `utils/url_parser.py`

**What it does:**
Accepts multiple input types: YouTube URLs, local audio files, or raw audio data.

**Input Types:**
| Input | Handling |
|---|---|
| `https://youtube.com/...` | Extract video ID, try API, then Whisper |
| `audio.mp3` | Transcribe directly with Whisper |
| `audio.m4a` | Transcribe directly with Whisper |
| `audio.wav` | Transcribe directly with Whisper |

**URL Extraction:**
```python
def extract_video_id(url: str) -> Optional[str]:
    """Handles all YouTube URL formats."""
    patterns = [
        r'youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    # Try each pattern, return first match
```

**Why this matters:**
Flexibility means the module can be used in various contexts:
- Batch processing YouTube URLs (Fusion Module integration)
- Transcribing local audio files (offline content)
- Testing with downloaded audio (development)

---

## 5. Complete End-to-End Workflow

Below is the complete sequence of operations from input to transcript:

```
STEP 1 — Parse Input
─────────────────────────────────────────────────────────────────
  User calls: TranscriptionEngine.transcribe(input_source)

  If input_source is a YouTube URL:
    → extract_video_id(url) → 11-character video ID

  Result: video_id identified (or "unknown" for local files)


STEP 2 — Try YouTube Transcript API (Fast Path)
─────────────────────────────────────────────────────────────────
  api_fetcher.fetch(video_id)

  API returns: List of {start, duration, text} entries

  If success:
    → Wrap in TranscriptEntry objects
    → Return TranscriptionResult(method="api", ...)
    → DONE (total time: ~0.1–0.5s)


STEP 3 — Fallback to Whisper AI (Slow Path)
─────────────────────────────────────────────────────────────────
  If API fails or skip_api=True:

  whisper_runner.transcribe(input_source, mode="auto")

  Mode decision:
    - If streaming preferred AND input is URL → use streaming
    - Otherwise → download then transcribe


STEP 4a — Streaming Mode
─────────────────────────────────────────────────────────────────
  Build command: yt-dlp | ffmpeg → raw PCM

  Execute: subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

  Read: raw_audio = process.stdout.read()

  Convert: audio_np = np.frombuffer(raw_audio) / 32768.0

  Transcribe: model.transcribe(audio_np, beam_size=1, vad_filter=True)

  Result: List of TranscriptEntry objects


STEP 4b — Download Mode
─────────────────────────────────────────────────────────────────
  Download: yt_dlp.download([url]) → temp_audio.m4a

  Transcribe: model.transcribe("temp_audio.m4a", beam_size=5, vad_filter=True)

  Cleanup: os.remove("temp_audio.m4a")

  Result: List of TranscriptEntry objects


STEP 5 — Return Result
─────────────────────────────────────────────────────────────────
  TranscriptionResult(
    video_id="...",
    entries=[...],
    method="whisper",
    language="en",
    duration_seconds=12.3,
    error=None
  )

  Result can be:
    → Printed to console
    → Saved to JSON file
    → Passed to Fusion Module
```

---

## 6. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│                                                                  │
│  YouTube URL              Local Audio File                       │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ https://... │         │ audio.mp3    │                      │
│  │ watch?v=XYZ │         │              │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
└─────────┼──────────────────────┬─┘──────────────────────────────┘
          │    extract_video_id  │
          ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TRANSCRIPTION ENGINE                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  1. Try YouTube Transcript API (0.1–0.5s)           │       │
│  │     ┌─────────────────────────────────────────┐    │       │
│  │     │ APIFetcher.fetch(video_id)              │    │       │
│  │     │                                        │    │       │
│  │     │ Success? → Return immediately          │    │       │
│  │     └─────────────────────────────────────────┘    │       │
│  │                     │                              │       │
│  │                     │ Fail → Fallback               │       │
│  │                     ▼                              │       │
│  │  2. Whisper AI Transcription (5–30s)              │       │
│  │     ┌─────────────────────────────────────────┐    │       │
│  │     │ WhisperRunner.transcribe(...)           │    │       │
│  │     │                                        │    │       │
│  │     │ Mode: Streaming OR Download            │    │       │
│  │     │ Model: base / medium / large-v3-turbo  │    │       │
│  │     │ Hardware: CUDA (GPU) or CPU            │    │       │
│  │     └─────────────────────────────────────────┘    │       │
│  └──────────────────────────────────────────────────────┘       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
┌──────────────────────┐    ┌──────────────────────────┐
│   Streaming Mode     │    │    Download Mode         │
│                      │    │                          │
│  yt-dlp → ffmpeg     │    │  yt-dlp → temp_audio.m4a │
│  → raw PCM → Whisper │    │  → Whisper               │
│                      │    │  → delete temp file      │
└──────────────────────┘    └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │     TranscriptResult     │
              │                          │
              │  video_id                │
              │  entries: [{time, text}] │
              │  method: "api"/"whisper" │
              │  language                │
              │  duration_seconds        │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │        OUTPUT            │
              │                          │
              │  - Print to console      │
              │  - Save to JSON          │
              │  - Pass to Fusion Module │
              └──────────────────────────┘
```

---

## 7. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.x | Primary programming language |
| API Client | `youtube-transcript-api` | YouTube Transcript API access |
| AI Model | `faster-whisper` | Optimized Whisper implementation |
| Base Model | OpenAI Whisper | Speech-to-text foundation model |
| Video Download | `yt-dlp` | YouTube audio extraction |
| Audio Processing | `ffmpeg` | Audio format conversion |
| Hardware Detect | `torch` (PyTorch) | CUDA availability detection |
| NumPy Support | `numpy` | Audio buffer handling |
| Serialization | `json` (stdlib) | JSON output format |

**Model Sizes:**
| Model | Parameters | Speed | Accuracy |
|---|---|---|---|
| tiny | ~39M | Fastest | Lower |
| base | ~74M | Fast | Good |
| small | ~244M | Moderate | Better |
| medium | ~769M | Slow | Very Good |
| large-v3 | ~1.5B | Slowest | Excellent |
| large-v3-turbo | ~1.5B | Moderate | Excellent |

---

## 8. Design Decisions & Notes

### Why `faster-whisper` over `openai/whisper`?
`faster-whisper` is a C++ reimplementation of Whisper that's 4–5× faster while maintaining the same accuracy. It also has:
- Better memory efficiency
- Easier installation (no PyTorch CUDA build headaches)
- Better VAD (Voice Activity Detection) support

### Why two modes (streaming vs download)?
- **Streaming**: Faster, no disk I/O, but requires stable network
- **Download**: More reliable for flaky connections, allows re-transcription with different models

The module defaults to streaming for URLs but can be configured.

### Why beam_size=1 for streaming but 5 for download?
- `beam_size=1` (greedy search) is fastest — good for streaming where latency matters
- `beam_size=5` provides better accuracy — good for download mode where we want quality

### Why separate `TranscriptEntry` and `WhisperSegment`?
They serve different purposes:
- `TranscriptEntry`: Standardized output format for the entire module
- `WhisperSegment`: Internal Whisper-specific representation

This allows changing the underlying AI model without affecting the output format.

### Lazy Model Loading
The Whisper model is loaded on first use, not at module import. This:
- Reduces startup time
- Saves memory if transcription isn't needed
- Allows per-instance configuration

### VAD (Voice Activity Detection)
Enabled by default in Whisper transcriptions. This:
- Skips silence periods
- Reduces processing time
- Produces cleaner segment boundaries

---

*Documentation generated: April 2, 2026*
