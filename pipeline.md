# Synapse Pipeline - Complete Documentation

## Overview

Synapse is an AI-powered accessibility pipeline for blind students. It processes educational STEM lecture videos from YouTube and detects "suffer points" where visual content is not accessible to blind students. At these points, it generates audio descriptions that seamlessly integrate with the original lecture.

### The Problem

Blind students lose access to visual content when:
- Teacher says "look at this graph" without describing it (Deictic)
- Teacher draws silently on the board (Silent Drawing)
- Visual content is complex but teacher ignores it (High Complexity)

### The Solution

Synapse automatically:
1. Extracts audio and video from YouTube in parallel
2. Transcribes speech to text
3. Identifies pedagogically important keyframes
4. Detects suffer points using a scoring system
5. Analyzes visual content with Vision Language Models (VLM) - 10 content types
6. Weaves visual descriptions into the spoken narrative using 3-mode explanation system
7. Generates audio with TTS provider fallback (Edge-TTS + SpeechT5)
8. Produces complete audio-described track or intervention-only output

---

## Pipeline Architecture

```
YouTube URL
    ↓
┌─────────────────────────────────┐
│     Parallel Extraction       │
│  Thread A: Audio Transcription │
│         ↓                       │
│  transcript.json               │
│                                │
│  Thread B: Video Keyframes     │
│         ↓                       │
│  keyframes/ + keyframes.json    │
└─────────────────────────────────┘
                ↓
        Synchronizer (M1)
        Detect suffer points → intervention_points
                ↓
        VLM Interface (M5a)
        Load & process images for VLM
                ↓
        VLM Engine (M2)
        Analyze ONLY intervention points (optimization)
        → vlm_snapshots
                ↓
        LLM Interface (M5b)
        Generate synthesized text via Ollama
                ↓
        Synthesizer (M3)
        Match VLM to transcript segments + weave
        → synthesized_text for ALL segments
                ↓
        TTS Engine (M6)
        Generate MP3 for all segments
                ↓
        Orchestrator (M4)
        Create manifest.json
                ↓
        Final Output
        - manifest.json
        - audio_segments/*.mp3
```

---

## All Modules (9 Total)

### M0: Frame Extraction
**File:** `frame-extraction/pedagogical_extractor_v2.py`
**Status:** ✅ EXISTS

| Attribute | Details |
|-----------|---------|
| **Input** | YouTube URL |
| **Output** | `keyframes/` directory + `keyframes.json` |
| **Technology** | CLIP-based keyframe extraction with complexity scoring |
| **Key Feature** | 7-phase pipeline selects pedagogically important frames |

**What it does:**
- Streams video from YouTube
- Extracts frames at regular intervals
- Uses CLIP embeddings to score each frame's pedagogical value
- Applies complexity scoring
- Selects keyframes representing important visual moments
- **Critical:** Outputs `timestamp_seconds` as float for synchronization

---

### M0b: Transcription
**File:** `Transcription-Module--Synapse-/transcriber/transcription_engine.py`
**Status:** ✅ EXISTS

| Attribute | Details |
|-----------|---------|
| **Input** | YouTube URL |
| **Output** | `transcript.json` |
| **Technology** | Whisper API (hybrid with local Whisper) |
| **Key Feature** | Timestamps as floats for synchronization |

**What it does:**
- Extracts audio track from video
- Transcribes speech using Whisper
- Generates timestamps (`start`, `end`) as floats
- Outputs transcript segments with timing and text

---

### M1: Synchronizer
**File:** `synchronizer/synchronizer.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | `transcript.json`, `keyframes.json` |
| **Output** | `List[InterventionPoint]` |
| **Technology** | Scoring system with multiple factors |
| **Key Feature** | Detects when blind students lose visual access |

**Scoring System:**

| Trigger | Score | Condition |
|---------|-------|-----------|
| Deictic phrases | +50 | "look at", "here we see", "this graph", "notice", "observe", "check this", "see this", "here's", "look here" |
| Silent drawing | +80 | Text < 10 chars AND complexity > 0.3 |
| High complexity | +50 | Complexity score > 0.30 |
| Negative deictic | -50 | "next slide", "moving on", "let's continue", "moving forward", "now let's", "next we'll", "moving to", "next topic" |

**Threshold:** Score >= 50 triggers intervention

**Redundancy Check:** Skips interventions within 5 seconds of previous intervention

**What it does:**
- Loads transcript and keyframes
- Matches each frame to transcript segment by timestamp
- Calculates intervention score for each frame
- Returns list of intervention points requiring visual description

---

### M5a: VLM Interface
**File:** `vlm_interface/vlm_interface.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | Image path/PIL, prompt, generation_params |
| **Output** | `VLMResponse` (text + metadata) |
| **Technology** | Transformers, Qwen2-VL-7B-Instruct |
| **Key Feature** | Supports 4-bit quantization, CUDA/CPU auto-detection |

**What it does:**
- Loads Qwen2-VL model with device management
- Handles image loading and preprocessing
- Formats prompts with chat templates
- Generates text responses from visual input
- Manages GPU memory with cleanup

**Optimizations:**
- 4-bit quantization: ~5-6GB VRAM (vs 13GB full precision)
- Flash Attention 2 support
- Low CPU memory usage during loading
- Tensor cleanup after each inference

---

### M2: VLM Engine (Snapshot Engine)
**File:** `vlm_engine/snapshot_engine.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | `frame_path`, `primary_type` (content category) |
| **Output** | JSON dict with visual analysis |
| **Technology** | Qwen2-VL with category-specific prompts |
| **Key Feature** | Chain-of-Regions (2x2 grid) scanning |

**Output Structure:**
```json
{
  "content_type": "dominant content type",
  "detected_types": ["all", "types", "present"],
  "visual_analysis": {
    "graph": { ... },
    "equation": [ ... ],
    "circuit": { ... },
    "diagram": { ... },
    "code": { ... },
    "handwritten_notes": { ... },
    "biology": { ... },
    "chemistry": { ... },
    "physics": { ... },
    "text": { ... }
  },
  "structural_description": "Complete spatial narrative for mental model building",
  "reading_order": ["Step 1", "Step 2", "Step 3"],
  "conceptual_hints": "What this visual demonstrates conceptually",
  "layout": "Overall physical arrangement",
  "text_readout": "Linear text from ALL regions",
  "spatial_map": "Geometry/relationships in ALL regions",
  "colors_styles": "Visual descriptors (colors, line styles)",
  "missing_elements": "null if successful, 'No clear STEM visual' if irrelevant/blurry/empty"
}
```

**Content Types (10 categories):**

| Type | Analysis Focus |
|------|----------------|
| `equation` | Read linearly, convert symbols to words (squared, subscript, etc.) |
| `graph` | Identify axes, labels, trends, key data points |
| `circuit` | Trace flow, identify components, values, topology |
| `diagram` | List parts, arrows, spatial relationships |
| `code` | Read literally, identify language and structure |
| `handwritten_notes` | Mark [ILLEGIBLE] text, describe layout |
| `biology` | Identify structures, parts, scale, labels |
| `chemistry` | Describe molecules, bonds, geometry, formulas |
| `physics` | Identify physical objects, forces/vectors, spatial relationships, measurements |
| `text` | Hierarchy, formatting, structure, key terms |

**Chain-of-Regions Method:**
- Divides frame into 2x2 grid (top-left, top-right, bottom-left, bottom-right)
- Systematically analyzes each quadrant
- Ensures small details like labels and arrows are not missed

---

### M5b: LLM Interface
**File:** `llm_interface/ollama_client.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | Prompt, model, stream, options |
| **Output** | `OllamaResponse` (text + metadata) |
| **Technology** | HTTP to Ollama API (localhost:11434) |
| **Key Feature** | Retry logic with exponential backoff |

**Default Configuration:**
- Model: `mistral`
- Base URL: `http://localhost:11434`
- Timeout: 30s
- Max retries: 3
- Retry delays: [1s, 2s, 4s]

**Error Handling:**
| Error Type | Action |
|-----------|--------|
| ConnectionError | Retry 3x with exponential backoff |
| TimeoutError | Retry 3x with exponential backoff |
| Server Error (5xx) | Retry 3x with exponential backoff |
| Client Error (4xx) | Fail immediately (no retry) |

**Fusion Module Integration:**
- Reuses `Fusion-Module--Synapse-/llm/mistral_runner.py` when available
- Adds structured response wrapper and timing
- Maintains backward compatibility

---

### M3: Synthesizer
**File:** `synthesizer/llm_synthesizer.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | `transcript_context`, `vlm_snapshot`, `content_type`, `intervention_reason`, `output_mode` |
| **Output** | `audio_script` string ready for TTS |
| **Technology** | Mistral via Ollama with three weaving modes |
| **Key Feature** | Unified Injection Rule - VLM data never discarded for interventions |

**Three Weaving Modes:**

| Mode | Trigger | Action | Example |
|------|---------|--------|---------|
| **Integration** | Deictic | Replace vague gestures with facts | "Look at this curve..." → "Look at this curve, which rises steeply..." |
| **Bridge** | High Complexity | Add bridge phrase | "This proves the theory. Visually, the diagram..." |
| **Narrator** | Silent/Drawing | Standalone narration | "On the screen, the equation F equals m times a..." |

**Output Modes:**

| Mode | Description |
|------|-------------|
| `brief` | Surface-level explanation - essential visual elements + main concept + key connection |
| `explanatory` | Standard-depth explanation - complete visual description + full concept explanation + clear connections |
| `detailed` | Deep exploration - exhaustive visual description + deep conceptual framework + multiple connections + cross-domain |

**Safety Heuristics:**

1. **Missing Elements:** If VLM marks `missing_elements`, fall back to transcript-only
2. **Cross-Domain Hallucination:** If VLM describes "person" for a "circuit", fall back to transcript-only
3. **Empty LLM Response:** Retry once, then fall back to transcript

**Cross-Domain Mappings:**
- `equation` should not describe: person, man, woman, face, people, teacher
- `graph` should not describe: person, man, woman, face, people
- `circuit` should not describe: person, man, woman, face, text, paragraph
- (and similar mappings for other types)

---

### M6: TTS Engine
**File:** `tts_engine/tts_engine.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | `text`, `output_path` |
| **Output** | `.mp3` file |
| **Technology** | Microsoft Edge-TTS (primary) + SpeechT5 (fallback) |
| **Key Feature** | Provider fallback, async/sync bridging, GPU memory management |

**Providers:**
- **Primary:** Edge-TTS (high quality, requires internet)
- **Fallback:** SpeechT5 (good quality, works offline)

**Default Voice:** `en-US-AndrewMultilingualNeural`

**Voice Alternatives (Edge-TTS):**

| Voice ID | Gender | Style | Use Case |
|----------|--------|-------|----------|
| `en-US-AndrewMultilingualNeural` | Male | Authoritative | Default, STEM content |
| `en-US-AriaNeural` | Female | Expressive | Narrative, storytelling |
| `en-US-GuyNeural` | Male | Conversational | Casual explanations |
| `en-US-JennyNeural` | Female | Friendly | Welcoming introductions |

**Provider Fallback:**
- Edge-TTS requires internet connection
- SpeechT5 works offline with local models
- Auto-fallback when Edge-TTS fails (network issues, timeout)

**Performance:**
- Short segment (10-30 words): 0.3-0.8s
- Medium segment (30-100 words): 0.8-2s
- Long segment (100-300 words): 2-5s

---

### M4: Orchestrator
**File:** `orchestrator/orchestrator.py`
**Status:** ✅ COMPLETE

| Attribute | Details |
|-----------|---------|
| **Input** | `youtube_url`, `base_dir` |
| **Output** | `manifest.json` + `audio_segments/*.mp3` (or `interventions/` for intervention-only mode) |
| **Technology** | Threading for parallel extraction |
| **Key Feature** | Central lifecycle manager with intervention-only workflow |

**The 6-Step Workflow:**

1. **Ingestion**
   - Extract video ID from YouTube URL
   - Create directory structure: `data/{video_id}/keyframes/`, `audio_segments/`, `interventions/`

2. **Parallel Extraction** (Threading)
   - Thread A: Transcription (audio)
   - Thread B: Frame extraction (video)
   - Benefits: 10-20s time savings on 10-minute video

3. **Synchronization**
   - Load transcript and keyframes
   - Run Synchronizer to detect suffer points
   - Output: `List[InterventionPoint]`

4. **VLM Snapshots** (Optimized)
   - Initialize Qwen2-VL (7B or 2B fallback)
   - Analyze **ONLY intervention points** (not all keyframes)
   - 90-95% reduction in VLM calls
   - Map: `timestamp → vlm_snapshot`

5. **Synthesis & TTS**
   - Process using 3-mode explanation system (`brief`, `explanatory`, `detailed`)
   - Full pipeline mode: Generate audio for all segments
   - Intervention-only mode: Generate audio and metadata for interventions only
   - Result: Seamless "Audio Described Track" or intervention collection

6. **Manifest Creation**
   - Full pipeline: Build `manifest.json` with all segments
   - Intervention-only: Build `interventions_manifest.json` with intervention metadata
   - Save to `data/{video_id}/`

**Spatiotemporal Matching:**
- Finds VLM snapshot with minimal time difference
- Maximum window: 30 seconds (configurable)
- Returns None if outside window

---

## Data Flow Example

```
YouTube: https://www.youtube.com/watch?v=abc123
    ↓
[Parallel Extraction]
├─ Transcription: "This graph shows velocity increasing..."
└─ Keyframes: frame_001.jpg (t=15.5, content_type=graph, complexity=0.45)
    ↓
[Synchronizer]
Frame at 15.5s matches transcript: "Look at this graph"
Score: +50 (Deictic) + 30 (High Complexity) = 80 ≥ 50 ✓
InterventionPoint created at t=15.5
    ↓
[VLM Engine]
Analyzes frame_001.jpg with graph prompt
Output: {"content_type": "graph", "detected_types": ["graph"], "visual_analysis": {...}, "structural_description": "...", "reading_order": [...], "conceptual_hints": "...", "layout": "Line graph on white", "text_readout": "X-axis: Time, Y-axis: Velocity", ...}
    ↓
[Synthesizer - Explanatory Mode]
Input: "Look at this graph" + VLM data
Output: "Look at this graph, which shows velocity increasing linearly with time on the x-axis and reaching 50 meters per second on the y-axis."
    ↓
[TTS Engine]
Generates: segment_0005.mp3 (15 seconds) using Edge-TTS or SpeechT5
    ↓
[Manifest]
{
  "video_id": "abc123",
  "total_segments": 150,
  "intervention_count": 24,
  "segments": [...]
}
```

---

## How to Run the Pipeline

### Prerequisites

1. **Python 3.9+**
2. **Virtual Environment:** `synapse-env`
3. **Ollama running with Mistral model**
4. **Qwen2-VL model downloaded** (already at `models/Qwen2-VL-7B-Instruct/`)
5. **FFmpeg** (bundled in `frame-extraction/ffmpeg/bin/`)
6. **Sufficient disk space** (~20GB for models + data)

### Step 1: Activate Virtual Environment

**Windows:**
```bash
F:\Prototyping\Synapse\synapse-env\Scripts\activate
```

**Linux/Mac:**
```bash
source synapse-env/bin/activate
```

### Step 2: Start Ollama (if not running)

```bash
ollama serve
```

In a separate terminal, verify Mistral is available:
```bash
ollama list
# Should show: mistral
```

### Step 3: Verify Dependencies

```bash
pip list | findstr -i "transformers torch edge-tts qwen-vl-utils sentence-transformers"
```

Required packages:
- `transformers>=2.0.0`
- `torch>=2.0.0`
- `edge-tts`
- `qwen-vl-utils`
- `sentence-transformers`
- `openai-clip`
- `yt-dlp`
- `faster-whisper`

### Step 4: Run the Pipeline

Create a Python script `run_synapse.py`:

**Full Pipeline Mode:**
```python
from orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(base_dir="data")

# Process a YouTube video with output mode
video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
output_mode = "explanatory"  # Options: brief, explanatory, detailed

try:
    manifest = orchestrator.process_video(video_url, output_mode=output_mode)

    print("\n=== Pipeline Summary ===")
    print(f"Video ID: {manifest['video_id']}")
    print(f"Total segments: {manifest['total_segments']}")
    print(f"Interventions: {manifest['intervention_count']}")
    print(f"\nOutput directory: data/{manifest['video_id']}/")

except Exception as e:
    print(f"Pipeline failed: {e}")
```

**Intervention-Only Mode:**
```python
from orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(base_dir="data")
video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
output_mode = "detailed"  # Options: brief, explanatory, detailed

try:
    result = orchestrator.process_video_intervention_only(video_url, output_mode=output_mode)

    print("\n=== Intervention Summary ===")
    print(f"Video ID: {result['video_id']}")
    print(f"Total interventions: {len(result['interventions'])}")
    print(f"Output directory: data/{result['video_id']}/interventions/")

except Exception as e:
    print(f"Pipeline failed: {e}")
```

Run it:
```bash
python run_synapse.py
```

### Step 5: Check Output

After completion, check:

```bash
# Navigate to output directory
cd data/YOUR_VIDEO_ID/

# Check files
ls -l

# Expected output:
# keyframes/           # Extracted images
# audio_segments/      # Generated MP3s
# transcript.json      # Original transcript
# keyframes.json       # Keyframe metadata
# manifest.json        # Master manifest
```

### Example: Interactive Run

```python
from orchestrator import PipelineOrchestrator

# Get YouTube URL from user
video_url = input("Enter YouTube URL: ")

# Run pipeline
orchestrator = PipelineOrchestrator()
manifest = orchestrator.process_video(video_url)

# Display results
print(f"\n✅ Processing complete!")
print(f"   Video ID: {manifest['video_id']}")
print(f"   Segments generated: {manifest['total_segments']}")
print(f"   Interventions added: {manifest['intervention_count']}")
print(f"   Output: data/{manifest['video_id']}/manifest.json")
```

---

## Output Structure

```
data/
└── {video_id}/
    ├── keyframes/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   └── ...
    ├── audio_segments/
    │   ├── segment_0000.mp3
    │   ├── segment_0001.mp3
    │   └── ...
    ├── transcript.json
    ├── keyframes.json
    └── manifest.json
```

### Manifest.json Structure

```json
{
  "video_id": "abc123",
  "total_segments": 150,
  "intervention_count": 24,
  "created_at": "2026-04-20T12:34:56.789",
  "segments": [
    {
      "chunk_id": 0,
      "timestamp": 0.0,
      "text": "Welcome to today's lecture...",
      "is_intervention": false,
      "audio_file": "audio_segments/segment_0000.mp3",
      "original_transcript": "Welcome to today's lecture...",
      "vlm_data": null
    },
    {
      "chunk_id": 5,
      "timestamp": 15.5,
      "text": "Look at this graph, which shows velocity increasing linearly...",
      "is_intervention": true,
      "audio_file": "audio_segments/segment_0005.mp3",
      "original_transcript": "Look at this graph",
      "vlm_data": {
        "layout": "Line graph on white background",
        "text_readout": "X-axis: Time (seconds), Y-axis: Velocity (m/s)",
        "spatial_map": "Line starts at origin, rises to top-right",
        "colors_styles": "Blue line, black axes",
        "missing_elements": null
      }
    }
  ]
}
```

---

## Performance Expectations

### Processing Time (10-minute video)

| Step | Time | Notes |
|------|------|-------|
| Ingestion | < 100ms | URL parsing, directory creation |
| Parallel Extraction | 30-60s | Audio (20-40s) + Video (10-20s) |
| Synchronization | < 100ms | Scoring and matching |
| VLM Snapshots | 50-250s | ~2-5s per intervention (25-50 expected) |
| Synthesis + TTS | 30-150s | ~1-2s LLM + ~0.5s TTS per segment |
| Manifest Creation | < 100ms | JSON serialization |
| **Total** | **110-460s** | ~2-8 minutes |

### Key Optimizations

1. **Parallel Extraction:** Saves 10-20s by running audio and video extraction simultaneously
2. **VLM Optimization:** Analyzing only intervention points saves 900-1900s (assuming 1000 keyframes, 50 interventions)
3. **Caching:** Transcript embeddings can be cached for faster subsequent runs (future enhancement)

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| VRAM (GPU) | 6GB (with 4-bit) | 12GB+ |
| Storage | 30GB | 50GB+ |
| CPU | 4 cores | 8+ cores |
| Internet | Required (Edge-TTS) | Optional (SpeechT5 fallback) |

---

## Troubleshooting

### Issue: "Ollama connection failed"

**Solution:**
```bash
# Start Ollama
ollama serve

# In another terminal, verify
curl http://localhost:11434/api/tags

# Ensure Mistral is installed
ollama pull mistral
```

### Issue: "CUDA out of memory"

**Solution:**
- Enable 4-bit quantization (already default in VLM Interface)
- Reduce `max_pixels` in VLM Interface
- Use CPU fallback (slower but works)

### Issue: "Missing timestamp_seconds in keyframes.json"

**Solution:**
- This should be fixed in `pedagogical_extractor_v2.py`
- Verify line 939 has the fix:
```python
report = [{
    "timestamp": r.timestamp,
    "timestamp_seconds": float(r.timestamp_seconds),  # This line
    "frame_path": r.frame_path,
    ...
} for r in results]
```

### Issue: "Transcription module not found"

**Solution:**
- Ensure project root is in Python path
- Check `Transcription-Module--Synapse-` directory exists

### Issue: "TTS Engine timeout"

**Solution:**
- Check internet connection (Edge-TTS requires online access)
- Try a different voice
- The engine has built-in retry logic

---

## Advanced Usage

### Custom Voice

```python
from tts_engine import TTSEngine

# Use different voice
engine = TTSEngine(voice="en-US-AriaNeural")

# List all available voices
TTSEngine.list_available_voices()
```

### Custom Output Mode

```python
from synthesizer import LLMSynthesizer

synthesizer = LLMSynthesizer()

# Use brief mode
audio_script = synthesizer.weave(
    transcript_context=transcript,
    vlm_snapshot=vlm_data,
    content_type="graph",
    intervention_reason="Deictic Phrase",
    output_mode="brief"  # Options: brief, explanatory, detailed
)
```

### Output Modes Explained

| Mode | Depth | Best For |
|------|--------|-----------|
| `brief` | Surface level | Quick overviews, rapid understanding |
| `explanatory` | Standard depth | Complete coverage, normal teaching depth |
| `detailed` | Deep exploration | Thorough understanding, tutoring sessions |

### Batch Processing

```python
from orchestrator import PipelineOrchestrator

video_urls = [
    "https://www.youtube.com/watch?v=video1",
    "https://www.youtube.com/watch?v=video2",
    "https://www.youtube.com/watch?v=video3",
]

orchestrator = PipelineOrchestrator()

for url in video_urls:
    print(f"\nProcessing: {url}")
    manifest = orchestrator.process_video(url)
    print(f"✅ Complete: {manifest['video_id']}")
```

---

## Module Import Paths

```python
# M0: Frame Extraction (existing)
from frame_extraction.pedagogical_extractor_v2 import extract_pedagogical_keyframes

# M0b: Transcription (existing)
from transcriber.transcription_engine import TranscriptionEngine

# M1: Synchronizer
from synchronizer import Synchronizer, InterventionPoint, TranscriptEntry

# M2: VLM Engine
from vlm_engine import SnapshotEngine

# M5a: VLM Interface
from vlm_interface import Qwen2VLInterface, VLMResponse

# M5b: LLM Interface
from llm_interface import OllamaClient, OllamaResponse

# M3: Synthesizer
from synthesizer import LLMSynthesizer

# M6: TTS Engine
from tts_engine import TTSEngine

# M4: Orchestrator
from orchestrator import PipelineOrchestrator
```

---

## Future Enhancements

### Planned Features

1. **Semantic Retrieval** - Use embeddings instead of timestamps for VLM-transcript matching
2. **Vector Persistence** - Cache transcript embeddings for faster subsequent runs
3. **Progress Tracking** - Emit progress events for UI feedback
4. **Resume Capability** - Check for partial results and resume
5. **Batch VLM Processing** - Process multiple frames at once
6. **Distributed Processing** - Distribute across multiple machines

### Recently Implemented

1. **3-Mode Explanation System** - Brief, Explanatory, and Detailed output modes for flexible intervention depth
2. **Intervention-Only Workflow** - Dedicated workflow for generating only interventions with separate metadata and directory structure
3. **TTS Provider Fallback** - SpeechT5 fallback for offline TTS generation when Edge-TTS is unavailable
4. **Physics Content Type** - Added physics as a 10th content type for physical diagrams and force vectors
5. **Unified VLM Output Format** - Enhanced JSON structure with visual_analysis, reading_order, and conceptual_hints

### Enhancement 1: Semantic Retrieval

Current: Timestamp-based matching
Future: Cosine similarity of embeddings from `all-MiniLM-L6-v2`

Benefits:
- Finds right context even if timing doesn't match
- Captures meaning instead of just timestamps
- Robust to timing drift

### Enhancement 2: Vector Persistence

Current: Recompute embeddings every run
Future: Save to `transcript_embeddings.pkl`

Benefits:
- First run: ~5-10s
- Subsequent runs: ~100ms
- Massive performance improvement

---

## Summary

The Synapse pipeline is **fully functional** and can process YouTube videos end-to-end with a single command:

```python
from orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
manifest = orchestrator.process_video("YOUTUBE_URL", output_mode="explanatory")
```

All 9 modules are implemented and integrated. The pipeline automatically:

1. ✅ Extracts audio and video in parallel
2. ✅ Transcribes speech
3. ✅ Identifies pedagogically important frames
4. ✅ Detects suffer points where blind students lose access
5. ✅ Analyzes visual content with VLM (10 content types)
6. ✅ Weaves descriptions into the narrative using 3-mode explanation system
7. ✅ Generates audio with TTS provider fallback (Edge-TTS + SpeechT5)
8. ✅ Produces a manifest for frontend consumption

**Available Modes:**
- **Full Pipeline**: Generates complete audio-described track
- **Intervention-Only**: Generates only interventions with metadata

The result is a seamless, accessible audio-described version of the original lecture that blind students can listen to from start to finish without missing visual information.

---

*Last Updated: April 26, 2026*
