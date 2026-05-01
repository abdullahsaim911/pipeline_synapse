# Orchestrator Module (M4) Documentation

## Overview

The Orchestrator Module serves as the "Central Brain" of the Synapse project, managing the entire pipeline lifecycle from ingestion to final audio-visual manifest. It coordinates all modules, handles parallel execution, manages GPU memory, and provides multiple workflow options including full pipeline, detection-only, and intervention-only modes.

## Purpose

- Coordinate all pipeline modules in correct sequence
- Manage parallel extraction (audio + video)
- Handle GPU memory allocation and cleanup
- Provide multiple workflow options
- Generate final manifest files
- Support single-intervention generation (for API)
- Handle Ollama process lifecycle

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **Threading**: Parallel execution
- **Subprocess**: External process management
- **JSON**: Data serialization
- **PyTorch**: GPU memory management
- **Ollama**: LLM process management
- **SQLite**: Database for state management (optional)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| GPU (VRAM) | 6GB | 12GB+ |
| CPU | 4 cores | 8+ cores |
| Disk Space | 30GB | 50GB+ |
| Internet | Required (Edge-TTS) | Optional (SpeechT5 fallback) |

## Implementation Details

### Module Structure

```
orchestrator/
├── __init__.py
├── orchestrator.py             # Main orchestration logic
└── gpu_memory_manager.py       # GPU memory utilities
```

### Core Components

#### 1. PipelineOrchestrator Class

**Main Orchestrator Class**: Manages complete pipeline lifecycle

```python
class PipelineOrchestrator:
    """
    Central lifecycle manager for the entire Synapse pipeline.

    Coordinates: Parallel Extraction → Synchronization → VLM Analysis →
    Synthesis → TTS Generation → Manifest Creation
    """

    def __init__(self, base_dir: str = "data"):
        """
        Initialize Orchestrator.

        Args:
            base_dir: Root directory for data storage (default: data)
        """
        self.base_dir = base_dir
        print(f"[Orchestrator] Initialized with base_dir: {base_dir}")
```

#### 2. Video ID Extraction

```python
@staticmethod
def _extract_video_id(youtube_url: str) -> str:
    """
    Extract video ID from YouTube URL using Regex.

    Args:
        youtube_url: Full YouTube URL

    Returns:
        11-character video ID

    Raises:
        ValueError: If URL format is invalid
    """
    pattern = r'(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([\w-]{11})'
    match = re.search(pattern, youtube_url)

    if not match:
        raise ValueError(f"Invalid YouTube URL format: {youtube_url}")

    return match.group(1)
```

#### 3. Ingestion Phase

```python
def _ingest(self, youtube_url: str) -> str:
    """
    Ingest video and extract video ID.

    Args:
        youtube_url: YouTube URL to process

    Returns:
        Video ID (11 characters)
    """
    video_id = self._extract_video_id(youtube_url)
    print(f"[Orchestrator] Extracted video ID: {video_id}")

    # Create video directory structure
    video_dir = f"{self.base_dir}/{video_id}"
    os.makedirs(f"{video_dir}/keyframes", exist_ok=True)
    os.makedirs(f"{video_dir}/audio_segments", exist_ok=True)

    return video_id
```

#### 4. Parallel Extraction

```python
def _execute_parallel_extraction(
    self,
    youtube_url: str,
    video_dir: str
) -> tuple:
    """
    Execute parallel extraction of audio and video.

    Args:
        youtube_url: YouTube URL
        video_dir: Video directory for output

    Returns:
        Tuple of (transcript_result, keyframes_result)
    """
    print("[Orchestrator] Starting parallel extraction...")

    # Results containers
    transcript_result = None
    keyframes_result = None

    # Define thread functions
    def extract_audio():
        nonlocal transcript_result
        try:
            # Use subprocess to run transcription module's main.py
            transcription_main = os.path.join(project_root, "transcription", "main.py")
            python_exe = sys.executable

            cmd = [
                python_exe,
                transcription_main,
                youtube_url,
                "-o", f"{video_dir}/transcript.json",
                "-m", "base",
                "-q"
            ]

            print(f"[Orchestrator] Running transcription...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                if os.path.exists(f"{video_dir}/transcript.json"):
                    with open(f"{video_dir}/transcript.json", "r") as f:
                        transcript_result = json.load(f)
                    print("[Orchestrator] Audio extraction complete")
        except Exception as e:
            print(f"[Orchestrator] Audio extraction failed: {e}")

    def extract_video():
        nonlocal keyframes_result
        try:
            # Import the frame extraction function
            from frame_extraction.pedagogical_extractor_v2 import extract_pedagogical_keyframes

            # Extract video ID from URL
            video_id = self._extract_video_id(youtube_url)

            # Call the extraction function
            result = extract_pedagogical_keyframes(
                video_id=video_id,
                output_dir=f"{video_dir}/keyframes",
                verbose=False
            )
            keyframes_result = {"keyframes": result}
            print("[Orchestrator] Video extraction complete")
        except Exception as e:
            print(f"[Orchestrator] Video extraction failed: {e}")

    # Run threads in parallel
    audio_thread = threading.Thread(target=extract_audio)
    video_thread = threading.Thread(target=extract_video)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    print("[Orchestrator] Parallel extraction complete")

    # CRITICAL: Clear GPU memory after Whisper before VLM loads
    print("[Orchestrator] Cleaning GPU memory after transcription...")
    clear_gpu_memory()

    return transcript_result, keyframes_result
```

#### 5. Synchronization Phase

```python
def _synchronize(self, video_dir: str) -> List:
    """
    Detect suffer points where blind students lose visual access.

    Args:
        video_dir: Video directory with transcript.json and keyframes.json

    Returns:
        List of InterventionPoint objects
    """
    print("[Orchestrator] Synchronizing transcript and keyframes...")

    from synchronizer import Synchronizer, TranscriptEntry

    synchronizer = Synchronizer(
        transcript_path=f"{video_dir}/transcript.json",
        keyframes_path=f"{video_dir}/keyframes/keyframes.json"
    )

    intervention_points = synchronizer.detect_suffer_points()
    print(f"[Orchestrator] Found {len(intervention_points)} suffer points")

    return intervention_points
```

#### 6. VLM Analysis Phase

```python
def _analyze_vlm_snapshots(
    self,
    intervention_points: List,
    video_dir: str
) -> Dict:
    """
    Analyze ONLY intervention points with VLM (optimization).

    Uses GGUF backend with fallback to 2B model.

    Args:
        intervention_points: List of InterventionPoint objects
        video_dir: Video directory

    Returns:
        Dictionary mapping timestamp → VLM snapshot
    """
    print("[Orchestrator] Analyzing VLM snapshots (only intervention points)...")

    # CRITICAL: Ensure GPU is clean and has enough memory before VLM loads
    try:
        ensure_gpu_free(5.0)
    except Exception as e:
        print(f"[Orchestrator] GPU memory check failed: {e}")
        print("[Orchestrator] Will try with CPU offloading or smaller model")

    # Model paths with fallback chain
    model_configs = [
        {
            "path": os.path.join(project_root, "models", "Qwen2-VL-7B-Instruct-GGUF"),
            "use_gguf": True,
            "n_gpu_layers": 24,
            "name": "Qwen2-VL-7B-GGUF"
        }
    ]

    vlm_engine = None
    vlm_snapshots = {}
    used_model = None

    for config in model_configs:
        try:
            print(f"[Orchestrator] Attempting to load model: {config['name']}")

            vlm_engine = SnapshotEngine(
                model_path=config["path"],
                use_gguf=config.get("use_gguf", False),
                n_gpu_layers=config.get("n_gpu_layers", 24),
                use_4bit=config.get("use_4bit", True)
            )

            used_model = config["name"]
            print(f"[Orchestrator] Successfully loaded: {used_model}")
            break

        except Exception as e:
            print(f"[Orchestrator] Failed to load {config['name']}: {e}")
            if vlm_engine:
                try:
                    vlm_engine.cleanup()
                except:
                    pass
                vlm_engine = None
            continue

    if vlm_engine is None:
        raise RuntimeError("Failed to load any VLM model")

    # Log memory state after VLM loaded successfully
    log_memory_state("VLM Model Loaded")

    try:
        for intervention in intervention_points:
            vlm_data = vlm_engine.analyze_frame(
                intervention.frame_path,
                intervention.content_type
            )
            vlm_snapshots[intervention.timestamp] = vlm_data

        print(f"[Orchestrator] VLM analysis complete: {len(vlm_snapshots)} snapshots (using {used_model})")

    finally:
        # Cleanup VLM Engine
        vlm_engine.cleanup()

        # CRITICAL: Clear GPU memory after VLM completes
        print("[Orchestrator] Cleaning GPU memory after VLM...")
        clear_gpu_memory()

    return vlm_snapshots
```

#### 7. Synthesis and TTS Phase

```python
def _synthesize_and_generate_audio(
    self,
    transcript_entries: List,
    vlm_snapshots: Dict,
    video_dir: str,
    intervention_only: bool,
    output_mode: str
) -> List[Dict]:
    """
    Synthesize text and generate audio for ALL segments or interventions ONLY.

    Args:
        transcript_entries: List of TranscriptEntry objects
        vlm_snapshots: Dictionary of VLM snapshots
        video_dir: Video directory
        intervention_only: If True, process interventions ONLY
        output_mode: "brief", "explanatory", or "detailed"

    Returns:
        List of segment dictionaries or interventions
    """
    print(f"[Orchestrator] Synthesizing...")
    print(f"[Orchestrator] Mode: {'intervention-only' if intervention_only else 'full'}")
    print(f"[Orchestrator] Output mode: {output_mode}")

    # Clear GPU before TTS
    clear_gpu_memory()

    # Initialize Synthesizer and TTS Engine
    synthesizer = LLMSynthesizer(
        ollama_url="http://localhost:11434",
        model="mistral",
        timeout=120,
        max_retries=3,
        transcripts=transcript_entries,
        output_mode=output_mode
    )

    tts_engine = TTSEngine()

    # Convert raw JSON dict entries to TranscriptEntry objects
    transcript_entries = [
        e if isinstance(e, TranscriptEntry) else TranscriptEntry(
            start=e.get("start", 0.0),
            end=e.get("start", 0.0) + e.get("duration", 0.0),
            text=e.get("text", "")
        )
        for e in transcript_entries
    ]

    # IF intervention-only mode
    if intervention_only:
        return self._synthesize_interventions_only(
            synthesizer, tts_engine, transcript_entries,
            vlm_snapshots, video_dir, output_mode
        )

    # Full pipeline mode
    segments = []

    for idx, segment in enumerate(transcript_entries):
        # Find matching VLM snapshot
        vlm_match = self._find_closest_vlm(segment.start, vlm_snapshots)

        # Only process if there's an intervention
        if vlm_match:
            # Synthesize text for intervention
            synthesized_text = synthesizer.weave(
                transcript_context=segment.text,
                vlm_snapshot=vlm_match,
                content_type=vlm_match.get("content_type", "text"),
                intervention_reason=vlm_match.get("intervention_reason", "Unknown"),
                output_mode=output_mode
            )

            # Generate MP3
            audio_file = f"intervention_{idx:04d}.mp3"
            audio_path = f"{video_dir}/audio_segments/{audio_file}"
            tts_engine.generate(synthesized_text, audio_path)

            # Store intervention segment
            segments.append({
                "chunk_id": idx,
                "timestamp": segment.start,
                "text": synthesized_text,
                "is_intervention": True,
                "audio_file": f"audio_segments/{audio_file}",
                "original_transcript": segment.text,
                "vlm_data": vlm_match
            })
        else:
            # No VLM match — store as non-intervention
            segments.append({
                "chunk_id": idx,
                "timestamp": segment.start,
                "text": segment.text,
                "is_intervention": False,
                "audio_file": None,
                "original_transcript": segment.text,
                "vlm_data": None
            })

    print(f"[Orchestrator] Generated {len(segments)} audio segments")

    # Cleanup TTS engine
    print("[Orchestrator] Cleaning up TTS engine...")
    tts_engine.cleanup()

    return segments
```

#### 8. Manifest Creation

```python
def _create_manifest(
    self,
    video_id: str,
    segments: List[Dict],
    video_dir: str
) -> Dict:
    """
    Create master manifest file for frontend consumption.

    Args:
        video_id: Video ID
        segments: List of segment dictionaries
        video_dir: Video directory

    Returns:
        Complete manifest dictionary
    """
    print("[Orchestrator] Creating manifest.json...")

    intervention_count = sum(1 for s in segments if s["is_intervention"])

    manifest = {
        "video_id": video_id,
        "total_segments": len(segments),
        "intervention_count": intervention_count,
        "created_at": datetime.now().isoformat(),
        "segments": segments
    }

    # Write manifest
    manifest_path = f"{video_dir}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Orchestrator] Manifest saved to: {manifest_path}")

    return manifest
```

#### 9. Full Pipeline Method

```python
def process_video(self, youtube_url: str, output_mode: str = "explanatory") -> Dict:
    """
    Full pipeline: Ingest → Extract → Sync → VLM → Synthesize → TTS → Manifest.

    Args:
        youtube_url: YouTube URL of STEM lecture video
        output_mode: "brief", "explanatory", or "detailed" (default: explanatory)

    Returns:
        Complete manifest dictionary with all pipeline outputs
    """
    start_time = time.perf_counter()

    print("=" * 60)
    print(f"SYNAPSE PIPELINE: {output_mode.upper()} MODE")
    print("=" * 60)
    print()

    # Print system information
    print_system_info()
    print()

    # Step 1: Ingest
    video_id = self._ingest(youtube_url)
    video_dir = f"{self.base_dir}/{video_id}"

    print(f"Processing video ID: {video_id}")
    print(f"Output directory: {video_dir}")
    print()

    # Step 2: Parallel Extraction
    transcript_result, keyframes_result = self._execute_parallel_extraction(
        youtube_url, video_dir
    )

    # Step 3: Synchronization
    intervention_points = self._synchronize(video_dir)

    # Step 4: VLM Snapshots
    vlm_snapshots = self._analyze_vlm_snapshots(intervention_points, video_dir)

    # Step 5: Synthesis & TTS
    segments = self._synthesize_and_generate_audio(
        transcript_entries=transcript_result["entries"] if transcript_result else [],
        vlm_snapshots=vlm_snapshots,
        video_dir=video_dir,
        intervention_only=False,
        output_mode=output_mode
    )

    # Step 6: Manifest Creation
    manifest = self._create_manifest(video_id, segments, video_dir)

    # Final summary
    total_time = time.perf_counter() - start_time
    print()
    print("=" * 60)
    print("SYNAPSE PIPELINE: Processing complete")
    print("=" * 60)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total segments: {manifest['total_segments']}")
    print(f"Intervention points: {manifest['intervention_count']}")
    print()
    print(f"Manifest saved to: {video_dir}/manifest.json")
    print()

    return manifest
```

#### 10. Intervention-Only Workflow

```python
def process_video_intervention_only(self, youtube_url: str, output_mode: str = "explanatory") -> Dict:
    """
    Process video and generate intervention-only outputs.

    Args:
        youtube_url: YouTube URL to process
        output_mode: "brief", "explanatory", or "detailed"

    Returns:
        Dictionary with intervention results
    """
    start_time = time.perf_counter()

    print("=" * 60)
    print(f"INTERVENTION-ONLY PIPELINE: {output_mode.upper()} MODE")
    print("=" * 60)

    try:
        # Reuse existing orchestration steps
        video_id = self._ingest(youtube_url)
        video_dir = f"{self.base_dir}/{video_id}"

        transcript_result, keyframes_result = self._execute_parallel_extraction(
            youtube_url, video_dir
        )

        intervention_points = self._synchronize(video_dir)

        vlm_snapshots = self._analyze_vlm_snapshots(intervention_points, video_dir)

        # Use enhanced synthesis for interventions only
        segments = self._synthesize_and_generate_audio(
            transcript_entries=transcript_result["entries"] if transcript_result else [],
            vlm_snapshots=vlm_snapshots,
            video_dir=video_dir,
            intervention_only=True,
            output_mode=output_mode
        )

        # Create interventions manifest
        manifest = self._create_interventions_manifest(
            video_id, segments, video_dir, output_mode
        )

        total_time = time.perf_counter() - start_time

        print("\n" + "=" * 60)
        print("INTERVENTION-ONLY PIPELINE: Complete")
        print("=" * 60)
        print(f"Total interventions: {len(segments)}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Manifest saved to: {video_dir}/interventions_manifest.json")
        print()

        return {
            "video_id": video_id,
            "interventions": segments,
            "manifest": manifest
        }

    except Exception as e:
        print(f"[Orchestrator] Error in intervention-only pipeline: {e}", exc_info=True)
        return {
            "video_id": None,
            "interventions": [],
            "manifest": None,
            "error": str(e)
        }
```

#### 11. Single Intervention Generation

```python
def generate_intervention_explanation(
    self,
    video_id: str,
    intervention_id: str,
    intervention_timestamp: float,
    frame_path: str,
    content_type: str,
    trigger_reason: str,
    transcript_entries: List,
    output_mode: str = "explanatory"
) -> Dict:
    """
    Generate explanation for a single intervention point.

    This function is used by Endpoint 3 to generate explanations on-demand.
    It reuses existing pipeline components without running the full pipeline.

    Args:
        video_id: YouTube video ID
        intervention_id: UUID of the intervention
        intervention_timestamp: Timestamp of the intervention
        frame_path: Path to the intervention's frame image
        content_type: Type of visual content
        trigger_reason: Reason for intervention
        transcript_entries: List of all transcript entries for context
        output_mode: "brief", "explanatory", or "detailed"

    Returns:
        Dictionary with text explanation and audio file path
    """
    print(f"[Orchestrator] Generating explanation for intervention {intervention_id}")
    print(f"[Orchestrator] Mode: {output_mode}, Content Type: {content_type}")

    # Initialize process handles
    ollama_process = None
    vlm_engine = None
    used_model = None

    # Clear GPU before VLM load
    clear_gpu_memory()
    log_memory_state("Before VLM Load for Single Intervention")

    # Load VLM engine
    for config in model_configs:
        try:
            vlm_engine = SnapshotEngine(
                model_path=config["path"],
                use_gguf=config.get("use_gguf", False),
                n_gpu_layers=config.get("n_gpu_layers", 24),
                use_4bit=config.get("use_4bit", True)
            )
            used_model = config["name"]
            break
        except Exception as e:
            if vlm_engine:
                vlm_engine.cleanup()
                vlm_engine = None
            continue

    # Verify frame exists
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    try:
        # Step 1: VLM Analysis
        vlm_snapshot = vlm_engine.analyze_frame(frame_path, content_type)
        log_memory_state("After VLM Analysis")

        # Cleanup VLM before Ollama
        vlm_engine.cleanup()
        clear_gpu_memory()
        log_memory_state("After VLM Cleanup, Before Ollama Start")

        # Create intervention directory
        intervention_dir = os.path.join(self.base_dir, video_id, "interventions", intervention_id[:8])
        os.makedirs(intervention_dir, exist_ok=True)

        # Save VLM snapshot
        vlm_snapshot_path = os.path.join(intervention_dir, f"vlm_snapshot.json")
        with open(vlm_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(vlm_snapshot, f, indent=2, ensure_ascii=False)

        # Step 2: Start Ollama for LLM Synthesis
        ollama_process = start_ollama()
        time.sleep(3)  # Wait for Ollama to be ready

        # Step 3: Text Synthesis
        synthesizer = LLMSynthesizer(
            ollama_url="http://localhost:11434",
            model="mistral",
            timeout=120,
            max_retries=3,
            transcripts=transcript_entries,
            output_mode=output_mode
        )

        explanation = synthesizer.weave(
            transcript_context=transcript_context,
            vlm_snapshot=vlm_snapshot,
            content_type=content_type,
            intervention_reason=trigger_reason,
            output_mode=output_mode
        )

        # Save text explanation
        text_file_path = os.path.join(intervention_dir, f"{output_mode}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(explanation)

        # Step 4: TTS Generation
        tts_engine = TTSEngine()
        audio_file_path = os.path.join(intervention_dir, f"{output_mode}.mp3")
        tts_engine.generate(explanation, audio_file_path)
        tts_engine.cleanup()

        # Stop Ollama
        stop_ollama(ollama_process)
        log_memory_state("After Ollama Stop (success path)")

        return {
            "intervention_id": intervention_id,
            "text_explanation": explanation,
            "audio_file_path": audio_file_path,
            "text_file_path": text_file_path,
            "vlm_snapshot_path": vlm_snapshot_path,
            "output_mode": output_mode
        }

    except Exception as e:
        print(f"[Orchestrator] Error generating explanation: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        if 'ollama_process' in locals() and ollama_process is not None:
            stop_ollama(ollama_process)

        if vlm_engine is not None:
            vlm_engine.cleanup()

        clear_gpu_memory()
        log_memory_state("After Single Intervention Cleanup")
```

## Configuration Parameters

### Output Modes

| Mode | Depth | Best For |
|------|-------|----------|
| brief | Surface level | Quick overviews |
| explanatory | Standard depth | Complete coverage |
| detailed | Deep exploration | Thorough understanding |

### Workflow Options

| Workflow | Description | Use Case |
|----------|-------------|----------|
| process_video() | Full pipeline with all segments | Complete audio-described track |
| process_video_intervention_only() | Interventions only | Review and intervention collection |
| process_detection_only() | Detection up to intervention points | Quick analysis |
| generate_intervention_explanation() | Single intervention on-demand | API endpoint |

## Usage Examples

### Basic Usage (Full Pipeline)

```python
from orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(base_dir="data")

# Process a YouTube video
manifest = orchestrator.process_video(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID",
    output_mode="explanatory"  # brief, explanatory, or detailed
)

print(f"Video ID: {manifest['video_id']}")
print(f"Total segments: {manifest['total_segments']}")
print(f"Interventions: {manifest['intervention_count']}")
```

### Intervention-Only Mode

```python
# Generate only interventions
result = orchestrator.process_video_intervention_only(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID",
    output_mode="detailed"
)

print(f"Total interventions: {len(result['interventions'])}")
print(f"Output directory: data/{result['video_id']}/interventions/")
```

### Detection-Only Mode

```python
# Detect intervention points only
result = orchestrator.process_detection_only(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID"
)

print(f"Total keyframes: {result['video_metadata']['total_keyframes']}")
print(f"Interventions detected: {result['video_metadata']['total_interventions']}")
```

### Single Intervention Generation

```python
# Generate explanation for a specific intervention
from synchronizer import TranscriptEntry

# Load transcript entries
transcript_entries = [...]  # Your transcript entries

result = orchestrator.generate_intervention_explanation(
    video_id="VIDEO_ID",
    intervention_id="abc123",
    intervention_timestamp=135.5,
    frame_path="data/VIDEO_ID/keyframes/frame_0005.jpg",
    content_type="graph",
    trigger_reason="Deictic Phrase",
    transcript_entries=transcript_entries,
    output_mode="explanatory"
)

print(f"Explanation: {result['text_explanation']}")
print(f"Audio: {result['audio_file_path']}")
```

## Performance Characteristics

### Processing Time (10-minute video)

| Step | Time | Notes |
|------|------|-------|
| Ingestion | < 100ms | URL parsing, directory creation |
| Parallel Extraction | 30-60s | Audio (20-40s) + Video (10-20s) |
| Synchronization | < 100ms | Scoring and matching |
| VLM Snapshots | 50-250s | ~2-5s per intervention (25-50 expected) |
| Synthesis + TTS | 30-150s | ~1-2s LLM + ~0.5s TTS per segment |
| Manifest Creation | < 100ms | JSON serialization |
| **Total** | **110-460s** | **~2-8 minutes** |

### Memory Usage

| Phase | RAM | VRAM | Notes |
|-------|-----|------|-------|
| Ingestion | 100MB | 0GB | Minimal |
| Parallel Extraction | 2-4GB | 0-2GB | Whisper uses GPU if available |
| Synchronization | 100MB | 0GB | Minimal |
| VLM Snapshots | 4-8GB | 5-8GB | Peak during VLM load |
| Synthesis + TTS | 2-4GB | 0-4GB | SpeechT5 if used |
| Manifest Creation | 100MB | 0GB | Minimal |

## Troubleshooting

### Common Issues

**Issue**: "Ollama connection failed"
- **Solution**:
  ```bash
  # Start Ollama
  ollama serve

  # Verify
  curl http://localhost:11434/api/tags
  ```

**Issue**: "CUDA out of memory"
- **Solutions**:
  - Clear GPU memory before VLM load
  - Use 4-bit quantization
  - Reduce number of interventions
  - Use CPU offloading

**Issue**: "Pipeline stuck at VLM phase"
- **Solutions**:
  - Check VLM model files exist
  - Verify GPU memory is available
  - Try smaller model variant
  - Check system logs

**Issue**: "TTS generation failing"
- **Solutions**:
  - Check internet connection (Edge-TTS)
  - Verify SpeechT5 models are downloaded
  - Try different provider
  - Check output directory permissions

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system information
from orchestrator.gpu_memory_manager import print_system_info
print_system_info()

# Test individual phases
orchestrator = PipelineOrchestrator()

# Test ingestion only
video_id = orchestrator._ingest("https://www.youtube.com/watch?v=VIDEO_ID")
print(f"Video ID: {video_id}")

# Test synchronizer only
interventions = orchestrator._synchronize(f"data/{video_id}")
print(f"Interventions: {len(interventions)}")
```

## Best Practices

### 1. Memory Management

```python
# Always cleanup after processing
orchestrator = PipelineOrchestrator()

try:
    manifest = orchestrator.process_video(video_url)
    # Use results
finally:
    # Ensure cleanup happens
    pass  # Orchestrator handles internal cleanup
```

### 2. Error Handling

```python
# Comprehensive error handling
def safe_process_video(orchestrator, video_url):
    """Process video with robust error handling."""
    try:
        manifest = orchestrator.process_video(video_url)
        return {"success": True, "manifest": manifest}
    except ValueError as e:
        return {"success": False, "error": f"Invalid URL: {e}"}
    except RuntimeError as e:
        return {"success": False, "error": f"Runtime error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

### 3. Progress Tracking

```python
# Custom progress tracking
class ProgressOrchestrator(PipelineOrchestrator):
    """Orchestrator with progress tracking."""

    def process_video(self, youtube_url: str, output_mode: str = "explanatory"):
        """Process with progress callbacks."""
        print("[Progress] Starting pipeline...")

        video_id = self._ingest(youtube_url)
        print(f"[Progress] Ingested: {video_id}")

        transcript_result, keyframes_result = self._execute_parallel_extraction(
            youtube_url, f"{self.base_dir}/{video_id}"
        )
        print("[Progress] Extraction complete")

        intervention_points = self._synchronize(f"{self.base_dir}/{video_id}")
        print(f"[Progress] Found {len(intervention_points)} interventions")

        # ... continue with tracking
```

## Future Enhancements

1. **Distributed Processing**: Spread across multiple machines
2. **Resume Capability**: Continue from partial results
3. **Progress Events**: Emit real-time progress updates
4. **Batch Processing**: Process multiple videos efficiently
5. **Caching**: Cache intermediate results
6. **Quality Metrics**: Automatically assess output quality
7. **Parallel VLM Processing**: Process multiple interventions simultaneously
8. **Streaming Output**: Generate results as they're ready
