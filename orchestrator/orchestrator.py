"""
Pipeline Orchestrator Module (M4)

Central Brain of Synapse project - manages entire pipeline lifecycle
from ingestion to final audio-visual manifest.
Enhanced with intervention-only workflow for 3-mode explanation system.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from synchronizer import Synchronizer, TranscriptEntry
from vlm_engine import SnapshotEngine
from synthesizer import LLMSynthesizer
from tts_engine import TTSEngine
from .gpu_memory_manager import (
    clear_gpu_memory,
    log_memory_state,
    before_vlm_load,
    after_whisper,
    ensure_gpu_free,
    print_system_info
)


def import_module_from_path(module_name: str, file_path: str) -> Any:
    """
    Import a module from a file path (handles hyphens in directory names).

    Args:
        module_name: Name to give the imported module
        file_path: Path to the Python file

    Returns:
        The imported module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Ollama Process Management (Per-Request Lifecycle)
# ============================================================================

def start_ollama():
    """
    Start Ollama as a subprocess for per-request execution.

    Ollama is started fresh for each LLM request and terminated after
    to avoid VRAM conflicts with VLM on 6GB GPU systems.

    Returns:
        subprocess.Popen: Ollama process handle
    """
    print("[Ollama] Starting Ollama server...")
    try:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        print("[Ollama] Server started (PID: {})".format(process.pid))
        return process
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found. Install with: winget install Ollama.Ollama"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start Ollama: {e}")


def stop_ollama(process):
    """
    Stop Ollama subprocess gracefully, then force kill if needed.

    Args:
        process: subprocess.Popen handle from start_ollama()
    """
    if process is None:
        return

    print("[Ollama] Stopping Ollama server (PID: {})...".format(process.pid))

    try:
        # Try graceful termination first
        process.terminate()
        try:
            process.wait(timeout=5)
            print("[Ollama] Server terminated gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            print("[Ollama] Force killing server...")
            process.kill()
            process.wait()
            print("[Ollama] Server killed")
    except Exception as e:
        print(f"[Ollama] Error stopping server: {e}")


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
                # This avoids relative import issues
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
                    # Load the generated transcript
                    if os.path.exists(f"{video_dir}/transcript.json"):
                        with open(f"{video_dir}/transcript.json", "r") as f:
                            transcript_result = json.load(f)
                        print("[Orchestrator] Audio extraction complete")
                    else:
                        print(f"[Orchestrator] Transcript file not created")
                else:
                    print(f"[Orchestrator] Audio extraction failed: {result.stderr}")

            except Exception as e:
                print(f"[Orchestrator] Audio extraction failed: {e}")

        def extract_video():
            nonlocal keyframes_result
            try:
                # Import the frame extraction function
                frame_extraction_pkg_path = os.path.join(project_root, "frame_extraction")
                if frame_extraction_pkg_path not in sys.path:
                    sys.path.insert(0, frame_extraction_pkg_path)
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
        # Whisper runs in subprocess and doesn't clean up after exit
        print("[Orchestrator] Cleaning GPU memory after transcription...")
        clear_gpu_memory()

        return transcript_result, keyframes_result

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
        # Main: GGUF 7B (2B transformers fallback disabled due to segfaults on this system)
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
            # This ensures clean slate for TTS and any subsequent operations
            print("[Orchestrator] Cleaning GPU memory after VLM...")
            clear_gpu_memory()

        return vlm_snapshots

    def _find_closest_vlm(
        self,
        timestamp: float,
        vlm_snapshots: Dict
    ) -> Optional[Dict]:
        """
        Find VLM snapshot closest to given timestamp.

        Args:
            timestamp: Target timestamp in seconds
            vlm_snapshots: Dictionary of VLM snapshots keyed by timestamp

        Returns:
            Matching VLM snapshot or None
        """
        if not vlm_snapshots:
            return None

        # Find closest timestamp
        closest_timestamp = min(
            vlm_snapshots.keys(),
            key=lambda t: abs(t - timestamp)
        )

        # Optional: Check if within reasonable window (e.g., 30 seconds)
        max_window = 30.0
        if abs(closest_timestamp - timestamp) > max_window:
            return None

        return vlm_snapshots[closest_timestamp]

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
            intervention_only: If True, process interventions ONLY - REQUIRED
            output_mode: "brief", "explanatory", or "detailed" - REQUIRED

        Returns:
            List of segment dictionaries or interventions
        """
        print(f"[Orchestrator] Synthesizing...")
        print(f"[Orchestrator] Mode: {'intervention-only' if intervention_only else 'full'}")
        print(f"[Orchestrator] Output mode: {output_mode}")

        # Clear GPU before TTS (if SpeechT5 fallback is used)
        # This prevents SpeechT5 from loading on dirty GPU state
        clear_gpu_memory()

        # Initialize Synthesizer (with context for new modes)
        synthesizer = LLMSynthesizer(
            ollama_url="http://localhost:11434",
            model="mistral",
            timeout=120,
            max_retries=3,
            transcripts=transcript_entries,
            output_mode=output_mode
        )

        # Initialize TTS Engine
        tts_engine = TTSEngine()

        # Convert raw JSON dict entries to TranscriptEntry objects if needed.
        # The transcription module saves JSON, so transcript_result["entries"] are
        # plain dicts. segment.start / segment.text would raise AttributeError otherwise.
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

        # Full pipeline mode - only generate audio for interventions
        segments = []

        for idx, segment in enumerate(transcript_entries):
            # Find matching VLM snapshot (15-second window)
            vlm_match = self._find_closest_vlm(
                segment.start,
                vlm_snapshots
            )

            # Only process if there's an intervention (VLM match)
            if vlm_match:
                # Synthesize text for intervention
                synthesized_text = synthesizer.weave(
                    transcript_context=segment.text,
                    vlm_snapshot=vlm_match,
                    content_type=vlm_match.get("content_type", "text"),
                    intervention_reason=vlm_match.get("intervention_reason", "Unknown"),
                    output_mode=output_mode
                )

                # Generate MP3 for intervention only
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
                # No VLM match — skip this segment, no audio generated
                # Just store as non-intervention without audio
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

        # Cleanup TTS engine to free SpeechT5 memory
        print("[Orchestrator] Cleaning up TTS engine...")
        tts_engine.cleanup()

        return segments

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

        # Print system information for diagnostics
        print_system_info()
        print()
        print()

        # Step 1: Ingest (extract video ID)
        video_id = self._ingest(youtube_url)
        video_dir = f"{self.base_dir}/{video_id}"

        print(f"Processing video ID: {video_id}")
        print(f"Output directory: {video_dir}")
        print()

        # Step 2: Parallel Extraction (Audio + Video)
        transcript_result, keyframes_result = self._execute_parallel_extraction(
            youtube_url,
            video_dir
        )

        # Step 3: Synchronization
        intervention_points = self._synchronize(video_dir)

        # Step 4: VLM Snapshots (only intervention points)
        vlm_snapshots = self._analyze_vlm_snapshots(
            intervention_points,
            video_dir
        )

        # Step 5: Synthesis & TTS (audio for interventions only)
        segments = self._synthesize_and_generate_audio(
            transcript_entries=transcript_result["entries"] if transcript_result else [],
            vlm_snapshots=vlm_snapshots,
            video_dir=video_dir,
            intervention_only=False,
            output_mode=output_mode
        )

        # Step 6: Manifest Creation
        manifest = self._create_manifest(
            video_id=video_id,
            segments=segments,
            video_dir=video_dir
        )

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

    # ========================================================================
    # NEW METHODS: Detection-Only Workflow (for API Endpoint 2)
    # ========================================================================

    def process_detection_only(self, youtube_url: str) -> Dict:
        """
        Process video up to intervention point detection only.

        Stops after M0 (Frame Extraction) + M0b (Transcription) + M1 (Synchronization).
        Does NOT run VLM, Synthesis, or TTS.

        Args:
            youtube_url: YouTube URL to process

        Returns:
            Dictionary with video metadata and intervention points
        """
        start_time = time.perf_counter()

        print("=" * 60)
        print("DETECTION-ONLY PIPELINE")
        print("=" * 60)
        print()

        try:
            # Step 1: Ingest (extract video ID)
            video_id = self._ingest(youtube_url)
            video_dir = f"{self.base_dir}/{video_id}"

            print(f"Processing video ID: {video_id}")
            print(f"Output directory: {video_dir}")
            print()

            # Step 2: Parallel Extraction (Audio + Video)
            transcript_result, keyframes_result = self._execute_parallel_extraction(
                youtube_url,
                video_dir
            )

            # Step 3: Synchronization
            intervention_points = self._synchronize(video_dir)

            # Extract metadata from keyframes result
            total_keyframes = len(keyframes_result.get("keyframes", [])) if keyframes_result else 0

            # Try to grab metadata from transcript if available
            transcript_metadata = transcript_result.get("metadata", {}) if transcript_result else {}
            title = transcript_metadata.get("title", f"Video {video_id}")
            duration_sec = int(transcript_metadata.get("duration", 0.0))
            duration_fmt = transcript_metadata.get("duration_formatted", "00:00")

            # Build video metadata
            video_metadata = {
                "video_id": video_id,
                "total_keyframes": total_keyframes,
                "total_interventions": len(intervention_points),
                "video_dir": video_dir
            }

            total_time = time.perf_counter() - start_time

            print()
            print("=" * 60)
            print("DETECTION-ONLY PIPELINE: Complete")
            print("=" * 60)
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Total keyframes: {total_keyframes}")
            print(f"Intervention points detected: {len(intervention_points)}")
            print(f"Output directory: {video_dir}")
            print()

            return {
                "video_id": video_id,
                "video_dir": video_dir,
                "title": title,
                "duration_seconds": duration_sec,
                "duration_formatted": duration_fmt,
                "transcript": transcript_result,
                "keyframes": keyframes_result,
                "intervention_points": intervention_points,
                "metadata": video_metadata
            }

        except Exception as e:
            print(f"[Orchestrator] Error in detection-only pipeline: {e}", exc_info=True)
            return {
                "video_id": None,
                "intervention_points": [],
                "error": str(e)
            }

    # ========================================================================
    # NEW METHODS: Intervention-Only Workflow
    # ========================================================================

    def _synthesize_interventions_only(
        self,
        synthesizer,
        tts_engine,
        transcript_entries: List,
        vlm_snapshots: Dict,
        video_dir: str,
        output_mode: str
    ) -> List[Dict]:
        """
        Synthesize and generate audio for interventions ONLY.

        Args:
            synthesizer: LLMSynthesizer instance
            tts_engine: TTSEngine instance
            transcript_entries: List of transcript entries
            vlm_snapshots: Dictionary of VLM snapshots
            video_dir: Video directory
            output_mode: "brief", "explanatory", or "detailed"

        Returns:
            List of intervention dictionaries
        """
        print(f"[Orchestrator] Processing {len(vlm_snapshots)} interventions in {output_mode} mode")

        interventions = []
        intervention_idx = 0

        try:
            for timestamp, vlm_data in vlm_snapshots.items():
                # Find corresponding transcript segment
                transcript_segment = self._find_transcript_segment(timestamp, transcript_entries)
                transcript_context = transcript_segment.text if transcript_segment else ""

                # Determine content type and reason
                content_type = vlm_data.get("content_type", "text")
                intervention_reason = "VLM Analysis"  # Default reason

                # Generate explanation
                explanation = synthesizer.weave(
                    transcript_context=transcript_context,
                    vlm_snapshot=vlm_data,
                    content_type=content_type,
                    intervention_reason=intervention_reason,
                    output_mode=output_mode
                )

                # Create intervention directory
                intervention_dir = os.path.join(video_dir, "interventions",
                                            f"intervention_{intervention_idx:04d}")
                os.makedirs(intervention_dir, exist_ok=True)

                # Save description
                description_path = os.path.join(intervention_dir, "description.txt")
                try:
                    with open(description_path, "w", encoding="utf-8") as f:
                        f.write(explanation)
                except Exception as e:
                    print(f"[Orchestrator] Error saving description: {e}")
                    description_path = None

                # Generate audio
                audio_path = os.path.join(intervention_dir, "audio.mp3")
                try:
                    tts_engine.generate(explanation, audio_path)
                except Exception as e:
                    print(f"[Orchestrator] Error generating audio: {e}")
                    audio_path = None

                # Find and copy reference frame
                frame_path = self._find_frame_for_timestamp(timestamp, video_dir)
                if frame_path:
                    try:
                        frame_dest = os.path.join(intervention_dir, "frame.jpg")
                        shutil.copy2(frame_path, frame_dest)
                    except Exception as e:
                        print(f"[Orchestrator] Error copying frame: {e}")
                        frame_dest = None
                else:
                    frame_dest = None

                # Create intervention metadata
                metadata = {
                    "intervention_id": intervention_idx,
                    "timestamp": timestamp,
                    "timestamp_formatted": f"{timestamp // 60:02d}:{timestamp % 60:02d}",
                    "frame_path": frame_dest,
                    "content_type": content_type,
                    "transcript_context": transcript_context,
                    "vlm_data": vlm_data,
                    "explanation": explanation,
                    "explanation_mode": output_mode
                }

                metadata_path = os.path.join(intervention_dir, "metadata.json")
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    print(f"[Orchestrator] Error saving metadata: {e}")
                    metadata_path = None

                interventions.append({
                    "intervention_id": intervention_idx,
                    "timestamp": timestamp,
                    "timestamp_formatted": f"{timestamp // 60:02d}:{timestamp % 60:02d}",
                    "content_type": content_type,
                    "audio_path": audio_path,
                    "description_path": description_path,
                    "frame_path": frame_dest,
                    "explanation": explanation,
                    "metadata_path": metadata_path
                })

                print(f"  ✓ Intervention {intervention_idx}: {timestamp:.1f}s - {content_type}")
                intervention_idx += 1

            print(f"[Orchestrator] Generated {len(interventions)} interventions")

            # Cleanup TTS engine to free SpeechT5 memory
            print("[Orchestrator] Cleaning up TTS engine...")
            tts_engine.cleanup()

            return interventions

        except Exception as e:
            print(f"[Orchestrator] Error synthesizing interventions: {e}", exc_info=True)
            return interventions

    def _find_transcript_segment(self, timestamp: float, transcript_entries: List) -> Optional[Any]:
        """
        Find transcript segment containing timestamp.

        Args:
            timestamp: Target timestamp in seconds
            transcript_entries: List of transcript entries

        Returns:
            Matching TranscriptEntry or None
        """
        if not transcript_entries:
            return None

        try:
            # Find segment where timestamp falls within start-end range
            for segment in transcript_entries:
                try:
                    segment_start = float(segment.start)
                    segment_end = float(segment.end)
                    if segment_start <= timestamp <= segment_end:
                        return segment
                except (ValueError, AttributeError) as e:
                    print(f"[Orchestrator] Error comparing timestamps: {e}")
                    continue

            # If no exact match, find nearest segment
            try:
                nearest = min(transcript_entries,
                           key=lambda s: min(abs(float(s.start) - timestamp),
                                             abs(float(s.end) - timestamp)))
                return nearest
            except Exception as e:
                print(f"[Orchestrator] Error finding nearest segment: {e}")
                return transcript_entries[0] if transcript_entries else None

        except Exception as e:
            print(f"[Orchestrator] Error finding transcript segment: {e}", exc_info=True)
            return transcript_entries[0] if transcript_entries else None

    def _find_frame_for_timestamp(self, timestamp: float, video_dir: str) -> Optional[str]:
        """
        Find keyframe image for given timestamp.

        Args:
            timestamp: Target timestamp in seconds
            video_dir: Video directory containing keyframes

        Returns:
            Path to closest keyframe image, or None if not found
        """
        try:
            keyframes_path = os.path.join(video_dir, "keyframes", "keyframes.json")

            if not os.path.exists(keyframes_path):
                print(f"[Orchestrator] Keyframes file not found: {keyframes_path}")
                return None

            with open(keyframes_path, "r", encoding="utf-8") as f:
                keyframes = json.load(f)

            # Find closest keyframe by timestamp_seconds
            closest = min(
                keyframes,
                key=lambda k: abs(float(k.get("timestamp_seconds", 0)) - timestamp)
            )

            frame_path = closest.get("frame_path")
            if frame_path and os.path.exists(frame_path):
                return frame_path
            else:
                print(f"[Orchestrator] Frame path not found or doesn't exist: {frame_path}")
                return None

        except Exception as e:
            print(f"[Orchestrator] Error finding frame for timestamp: {e}", exc_info=True)
            return None

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

    def _create_interventions_manifest(
        self,
        video_id: str,
        interventions: List[Dict],
        video_dir: str,
        output_mode: str
    ) -> Dict:
        """
        Create interventions manifest.

        Args:
            video_id: Video identifier
            interventions: List of intervention dictionaries
            video_dir: Video directory
            output_mode: Explanation mode used

        Returns:
            Manifest dictionary
        """
        print("[Orchestrator] Creating interventions manifest...")

        manifest = {
            "video_id": video_id,
            "total_interventions": len(interventions),
            "explanation_mode": output_mode,
            "created_at": datetime.now().isoformat(),
            "interventions": [
                {
                    "intervention_id": inv["intervention_id"],
                    "timestamp": inv["timestamp"],
                    "timestamp_formatted": inv["timestamp_formatted"],
                    "content_type": inv["content_type"],
                    "audio_file": f"interventions/intervention_{inv['intervention_id']:04d}/audio.mp3",
                    "description_file": f"interventions/intervention_{inv['intervention_id']:04d}/description.txt",
                    "frame_file": f"interventions/intervention_{inv['intervention_id']:04d}/frame.jpg",
                    "explanation": inv["explanation"]
                }
                for inv in interventions
            ]
        }

        manifest_path = os.path.join(video_dir, "interventions_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[Orchestrator] Interventions manifest saved: {manifest_path}")
        except Exception as e:
            print(f"[Orchestrator] Error saving interventions manifest: {e}")

        return manifest

    # ========================================================================
    # NEW METHODS: Single Intervention Explanation for Endpoint 3
    # ========================================================================

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
        It reuses existing pipeline components (VLM, Synthesizer, TTS) without
        running the full pipeline.

        Args:
            video_id: YouTube video ID
            intervention_id: UUID of the intervention
            intervention_timestamp: Timestamp of the intervention
            frame_path: Path to the intervention's frame image
            content_type: Type of visual content (equation, graph, etc.)
            trigger_reason: Reason for intervention
            transcript_entries: List of all transcript entries for context
            output_mode: "brief", "explanatory", or "detailed"

        Returns:
            Dictionary with:
            - intervention_id: UUID of the intervention
            - text_explanation: Generated text explanation
            - audio_file_path: Path to generated MP3 (or None if failed)
            - output_mode: Mode used for generation
        """
        print(f"[Orchestrator] Generating explanation for intervention {intervention_id}")
        print(f"[Orchestrator] Mode: {output_mode}, Content Type: {content_type}")

        # Initialize process handles to None for cleanup
        ollama_process = None
        vlm_engine = None
        used_model = None

        # CRITICAL: Clear GPU before VLM load
        clear_gpu_memory()
        log_memory_state("Before VLM Load for Single Intervention")

        # Model paths with fallback chain (same as full pipeline)
        # Main: GGUF 7B (2B transformers fallback disabled due to segfaults on this system)
        model_configs = [
            {
                "path": os.path.join(project_root, "models", "Qwen2-VL-7B-Instruct-GGUF"),
                "use_gguf": True,
                "n_gpu_layers": 24,
                "name": "Qwen2-VL-7B-GGUF"
            }
        ]

        # Try to load VLM engine
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

        # Verify frame exists
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # Find transcript context
        from synchronizer import TranscriptEntry
        transcript_entries = [
            e if isinstance(e, TranscriptEntry) else TranscriptEntry(
                start=e.get("start", 0.0),
                end=e.get("start", 0.0) + e.get("duration", 0.0),
                text=e.get("text", "")
            )
            for e in transcript_entries
        ]

        # Find matching transcript segment
        transcript_segment = self._find_transcript_segment(
            intervention_timestamp,
            transcript_entries
        )
        transcript_context = transcript_segment.text if transcript_segment else ""

        try:
            # Step 1: VLM Analysis (reuses existing SnapshotEngine with existing prompts)
            log_memory_state("Before VLM Analysis")
            vlm_snapshot = vlm_engine.analyze_frame(
                frame_path,
                content_type
            )
            log_memory_state("After VLM Analysis")

            print(f"[Orchestrator] VLM analysis complete (using {used_model})")

            # CRITICAL: Cleanup VLM engine to free VRAM before Ollama starts
            vlm_engine.cleanup()
            clear_gpu_memory()
            log_memory_state("After VLM Cleanup, Before Ollama Start")

            # Create intervention directory early (for VLM snapshot)
            intervention_dir = os.path.join(
                self.base_dir,
                video_id,
                "interventions",
                intervention_id[:8]
            )
            os.makedirs(intervention_dir, exist_ok=True)

            # Save VLM snapshot to disk
            vlm_snapshot_path = os.path.join(
                intervention_dir,
                f"vlm_snapshot.json"
            )
            with open(vlm_snapshot_path, "w", encoding="utf-8") as f:
                json.dump(vlm_snapshot, f, indent=2, ensure_ascii=False)
            print(f"[Orchestrator] VLM snapshot saved: {vlm_snapshot_path}")

            # Step 2: Start Ollama for LLM Synthesis
            ollama_process = start_ollama()

            # Wait for Ollama to be ready
            print("[Orchestrator] Waiting for Ollama to be ready...")
            time.sleep(3)  # Give Ollama time to start up

            # Step 3: Text Synthesis (reuses existing LLMSynthesizer with existing prompts)
            from synthesizer import LLMSynthesizer
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

            print(f"[Orchestrator] Text synthesis complete")

            # Cleanup synthesizer local caches
            synthesizer.cleanup()

            # Step 4: TTS Generation (reuses existing TTSEngine)
            from tts_engine import TTSEngine
            tts_engine = TTSEngine()

            # intervention_dir already created earlier for VLM snapshot

            audio_file_path = os.path.join(
                intervention_dir,
                f"{output_mode}.mp3"
            )

            tts_engine.generate(explanation, audio_file_path)
            tts_engine.cleanup()

            print(f"[Orchestrator] Audio generated: {audio_file_path}")

            # Save text explanation to disk
            text_file_path = os.path.join(
                intervention_dir,
                f"{output_mode}.txt"
            )
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(explanation)
            print(f"[Orchestrator] Text explanation saved: {text_file_path}")

            # Return result (Ollama cleanup happens in finally block)
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
            # Cleanup Ollama (if it was started)
            if ollama_process is not None:
                try:
                    stop_ollama(ollama_process)
                except Exception as cleanup_error:
                    print(f"[Orchestrator] Error stopping Ollama during cleanup: {cleanup_error}")
            # Cleanup VLM engine
            if vlm_engine is not None:
                try:
                    vlm_engine.cleanup()
                except Exception as cleanup_error:
                    print(f"[Orchestrator] Error cleaning up VLM: {cleanup_error}")
            clear_gpu_memory()
            log_memory_state("After Single Intervention Cleanup")


# Example usage
if __name__ == "__main__":
    orchestrator = PipelineOrchestrator(base_dir="data")

    # Example: Process a YouTube video
    # video_url = "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID"
    # manifest = orchestrator.process_video(video_url)

    # print("\n=== Pipeline Summary ===")
    # print(f"Video ID: {manifest['video_id']}")
    # print(f"Total segments: {manifest['total_segments']}")
    # print(f"Interventions: {manifest['intervention_count']}")
