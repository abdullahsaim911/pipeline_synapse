"""
Pipeline Orchestrator Module (M4)

Central Brain of Synapse project - manages entire pipeline lifecycle
from ingestion to final audio-visual manifest.
Enhanced with intervention-only workflow for 3-mode explanation system.
"""

import importlib.util
import json
import logging
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

from synchronizer import Synchronizer, TranscriptEntry, reconcile_visual_evidence
from vlm_engine import SnapshotEngine
from synthesizer import LLMSynthesizer, VisualEvidenceUnavailableError
from visual_evidence import assess_snapshot, load_usable_snapshot
from stage_tracing import current_trace, trace_stage
from youtube_runtime import youtube_runtime_options
try:
    from tts_engine import TTSEngine
except ImportError:
    TTSEngine = None
from .ollama_service import OllamaService, shared_ollama_service
from .gpu_memory_manager import (
    clear_gpu_memory,
    log_memory_state,
    before_vlm_load,
    after_whisper,
    ensure_gpu_free,
    print_system_info
)


logger = logging.getLogger(__name__)


def print(*values, **_kwargs):  # noqa: A001
    """Route legacy operational banners to diagnostic logging."""
    logger.debug(" ".join(str(value) for value in values))


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
# Ollama Process Management (one shared server, see ollama_service.py)
# ============================================================================

OLLAMA_PATH_ENV_VAR = "SYNAPSE_OLLAMA_PATH"


def resolve_ollama_executable() -> str:
    """Resolve Ollama without requiring its install directory on ``PATH``.

    An explicit configuration is authoritative: a bad ``SYNAPSE_OLLAMA_PATH``
    fails clearly instead of silently selecting a different installation.
    Otherwise use normal command discovery, followed by Ollama's standard
    per-user Windows install locations.
    """
    configured = os.environ.get(OLLAMA_PATH_ENV_VAR, "").strip().strip('"')
    if configured:
        configured = os.path.abspath(os.path.expanduser(os.path.expandvars(configured)))
        if os.path.isfile(configured):
            return configured
        raise RuntimeError(
            f"{OLLAMA_PATH_ENV_VAR} does not point to an Ollama executable: "
            f"{configured}"
        )

    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        program_files = os.environ.get("ProgramFiles", "").strip()
        candidates = []
        if local_app_data:
            candidates.extend([
                os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe"),
                os.path.join(local_app_data, "Ollama", "ollama.exe"),
            ])
        if program_files:
            candidates.append(os.path.join(program_files, "Ollama", "ollama.exe"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    raise RuntimeError(
        "Ollama executable was not found. Set SYNAPSE_OLLAMA_PATH to the full "
        "executable path or install Ollama with: winget install Ollama.Ollama"
    )


def start_ollama():
    """
    Start an Ollama server owned by this backend.

    Returns:
        subprocess.Popen: Ollama process handle
    """
    print("[Ollama] Starting Ollama server...")
    try:
        executable = resolve_ollama_executable()
        process = subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        print("[Ollama] Server started (PID: {})".format(process.pid))
        return process
    except RuntimeError:
        raise
    except FileNotFoundError as e:
        raise RuntimeError(f"Resolved Ollama executable could not be started: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to start Ollama: {e}")


def stop_ollama(process):
    """
    Stop an Ollama server this backend started, including its model runners.

    Only the given process tree is touched, so an Ollama started independently
    by the user keeps running.

    Args:
        process: subprocess.Popen handle from start_ollama()
    """
    if process is None:
        return

    print("[Ollama] Stopping Ollama server (PID: {})...".format(process.pid))

    try:
        if os.name == 'nt':
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
            print("[Ollama] Server stopped")
        except subprocess.TimeoutExpired:
            print("[Ollama] Force killing server...")
            process.kill()
            process.wait()
            print("[Ollama] Server killed")

    except Exception as e:
        print(f"[Ollama] Error stopping server: {e}")


def get_ollama_service() -> OllamaService:
    """Return the process-wide Ollama service."""
    return shared_ollama_service(
        start=lambda: start_ollama(),
        stop=lambda process: stop_ollama(process),
    )


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
        trace_context = current_trace()

        # Results containers
        transcript_result = None
        keyframes_result = None
        audio_error = None
        video_error = None

        # Define thread functions
        def extract_audio():
            nonlocal transcript_result, audio_error
            stage_started = time.monotonic()
            trace_stage("transcription", "started", context=trace_context)
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
                        trace_stage(
                            "transcription", "completed", started_at=stage_started,
                            context=trace_context, status="success",
                        )
                    else:
                        print(f"[Orchestrator] Transcript file not created")
                else:
                    print(f"[Orchestrator] Audio extraction failed: {result.stderr}")

            except Exception as e:
                audio_error = e
                print(f"[Orchestrator] Audio extraction failed: {e}")
                trace_stage(
                    "transcription", "failed", started_at=stage_started,
                    context=trace_context, level=logging.DEBUG,
                    failure_type=type(e).__name__,
                )

        def extract_video():
            nonlocal keyframes_result, video_error
            stage_started = time.monotonic()
            trace_stage("frame_extraction", "started", context=trace_context)
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
                trace_stage(
                    "frame_extraction", "completed", started_at=stage_started,
                    context=trace_context, status="success", frame_count=len(result),
                )
            except Exception as e:
                video_error = e
                print(f"[Orchestrator] Video extraction failed: {e}")
                trace_stage(
                    "frame_extraction", "failed", started_at=stage_started,
                    context=trace_context, level=logging.DEBUG,
                    failure_type=type(e).__name__,
                )

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

        if video_error is not None or keyframes_result is None:
            raise RuntimeError(
                "Video frame extraction failed before keyframe synchronization"
            ) from video_error
        if audio_error is not None or transcript_result is None:
            raise RuntimeError(
                "Audio transcription failed before keyframe synchronization"
            ) from audio_error

        if os.environ.get("SYNAPSE_INTERACTIONS") != "0":
            detection_started = time.monotonic()
            trace_stage("intervention_detection", "started", context=trace_context)
            try:
                from synapse_interaction import Config, detect_interactions
                out_path = os.path.join(video_dir, "interactions.json")
                cfg = Config(enable_hand=False)
                print("[Orchestrator] Detecting visual interactions...")
                try:
                    interactions = detect_interactions(
                        youtube_url, out_path, cfg, verbose=True
                    )
                except Exception:
                    # OpenCV often cannot decode a live YouTube stream; download
                    # once into this video's existing data dir (not demo_videos).
                    local_mp4 = os.path.join(video_dir, "video.mp4")
                    if not os.path.exists(local_mp4):
                        import yt_dlp
                        with yt_dlp.YoutubeDL({
                            "format": "bv*[height<=720]",
                            "outtmpl": local_mp4,
                            "quiet": True,
                            "no_warnings": True,
                            **youtube_runtime_options(),
                        }) as ydl:
                            ydl.download([youtube_url])
                    interactions = detect_interactions(
                        local_mp4, out_path, cfg, verbose=True
                    )
                print(
                    f"[Orchestrator] Interaction detection complete: "
                    f"{len(interactions)} events -> {out_path}"
                )
                trace_stage(
                    "intervention_detection", "completed", started_at=detection_started,
                    context=trace_context, status="success",
                    interaction_count=len(interactions),
                )
            except Exception as e:
                print(f"[Orchestrator] Interaction detection failed: {e}")
                trace_stage(
                    "intervention_detection", "degraded", started_at=detection_started,
                    context=trace_context, level=logging.WARNING,
                    failure_type=type(e).__name__,
                )

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
        if os.environ.get("SYNAPSE_INTERACTIONS") != "0":
            try:
                from synapse_interaction import fuse_interactions
                before = len(intervention_points)
                intervention_points = fuse_interactions(
                    intervention_points,
                    os.path.join(video_dir, "interactions.json"),
                    transcript_path=f"{video_dir}/transcript.json",
                    keyframes_path=f"{video_dir}/keyframes/keyframes.json"
                )
                print(
                    f"[Orchestrator] Fused interactions into M1 points: "
                    f"{before} -> {len(intervention_points)}"
                )
            except Exception as e:
                print(f"[Orchestrator] Interaction fusion failed: {e}")
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

        from vlm_interface import PROVIDER_ENV_VAR, PROVIDER_MODAL, PROVIDER_LOCAL, VALID_PROVIDERS
        provider = os.environ.get(PROVIDER_ENV_VAR, PROVIDER_MODAL).strip().lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported VLM provider '{provider}'. Must be one of: {sorted(VALID_PROVIDERS)}"
            )

        vlm_engine = None
        vlm_snapshots = {}
        used_model = None

        if provider == PROVIDER_MODAL:
            print("[Orchestrator] Using remote Modal VLM backend (SYNAPSE_VLM_PROVIDER=modal)")
            vlm_engine = SnapshotEngine(provider=PROVIDER_MODAL)
            used_model = vlm_engine.model_name

            try:
                for intervention in intervention_points:
                    try:
                        vlm_data = vlm_engine.analyze_frame(
                            intervention.frame_path,
                            intervention.content_type,
                            classification_confidence=getattr(
                                intervention, "classification_confidence", None
                            ),
                            confidence_semantics=getattr(
                                intervention, "confidence_semantics", "unknown"
                            ),
                            frame_id=os.path.basename(intervention.frame_path),
                            timestamp_seconds=intervention.timestamp,
                            raw_category_labels=getattr(
                                intervention, "raw_category_labels", None
                            ) or [intervention.content_type],
                            canonical_category_ids=getattr(
                                intervention, "canonical_categories", None
                            ) or [],
                        )
                        vlm_snapshots[intervention.timestamp] = vlm_data
                        if vlm_data.get("_error"):
                            print(
                                f"[Orchestrator] Warning: Visual accessibility info unavailable "
                                f"for timestamp {intervention.timestamp}"
                            )
                    except Exception as e:
                        sanitized_err = f"{type(e).__name__}: {e}"
                        print(
                            f"[Orchestrator] Error analyzing frame {intervention.frame_path} "
                            f"at timestamp {intervention.timestamp}: {sanitized_err}"
                        )
                        vlm_snapshots[intervention.timestamp] = vlm_engine._create_failed_snapshot(
                            sanitized_err, intervention.content_type
                        )

                print(f"[Orchestrator] VLM analysis complete: {len(vlm_snapshots)} snapshots (using {used_model})")
            finally:
                vlm_engine.cleanup()

            return vlm_snapshots

        # Local provider path (GGUF / Transformers)
        # CRITICAL: Ensure GPU is clean and has enough memory before local VLM loads
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

        for config in model_configs:
            try:
                print(f"[Orchestrator] Attempting to load model: {config['name']}")

                vlm_engine = SnapshotEngine(
                    model_path=config["path"],
                    use_gguf=config.get("use_gguf", False),
                    n_gpu_layers=config.get("n_gpu_layers", 24),
                    use_4bit=config.get("use_4bit", True),
                    provider=PROVIDER_LOCAL
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
                    intervention.content_type,
                    classification_confidence=getattr(
                        intervention, "classification_confidence", None
                    ),
                    confidence_semantics=getattr(
                        intervention, "confidence_semantics", "unknown"
                    ),
                    frame_id=os.path.basename(intervention.frame_path),
                    timestamp_seconds=intervention.timestamp,
                    raw_category_labels=getattr(
                        intervention, "raw_category_labels", None
                    ) or [intervention.content_type],
                    canonical_category_ids=getattr(
                        intervention, "canonical_categories", None
                    ) or [],
                )
                vlm_snapshots[intervention.timestamp] = vlm_data

            print(f"[Orchestrator] VLM analysis complete: {len(vlm_snapshots)} snapshots (using {used_model})")

        finally:
            # Cleanup VLM Engine
            vlm_engine.cleanup()

            # CRITICAL: Clear GPU memory after local VLM completes
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

    @staticmethod
    def _attach_transcript_coverage(
        vlm_snapshot: Dict,
        transcript_context: str,
    ) -> Dict:
        """Attach post-VLM reconciliation without altering canonical evidence."""

        vlm_snapshot["coverage_plan"] = reconcile_visual_evidence(
            vlm_snapshot, transcript_context
        )
        return vlm_snapshot

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
                self._attach_transcript_coverage(vlm_match, segment.text)
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

    def process_detection_only(
        self, youtube_url: str, should_continue: Optional[Any] = None
    ) -> Dict:
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
            def require_active() -> None:
                if should_continue is not None and not should_continue():
                    raise RuntimeError("Video processing was cancelled")

            # Step 1: Ingest (extract video ID)
            require_active()
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
            require_active()

            # Step 3: Synchronization
            intervention_points = self._synchronize(video_dir)
            require_active()

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
            print(f"[Orchestrator] Error in detection-only pipeline: {e}")
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

                self._attach_transcript_coverage(vlm_data, transcript_context)

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
                    "timestamp_formatted": f"{int(timestamp) // 60:02d}:{int(timestamp) % 60:02d}",
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
                    "timestamp_formatted": f"{int(timestamp) // 60:02d}:{int(timestamp) % 60:02d}",
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
            print(f"[Orchestrator] Error synthesizing interventions: {e}")
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
            print(f"[Orchestrator] Error finding transcript segment: {e}")
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
            print(f"[Orchestrator] Error finding frame for timestamp: {e}")
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
            print(f"[Orchestrator] Error in intervention-only pipeline: {e}")
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

    def _intervention_dir(self, video_id: str, intervention_id: str) -> str:
        return os.path.join(
            self.base_dir, video_id, "interventions", intervention_id[:8]
        )

    @staticmethod
    def _write_json_atomic(path: str, payload: Dict) -> None:
        """Replace *path* whole so readers never see a half-written snapshot."""
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # Windows refuses to replace a file another thread has open for reading.
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    @staticmethod
    def _valid_mp3(path: str) -> bool:
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < 3:
                return False
            with open(path, "rb") as handle:
                header = handle.read(3)
        except OSError:
            return False
        return header == b"ID3" or (
            len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
        )

    @staticmethod
    def _create_intervention_vlm_engine(provider: str):
        """Create the engine for one extraction. Returns (engine, model name)."""
        from vlm_interface import PROVIDER_LOCAL, PROVIDER_MODAL

        if provider == PROVIDER_MODAL:
            print("[Orchestrator] Using remote Modal VLM backend for intervention explanation")
            engine = SnapshotEngine(provider=PROVIDER_MODAL)
            return engine, engine.model_name

        # A local VLM needs the GPU, so release anything Mistral still holds.
        get_ollama_service().shutdown()
        clear_gpu_memory()
        log_memory_state("Before VLM Load for Single Intervention")
        name = "Qwen2-VL-7B-GGUF"
        print(f"[Orchestrator] Attempting to load model: {name}")
        engine = SnapshotEngine(
            model_path=os.path.join(project_root, "models", "Qwen2-VL-7B-Instruct-GGUF"),
            use_gguf=True,
            n_gpu_layers=24,
            use_4bit=True,
            provider=PROVIDER_LOCAL,
        )
        print(f"[Orchestrator] Successfully loaded: {name}")
        return engine, name

    def extract_visual_evidence(
        self,
        video_id: str,
        intervention_id: str,
        intervention_timestamp: float,
        frame_path: str,
        content_type: str,
        classification_confidence: Optional[float] = None,
        confidence_semantics: str = "unknown",
        raw_category_labels: Optional[List[str]] = None,
        canonical_category_ids: Optional[List[str]] = None,
        before_inference: Optional[Any] = None,
        ownership_check: Optional[Any] = None,
        checkpoint_commit: Optional[Any] = None,
    ) -> Dict:
        """
        Produce and persist the visual snapshot for one intervention.

        A saved snapshot that passes validation is reused, so every output mode
        shares one VLM call. Failed evidence is persisted too, with its
        diagnostics, but is reported as unusable.

        Args:
            before_inference: Optional callable run only when the model will
                actually be called. Returning False skips the call.

        Returns:
            Dictionary with snapshot, snapshot_path, usable, reused and failure
        """
        intervention_dir = self._intervention_dir(video_id, intervention_id)
        snapshot_path = os.path.join(intervention_dir, "vlm_snapshot.json")

        reusable = load_usable_snapshot(snapshot_path)
        if reusable is not None:
            trace_stage(
                "snapshot_validation", "complete", usable=True, reused=True,
                snapshot_path=snapshot_path,
            )
            print(f"[Orchestrator] Reusing validated VLM snapshot: {snapshot_path}")
            return {
                "snapshot": reusable,
                "snapshot_path": snapshot_path,
                "usable": True,
                "reused": True,
                "failure": None,
            }

        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        if ownership_check is not None:
            ownership_check()
        if before_inference is not None and not before_inference():
            return {
                "snapshot": None,
                "snapshot_path": None,
                "usable": False,
                "reused": False,
                "failure": {"code": "vlm_remote_unavailable", "retryable": True},
            }

        from vlm_interface import PROVIDER_ENV_VAR, PROVIDER_MODAL, VALID_PROVIDERS
        provider = os.environ.get(PROVIDER_ENV_VAR, PROVIDER_MODAL).strip().lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported VLM provider '{provider}'. Must be one of: {sorted(VALID_PROVIDERS)}"
            )

        extraction_started = time.monotonic()
        trace_stage("vlm_inference", "start", provider=provider)
        vlm_engine, used_model = self._create_intervention_vlm_engine(provider)
        try:
            snapshot = vlm_engine.analyze_frame(
                frame_path,
                content_type,
                classification_confidence=classification_confidence,
                confidence_semantics=confidence_semantics,
                frame_id=intervention_id,
                timestamp_seconds=intervention_timestamp,
                raw_category_labels=raw_category_labels or [content_type],
                canonical_category_ids=canonical_category_ids or [],
            )
        finally:
            vlm_engine.cleanup()
            if provider != PROVIDER_MODAL:
                clear_gpu_memory()
        if ownership_check is not None:
            try:
                ownership_check()
            except Exception:
                trace_stage("video_cancellation", "late_vlm_response_discarded")
                raise
        trace_stage(
            "vlm_inference", "complete", started_at=extraction_started,
            model=used_model,
            finish_reason=(snapshot.get("_vlm_metrics") or {}).get("finish_reason", "missing"),
            prompt_tokens=(snapshot.get("_vlm_metrics") or {}).get("prompt_tokens", 0),
            completion_tokens=(snapshot.get("_vlm_metrics") or {}).get("completion_tokens", 0),
        )

        os.makedirs(intervention_dir, exist_ok=True)
        persistence_started = time.monotonic()
        trace_stage("snapshot_persistence", "start", snapshot_path=snapshot_path)
        persist_snapshot = lambda: self._write_json_atomic(snapshot_path, snapshot)
        if checkpoint_commit is not None:
            checkpoint_commit(persist_snapshot)
        else:
            persist_snapshot()
        trace_stage(
            "snapshot_persistence", "complete", started_at=persistence_started,
            snapshot_path=snapshot_path,
        )
        assessment = assess_snapshot(snapshot)
        trace_stage(
            "snapshot_validation", "complete", usable=assessment.usable,
            reason=assessment.reason, reused=False,
            extraction_status=(snapshot.get("visual_evidence") or {}).get("extraction_status"),
            evidence_status=(snapshot.get("visual_evidence") or {}).get("evidence_status"),
        )
        print(
            f"[Orchestrator] VLM snapshot saved using {used_model}: {snapshot_path} "
            f"(usable={assessment.usable}, reason={assessment.reason})"
        )
        return {
            "snapshot": snapshot,
            "snapshot_path": snapshot_path,
            "usable": assessment.usable,
            "reused": False,
            "failure": snapshot.get("_failure") or (
                None if assessment.usable else {"code": assessment.reason, "retryable": True}
            ),
        }

    def synthesize_intervention_explanation(
        self,
        video_id: str,
        intervention_id: str,
        intervention_timestamp: float,
        content_type: str,
        trigger_reason: str,
        transcript_entries: List,
        output_mode: str = "explanatory",
        vlm_snapshot: Optional[Dict] = None,
        on_stage: Optional[Any] = None,
        ownership_check: Optional[Any] = None,
        checkpoint_commit: Optional[Any] = None,
    ) -> Dict:
        """
        Reconcile saved evidence with the transcript, then run Mistral and TTS.

        Args:
            vlm_snapshot: Evidence already in memory from extraction. When
                omitted, the saved snapshot is loaded and must validate.
            on_stage: Optional callable receiving the next stage name

        Returns:
            Dictionary with intervention_id, text_explanation, audio_file_path,
            text_file_path, vlm_snapshot_path and output_mode
        """
        intervention_dir = self._intervention_dir(video_id, intervention_id)
        vlm_snapshot_path = os.path.join(intervention_dir, "vlm_snapshot.json")
        fresh_snapshot = vlm_snapshot is not None
        if vlm_snapshot is None:
            vlm_snapshot = load_usable_snapshot(vlm_snapshot_path)
        if vlm_snapshot is None:
            raise VisualEvidenceUnavailableError("No usable visual evidence is saved")
        if ownership_check is not None:
            ownership_check()

        from synchronizer import TranscriptEntry
        transcript_entries = [
            e if isinstance(e, TranscriptEntry) else TranscriptEntry(
                start=e.get("start", 0.0),
                end=e.get("start", 0.0) + e.get("duration", 0.0),
                text=e.get("text", "")
            )
            for e in transcript_entries
        ]
        transcript_segment = self._find_transcript_segment(
            intervention_timestamp,
            transcript_entries
        )
        transcript_context = transcript_segment.text if transcript_segment else ""

        # Reconcile canonical visual facts with the matched teacher speech
        # after VLM extraction and before synthesis.
        reconciliation_started = time.monotonic()
        trace_stage("reconciliation", "start")
        self._attach_transcript_coverage(vlm_snapshot, transcript_context)
        # A reused canonical snapshot is immutable cache evidence. Reconcile it
        # in memory for this mode without rewriting a previously valid file.
        if fresh_snapshot:
            commit_reconciliation = lambda: self._write_json_atomic(
                vlm_snapshot_path, vlm_snapshot
            )
            if checkpoint_commit is not None:
                checkpoint_commit(commit_reconciliation)
            else:
                commit_reconciliation()
        trace_stage(
            "reconciliation", "complete", started_at=reconciliation_started,
            fact_count=(vlm_snapshot.get("coverage_plan") or {}).get("fact_count", 0),
        )

        ollama_started = time.monotonic()
        if ownership_check is not None:
            ownership_check()
        trace_stage("ollama_readiness", "start")
        get_ollama_service().ensure_ready()
        trace_stage("ollama_readiness", "complete", started_at=ollama_started)

        from synthesizer import LLMSynthesizer
        synthesizer = LLMSynthesizer(
            ollama_url="http://localhost:11434",
            model="mistral",
            timeout=120,
            max_retries=3,
            transcripts=transcript_entries,
            output_mode=output_mode
        )
        text_file_path = os.path.join(intervention_dir, f"{output_mode}.txt")
        explanation = ""
        try:
            if os.path.isfile(text_file_path):
                with open(text_file_path, "r", encoding="utf-8") as handle:
                    explanation = handle.read().strip()
            if explanation:
                trace_stage("mistral_generation", "reused", text_chars=len(explanation))
            else:
                mistral_started = time.monotonic()
                if ownership_check is not None:
                    ownership_check()
                trace_stage("mistral_generation", "request")
                explanation = synthesizer.weave(
                    transcript_context=transcript_context,
                    vlm_snapshot=vlm_snapshot,
                    content_type=content_type,
                    intervention_reason=trigger_reason,
                    output_mode=output_mode,
                    require_visual_explanation=True,
                )
                trace_stage(
                    "mistral_generation", "response", started_at=mistral_started,
                    text_chars=len(explanation),
                )
                if ownership_check is not None:
                    try:
                        ownership_check()
                    except Exception:
                        trace_stage("video_cancellation", "late_mistral_response_discarded")
                        raise
                temporary_text = f"{text_file_path}.tmp"
                try:
                    with open(temporary_text, "w", encoding="utf-8") as handle:
                        handle.write(explanation)
                    commit_text = lambda: os.replace(temporary_text, text_file_path)
                    if checkpoint_commit is not None:
                        checkpoint_commit(commit_text)
                    else:
                        commit_text()
                finally:
                    if os.path.exists(temporary_text):
                        try:
                            os.remove(temporary_text)
                        except OSError:
                            pass
                trace_stage(
                    "explanation_text_persistence", "complete",
                    text_path=text_file_path, text_chars=len(explanation),
                )
        finally:
            synthesizer.cleanup()
        print("[Orchestrator] Text synthesis complete")

        if on_stage is not None:
            on_stage("generating_tts")
        if ownership_check is not None:
            ownership_check()

        from tts_engine import TTSEngine
        tts_engine = TTSEngine()
        audio_file_path = os.path.join(intervention_dir, f"{output_mode}.mp3")
        if self._valid_mp3(audio_file_path):
            trace_stage(
                "tts_creation", "reused", audio_path=audio_file_path,
                audio_bytes=os.path.getsize(audio_file_path),
            )
            tts_engine.cleanup()
        else:
            tts_started = time.monotonic()
            temporary_audio = f"{audio_file_path}.tmp.mp3"
            trace_stage("tts_creation", "start", audio_path=audio_file_path)
            try:
                tts_engine.generate(explanation, temporary_audio)
                if not self._valid_mp3(temporary_audio):
                    raise RuntimeError("TTS provider did not create a valid MP3 artifact")
                if ownership_check is not None:
                    ownership_check()
                commit_audio = lambda: os.replace(temporary_audio, audio_file_path)
                if checkpoint_commit is not None:
                    checkpoint_commit(commit_audio)
                else:
                    commit_audio()
            finally:
                tts_engine.cleanup()
                try:
                    if os.path.exists(temporary_audio):
                        os.remove(temporary_audio)
                        trace_stage("tts_creation", "temporary_artifact_removed")
                except OSError:
                    pass
            trace_stage(
                "tts_creation", "complete", started_at=tts_started,
                audio_path=audio_file_path, audio_bytes=os.path.getsize(audio_file_path),
            )
        print(f"[Orchestrator] Audio generated: {audio_file_path}")
        print(f"[Orchestrator] Text explanation saved: {text_file_path}")

        return {
            "intervention_id": intervention_id,
            "text_explanation": explanation,
            "audio_file_path": audio_file_path,
            "text_file_path": text_file_path,
            "vlm_snapshot_path": vlm_snapshot_path,
            "output_mode": output_mode
        }

    def generate_intervention_explanation(
        self,
        video_id: str,
        intervention_id: str,
        intervention_timestamp: float,
        frame_path: str,
        content_type: str,
        trigger_reason: str,
        transcript_entries: List,
        output_mode: str = "explanatory",
        classification_confidence: Optional[float] = None,
        confidence_semantics: str = "unknown",
        raw_category_labels: Optional[List[str]] = None,
        canonical_category_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run both stages for one intervention in the calling thread.

        Raises:
            VisualEvidenceUnavailableError: if extraction gave no usable evidence
        """
        print(f"[Orchestrator] Generating explanation for intervention {intervention_id}")
        print(f"[Orchestrator] Mode: {output_mode}, Content Type: {content_type}")

        evidence = self.extract_visual_evidence(
            video_id=video_id,
            intervention_id=intervention_id,
            intervention_timestamp=intervention_timestamp,
            frame_path=frame_path,
            content_type=content_type,
            classification_confidence=classification_confidence,
            confidence_semantics=confidence_semantics,
            raw_category_labels=raw_category_labels,
            canonical_category_ids=canonical_category_ids,
        )
        if not evidence["usable"]:
            error = VisualEvidenceUnavailableError("Visual evidence is unavailable")
            error.failure = evidence["failure"]
            raise error
        return self.synthesize_intervention_explanation(
            video_id=video_id,
            intervention_id=intervention_id,
            intervention_timestamp=intervention_timestamp,
            content_type=content_type,
            trigger_reason=trigger_reason,
            transcript_entries=transcript_entries,
            output_mode=output_mode,
            vlm_snapshot=None if evidence["reused"] else evidence["snapshot"],
        )


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
