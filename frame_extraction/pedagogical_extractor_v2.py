import os
import json
import subprocess
import platform
import threading
import time
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Generator
from collections import deque
from datetime import datetime

logger = logging.getLogger("synapse.extractor")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class FrameData:
    """Raw frame with basic metadata."""

    timestamp: float
    frame: np.ndarray
    embedding: Optional[np.ndarray] = None


@dataclass
class ScoredFrame:
    """Frame with computed scores for selection."""

    timestamp: float
    frame: np.ndarray
    embedding: np.ndarray
    complexity_score: float = 0.0
    is_educational: bool = True
    content_label: str = "educational"


@dataclass
class Scene:
    """A conceptual scene containing multiple frames."""

    start_time: float
    end_time: float
    frames: List[ScoredFrame] = field(default_factory=list)
    best_frame: Optional[ScoredFrame] = None


@dataclass
class ExtractedKeyframe:
    """Final output keyframe."""

    timestamp: str
    frame_path: str
    content_type: str
    timestamp_seconds: float
    complexity_score: float
    clip_confidence: float


# =============================================================================
# CLIP MODEL (SINGLETON)
# =============================================================================

_clip_model = None
_clip_preprocess = None
_clip_device = None
_text_embeddings_cache = {}  # Cache for text prompt embeddings


def get_clip():
    """Lazy-load CLIP model."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import clip

        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Loading CLIP ViT-B/32 on {_clip_device}...")
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
    return _clip_model, _clip_preprocess, _clip_device


def compute_embedding(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized CLIP embedding for a frame."""
    model, preprocess, device = get_clip()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten()


# Educational content prompts with ENSEMBLING (multiple phrasings per category)
# Each category has 3 prompt variations for more robust classification
EDU_CATEGORIES = {
    "diagram": [
        "a diagram explaining a concept with labels and arrows",
        "an educational diagram with annotations",
        "a labeled illustration showing a concept",
    ],
    "equation": [
        "mathematical equations with equals signs and variables like x y z",
        "algebra with symbols like plus minus equals square root",
        "calculus formulas with integrals summations and derivatives",
    ],
    "graph": [
        "a scientific graph with axes and plotted data",
        "a chart showing data visualization",
        "a plot with x and y axes displaying information",
    ],
    "flowchart": [
        "a flowchart showing a process or algorithm",
        "a process diagram with connected steps",
        "boxes and arrows showing a workflow",
    ],
    "biology": [
        "a biological cell or organism diagram with labeled parts",
        "anatomy or biology illustration",
        "a diagram of living organisms or cells",
    ],
    "chemistry": [
        "chemical structures or molecular diagrams",
        "molecule structures and chemical bonds",
        "periodic table or chemical formulas",
    ],
    "code": [
        "code or programming syntax on a screen",
        "source code with syntax highlighting",
        "programming code in an editor or terminal",
    ],
    "circuit": [
        "a circuit diagram with electronic components",
        "electrical circuit schematic",
        "electronic wiring diagram",
    ],
    "geometry": [
        "geometric shapes with measurements and angles",
        "geometry proof with triangles or circles",
        "mathematical shapes and constructions",
    ],
    "table": [
        "a table or chart showing organized data",
        "a data table with rows and columns",
        "organized information in tabular format",
    ],
    "notes": [
        "handwritten notes explaining a topic",
        "written explanations on a whiteboard",
        "hand-drawn diagrams and text",
    ],
    "text": [
        "a slide with only text and bullet points",
        "written paragraphs explaining something",
        "text-based content without diagrams or formulas",
    ],
    "slide": [
        "an educational slide with both images and text",
        "a presentation slide with graphics and words",
        "visual educational content with diagrams",
    ],
}

# Flatten edu prompts for embedding (maintain order)
EDU_LABELS = list(EDU_CATEGORIES.keys())  # ['diagram', 'equation', 'graph', ...]
EDU_PROMPTS = [prompt for prompts in EDU_CATEGORIES.values() for prompt in prompts]
PROMPTS_PER_CATEGORY = 3  # Each category has 3 prompts

# Non-educational content prompts (also expanded)
NON_EDU_PROMPTS = [
    "a person's face talking to the camera",
    "a close-up of someone speaking",
    "a YouTube video thumbnail or title screen",
    "a subscribe button or end screen with social links",
    "like and subscribe reminder",
    "a blank or empty screen",
    "a solid color background",
    "a plain title slide with only text",
    "a classroom or background without content",
    "video intro or outro animation",
    "channel logo or branding",
]


def get_cached_text_embeddings() -> torch.Tensor:
    """
    Get cached text embeddings for classification prompts.
    Computes once on first call, reuses thereafter.
    """
    global _text_embeddings_cache

    cache_key = "edu_classifier"
    if cache_key not in _text_embeddings_cache:
        model, _, device = get_clip()
        import clip

        all_prompts = EDU_PROMPTS + NON_EDU_PROMPTS
        text_tokens = clip.tokenize(all_prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        _text_embeddings_cache[cache_key] = text_features
        print("    Text embeddings cached for classification.")

    return _text_embeddings_cache[cache_key]


def classify_educational(frame_bgr: np.ndarray) -> Tuple[bool, str, float]:
    """
    Binary classification: Is this frame educational STEM content?

    Uses carefully designed prompts that work across all video types:
    - Black backgrounds (Khan Academy)
    - White/light backgrounds (slides, whiteboards)
    - Colorful diagrams (biology, chemistry)
    - Code on screens
    - Handwritten content

    Returns: (is_educational, content_label, confidence)
    """
    model, preprocess, device = get_clip()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    # Get cached text embeddings (computed only once)
    text_features = get_cached_text_embeddings()

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T)[0]

        # Get scores for educational vs non-educational
        n_edu = len(EDU_PROMPTS)
        edu_similarities = similarities[:n_edu]
        non_edu_similarities = similarities[n_edu:]

        # ENSEMBLE: Compute mean similarity per category (more robust than max)
        category_scores = []
        for i in range(len(EDU_LABELS)):
            start_idx = i * PROMPTS_PER_CATEGORY
            end_idx = start_idx + PROMPTS_PER_CATEGORY
            category_mean = edu_similarities[start_idx:end_idx].mean().item()
            category_scores.append(category_mean)

        # Best educational category
        best_category_idx = max(
            range(len(category_scores)), key=lambda i: category_scores[i]
        )
        best_edu_score = category_scores[best_category_idx]

        # Non-educational score (mean for stability)
        mean_non_edu_score = non_edu_similarities.mean().item()
        max_non_edu_score = non_edu_similarities.max().item()

        # Determine if educational (use best category vs non-edu)
        is_educational = best_edu_score > max_non_edu_score

        if is_educational:
            content_label = EDU_LABELS[best_category_idx]
            confidence = best_edu_score
        else:
            content_label = "non-educational"
            confidence = max_non_edu_score

    return is_educational, content_label, confidence


# =============================================================================
# VISUAL COMPLEXITY SCORING
# =============================================================================


def compute_complexity_score(frame: np.ndarray) -> float:
    """
    Compute visual complexity score that works across ALL background types.

    Uses multiple features that are background-agnostic:
    1. Edge density (Canny) - works on any background
    2. Gradient magnitude - captures drawing/writing detail
    3. Color variance - captures colorful diagrams
    4. Local contrast - captures text and fine details

    Returns: Normalized score 0-1 (higher = more complex/content-rich)
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape
    total_pixels = h * w

    # 1. Edge density (adaptive Canny)
    # Use median-based thresholds for robustness
    median_val = np.median(gray)
    lower = int(max(0, 0.7 * median_val))
    upper = int(min(255, 1.3 * median_val))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = np.sum(edges > 0) / total_pixels

    # 2. Gradient magnitude (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_score = np.mean(gradient_mag) / 255.0

    # 3. Color variance (for colorful diagrams)
    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_variance = np.std(hsv[:, :, 0]) / 180.0  # Hue variance
    else:
        color_variance = 0.0

    # 4. Local contrast (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast_score = np.var(laplacian) / 10000.0  # Normalize
    contrast_score = min(1.0, contrast_score)

    # 5. Text-like regions (high-frequency content)
    # Use morphological operations to detect text regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Count transitions (text has many)
    transitions = np.sum(np.abs(np.diff(binary.astype(float), axis=1))) / total_pixels
    text_score = min(1.0, transitions * 10)

    # Weighted combination
    complexity = (
        0.30 * edge_density
        + 0.25 * text_score
        + 0.20 * contrast_score
        + 0.15 * gradient_score
        + 0.10 * color_variance
    )

    return min(1.0, complexity)


def is_mostly_blank(frame: np.ndarray, threshold: float = 0.02) -> bool:
    """Check if frame is mostly blank/uniform (any color)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Low variance = uniform/blank
    return np.std(gray) < 20 or compute_complexity_score(frame) < threshold


# =============================================================================
# VIDEO STREAMING
# =============================================================================


def get_video_info(video_id: str) -> Tuple[str, int, str, int, int]:
    """Get video stream URL and metadata using yt-dlp."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Get direct video URL
        if "requested_formats" in info:
            stream_url = info["requested_formats"][0]["url"]
            width = info["requested_formats"][0].get("width", 1280)
            height = info["requested_formats"][0].get("height", 720)
        else:
            stream_url = info["url"]
            width = info.get("width", 1280)
            height = info.get("height", 720)

        duration = info.get("duration", 600)
        title = info.get("title", "Unknown")

    return stream_url, duration, title, width, height


def stream_frames(
    stream_url: str,
    width: int,
    height: int,
    fps: float = 0.5,
    max_retries: int = 3,
    timeout_seconds: int = 30,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Stream frames from video URL at specified FPS.
    Retries up to max_retries times if stream is interrupted or times out.
    """
    ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg"

    frame_size = width * height * 3
    frame_interval = 1.0 / fps
    frame_idx = 0
    retry_count = 0

    while retry_count <= max_retries:

        # On retry, seek to where we left off so we don't re-process frames
        seek_args = (
            ["-ss", str(int(frame_idx * frame_interval))] if frame_idx > 0 else []
        )

        cmd = [
            ffmpeg_path,
            *seek_args,
            "-i",
            stream_url,
            "-vf",
            f"fps={fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-loglevel",
            "error",
            "-",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                # ── Read with timeout (Windows vs Unix) ──────────────────────
                if platform.system() == "Windows":
                    result = [None]

                    def _read():
                        result[0] = process.stdout.read(frame_size)

                    t = threading.Thread(target=_read, daemon=True)
                    t.start()
                    t.join(timeout=timeout_seconds)

                    if t.is_alive():
                        # Read is still blocked — stream is frozen
                        logger.warning(
                            f"Stream frozen for {timeout_seconds}s "
                            f"at frame {frame_idx}, retrying..."
                        )
                        break  # exits inner while, triggers retry

                    raw_frame = result[0]
                else:
                    # Unix/macOS — use select for non-blocking check
                    import select

                    ready = select.select([process.stdout], [], [], timeout_seconds)
                    if not ready[0]:
                        logger.warning(
                            f"Stream timeout after {timeout_seconds}s "
                            f"at frame {frame_idx}, retrying..."
                        )
                        break
                    raw_frame = process.stdout.read(frame_size)
                # ─────────────────────────────────────────────────────────────

                if not raw_frame or len(raw_frame) != frame_size:
                    # Clean end of stream — finished normally
                    return

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                timestamp = frame_idx * frame_interval
                frame_idx += 1
                retry_count = 0  # Reset on every successful frame
                yield timestamp, frame.copy()

        except Exception as e:
            logger.error(f"Stream error at frame {frame_idx}: {e}")

        finally:
            process.terminate()
            process.wait()

        retry_count += 1
        if retry_count <= max_retries:
            wait_secs = 2**retry_count  # 2s, 4s, 8s
            logger.info(f"Retry {retry_count}/{max_retries} in {wait_secs}s...")
            time.sleep(wait_secs)

    logger.error(f"Stream failed after {max_retries} retries at frame {frame_idx}")


# =============================================================================
# ADAPTIVE SCENE SEGMENTATION
# =============================================================================


def segment_into_scenes(
    frames: List[FrameData], min_scene_gap: float = 4.0
) -> List[Scene]:
    """
    Segment frames into conceptual scenes using CLIP embeddings.

    A scene change is detected when the semantic similarity between
    consecutive frames drops below an adaptive threshold.

    The threshold is computed from the video's own statistics:
    - Mean similarity - 1.5 * std deviation

    This makes it work across all video types without hardcoding.
    """
    if not frames:
        return []

    # Compute consecutive similarities
    similarities = []
    for i in range(1, len(frames)):
        if frames[i].embedding is not None and frames[i - 1].embedding is not None:
            sim = float(np.dot(frames[i].embedding, frames[i - 1].embedding))
            similarities.append((i, sim))

    if not similarities:
        # No embeddings - treat as single scene
        scene = Scene(
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
        )
        return [scene]

    # Compute adaptive threshold from video statistics
    sim_values = [s for _, s in similarities]
    mean_sim = np.mean(sim_values)
    std_sim = np.std(sim_values)

    # Scene change threshold: below mean - 1.5*std (significant drops)
    # But not below 0.7 (always consider very different frames as new scenes)
    threshold = max(0.70, mean_sim - 1.5 * std_sim)

    # Detect scene boundaries
    scene_boundaries = [0]  # Start with first frame

    last_boundary_time = frames[0].timestamp
    for idx, sim in similarities:
        current_time = frames[idx].timestamp

        # Scene change if:
        # 1. Similarity drops below threshold, AND
        # 2. Minimum time has passed since last boundary
        if sim < threshold and (current_time - last_boundary_time) >= min_scene_gap:
            scene_boundaries.append(idx)
            last_boundary_time = current_time

    # Create scenes
    scenes = []
    for i in range(len(scene_boundaries)):
        start_idx = scene_boundaries[i]
        end_idx = (
            scene_boundaries[i + 1] if i + 1 < len(scene_boundaries) else len(frames)
        )

        scene_frames = frames[start_idx:end_idx]
        if scene_frames:
            scenes.append(
                Scene(
                    start_time=scene_frames[0].timestamp,
                    end_time=scene_frames[-1].timestamp,
                )
            )

    return scenes


# =============================================================================
# FRAME SELECTION WITHIN SCENES
# =============================================================================


def select_best_frame_in_window(
    frames: List[ScoredFrame], start_idx: int, end_idx: int
) -> Optional[ScoredFrame]:
    """
    Select the best (most content-rich) frame within a window.

    Selection criteria:
    1. Must be educational content
    2. Highest complexity score
    """
    window_frames = frames[start_idx:end_idx]

    # Filter to educational frames only
    edu_frames = [f for f in window_frames if f.is_educational]

    if not edu_frames:
        return None

    # Return frame with highest complexity
    return max(edu_frames, key=lambda f: f.complexity_score)


# =============================================================================
# SEMANTIC DEDUPLICATION
# =============================================================================


def is_semantically_redundant(
    new_embedding: np.ndarray,
    saved_embeddings: List[np.ndarray],
    threshold: float = 0.92,
) -> Tuple[bool, float]:
    """
    Check if new frame is semantically redundant with saved frames.

    Uses adaptive comparison:
    - Compares against the LAST saved frame only
    - This allows revisiting similar topics later in the video

    Returns: (is_redundant, similarity_to_last)
    """
    if not saved_embeddings:
        return False, 0.0

    # Compare to last saved only (allows topic revisits)
    last_embedding = saved_embeddings[-1]
    similarity = float(np.dot(new_embedding, last_embedding))

    return similarity > threshold, similarity


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================


def extract_pedagogical_keyframes(
    video_id: str,
    output_dir: str,
    verbose: bool = False,
    sample_fps: float = 0.5,
    min_frame_gap: float = 4.0,
    max_frame_gap: float = 30.0,
    similarity_threshold: float = 0.92,
) -> List[ExtractedKeyframe]:

    # Normalize path for Windows compatibility
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Get video info
    if verbose:
        print("\n  [1/7] Getting video stream...")

    stream_url, duration, title, width, height = get_video_info(video_id)

    # Calculate target frame count based on duration
    target_min = max(5, int(duration / 60 * 1.5))  # ~1.5 frames per minute
    target_max = max(8, int(duration / 60 * 4))  # ~4 frames per minute

    if verbose:
        print(f"    Title: {title[:60]}...")
        print(f"    Duration: {duration}s ({duration/60:.1f} min)")
        print(f"    Resolution: {width}x{height}")
        print(f"    Target frames: {target_min}-{target_max}")

    # Phase 2: Sample and process frames
    if verbose:
        print("\n  [2/7] Sampling frames...")

    all_frames: List[FrameData] = []
    frame_count = 0

    for timestamp, frame in stream_frames(stream_url, width, height, sample_fps):
        # Skip blank frames
        if is_mostly_blank(frame):
            continue

        frame_data = FrameData(timestamp=timestamp, frame=frame)
        all_frames.append(frame_data)
        frame_count += 1

        if verbose and frame_count % 30 == 0:
            print(f"    Processed {frame_count} frames ({timestamp:.0f}s)...")

    if verbose:
        print(f"    Total valid frames: {len(all_frames)}")

    if not all_frames:
        print("    Error: No valid frames found")
        return []

    # Phase 3: Compute embeddings in batches (4-6x faster than one-by-one)
    if verbose:
        print("\n  [3/7] Computing CLIP embeddings (batched)...")

    model, preprocess, device = get_clip()
    BATCH_SIZE = 8

    for batch_start in range(0, len(all_frames), BATCH_SIZE):
        batch = all_frames[batch_start : batch_start + BATCH_SIZE]

        # Preprocess all frames in this batch into a single tensor
        batch_tensors = []
        for frame_data in batch:
            frame_rgb = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            batch_tensors.append(preprocess(pil_image))

        image_batch = torch.stack(batch_tensors).to(device)

        # One GPU call for the whole batch
        with torch.no_grad():
            embeddings = model.encode_image(image_batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Assign each embedding back to its frame
        for i, frame_data in enumerate(batch):
            frame_data.embedding = embeddings[i].cpu().numpy().flatten()

        if verbose:
            done = min(batch_start + BATCH_SIZE, len(all_frames))
            print(f"    Embedded {done}/{len(all_frames)} frames...")

    # Phase 4: Score and classify frames
    if verbose:
        print("\n  [4/7] Scoring and classifying frames...")

    scored_frames: List[ScoredFrame] = []
    edu_count = 0

    for frame_data in all_frames:
        # Compute complexity
        complexity = compute_complexity_score(frame_data.frame)

        # Classify educational vs non-educational
        is_edu, label, confidence = classify_educational(frame_data.frame)

        scored = ScoredFrame(
            timestamp=frame_data.timestamp,
            frame=frame_data.frame,
            embedding=frame_data.embedding,
            complexity_score=complexity,
            is_educational=is_edu,
            content_label=label,
        )
        scored_frames.append(scored)

        if is_edu:
            edu_count += 1

    if verbose:
        print(f"    Educational frames: {edu_count}/{len(scored_frames)}")

    # Phase 5: Segment into scenes
    if verbose:
        print("\n  [5/7] Segmenting into conceptual scenes...")

    # Convert to FrameData for scene segmentation
    frame_data_list = [
        FrameData(timestamp=sf.timestamp, frame=sf.frame, embedding=sf.embedding)
        for sf in scored_frames
    ]

    scenes = segment_into_scenes(frame_data_list, min_scene_gap=min_frame_gap)

    if verbose:
        print(f"    Detected {len(scenes)} conceptual scenes")

    # Phase 6: Select best frame from each scene
    if verbose:
        print("\n  [6/7] Selecting representative frames...")

    selected_frames: List[ScoredFrame] = []
    saved_embeddings: List[np.ndarray] = []
    last_saved_time = -min_frame_gap

    # Create timestamp lookup for scored frames
    ts_to_frame = {sf.timestamp: sf for sf in scored_frames}

    for scene in scenes:
        # Find frames in this scene
        scene_scored = [
            sf
            for sf in scored_frames
            if scene.start_time <= sf.timestamp <= scene.end_time
        ]

        if not scene_scored:
            continue

        # Get educational frames with good complexity
        candidates = [
            sf
            for sf in scene_scored
            if sf.is_educational and sf.complexity_score > 0.05
        ]

        # Only use non-educational as fallback if:
        # 1. No educational frames in scene, AND
        # 2. Time since last saved frame exceeds max_frame_gap (coverage guarantee)
        time_since_last = scene.start_time - last_saved_time
        needs_coverage = time_since_last >= max_frame_gap

        if not candidates and needs_coverage:
            # Force save best frame for coverage, even if non-educational
            candidates = sorted(
                scene_scored, key=lambda x: x.complexity_score, reverse=True
            )[:1]
            if verbose and candidates:
                print(
                    f"    Force-including {scene.start_time:.0f}s for coverage (gap={time_since_last:.0f}s)"
                )

        if not candidates:
            continue

        # Select best candidate (highest complexity)
        best = max(candidates, key=lambda x: x.complexity_score)

        # Check temporal spacing
        if best.timestamp - last_saved_time < min_frame_gap:
            continue

        # Check semantic redundancy
        is_redundant, sim = is_semantically_redundant(
            best.embedding, saved_embeddings, similarity_threshold
        )

        # SCENE GUARANTEE: Save best frame from each scene regardless of similarity
        # This ensures consistent-style videos (same background, incremental content)
        # still get proper coverage across all detected scenes.
        # Without this, videos like Khan Academy or 3Blue1Brown would only get 1 frame
        # because consecutive scenes have CLIP similarity > 0.92
        if is_redundant:
            if verbose:
                print(
                    f"    Scene {best.timestamp:.0f}s similar (CLIP={sim:.3f}) but SAVED (scene guarantee)"
                )
            # Don't skip - still save this frame
        else:
            if verbose:
                print(
                    f"    Selected {best.timestamp:.0f}s: {best.content_label} (complexity={best.complexity_score:.3f})"
                )

        # Save this frame (always, since we passed temporal spacing check)
        selected_frames.append(best)
        saved_embeddings.append(best.embedding)
        last_saved_time = best.timestamp

    # Phase 7: Ensure temporal coverage
    if verbose:
        print("\n  [7/7] Ensuring temporal coverage...")

    # Check for gaps and force-save if needed
    final_frames = []
    last_time = -max_frame_gap

    for sf in selected_frames:
        # If gap is too large, we might need to fill it
        # (This is already handled by scene segmentation, but double-check)
        final_frames.append(sf)
        last_time = sf.timestamp

    # Check if we have too few frames
    if len(final_frames) < target_min:
        if verbose:
            print(f"    Too few frames ({len(final_frames)}), adding more...")

        # Add more frames by lowering thresholds
        remaining = [
            sf for sf in scored_frames if sf not in final_frames and sf.is_educational
        ]
        remaining.sort(key=lambda x: x.complexity_score, reverse=True)

        for sf in remaining:
            if len(final_frames) >= target_min:
                break

            # Check spacing
            too_close = any(
                abs(sf.timestamp - f.timestamp) < min_frame_gap / 2
                for f in final_frames
            )

            if not too_close:
                final_frames.append(sf)

        # Sort by timestamp
        final_frames.sort(key=lambda x: x.timestamp)

    # Trim if too many
    if len(final_frames) > target_max:
        # Keep most complex frames with good temporal distribution
        final_frames.sort(key=lambda x: x.complexity_score, reverse=True)
        final_frames = final_frames[:target_max]
        final_frames.sort(key=lambda x: x.timestamp)

    if verbose:
        print(f"    Final frame count: {len(final_frames)}")

    # Save frames
    results = []
    for idx, sf in enumerate(final_frames):
        mins = int(sf.timestamp // 60)
        secs = int(sf.timestamp % 60)

        filename = f"frame_{idx:04d}_{mins:02d}_{secs:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, sf.frame)

        results.append(
            ExtractedKeyframe(
                timestamp=f"{mins:02d}:{secs:02d}",
                frame_path=filepath,
                content_type=sf.content_label,
                timestamp_seconds=sf.timestamp,
                complexity_score=round(sf.complexity_score, 4),
                clip_confidence=round(sf.is_educational * sf.complexity_score, 4),
            )
        )

    # Save JSON report
    # CRITICAL FIX: Explicit dict construction to ensure timestamp_seconds is float
    report = [{
        "timestamp": r.timestamp,
        "timestamp_seconds": float(r.timestamp_seconds),  # CRITICAL: float for Synchronizer comparisons
        "frame_path": r.frame_path,
        "content_type": r.content_type,
        "complexity_score": r.complexity_score,
        "clip_confidence": r.clip_confidence
    } for r in results]
    report_path = os.path.join(output_dir, "keyframes.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        # ─── ADD FROM HERE ───────────────────────────────────────
    from collections import Counter

    content_types = [sf.content_label for sf in final_frames]

    subject_map = {
        "equation": "Mathematics",
        "graph": "Mathematics",
        "geometry": "Mathematics",
        "biology": "Biology",
        "chemistry": "Chemistry",
        "circuit": "Physics",
        "diagram": "Science",
        "code": "Computer Science",
        "table": "General",
        "flowchart": "General",
        "text": "General",
        "slide": "General",
        "notes": "General",
    }
    dominant_type = (
        Counter(content_types).most_common(1)[0][0] if content_types else "general"
    )

    meta = {
        "video_id": video_id,
        "title": title,
        "duration_seconds": duration,
        "duration_formatted": f"{int(duration // 60):02d}:{int(duration % 60):02d}",
        "total_keyframes": len(results),
        "subject": subject_map.get(dominant_type, "General"),
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "date_processed": datetime.now().isoformat(),
        "content_type_counts": dict(Counter(content_types)),
    }

    meta_path = os.path.join(output_dir, "..", "video_meta.json")
    meta_path = os.path.normpath(meta_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pedagogical_extractor_v2.py <video_id> [output_dir]")
        sys.exit(1)

    video_id = sys.argv[1]
    output_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join("data", video_id, "keyframes")
    )

    print(f"\nExtracting keyframes from: {video_id}")
    print(f"Output: {output_dir}\n")

    results = extract_pedagogical_keyframes(video_id, output_dir, verbose=True)

    print(f"\n✓ Extracted {len(results)} pedagogical keyframes")
    for r in results:
        print(f"  [{r.timestamp}] {r.content_type}: {os.path.basename(r.frame_path)}")
