# Frame Extraction Module (M0) Documentation

## Overview

The Frame Extraction Module is responsible for extracting pedagogically important keyframes from educational videos. It uses computer vision and machine learning techniques to identify frames that contain significant visual content relevant to the lecture.

## Purpose

- Extract representative frames from video lectures
- Identify pedagogically important content using CLIP-based classification
- Calculate visual complexity scores for frame selection
- Support educational content detection across multiple STEM subjects
- Optimize frame selection to reduce redundancy while maintaining coverage

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **OpenCV (cv2)**: Video processing and frame extraction
- **NumPy**: Numerical operations and array handling
- **PyTorch**: Deep learning framework for CLIP model
- **PIL (Pillow)**: Image processing
- **yt-dlp**: YouTube video downloading and streaming
- **CLIP (OpenAI)**: Vision-language model for content classification

### System Requirements
- **FFmpeg**: Required for video stream processing
  - Windows: Bundled in `frame_extraction/ffmpeg/bin/`
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster processing)
- **Memory**: 4GB+ RAM minimum, 8GB+ recommended
- **Storage**: 10-20GB per 60-minute video for temporary files

## Implementation Details

### Module Structure

```
frame_extraction/
├── __init__.py
├── main.py                    # CLI entry point
├── pedagogical_extractor_v2.py # Main extraction logic
├── stream.py                  # Video streaming utilities
├── batch_process.py           # Batch processing support
├── ffmpeg/                    # Bundled FFmpeg binaries
│   └── bin/
│       └── ffmpeg.exe
└── data/                      # Output directory
```

### Core Components

#### 1. Data Structures

**FrameData**: Raw frame with basic metadata
```python
@dataclass
class FrameData:
    timestamp: float
    frame: np.ndarray
    embedding: Optional[np.ndarray] = None
```

**ScoredFrame**: Frame with computed scores for selection
```python
@dataclass
class ScoredFrame:
    timestamp: float
    frame: np.ndarray
    embedding: np.ndarray
    complexity_score: float = 0.0
    is_educational: bool = True
    content_label: str = "educational"
```

**ExtractedKeyframe**: Final output keyframe
```python
@dataclass
class ExtractedKeyframe:
    timestamp: str
    frame_path: str
    content_type: str
    timestamp_seconds: float
    complexity_score: float
    clip_confidence: float
```

#### 2. CLIP Model Management

**Singleton Pattern**: CLIP model is loaded once and reused across all frames

```python
def get_clip():
    """Lazy-load CLIP model."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import clip
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
    return _clip_model, _clip_preprocess, _clip_device
```

**Text Embedding Cache**: Text prompt embeddings are cached to avoid recomputation

```python
_text_embeddings_cache = {}  # Cache for text prompt embeddings

def get_cached_text_embeddings() -> torch.Tensor:
    """Get cached text embeddings for classification prompts."""
    global _text_embeddings_cache
    cache_key = "edu_classifier"
    if cache_key not in _text_embeddings_cache:
        # Compute and cache embeddings
        ...
    return _text_embeddings_cache[cache_key]
```

#### 3. Educational Content Classification

**Categories with Ensembling**: Each category has 3 prompt variations for robustness

```python
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
    # ... more categories
}
```

**Classification Logic**:
- Computes CLIP embeddings for frame and text prompts
- Compares frame embeddings to educational and non-educational text embeddings
- Uses ensemble mean for category scoring
- Determines if frame is educational based on similarity comparison

```python
def classify_educational(frame_bgr: np.ndarray) -> Tuple[bool, str, float]:
    """Binary classification: Is this frame educational STEM content?"""
    # Compute CLIP embedding for frame
    # Get cached text embeddings
    # Calculate similarities
    # Determine best category
    # Return (is_educational, content_label, confidence)
```

#### 4. Visual Complexity Scoring

**Multi-Feature Approach**: Uses 5 different visual features for robustness

```python
def compute_complexity_score(frame: np.ndarray) -> float:
    """Compute visual complexity score that works across ALL background types."""
    # 1. Edge density (Canny) - works on any background
    # 2. Gradient magnitude (Sobel) - captures drawing/writing detail
    # 3. Color variance - captures colorful diagrams
    # 4. Local contrast (Laplacian) - captures text and fine details
    # 5. Text-like regions - high-frequency content detection

    # Weighted combination
    complexity = (
        0.30 * edge_density
        + 0.25 * text_score
        + 0.20 * contrast_score
        + 0.15 * gradient_score
        + 0.10 * color_variance
    )
    return min(1.0, complexity)
```

**Adaptive Thresholds**: Uses median-based Canny thresholds for robustness

```python
median_val = np.median(gray)
lower = int(max(0, 0.7 * median_val))
upper = int(min(255, 1.3 * median_val))
edges = cv2.Canny(gray, lower, upper)
```

#### 5. Video Streaming with Retry Logic

**Streaming with FFmpeg**: Direct video streaming without full download

```python
def stream_frames(
    stream_url: str,
    width: int,
    height: int,
    fps: float = 0.5,
    max_retries: int = 3,
    timeout_seconds: int = 30,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Stream frames from video URL at specified FPS."""
    # Uses FFmpeg subprocess to stream raw video frames
    # Implements retry logic with exponential backoff
    # Handles stream freezes and timeouts
    # Maintains frame position for resume capability
```

**Retry Strategy**:
- Up to 3 retries with exponential backoff (2s, 4s, 8s)
- Seeks to last successful frame position on retry
- Windows-specific timeout handling with threading
- Unix/macOS uses select() for non-blocking checks

#### 6. Adaptive Scene Segmentation

**CLIP-Based Scene Detection**: Uses semantic similarity for scene boundaries

```python
def segment_into_scenes(
    frames: List[FrameData],
    min_scene_gap: float = 4.0
) -> List[Scene]:
    """Segment frames into conceptual scenes using CLIP embeddings."""
    # Compute consecutive frame similarities
    # Calculate adaptive threshold from video statistics
    # Detect scene boundaries at significant similarity drops
    # Enforce minimum scene gap to avoid over-segmentation
```

**Adaptive Thresholding**:
```python
sim_values = [s for _, s in similarities]
mean_sim = np.mean(sim_values)
std_sim = np.std(sim_values)
threshold = max(0.70, mean_sim - 1.5 * std_sim)
```

#### 7. Frame Selection Pipeline

**7-Phase Extraction Process**:

1. **Video Info Extraction**: Get stream URL, duration, resolution
2. **Frame Sampling**: Extract frames at specified FPS
3. **Batch Embedding**: Compute CLIP embeddings in batches (4-6x faster)
4. **Scoring & Classification**: Compute complexity and educational classification
5. **Scene Segmentation**: Group frames into conceptual scenes
6. **Frame Selection**: Select best frame from each scene
7. **Coverage Guarantee**: Ensure temporal coverage and optimal frame count

**Selection Criteria**:
- Must be educational content
- Highest complexity score within scene
- Temporal spacing (minimum 4 seconds)
- Semantic deduplication (similarity threshold 0.92)
- Scene guarantee: Always save best frame from each scene

**Target Frame Calculation**:
```python
target_min = max(5, int(duration / 60 * 1.5))  # ~1.5 frames per minute
target_max = max(8, int(duration / 60 * 4))    # ~4 frames per minute
```

#### 8. Semantic Deduplication

**Last-Frame Comparison**: Only compares to the last saved frame to allow topic revisits

```python
def is_semantically_redundant(
    new_embedding: np.ndarray,
    saved_embeddings: List[np.ndarray],
    threshold: float = 0.92,
) -> Tuple[bool, float]:
    """Check if new frame is semantically redundant with saved frames."""
    if not saved_embeddings:
        return False, 0.0
    # Compare to last saved only (allows topic revisits)
    last_embedding = saved_embeddings[-1]
    similarity = float(np.dot(new_embedding, last_embedding))
    return similarity > threshold, similarity
```

**Scene Guarantee Override**: Even if redundant, saves best frame from each scene

```python
if is_redundant:
    if verbose:
        print(f"Scene {best.timestamp:.0f}s similar (CLIP={sim:.3f}) but SAVED (scene guarantee)")
    # Don't skip - still save this frame
```

## Key Algorithms

### 1. Pedagogical Keyframe Extraction Algorithm

```
Input: video_id, output_dir, sample_fps
Output: List[ExtractedKeyframe]

1. Get video stream URL and metadata
2. Calculate target frame count based on duration
3. Stream frames at sample_fps
4. For each frame:
   a. Skip if mostly blank
   b. Compute CLIP embedding
5. Batch compute embeddings for efficiency
6. For each frame:
   a. Compute visual complexity score
   b. Classify as educational/non-educational
7. Segment frames into scenes using CLIP similarities
8. For each scene:
   a. Find educational frames with good complexity
   b. Select frame with highest complexity
   c. Apply temporal spacing constraints
   d. Check semantic redundancy
   e. Save best frame (with scene guarantee)
9. Ensure temporal coverage (add frames if too few)
10. Trim if too many frames
11. Save frames to disk
12. Generate keyframes.json and video_meta.json
```

### 2. Visual Complexity Calculation

```
Input: frame (BGR image)
Output: complexity_score (0.0 - 1.0)

1. Convert to grayscale
2. Calculate edge density:
   a. Compute median pixel value
   b. Set Canny thresholds: lower=0.7*median, upper=1.3*median
   c. Apply Canny edge detection
   d. Count edge pixels / total pixels
3. Calculate gradient magnitude:
   a. Apply Sobel operators in x and y
   b. Compute sqrt(sobelx^2 + sobely^2)
   c. Normalize by 255.0
4. Calculate color variance:
   a. Convert to HSV
   b. Compute standard deviation of hue channel
   c. Normalize by 180.0
5. Calculate local contrast:
   a. Apply Laplacian operator
   b. Compute variance of Laplacian
   c. Normalize by 10000.0
6. Calculate text-like regions:
   a. Apply Otsu thresholding
   b. Count pixel transitions
   c. Normalize and cap at 1.0
7. Weighted combination:
   complexity = 0.30*edge + 0.25*text + 0.20*contrast + 0.15*gradient + 0.10*color
8. Return min(complexity, 1.0)
```

## Configuration Parameters

### Main Function Parameters

```python
def extract_pedagogical_keyframes(
    video_id: str,                    # YouTube video ID
    output_dir: str,                  # Output directory path
    verbose: bool = False,            # Print progress messages
    sample_fps: float = 0.5,          # Frame sampling rate (frames per second)
    min_frame_gap: float = 4.0,       # Minimum seconds between selected frames
    max_frame_gap: float = 30.0,      # Maximum seconds between frames (coverage)
    similarity_threshold: float = 0.92,  # CLIP similarity threshold for redundancy
) -> List[ExtractedKeyframe]
```

### Performance Tuning

- **sample_fps**: Lower values (0.3-0.5) for faster processing, higher (1.0-2.0) for better coverage
- **min_frame_gap**: Increase to reduce redundancy, decrease for more frames
- **similarity_threshold**: Higher (0.95) for stricter deduplication, lower (0.85) for more coverage
- **BATCH_SIZE**: Default 8 for CLIP embedding, adjust based on GPU memory

## Output Format

### Keyframes JSON Structure

```json
[
  {
    "timestamp": "02:15",
    "timestamp_seconds": 135.0,
    "frame_path": "data/video_id/keyframes/frame_0001_02_15.jpg",
    "content_type": "equation",
    "complexity_score": 0.4523,
    "clip_confidence": 0.3891
  },
  ...
]
```

### Video Metadata JSON Structure

```json
{
  "video_id": "abc123",
  "title": "Introduction to Calculus",
  "duration_seconds": 1800,
  "duration_formatted": "30:00",
  "total_keyframes": 15,
  "subject": "Mathematics",
  "source_url": "https://www.youtube.com/watch?v=abc123",
  "date_processed": "2026-04-30T12:34:56.789",
  "content_type_counts": {
    "equation": 5,
    "graph": 4,
    "diagram": 3,
    "text": 3
  }
}
```

## Usage Examples

### Basic Usage

```python
from frame_extraction.pedagogical_extractor_v2 import extract_pedagogical_keyframes

# Extract keyframes from YouTube video
keyframes = extract_pedagogical_keyframes(
    video_id="dQw4w9WgXcQ",
    output_dir="data/dQw4w9WgXcQ/keyframes",
    verbose=True
)

print(f"Extracted {len(keyframes)} keyframes")
for kf in keyframes:
    print(f"[{kf.timestamp}] {kf.content_type}: {kf.frame_path}")
```

### Command Line Interface

```bash
# Basic extraction
python frame_extraction/pedagogical_extractor_v2.py dQw4w9WgXcQ

# Custom output directory
python frame_extraction/pedagogical_extractor_v2.py dQw4w9WgXcQ ./output/keyframes
```

### Batch Processing

```python
from frame_extraction.batch_process import process_video_list

video_ids = ["video1", "video2", "video3"]
results = process_video_list(video_ids, output_base_dir="data")
```

## Performance Characteristics

### Processing Time (10-minute video)

| Component | Time | Notes |
|-----------|------|-------|
| Video Info Extraction | < 100ms | yt-dlp metadata fetch |
| Frame Streaming | 10-20s | Depends on internet speed |
| CLIP Embeddings | 15-30s | Batch processing (BATCH_SIZE=8) |
| Scoring & Classification | 5-10s | Complexity + educational check |
| Scene Segmentation | < 1s | CLIP similarity calculations |
| Frame Selection | < 1s | Complexity ranking |
| Total | **30-60s** | Without download time |

### Memory Usage

- **CPU-only**: 2-4GB RAM
- **GPU (CUDA)**: 4-6GB VRAM for CLIP model
- **Disk**: 50-100MB for keyframes (15-30 frames)

### Optimization Techniques

1. **Batch Embedding**: 4-6x faster than one-by-one
2. **Text Embedding Cache**: Computed once, reused
3. **Lazy Loading**: CLIP model loaded on first use
4. **Streaming**: No full video download required
5. **Blank Frame Skipping**: Reduces processing overhead

## Troubleshooting

### Common Issues

**Issue**: "FFmpeg not found"
- **Solution**: Ensure FFmpeg is in PATH or bundled in `frame_extraction/ffmpeg/bin/`

**Issue**: "CUDA out of memory"
- **Solution**: Reduce BATCH_SIZE or use CPU mode

**Issue**: "Too few frames extracted"
- **Solution**: Lower `similarity_threshold` or `min_frame_gap`

**Issue**: "Frames are blurry/low quality"
- **Solution**: Increase sample_fps or check video source quality

### Debugging

```python
# Enable verbose output
keyframes = extract_pedagogical_keyframes(
    video_id="abc123",
    output_dir="data/abc123/keyframes",
    verbose=True  # Shows detailed progress
)
```

## Future Enhancements

1. **Adaptive Sampling**: Dynamically adjust FPS based on scene complexity
2. **Multi-Scale Processing**: Analyze frames at multiple resolutions
3. **Content-Specific Models**: Use specialized models for different subjects
4. **Progressive Refinement**: Iterative frame selection with user feedback
5. **Semantic Clustering**: Group frames by semantic similarity
6. **Temporal Coherence**: Ensure smooth progression of visual content
