# SYNAPSE — Project Documentation

> **Pedagogical Keyframe Extraction for STEM Education**  
> An AI-powered pipeline that extracts conceptually significant frames from STEM YouTube lectures to assist visually impaired learners.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Architecture & Complete Workflow](#3-architecture--complete-workflow)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
   - [main.py](#41-mainpy)
   - [batch_process.py](#42-batch_processpy)
   - [pedagogical_extractor_v2.py](#43-pedagogical_extractor_v2py)
   - [stream.py](#44-streampy)
   - [urls.txt](#45-urlstxt)
5. [Feature Deep Dives](#5-feature-deep-dives)
   - [CLIP-Based Content Classification](#51-clip-based-content-classification)
   - [Visual Complexity Scoring](#52-visual-complexity-scoring)
   - [Adaptive Scene Segmentation](#53-adaptive-scene-segmentation)
   - [Semantic Deduplication](#54-semantic-deduplication)
   - [Temporal Coverage Guarantee](#55-temporal-coverage-guarantee)
   - [Resilient Video Streaming](#56-resilient-video-streaming)
   - [SSIM-Based Extraction (Legacy)](#57-ssim-based-extraction-legacy)
   - [Output & Metadata Generation](#58-output--metadata-generation)
6. [Data Structures](#6-data-structures)
7. [Output Format](#7-output-format)
8. [Configuration & Tunable Parameters](#8-configuration--tunable-parameters)
9. [Extractor Modes Comparison](#9-extractor-modes-comparison)
10. [Usage Examples](#10-usage-examples)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

SYNAPSE is designed around one core mission: **given a STEM educational YouTube video, automatically extract the small set of frames that carry the most pedagogical value** — equations being written, diagrams being labeled, graphs being plotted — and discard everything else (talking-head filler, blank slides, subscribe screens, outro animations).

The extracted frames are intended to be fed downstream to an AI description/accessibility pipeline so that visually impaired learners receive a concise, content-rich visual summary of a lecture.

### Design Goals

| Goal | How it's achieved |
|---|---|
| Content-aware, not just motion-aware | CLIP embeddings classify *what* is in each frame |
| Generalize across all STEM video styles | Adaptive thresholds derived from each video's own statistics |
| Work on dark AND light backgrounds | Background-agnostic complexity signals (edges, gradients, contrast) |
| Avoid redundant "similar" frames | Cosine similarity deduplication on CLIP embeddings |
| Never miss important content | Temporal coverage guarantee + scene-level forced inclusion |
| Produce human-readable output | JSON reports with content type, timestamp, complexity scores |

---

## 2. Project Structure

```
synapse/
├── main.py                        # CLI entry point
├── batch_process.py               # Batch orchestration pipeline
├── pedagogical_extractor_v2.py    # Core AI extraction engine (default)
├── stream.py                      # Legacy SSIM extractor (fallback)
├── urls.txt                       # List of YouTube URLs to process
└── data/                          # Output directory (auto-created)
    ├── batch_report.json          # Batch run summary
    └── {video_id}/
        ├── video_meta.json        # Per-video metadata
        └── keyframes/
            ├── keyframes.json     # Keyframe report with scores
            ├── frame_0001_01_23.jpg
            ├── frame_0002_03_45.jpg
            └── ...
```

---

## 3. Architecture & Complete Workflow

The full end-to-end workflow from CLI invocation to saved keyframes:

```
User runs: python main.py [--extractor pedagogical] [--start 0] [--end 5]
                │
                ▼
        main.py — Parse CLI args, inject ffmpeg into PATH
                │
                ▼
        batch_process.run_batch()
                │
                ├── Load URLs from urls.txt
                ├── Determine extractor (v2 → pedagogical → SSIM fallback)
                │
                └── For each URL:
                        │
                        ├── extract_video_id(url)
                        │
                        └── extract_pedagogical_keyframes(video_id, output_dir)
                                │
                                ├── Phase 1: get_video_info()
                                │     └─ yt-dlp resolves stream URL + metadata
                                │
                                ├── Phase 2: stream_frames()
                                │     └─ FFmpeg pipes raw BGR frames at 0.5 fps
                                │     └─ Skip blank/uniform frames
                                │
                                ├── Phase 3: Batched CLIP embeddings
                                │     └─ Batch size 8, GPU if available
                                │
                                ├── Phase 4: Score & Classify
                                │     ├─ compute_complexity_score() per frame
                                │     └─ classify_educational() via CLIP
                                │
                                ├── Phase 5: segment_into_scenes()
                                │     └─ Adaptive CLIP similarity threshold
                                │
                                ├── Phase 6: Select best frame per scene
                                │     ├─ Filter to educational candidates
                                │     ├─ Pick highest complexity
                                │     ├─ Enforce min time gap (4s)
                                │     ├─ Semantic deduplication (CLIP cosine > 0.92)
                                │     └─ Scene guarantee override
                                │
                                ├── Phase 7: Temporal coverage check
                                │     ├─ Too few? Add more (lower thresholds)
                                │     └─ Too many? Keep most complex, trim
                                │
                                └── Save JPEGs + keyframes.json + video_meta.json
                                        │
                        batch_process saves data/batch_report.json
```

---

## 4. File-by-File Deep Dive

### 4.1 `main.py`

**Role:** CLI entry point and environment setup.

**What it does:**

1. Adds `./ffmpeg/bin` to the system `PATH` so the bundled FFmpeg binary is found without a system-wide installation.
2. Defines a full `argparse` CLI with these arguments:

| Argument | Default | Description |
|---|---|---|
| `--urls` / `-u` | `urls.txt` | Path to the URL list file |
| `--extractor` / `-e` | `pedagogical` | Extraction algorithm: `pedagogical`, `semantic`, `scene`, `ssim` |
| `--method` / `-m` | `hybrid` | Sub-method for the semantic extractor: `hybrid`, `cluster`, `temporal` |
| `--start` / `-s` | `None` | 0-based start index to slice the URL list |
| `--end` / `-n` | `None` | Exclusive end index to slice the URL list |

3. Constructs the `url_range` tuple and delegates all work to `batch_process.run_batch()`.

**Key design choice:** `main.py` contains zero extraction logic — it's purely a thin launcher. All business logic lives in the module it calls.

---

### 4.2 `batch_process.py`

**Role:** Batch orchestration — loops over all URLs, picks the right extractor, collects results, writes the summary report.

**Responsibilities:**

#### URL Loading (`load_urls`)
- Opens `urls.txt`
- Skips blank lines and comment lines (starting with `#`)
- Returns a clean list of URL strings

#### Video ID Extraction (`extract_video_id`)
Parses any YouTube URL format to get the 11-character video ID:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `/v/VIDEO_ID` paths

Uses two regex patterns with fallback.

#### Extractor Resolution
Checks module availability at import time using `try/except ImportError` flags:
- `PEDAGOGICAL_V2_AVAILABLE` — for `pedagogical_extractor_v2`
- `SSIM_AVAILABLE` — for `stream`

Fallback chain: **Pedagogical v2 → Pedagogical v1 → SSIM**

This means the system degrades gracefully if a dependency (like `clip`) is unavailable.

#### Main Loop (`run_batch`)
For each URL:
1. Extracts the video ID
2. Constructs `output_dir = data/{video_id}/keyframes`
3. Calls the selected extractor
4. Records: `{index, url, video_id, status, keyframes, time_sec}` or `{..., error}` on failure
5. Catches all exceptions with full traceback, marks as `"failed"`, and continues with the next URL

#### Summary & Reporting
After all videos:
- Prints a formatted table with per-video status, frame count, and time
- Prints totals: success rate, total frames extracted, total time
- Saves `data/batch_report.json` — a JSON array of all per-video result records

---

### 4.3 `pedagogical_extractor_v2.py`

**Role:** The primary AI extraction engine. Implements the complete 7-phase pipeline.

This is the most complex and important file. It is broken into distinct sections:

---

#### Data Structures

| Class | Purpose |
|---|---|
| `FrameData` | Raw frame + timestamp + optional CLIP embedding |
| `ScoredFrame` | Frame + embedding + `complexity_score` + `is_educational` flag + `content_label` |
| `Scene` | Time-bounded group of frames (start/end time) |
| `ExtractedKeyframe` | Final output record: timestamp string, file path, content type, seconds, complexity score, CLIP confidence |

---

#### CLIP Model Management

The CLIP model (`ViT-B/32`) is a **singleton** — loaded once on the first call to `get_clip()` and reused for all subsequent calls. This avoids repeated multi-second model loading across frames.

- Automatically uses **CUDA** if a GPU is available, otherwise **CPU**
- Text embeddings (the category prompts) are additionally cached in `_text_embeddings_cache` — computed once per session on the first `classify_educational()` call

---

#### Phase 1 — Video Info (`get_video_info`)

Uses `yt-dlp` to extract:
- Direct video stream URL (bypasses YouTube's DRM-free stream directly)
- Video resolution (width × height)
- Duration in seconds
- Title string

**Format preference:** `bestvideo[height<=720][ext=mp4]` — caps at 720p to reduce download bandwidth without sacrificing content legibility.

From the duration, it also pre-calculates the **target frame count range**:
- `target_min = max(5, duration_minutes × 1.5)` ← lower bound
- `target_max = max(8, duration_minutes × 4.0)` ← upper bound

This scales naturally: a 10-minute video targets 15–40 frames; a 60-minute lecture targets 90–240 frames.

---

#### Phase 2 — Frame Sampling (`stream_frames`)

Streams raw video frames using **FFmpeg** piped directly into Python — no temporary file is written to disk.

- Samples at **0.5 fps** (one frame every 2 seconds) — sufficient temporal resolution for lecture-style content where changes happen slowly
- Outputs raw **BGR24** pixel data directly into `numpy` arrays via `stdout`
- **Blank frame detection** before any expensive processing: frames where `np.std(gray) < 20` (near-uniform color) or complexity < 0.02 are immediately discarded

**Retry mechanism:**
- On stream freeze or read timeout, the process is killed and restarted
- On retry, it seeks to `frame_idx × frame_interval` seconds — resumes from where it left off rather than re-processing from the beginning
- Retries up to 3 times with **exponential backoff**: 2s → 4s → 8s wait

**Cross-platform timeout handling:**
- **Windows:** Spawns a daemon thread to perform the blocking `stdout.read()`; if the thread doesn't complete within `timeout_seconds` (30s), the stream is considered frozen
- **Unix/macOS:** Uses `select.select()` with a timeout for non-blocking I/O monitoring

---

#### Phase 3 — CLIP Embeddings (Batched)

Rather than computing embeddings one frame at a time (which would invoke the GPU kernel once per frame), frames are processed in **batches of 8**:

1. All frames in the batch are preprocessed into PIL Images and stacked into a single `torch.Tensor`
2. A single `model.encode_image(image_batch)` call processes all 8 frames at once
3. Each output embedding is L2-normalized and stored as a flat `numpy` array on the `FrameData`

This gives a **4–6× throughput improvement** over per-frame inference.

---

#### Phase 4 — Scoring & Classification

Every frame (that survived the blank-frame filter) goes through two scoring functions:

##### `compute_complexity_score(frame)`

Returns a float 0–1 representing how much visual content the frame contains. Uses 5 **background-agnostic** signals:

| Signal | Method | Weight | What it captures |
|---|---|---|---|
| Edge density | Canny (adaptive median-based thresholds) | 30% | Lines, borders, equation strokes |
| Text transitions | Otsu binary + horizontal diff count | 25% | Dense text, formulas |
| Local contrast | Laplacian variance | 20% | Fine detail sharpness |
| Gradient magnitude | Sobel X+Y | 15% | General drawing/writing detail |
| Color variance | HSV hue channel std deviation | 10% | Colorful diagrams, biology figures |

**Why adaptive Canny thresholds?** Standard Canny uses fixed thresholds, which fail on dark backgrounds (Khan Academy black chalkboard) or very bright slides. Here, `lower = 0.7 × median_pixel` and `upper = 1.3 × median_pixel`, making edge detection self-calibrating to each frame's actual brightness.

##### `classify_educational(frame)`

Binary question: *Is this frame educational STEM content?*

Uses the pre-loaded CLIP model to compare the frame's image embedding against two sets of text embeddings:

**Educational categories (13 total, 3 prompt variations each = 39 prompts):**

| Category | Example Prompts |
|---|---|
| `equation` | "mathematical equations with equals signs and variables like x y z" |
| `diagram` | "a diagram explaining a concept with labels and arrows" |
| `graph` | "a scientific graph with axes and plotted data" |
| `flowchart` | "a flowchart showing a process or algorithm" |
| `biology` | "a biological cell or organism diagram with labeled parts" |
| `chemistry` | "chemical structures or molecular diagrams" |
| `code` | "code or programming syntax on a screen" |
| `circuit` | "a circuit diagram with electronic components" |
| `geometry` | "geometric shapes with measurements and angles" |
| `table` | "a table or chart showing organized data" |
| `notes` | "handwritten notes explaining a topic" |
| `text` | "a slide with only text and bullet points" |
| `slide` | "an educational slide with both images and text" |

**Non-educational prompts (11 prompts):**
- Person's face talking to camera
- YouTube thumbnail / title screen
- Subscribe button / end screen
- Blank or solid color background
- Intro/outro animation
- Channel logo or branding

**Ensemble scoring:** For each educational category, the **mean** similarity across its 3 prompt variations is taken (not max). This makes classification more robust and less sensitive to prompt phrasing quirks.

**Decision rule:** `is_educational = best_category_mean_score > max_non_edu_score`

Returns: `(is_educational: bool, content_label: str, confidence: float)`

---

#### Phase 5 — Adaptive Scene Segmentation (`segment_into_scenes`)

Groups the stream of frames into conceptual "scenes" — time windows where the content is semantically related.

**Algorithm:**
1. For every consecutive pair of frames, compute the CLIP embedding cosine similarity: `sim = dot(embedding_i, embedding_{i+1})`
2. Collect all similarity values and compute their mean and standard deviation
3. **Adaptive threshold** = `max(0.70, mean_sim - 1.5 × std_sim)`
   - This is derived from the video's own content distribution
   - A video with gradual changes (slow build-up on a whiteboard) will have a high mean and tight std → higher threshold → fewer, longer scenes
   - A video switching between slides and demos will have more variation → lower threshold → more, shorter scenes
4. A new scene starts when `similarity < threshold` AND at least `min_scene_gap` seconds (4.0s) have passed since the last boundary
5. Each detected boundary creates a new `Scene` object with `start_time` and `end_time`

**Why compare only consecutive frames?** Because a scene change is a *local* event. Comparing to a global centroid would be confused by the overall video topic.

---

#### Phase 6 — Frame Selection Per Scene

For each scene, the pipeline selects at most one representative keyframe:

1. **Filter** scene frames to `is_educational == True` AND `complexity_score > 0.05`
2. If no educational frames exist **and** the time since the last saved frame exceeds `max_frame_gap` (30s), fall back to the highest-complexity non-educational frame (coverage guarantee)
3. **Select** the frame with the highest `complexity_score` from the candidates
4. **Temporal spacing check:** Skip if `best.timestamp - last_saved_time < min_frame_gap` (4s)
5. **Semantic deduplication:** Check `cosine_similarity(best.embedding, last_saved_embedding)`
   - If similarity > 0.92 → frame would normally be skipped as redundant
   - **BUT:** "Scene guarantee" overrides this — if the scene segmenter detected a new scene, the best frame is saved regardless, to handle incremental-build videos (Khan Academy, 3Blue1Brown) where consecutive scenes look very similar but represent new content

---

#### Phase 7 — Temporal Coverage & Final Trim

After scene-based selection:

1. **Too few frames?** (`len < target_min`)
   - Collect all remaining educational frames not yet selected
   - Sort by complexity descending
   - Add highest-complexity frames that aren't too close (within `min_frame_gap / 2`) to already-selected frames
   - Re-sort final list by timestamp

2. **Too many frames?** (`len > target_max`)
   - Sort by complexity descending
   - Keep only the top `target_max` frames
   - Re-sort by timestamp (to restore chronological order)

3. **Save all selected frames** as JPEG files named `frame_{idx:04d}_{mm}_{ss}.jpg`

4. **Generate `keyframes.json`** — array of `ExtractedKeyframe` objects with full metadata

5. **Generate `video_meta.json`** — video-level summary (see Output Format section)

---

### 4.4 `stream.py`

**Role:** Legacy / standalone SSIM-based keyframe extractor.

This is an older, simpler approach that does not use ML models. It is still functional and useful as a fast fallback when CLIP is unavailable.

**How it works:**

1. **`get_stream_info(youtube_url)`** — Uses `yt-dlp` to get a direct stream URL, preferring H.264 (avc1) codec over AV1, at max 720p. Returns: stream URL, video ID, title, duration.

2. **`probe_video(stream_url)`** — Runs `ffprobe` to get the actual resolution and frame rate from the stream headers.

3. **`stream_and_extract(youtube_url)`** — Main function:
   - Scales frames to 640px wide (maintaining aspect ratio) to reduce processing load
   - Samples at exactly **1 fps** (configurable via `sample_fps`)
   - Pipes raw RGB24 frames from FFmpeg into Python
   - Converts each frame to grayscale for SSIM comparison

   **Garbage frame detection** (before SSIM):
   - `max_pixel ≤ 5` → purely black frame
   - `min_pixel ≥ 250` → purely white frame  
   - `mean < 0.1` → near-black noise
   - `mean > 254` → near-white

   **Keyframe decision logic:**
   - First valid frame is always saved
   - `gap_ok`: at least `MIN_GAP_SEC` (1.5s) since last keyframe
   - `gap_too_long`: more than `MAX_GAP_SEC` (15s) since last keyframe → force-save regardless of similarity
   - Otherwise: compute `ssim(last_saved_gray, current_gray)` — save if `ssim < SSIM_THRESHOLD` (0.50)

   **SSIM threshold reasoning:** 0.50 means the current frame must be structurally at least 50% different from the last saved frame. Lower values = more sensitive = more keyframes. This was tuned down from an earlier 0.65 to capture more incremental content changes.

4. Saves JPEGs with quality 88 to `data/{video_id}/keyframes/frame_{count:04d}_{hh}_{mm}_{ss}.jpg`

**Key parameters (all at top of file):**

| Parameter | Value | Meaning |
|---|---|---|
| `SCALE_WIDTH` | 640 | Resize all frames to 640px wide |
| `SSIM_THRESHOLD` | 0.50 | Frame is a keyframe if SSIM < this |
| `MIN_GAP_SEC` | 1.5 | Minimum seconds between keyframes |
| `MAX_GAP_SEC` | 15.0 | Force-save if no keyframe for this long |
| `JPEG_QUALITY` | 88 | JPEG compression quality |

---

### 4.5 `urls.txt`

A plain-text list of 14 YouTube STEM lecture URLs, one per line. Lines starting with `#` are treated as comments. Blank lines are ignored.

The `batch_process.py` currently hard-slices `[13:14]` from this list (line 113), limiting processing to the 14th URL. This is likely a development/testing override.

---

## 5. Feature Deep Dives

### 5.1 CLIP-Based Content Classification

**Technology:** OpenAI CLIP (Contrastive Language-Image Pre-training), `ViT-B/32` architecture.

**What CLIP does:** CLIP was trained on 400 million image-text pairs to learn a shared embedding space where an image of "a mathematical equation" and the text "mathematical equation" end up near each other in high-dimensional space. We exploit this to classify educational content *without training a custom classifier*.

**Workflow:**
```
Frame (BGR) 
    → BGR to RGB conversion
    → PIL Image
    → CLIP preprocessing (resize, normalize)
    → model.encode_image()         ← visual embedding (512-d vector)
    → L2 normalization

39 educational prompts + 11 non-edu prompts
    → clip.tokenize()
    → model.encode_text()          ← text embeddings (512-d each, cached)
    → L2 normalization

Similarity = image_embedding @ text_embeddings.T  (dot product = cosine sim)

Per educational category:
    mean_score = average of 3 prompt similarities

Best category = argmax(mean_scores)
Decision = best_edu_score > max_non_edu_score
```

**Why 3 prompts per category?** Single prompts can be brittle — "mathematical equations with equals signs" might not strongly activate for Roman numeral equations or Matrix equations. Having 3 diverse phrasings and taking the mean provides more robust coverage.

**Why compare against max of non-edu vs mean of edu?** Non-educational content can be confidently identified even from a single matching prompt (a face is a face). Educational content benefits from the stability of averaging.

---

### 5.2 Visual Complexity Scoring

The complexity score is designed to answer: *"How much information is packed into this frame?"*

It uses 5 independent signals combined with a weighted sum, designed to work on **any background color**:

```
complexity = 0.30 × edge_density
           + 0.25 × text_transitions
           + 0.20 × contrast_score
           + 0.15 × gradient_score
           + 0.10 × color_variance
```

**Why background-agnostic matters:**
- Traditional methods fail on Khan Academy's black background (white text on black looks very different from black text on white)
- Adaptive Canny uses `lower = 0.7 × median_pixel`, `upper = 1.3 × median_pixel` — automatically adjusts to the frame's luminance distribution
- Otsu thresholding for text detection is self-calibrating
- Laplacian and Sobel gradients detect changes relative to neighbors, not absolute brightness

**Score interpretation:**
- `< 0.02` → blank/uniform frame (filtered out before scoring)
- `0.02 – 0.15` → sparse content (title slide, minimal text)
- `0.15 – 0.40` → moderate content (annotated diagram, formula)
- `> 0.40` → dense content (complex equation, multi-panel figure)

---

### 5.3 Adaptive Scene Segmentation

The goal is to group the 1-fps frame stream into "moments" — stretches of video where the same concept is being developed — and then pick one representative frame per moment.

**Why not just sample every N seconds?** Fixed-interval sampling misses the structure of a lecture. A professor might spend 3 seconds on a title slide and 45 seconds building a complex proof. Fixed intervals would over-represent the title and under-represent the proof.

**Adaptive threshold derivation:**
```python
mean_sim = mean(all_consecutive_similarities)
std_sim  = std(all_consecutive_similarities)
threshold = max(0.70, mean_sim - 1.5 × std_sim)
```

This is a **statistics-based outlier detection** approach. Similarities that are more than 1.5 standard deviations below the mean are considered scene changes. The `max(0.70, ...)` floor ensures that even in very stable videos, genuinely different frames are always considered new scenes.

---

### 5.4 Semantic Deduplication

After scene-based selection picks one frame per scene, a secondary check ensures selected frames aren't semantically too similar to the *previous* selected frame.

```python
similarity = dot(new_embedding, last_saved_embedding)
is_redundant = similarity > 0.92
```

**Why compare only to the last saved frame?** Not to the entire history. This allows the video to "revisit" a topic later — if a professor returns to an equation from earlier in the lecture, that frame should be saved again, not suppressed.

**Scene Guarantee Override:** For videos like Khan Academy where every scene builds incrementally on the same visual field (same background, gradual annotation), consecutive scenes have CLIP similarity > 0.92. Without the override, most scenes would be dropped. The scene guarantee says: *"The scene segmenter detected a boundary here — trust it and save the frame regardless of similarity."*

---

### 5.5 Temporal Coverage Guarantee

Even after all the intelligent selection, the pipeline performs a final audit:

**Too few frames (< target_min):**
- Collects all educational frames not yet in the selection
- Sorts by complexity descending
- Iteratively adds the most complex frames that don't violate a half-minimum-gap spacing constraint
- Ensures the output reaches at least the target minimum count

**Too many frames (> target_max):**
- Sorts the selection by complexity score descending
- Trims to `target_max` entries (keeps the most complex ones)
- Re-sorts by timestamp to restore chronological order

**target_min / target_max scaling:**
```python
target_min = max(5, int(duration_seconds / 60 * 1.5))
target_max = max(8, int(duration_seconds / 60 * 4.0))
```

A 5-minute video targets 7–20 frames. A 30-minute video targets 45–120. This is calibrated to the typical density of conceptual transitions in STEM lectures.

---

### 5.6 Resilient Video Streaming

The streaming layer (`stream_frames`) is built to handle unreliable network conditions and YouTube CDN timeouts.

**Timeout detection (Windows):**
```python
result = [None]
def _read():
    result[0] = process.stdout.read(frame_size)

t = threading.Thread(target=_read, daemon=True)
t.start()
t.join(timeout=30)  # 30-second timeout

if t.is_alive():
    # read is still blocking → stream is frozen
    break  # trigger retry
```

**Timeout detection (Unix/macOS):**
```python
ready = select.select([process.stdout], [], [], 30)
if not ready[0]:
    # no data in 30s → stream frozen
    break
```

**Retry loop with seek-resume:**
```python
seek_args = ["-ss", str(int(frame_idx × frame_interval))] if frame_idx > 0 else []
# re-launches ffmpeg command with -ss to skip already-processed frames
```

**Exponential backoff:** waits `2^retry_count` seconds (2s, 4s, 8s) between retry attempts, up to 3 retries before giving up and logging an error.

---

### 5.7 SSIM-Based Extraction (Legacy)

The Structural Similarity Index (SSIM) measures perceptual similarity between two grayscale images, accounting for luminance, contrast, and structure. It produces values between 0 (completely different) and 1 (identical).

**Decision logic:**
```
If first frame         → always save
If gap > MAX_GAP_SEC   → force save (prevent large gaps in output)
If gap > MIN_GAP_SEC:
    ssim_score = ssim(last_saved_gray, current_gray)
    if ssim_score < SSIM_THRESHOLD (0.50):
        save this frame
```

**Strengths over simple pixel diff:**
- SSIM is not fooled by slight brightness shifts or video compression artifacts
- Captures structural changes (a new equation appearing) even if the overall brightness is similar

**Limitations vs pedagogical extractor:**
- Has no concept of *what* changed — a camera pan and a new equation score the same
- Doesn't classify content type
- May over-extract on high-motion content and under-extract on gradual text build-up

---

### 5.8 Output & Metadata Generation

#### `keyframes.json` (per video)

An array of `ExtractedKeyframe` objects:
```json
[
  {
    "timestamp": "01:23",
    "frame_path": "data/VIDEO_ID/keyframes/frame_0001_01_23.jpg",
    "content_type": "equation",
    "timestamp_seconds": 83.0,
    "complexity_score": 0.3412,
    "clip_confidence": 0.3412
  },
  ...
]
```

#### `video_meta.json` (per video)

```json
{
  "video_id": "VIDEO_ID",
  "title": "Full YouTube Video Title",
  "duration_seconds": 623,
  "duration_formatted": "10:23",
  "total_keyframes": 18,
  "subject": "Mathematics",
  "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "date_processed": "2026-03-29T14:32:11.123456",
  "content_type_counts": {
    "equation": 8,
    "diagram": 4,
    "graph": 3,
    "text": 3
  }
}
```

**Subject mapping:** The dominant content type is mapped to a subject:
| Content Type | Subject |
|---|---|
| equation, graph, geometry | Mathematics |
| biology | Biology |
| chemistry | Chemistry |
| circuit | Physics |
| diagram | Science |
| code | Computer Science |
| table, flowchart, text, slide, notes | General |

#### `data/batch_report.json`

Array of per-video run records:
```json
[
  {
    "index": 1,
    "url": "https://www.youtube.com/watch?v=...",
    "video_id": "VIDEO_ID",
    "status": "success",
    "keyframes": 18,
    "time_sec": 142.3
  },
  {
    "index": 2,
    "url": "...",
    "video_id": "unknown",
    "status": "failed",
    "keyframes": 0,
    "time_sec": 3.1,
    "error": "HTTPError: 403 Forbidden"
  }
]
```

---

## 6. Data Structures

```python
@dataclass
class FrameData:
    timestamp: float           # Seconds from video start
    frame: np.ndarray          # BGR image (H × W × 3)
    embedding: Optional[np.ndarray]  # 512-d CLIP vector, L2-normalized

@dataclass
class ScoredFrame:
    timestamp: float
    frame: np.ndarray
    embedding: np.ndarray
    complexity_score: float    # 0.0 – 1.0
    is_educational: bool       # CLIP classification result
    content_label: str         # e.g. "equation", "non-educational"

@dataclass
class Scene:
    start_time: float          # Start timestamp (seconds)
    end_time: float            # End timestamp (seconds)
    frames: List[ScoredFrame]  # All frames in this scene
    best_frame: Optional[ScoredFrame]  # Highest-complexity frame

@dataclass
class ExtractedKeyframe:
    timestamp: str             # "MM:SS" formatted
    frame_path: str            # Absolute path to saved JPEG
    content_type: str          # CLIP category label
    timestamp_seconds: float   # Raw seconds
    complexity_score: float    # 0.0 – 1.0
    clip_confidence: float     # is_educational × complexity
```

---

## 7. Output Format

### Directory Layout (after processing)

```
data/
├── batch_report.json
├── VIDEO_ID_1/
│   ├── video_meta.json
│   └── keyframes/
│       ├── keyframes.json
│       ├── frame_0001_00_04.jpg    ← frame #1 at 0:04
│       ├── frame_0002_01_20.jpg
│       └── ...
└── VIDEO_ID_2/
    └── ...
```

### Filename Convention

```
frame_{index:04d}_{mm:02d}_{ss:02d}.jpg
```
- `{index}` — sequential 1-based keyframe number (zero-padded to 4 digits)
- `{mm}` — minute (zero-padded to 2 digits)
- `{ss}` — second (zero-padded to 2 digits)

Example: `frame_0007_03_45.jpg` = keyframe #7, extracted at timestamp 3 minutes 45 seconds.

---

## 8. Configuration & Tunable Parameters

### `pedagogical_extractor_v2.py`

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `sample_fps` | `extract_pedagogical_keyframes()` arg | `0.5` | Frame sampling rate (frames/sec). Lower = faster but may miss fast transitions |
| `min_frame_gap` | arg | `4.0` s | Minimum time between saved keyframes |
| `max_frame_gap` | arg | `30.0` s | Force-save if no keyframe for this long |
| `similarity_threshold` | arg | `0.92` | CLIP cosine similarity above which a frame is considered redundant |
| `BATCH_SIZE` | hardcoded | `8` | Number of frames per CLIP inference call |
| `timeout_seconds` | `stream_frames()` arg | `30` | Seconds before stream is considered frozen |
| `max_retries` | arg | `3` | Maximum stream retry attempts |

### `stream.py`

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `SCALE_WIDTH` | module constant | `640` | Width to resize frames to before SSIM comparison |
| `SSIM_THRESHOLD` | module constant | `0.50` | SSIM below this → new keyframe |
| `MIN_GAP_SEC` | module constant | `1.5` | Minimum gap between keyframes |
| `MAX_GAP_SEC` | module constant | `15.0` | Force-save gap |
| `JPEG_QUALITY` | module constant | `88` | JPEG compression quality (0–100) |

---

## 9. Extractor Modes Comparison

| Mode | Algorithm | ML Required | Speed | Best For |
|---|---|---|---|---|
| `pedagogical` (default) | 7-phase CLIP pipeline | Yes (CLIP, PyTorch) | Moderate | Any STEM lecture |
| `semantic` | Adaptive clustering | Yes (CLIP) | Moderate | Dense, varied content |
| `scene` | PySceneDetect | No | Fast | High-motion/cut content |
| `ssim` | Structural Similarity | No | Fast | Simple slides, low motion |

---

## 10. Usage Examples

```bash
# Process all URLs with default pedagogical extractor
python main.py

# Process URLs 0–4 (first 5) with pedagogical extractor
python main.py --start 0 --end 5

# Use legacy SSIM extractor (fast, no ML)
python main.py --extractor ssim

# Use scene detection
python main.py --extractor scene

# Use a custom URL file
python main.py --urls my_urls.txt

# Run pedagogical extractor directly on a single video
python pedagogical_extractor_v2.py VIDEO_ID

# Run pedagogical extractor with custom output dir
python pedagogical_extractor_v2.py VIDEO_ID ./output/frames

# Run legacy SSIM extractor directly (interactive)
python stream.py
```

---

## 11. Dependencies

| Library | Used For |
|---|---|
| `yt-dlp` | Resolving YouTube stream URLs and metadata |
| `ffmpeg` / `ffprobe` | Video decoding, frame extraction, stream probing |
| `opencv-python` (cv2) | Image processing, Canny/Sobel/Laplacian, JPEG saving |
| `numpy` | Array operations, frame buffers |
| `torch` (PyTorch) | CLIP model inference |
| `clip` (OpenAI CLIP) | Visual content classification |
| `Pillow` (PIL) | Image format conversion for CLIP preprocessing |
| `scikit-image` | SSIM computation (stream.py legacy) |

**FFmpeg** must be available in PATH. `main.py` automatically appends `./ffmpeg/bin` so a local bundled binary is found without requiring a system installation.

---

*Documentation generated: March 29, 2026*
