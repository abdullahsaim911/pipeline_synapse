# Synapse Pipeline - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Design Choices](#design-choices)
5. [Algorithms](#algorithms)
6. [System Architecture](#system-architecture)
7. [Module Integration & Contracts](#module-integration)
8. [Step-by-Step Implementation Process](#step-by-step-implementation-process)
9. [Module Descriptions](#module-descriptions)
   - [M0: Frame Extraction](#m0-frame-extraction-module)
   - [M0b: Transcription](#m0b-transcription-module)
   - [M1: Synchronizer](#m1-synchronizer-module)
   - [M7: Visual Interaction Detection & Fusion](#m7-visual-interaction-detection--fusion-synapse_interaction)
   - [M5a: VLM Interface](#m5a-vlm-interface-module)
   - [M2: VLM Engine](#m2-vlm-engine-module)
   - [M5b: LLM Interface & Provider Abstraction](#m5b-llm-interface--provider-abstraction-module)
   - [M3: Synthesizer](#m3-synthesizer-module)
   - [M6: TTS Engine](#m6-tts-engine-module)
   - [M4: Pipeline Orchestrator](#m4-orchestrator-module)
   - [M8: Transcript-Grounded Q&A Subsystem](#m8-transcript-grounded-qa-subsystem-qa)
   - [M9: Screened Formative Quiz Subsystem](#m9-screened-formative-quiz-subsystem-quiz)
   - [M10: FastAPI Backend & Security](#m10-fastapi-backend--security-api)
10. [Data Flow, Database & File Persistence](#data-flow)
    - [Complete Pipeline Flow](#complete-pipeline-flow)
    - [Database Schema & SQLAlchemy Models](#database-schema--sqlalchemy-models)
    - [File System Hierarchy & On-Disk Saves](#file-system-hierarchy--on-disk-saves)
    - [Data Save Lifecycle & Transaction Boundaries](#data-save-lifecycle--transaction-boundaries)
11. [Installation](#installation)
12. [Usage Guide](#usage-guide)
13. [API Reference](#api-reference)
    - [FastAPI REST API Reference](#fastapi-rest-api-reference)
    - [PipelineOrchestrator Python API](#pipelineorchestrator)
14. [Performance](#performance)
15. [Troubleshooting](#troubleshooting)
16. [Future Enhancements](#future-enhancements)

---

## Overview

**Synapse** is an AI-powered accessibility pipeline designed specifically for blind students in STEM education. It processes educational lecture videos from YouTube and automatically generates audio descriptions at critical moments when visual content is not accessible through speech alone.

### What Makes Synapse Different

- **Intelligent Suffer Point Detection**: Identifies exactly when blind students lose access to visual information
- **Context-Aware Descriptions**: Explanations that weave naturally with the teacher's spoken words
- **3-Mode Explanation System**: Brief, explanatory, or detailed depth options
- **Multiple Content Types**: Handles equations, graphs, circuits, diagrams, code, and more
- **Provider Fallback**: Reliable audio generation with Edge-TTS + SpeechT5

### Target Users

- **Blind Students**: Primary beneficiaries who need audio descriptions
- **Educators**: Teachers creating accessible course materials
- **Content Creators**: YouTubers wanting to make content more accessible
- **Researchers**: Studying accessibility in STEM education

---

## Problem Statement

### The Challenge

Blind students face significant barriers in STEM education due to the heavy reliance on visual content in lectures:

1. **Deictic References**: Teachers say "look at this graph" without describing it
2. **Silent Drawing**: Teachers draw on the board without speaking
3. **Complex Visuals**: Detailed diagrams, equations, and charts are never described
4. **Rapid Progression**: Visual content changes faster than explanations can keep up

### Impact

- **Loss of Information**: Critical visual content remains inaccessible
- **Cognitive Load**: Students must constantly ask for descriptions
- **Social Exclusion**: Can't participate in visual discussions
- **Educational Disadvantage**: Fall behind sighted peers

### Existing Solutions Fall Short

- **Human Describers**: Expensive, not scalable, not always available
- **Auto-generated Captions**: Don't describe visual content
- **Screen Readers**: Can't interpret images, graphs, or diagrams
- **Manual Audio Description**: Time-consuming, requires specialized training

---

## Solution Architecture

### High-Level Pipeline

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
        VLM Engine (M2)
        Analyze ONLY intervention points
                ↓
        Synthesizer (M3)
        Generate woven explanations
                ↓
        TTS Engine (M6)
        Generate MP3 audio files
                ↓
        Orchestrator (M4)
        Create manifest.json
                ↓
        Final Output
        - manifest.json
        - audio_segments/*.mp3
        - interventions/ (optional)
```

### Module Map

| Module | ID | Purpose | Input | Output |
|--------|-----|---------|-------|--------|
| Frame Extraction | M0 | Extract pedagogical keyframes | YouTube URL | keyframes/*.jpg, keyframes.json |
| Transcription | M0b | Convert speech to text | YouTube URL | transcript.json |
| Synchronizer | M1 | Detect suffer points | transcript.json, keyframes.json | intervention_points |
| Interaction Detection | M7 | Detect visual writing, pointing & gestures | Video stream, transcript | Fused intervention_points |
| VLM Interface | M5a | VLM wrapper | Image, prompt | VLMResponse |
| VLM Engine | M2 | Analyze visual content | Frame, content_type | vlm_snapshot.json |
| LLM Interface & Providers | M5b | LLM wrapper & provider abstraction (Ollama, Gemini) | Prompt, GenerationSettings | GenerationResult / OllamaResponse |
| Synthesizer | M3 | Fuse transcript + VLM | Transcript, VLM data | audio_script |
| TTS Engine | M6 | Generate audio | audio_script | MP3 file |
| Orchestrator | M4 | Coordinate pipeline lifecycle | YouTube URL | manifest.json |
| Q&A Subsystem | M8 | Transcript-grounded student Q&A with citations | video_id, question, moment | AskResponse with citations |
| Quiz Subsystem | M9 | Screened formative practice quiz generation | video_id, question_count | QuizResponse with screened MCQs |
| FastAPI Backend | M10 | REST API gateway, SQLite persistence & CORS | HTTP requests | JSON responses, SQLite records, disk files |

---

## Design Choices

### Why These Design Decisions?

The Synapse pipeline's architecture is the result of deliberate choices to optimize for accessibility, performance, reliability, and user experience. Below are the key design decisions and their rationales.

#### 1. Intelligent Suffer Point Detection vs. Full Video Analysis

**Decision**: Analyze only intervention points (suffer points) rather than every frame in the video.

**Rationale**:
- **Performance**: Processing only ~25-50 intervention points from a 10-minute video saves 900-1900 seconds (90-95% reduction in VLM processing time)
- **Relevance**: Blind students only need descriptions when visual content is inaccessible, which occurs at suffer points
- **Cost**: Significantly reduces computational resources and energy consumption
- **Accuracy**: Focuses VLM attention on frames where descriptions are actually needed

**Implementation**: The Synchronizer module (M1) uses a multi-factor scoring system to identify suffer points based on:
- Deictic phrases (teacher pointing at visual content)
- Silent drawing (teacher drawing without speaking)
- High complexity (complex visual content)

#### 2. Multi-Factor Scoring for Suffer Point Detection

**Decision**: Use a weighted scoring system combining multiple signals rather than a single criterion.


**Rationale**:
- **Robustness**: Different teachers have different styles; a single rule would miss many cases
- **Flexibility**: Scoring allows tuning based on specific video types or subjects
- **Confidence Levels**: Enables filtering by confidence (high/medium/low)
- **Reduced False Positives**: Multiple triggers increase confidence

**Scoring Formula**:
```
score = base_score
if deictic_phrase: score += 50
if negative_deictic_phrase: score -= 50
if silent_drawing: score += 80
if high_complexity: score += 50

intervention_if: score >= 50 OR (score < 50 AND complexity > 0.25)
```

#### 3. Chain-of-Regions Scanning for VLM Analysis

**Decision**: Instruct VLM to scan the entire frame using a 2x2 grid (top-left, top-right, bottom-left, bottom-right).

**Rationale**:
- **Complete Coverage**: Prevents missing small but important elements like labels, arrows, or annotations
- **Mental Model Building**: Helps blind students understand spatial relationships
- **Navigation Support**: Reading order is critical for accessibility
- **Accurate OCR**: Text in corners is often missed by center-focused models

**Implementation**: The VLM Engine (M2) includes this in its system prompt:
```
Scan ENTIRE frame (not just center) using Chain-of-Regions:
Partition screen into distinct zones (2x2 grid: top-left, top-right, bottom-left, bottom-right)
to ensure small details like labels and arrows are not missed.
```

#### 4. 3-Mode Explanation System

**Decision**: Support brief, explanatory, and detailed output modes instead of a single explanation depth.

**Rationale**:
- **Context-Dependent Needs**: Different situations require different depth (quick review vs. deep study)
- **Performance Tradeoffs**: Brief mode is 2-3x faster than detailed mode
- **User Preference**: Some users prefer concise explanations, others want comprehensive coverage
- **Flexible Teaching**: Mimics how teachers adjust explanations based on student questions

**Token Ranges**:
- Brief: 150-250 tokens (~30-60s audio)
- Explanatory: 300-500 tokens (~60-120s audio)
- Detailed: 500-900 tokens (~120-240s audio)

#### 5. Dual TTS Provider Architecture

**Decision**: Use Edge-TTS as primary with SpeechT5 as fallback instead of a single provider.

**Rationale**:
- **Reliability**: Edge-TTS requires internet; SpeechT5 provides offline backup
- **Quality**: Edge-TTS offers high-quality voices; SpeechT5 provides consistent quality
- **Cost**: Both are free (no API costs)
- **Accessibility**: Multiple voice options for different preferences

**Implementation**: The TTS Engine (M6) tries Edge-TTS first, then falls back to SpeechT5 automatically.

#### 6. Hybrid Transcription Approach

**Decision**: Use YouTube Transcript API first (fast), then Whisper AI (slow but comprehensive) as fallback.

**Rationale**:
- **Speed**: YouTube API provides instant results for videos with captions
- **Coverage**: Whisper AI handles videos without captions or with poor auto-captions
- **Accuracy**: Whisper is more accurate than YouTube's auto-captions for STEM content
- **Flexibility**: Multiple Whisper models allow quality/accuracy tradeoffs

#### 7. GGUF Backend for VLM with Partial GPU Offloading

**Decision**: Use GGUF quantized model with configurable GPU layer offloading instead of full-precision model.

**Rationale**:
- **Memory Efficiency**: Reduces VRAM from 12-14GB to 4-5GB (60-70% reduction)
- **Accessibility**: Makes system usable on consumer GPUs (6GB+ instead of 12GB+)
- **Speed**: Quantized model is 2-3x faster than full precision
- **CPU Offloading**: Allows operation on systems without GPU or with limited VRAM

**Configuration**:
```python
use_gguf=True
n_gpu_layers=24  # Balance between speed and memory
n_ctx=4096        # Context window
```

#### 8. Context-Aware Descriptions

**Decision**: Generate explanations that weave naturally with teacher's spoken words rather than standalone descriptions.

**Rationale**:
- **Natural Flow**: Students hear one seamless audio track instead of jarring interruptions
- **Educational Continuity**: Maintains the teacher's narrative arc
- **Reduced Cognitive Load**: Students don't have to mentally merge separate descriptions
- **Better Retention**: Connected explanations are more memorable

**Implementation**: The Synthesizer (M3) receives both transcript context and VLM data, then weaves them together.

#### 9. Hallucination Detection

**Decision**: Implement cross-domain hallucination detection and missing elements checks.

**Rationale**:
- **Accuracy**: VLMs sometimes describe content that isn't in the image (hallucinations)
- **Trust**: False information damages educational value and user trust
- **Fallback**: When hallucination is detected, fall back to transcript-only explanation
- **Quality Control**: Missing elements flag indicates when OCR or detection failed

**Detection Logic**:
```python
# Cross-domain mismatch keywords
CROSS_DOMAIN_MISMATCHES = {
    "equation": ["person", "man", "woman", "face"],
    "graph": ["person", "man", "woman", "face"],
    # ... more mappings
}

# Missing elements flag
if vlm_snapshot.get("missing_elements") not in [None, "null", ""]:
    return transcript_only_fallback()
```

#### 10. Parallel Extraction Architecture

**Decision**: Execute audio transcription and video frame extraction simultaneously using threading.

**Rationale**:
- **Performance**: Saves 10-20 seconds on a 10-minute video (30-60s total instead of 40-80s)
- **Independence**: Two operations don't depend on each other
- **Resource Utilization**: Uses CPU/GPU more efficiently
- **No Downsides**: No data dependencies between the two operations

**Implementation**:
```python
audio_thread = threading.Thread(target=extract_audio)
video_thread = threading.Thread(target=extract_video)
audio_thread.start()
video_thread.start()
audio_thread.join()
video_thread.join()
```

#### 11. Temporal Redundancy Prevention

**Decision**: Enforce a 5-second minimum gap between interventions using redundancy window.

**Rationale**:
- **Avoid Overload**: Students can't process rapid-fire descriptions
- **Prevent Redundancy**: Consecutive frames often show similar content
- **Natural Pacing**: Mimics how teachers pause between points
- **Efficiency**: Reduces unnecessary VLM processing

**Implementation**:
```python
if abs(current_time - last_intervention_time) < REDUNDANCY_WINDOW:
    continue  # Skip redundant intervention
```

#### 12. Math Linearization for Speech

**Decision**: Convert mathematical notation (², √, ∫, etc.) to spoken form for TTS.

**Rationale**:
- **Accessibility**: Math notation is unintelligible when read literally
- **Natural Speech**: TTS voices handle text better than symbols
- **Educational Standards**: Matches how teachers read equations aloud
- **Universal Understanding**: Standardized spoken forms across content

**Example Mappings**:
```python
"²" → "squared"
"√" → "square root of"
"∫" → "integral of"
"π" → "pi"
```

#### 13. Unified VLM Output Format

**Decision**: Use a consistent JSON structure across all content types with standardized fields.

**Rationale**:
- **Simplification**: Downstream modules (Synthesizer) don't need content-type-specific parsing
- **Extensibility**: Easy to add new content types without breaking existing code
- **Comprehensiveness**: Captures all relevant information (spatial, structural, conceptual)
- **Reliability**: Parsing is robust to missing or malformed data

**Unified Fields**:
```json
{
  "content_type": "graph",
  "detected_types": ["graph", "text"],
  "visual_analysis": {...},
  "structural_description": "...",
  "reading_order": [...],
  "conceptual_hints": "..."
}
```

#### 14. Intervention-Only Workflow

**Decision**: Support a workflow that generates only interventions (not full track) as a separate mode.

**Rationale**:
- **Curator Review**: Educators can review and approve interventions before publishing
- **Editing**: Human editors can enhance or correct descriptions
- **Flexibility**: Different use cases (review vs. final production)
- **Efficiency**: Skips transcript segments that don't need intervention

---

## Algorithms

This section documents the key algorithms used throughout the Synapse pipeline, explaining their logic, complexity, and role in the system.

### 1. Suffer Point Detection Algorithm

**Module**: Synchronizer (M1)

**Purpose**: Identify moments when blind students lose visual access to content.

**Input**:
- `transcript.json`: List of transcript segments with timestamps
- `keyframes.json`: List of keyframes with metadata (content_type, complexity_score, clip_confidence)

**Output**:
- `List[InterventionPoint]`: Detected suffer points with confidence levels

**Algorithm**:
```
1. Load transcript segments and keyframes
2. Initialize empty interventions list
3. Set last_intervention_time = None

4. For each keyframe in keyframes:
   a. Find matching transcript segment within 30-second window
   b. Calculate intervention score:
      score = 0
      if deictic phrase in transcript: score += 50
      if negative deictic phrase in transcript: score -= 50
      if transcript length < 10 chars AND complexity > 0.3: score += 80
      if complexity > 0.30: score += 50

   c. Determine if intervention needed:
      meets_threshold = score >= 50
      visual_only = complexity > 0.25 AND NOT meets_threshold

      If meets_threshold OR visual_only:
         i. Check redundancy: if within 5s of last intervention, skip
         ii. Calculate confidence:
              if score >= 100: confidence = "high"
              elif meets_threshold: confidence = "medium"
              else: confidence = "low"
         iii. Create InterventionPoint
         iv. Append to interventions list
         v. Update last_intervention_time

5. Return interventions list
```

**Time Complexity**: O(N) where N = number of keyframes

**Space Complexity**: O(K) where K = number of interventions (output size)

**Key Parameters**:
- `MATCHING_WINDOW = 30.0`: Maximum gap for transcript matching
- `REDUNDANCY_WINDOW = 5.0`: Minimum time between interventions
- `DEICTIC_SCORE = 50`: Score for deictic phrases
- `SILENT_DRAWING_SCORE = 80`: Score for silent drawing
- `HIGH_COMPLEXITY_SCORE = 50`: Score for high complexity
- `NEGATIVE_DEICTIC_PENALTY = -50`: Penalty for negative deictic
- `SUFFER_POINT_THRESHOLD = 50`: Minimum score for intervention

---

### 2. Pedagogical Keyframe Extraction Algorithm

**Module**: Frame Extraction (M0)

**Purpose**: Extract pedagogically important keyframes from video lectures.

**Input**:
- YouTube video URL
- Output directory path
- Sampling FPS (default: 0.5 fps)

**Output**:
- `List[ExtractedKeyframe]`: Selected keyframes with metadata

**Algorithm**:
```
1. Get video stream URL and metadata (duration, resolution)
2. Calculate target frame count:
   target_min = max(5, duration/60 * 1.5)  // ~1.5 frames/minute
   target_max = max(8, duration/60 * 4)    // ~4 frames/minute

3. Stream frames at sample_fps
4. For each frame:
   a. Skip if mostly blank (low pixel variance)
   b. Compute CLIP embedding
   c. Store with metadata

5. Batch compute CLIP embeddings (BATCH_SIZE = 8)

6. For each frame:
   a. Compute complexity score:
      i.   Edge density (Canny with adaptive thresholds)
      ii.  Gradient magnitude (Sobel)
      iii. Color variance (HSV std deviation)
      iv.  Local contrast (Laplacian variance)
      v.   Text-like regions (Otsu threshold transitions)
      vi.  Weighted combination: 0.30*edge + 0.25*text + 0.20*contrast + 0.15*gradient + 0.10*color

   b. Classify as educational using CLIP:
      i.   Compare frame embedding to 10 educational category prompts
      ii.  Compare frame embedding to non-educational prompts
      iii. Determine best category

7. Segment frames into scenes using CLIP similarity:
   a. Compute consecutive frame similarities
   b. Calculate adaptive threshold: mean - 1.5*std
   c. Detect scene boundaries at similarity drops below threshold
   d. Enforce minimum 4-second gap between scenes

8. For each scene:
   a. Find educational frames with good complexity (>0.25)
   b. Select frame with highest complexity score
   c. Apply temporal spacing (minimum 4s from last saved)
   d. Check semantic redundancy:
      Compare CLIP embedding to last saved frame
      If similarity > 0.92, mark as redundant
   e. Save frame (override redundancy if scene guarantee applies)

9. Ensure temporal coverage:
   If saved frames < target_min, add frames to meet minimum
   If saved frames > target_max, trim excess

10. Save frames to disk and generate metadata JSON
```

**Time Complexity**: O(F) where F = number of sampled frames (depends on duration and fps)

**Space Complexity**: O(F) for storing embeddings during processing

**Key Parameters**:
- `sample_fps = 0.5`: Frames per second to sample
- `min_frame_gap = 4.0`: Minimum seconds between selected frames
- `similarity_threshold = 0.92`: CLIP similarity threshold for deduplication
- `BATCH_SIZE = 8`: CLIP embedding batch size

---

### 3. VLM Frame Analysis Algorithm

**Module**: VLM Engine (M2)

**Purpose**: Analyze STEM lecture frames with category-specific prompts.

**Input**:
- Frame image path (JPG)
- Primary content type (equation, graph, circuit, etc.)

**Output**:
- `vlm_snapshot`: Structured JSON with visual analysis

**Algorithm**:
```
1. Load frame image (PIL Image or numpy array)
2. Preprocess image:
   a. Resize to target resolution (256x256 to 896x896)
   b. Normalize pixel values
   c. Convert to model format

3. Build prompt:
   a. Start with system prompt (instructions for blind accessibility)
   b. Add category-specific directives (if available)
   c. Specify primary content type
   d. Require Chain-of-Regions scanning (2x2 grid)

4. Generate via VLM:
   a. Pass image and prompt to VLM
   b. Set generation parameters:
      max_new_tokens = 384
      temperature = 0.1
   c. Receive text response

5. Parse JSON response:
   a. Strip markdown blocks (```json...```)
   b. Remove trailing commas
   c. Parse JSON object
   d. If parse fails, return fallback structure

6. Ensure unified format:
   a. Set defaults for missing fields:
      - content_type: first detected type or "unknown"
      - detected_types: list containing content_type
      - visual_analysis: empty object
      - structural_description: layout or ""
      - reading_order: empty array
      - conceptual_hints: ""
   b. Set backward compatibility fields:
      - layout: structural_description or ""
      - text_readout: ""
      - spatial_map: ""
      - colors_styles: ""
      - missing_elements: null

7. Check for [ILLEGIBLE] text:
   a. Count [ILLEGIBLE] markers in text_readout
   b. Calculate illegible ratio: illegible_count / text_length
   c. If ratio > 0.2 for equation/handwritten:
      Set missing_elements = "OCR fallback required"

8. Return structured snapshot
```

**Time Complexity**: O(1) per frame (model inference)

**Space Complexity**: O(M) where M = model size in memory

**Key Parameters**:
- `max_new_tokens = 384`: Maximum tokens for VLM output
- `temperature = 0.1`: Low temperature for deterministic output
- `use_gguf = True`: Use quantized GGUF model
- `n_gpu_layers = 24`: Number of layers on GPU

---

### 4. Text Synthesis Algorithm

**Module**: Synthesizer (M3)

**Purpose**: Fuse transcript with VLM data into accessible audio scripts.

**Input**:
- Transcript context (teacher's spoken words)
- VLM snapshot (visual analysis)
- Content type
- Output mode (brief/explanatory/detailed)
- Full transcript list (for context gathering)

**Output**:
- `audio_script`: Text ready for TTS

**Algorithm**:
```
1. Safety Check 1: Missing Elements
   If vlm_snapshot.missing_elements is not null/empty:
      Return _apply_output_mode(transcript_context, mode)

2. Safety Check 2: Cross-Domain Hallucination
   a. Get content_type mismatch keywords
   b. Check text_readout for mismatch keywords
   c. If any mismatch keyword found:
      Return _apply_output_mode(transcript_context, mode)

3. Find current transcript index:
   Search transcripts for entry matching transcript_context

4. Gather context from ContextManager:
   Based on output_mode:
   - brief: 1 previous, 0 following segments
   - explanatory: 2 previous, 1 following
   - detailed: 3 previous, 2 following

5. Linearize VLM data:
   a. Convert visual_analysis to readable format
   b. Extract key fields based on content_type
   c. Format as structured text for LLM

6. Build mode-specific prompt:
   If mode == "brief":
      Use _build_brief_prompt()
   Else if mode == "detailed":
      Use _build_detailed_prompt()
   Else: // explanatory
      Use _build_explanatory_prompt()

   Prompt includes:
   - Detected content types
   - Visual analysis (linearized)
   - Structural description
   - Conceptual hints
   - Teacher's transcript
   - Previous concepts (from context)
   - Depth guidelines

7. Call LLM (Ollama):
   a. POST to /api/generate endpoint
   b. Include prompt in request
   c. Set generation parameters:
      num_predict = mode_specific_limit
      temperature = mode_specific_temp
   d. Receive generated text

8. Math Linearization:
   Replace mathematical symbols with spoken forms:
   "²" → "squared"
   "√" → "square root of"
   "∫" → "integral of"
   "π" → "pi"
   ... (see symbol mappings)

9. Guard: If LLM returned empty or very short:
   Return transcript_context only

10. Return generated audio_script
```

**Time Complexity**: O(1) per synthesis (LLM inference)

**Space Complexity**: O(C + L) where C = context size, L = LLM response

**Key Parameters**:
- `num_predict`: 200 (brief), 400 (explanatory), 800 (detailed)
- `temperature`: 0.6 (brief), 0.7 (explanatory), 0.8 (detailed)
- `top_k = 40`: Top-k sampling
- `top_p = 0.9`: Nucleus sampling

---

### 5. Audio Generation Algorithm

**Module**: TTS Engine (M6)

**Purpose**: Convert text scripts to high-fidelity audio files.

**Input**:
- Text script
- Output audio file path
- Voice preference (optional)

**Output**:
- MP3 audio file

**Algorithm**:
```
1. Try Edge-TTS first:
   a. Extract voice from request or use default
   b. Call Microsoft Edge-TTS API
   c. Stream audio response
   d. Save to MP3 file
   e. If successful, return file path
   f. If failed, log error and continue to step 2

2. Fallback to SpeechT5:
   a. Load SpeechT5 model and HiFi-GAN vocoder
   b. Tokenize input text
   c. Generate mel spectrogram
   d. Convert to audio waveform
   e. Save to MP3 file
   f. If successful, return file path
   g. If failed, raise error

3. Post-processing:
   a. Verify file exists and is not empty
   b. Check file size (> 1KB)
   c. If validation fails, attempt regeneration

4. Return audio file path
```

**Time Complexity**: O(L) where L = length of text (linear)

**Space Complexity**: O(M) where M = model size in memory

**Key Parameters**:
- Edge-TTS voice: "en-US-AriaNeural" (default)
- SpeechT5 sample rate: 16000 Hz
- SpeechT5 batch size: 1 (single audio generation)

---

### 6. Visual Complexity Calculation Algorithm

**Module**: Frame Extraction (M0)

**Purpose**: Compute a robust complexity score that works across different background types.

**Input**:
- Frame image (BGR numpy array)

**Output**:
- `complexity_score`: Float between 0.0 and 1.0

**Algorithm**:
```
1. Convert frame to grayscale

2. Calculate edge density:
   a. Compute median pixel value of grayscale image
   b. Set Canny thresholds:
      lower = max(0, 0.7 * median)
      upper = min(255, 1.3 * median)
   c. Apply Canny edge detection
   d. Count edge pixels / total pixels
   e. Store as edge_density

3. Calculate gradient magnitude:
   a. Apply Sobel operator in x direction
   b. Apply Sobel operator in y direction
   c. Compute sqrt(sobel_x² + sobel_y²)
   d. Normalize by 255.0
   e. Store as gradient_score

4. Calculate color variance:
   a. Convert frame to HSV color space
   b. Compute standard deviation of hue channel
   c. Normalize by 180.0 (max hue value)
   d. Store as color_variance

5. Calculate local contrast:
   a. Apply Laplacian operator
   b. Compute variance of Laplacian result
   c. Normalize by 10000.0
   d. Store as contrast_score

6. Calculate text-like regions:
   a. Apply Otsu thresholding
   b. Count pixel transitions (0→1 and 1→0)
   c. Normalize and cap at 1.0
   d. Store as text_score

7. Weighted combination:
   complexity = 0.30 * edge_density
              + 0.25 * text_score
              + 0.20 * contrast_score
              + 0.15 * gradient_score
              + 0.10 * color_variance

8. Return min(complexity, 1.0)
```

**Time Complexity**: O(W*H) where W = width, H = height

**Space Complexity**: O(W*H) for storing intermediate images

**Key Parameters**:
- Weights: edge (0.30), text (0.25), contrast (0.20), gradient (0.15), color (0.10)

---

### 7. CLIP-Based Scene Segmentation Algorithm

**Module**: Frame Extraction (M0)

**Purpose**: Group frames into conceptual scenes using semantic similarity.

**Input**:
- List of FrameData objects (frames with CLIP embeddings)
- Minimum scene gap (default: 4.0 seconds)

**Output**:
- `List[Scene]`: Scenes with start/end times and representative frame

**Algorithm**:
```
1. Initialize empty scenes list

2. Compute consecutive frame similarities:
   For i from 0 to len(frames) - 2:
      similarity = dot_product(frames[i].embedding, frames[i+1].embedding)
      Store similarity

3. Calculate adaptive threshold:
   a. Get all similarity values
   b. Compute mean_sim = mean(similarities)
   c. Compute std_sim = std(similarities)
   d. threshold = max(0.70, mean_sim - 1.5 * std_sim)

4. Detect scene boundaries:
   For each similarity:
      If similarity < threshold:
         Mark as potential scene boundary

5. Enforce minimum scene gap:
   For each pair of consecutive boundaries:
      If gap < min_scene_gap:
         Remove weaker boundary (lower similarity)

6. Create scenes:
   a. Start first scene at first frame
   b. At each boundary:
      i.  End current scene
      ii. Select best frame (highest complexity)
      iii. Add to scenes list
      iv. Start new scene at next frame
   c. End last scene at final frame

7. For each scene:
   Select representative frame:
   a. Find educational frames in scene
   b. Select frame with highest complexity score
   c. If no educational frames, select frame with highest overall score

8. Return scenes list
```

**Time Complexity**: O(N) where N = number of frames

**Space Complexity**: O(N) for storing scenes

**Key Parameters**:
- `min_scene_gap = 4.0`: Minimum seconds between scenes
- `threshold = max(0.70, mean - 1.5*std)`: Adaptive boundary threshold

---

## System Architecture

The Synapse system follows a modular, pipeline-based architecture designed for modularity, maintainability, and performance. Each module has a clear responsibility and well-defined interfaces.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL CLIENTS / FRONTEND                            │
│                     HTTP Requests / REST API / Electron                         │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ HTTP / JSON
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND GATEWAY (M10: api/)                         │
│   • CORS Security & Origin Validation (api/security.py)                          │
│   • Structured Logging & Sanitization (api/logging_setup.py)                     │
│   • SQLite Database Engine & Session Management (api/database.py)               │
│   • Routers: Video Pipeline (video.py), Quiz (quiz.py), Health                  │
└──────┬───────────────────────┬───────────────────────────┬──────────────────────┘
       │                       │                           │
       ▼                       ▼                           ▼
┌──────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ PIPELINE     │     │ Q&A SUBSYSTEM (M8)  │     │ QUIZ SUBSYSTEM (M9) │
│ ORCHESTRATOR │     │ qa/                 │     │ quiz/               │
│ (M4)         │     │ • Temporal / BM25   │     │ • Passage Builder   │
│              │     │ • Moment Context    │     │ • Screened MCQs/TF  │
│              │     │ • Faithfulness Gate │     │ • Duplicate Filter  │
└──────┬───────┘     └─────────┬───────────┘     └─────────┬───────────┘
       │                       │                           │
       ▼                       └─────────────┬─────────────┘
┌───────────────────────────────────┐        │
│ CORE EXTRACTION & DETECTION       │        │
│ • M0: Frame Extractor (OpenCV)    │        ▼
│ • M0b: Transcriber (Whisper/API)  │ ┌──────────────────────────────────────┐
│ • M7: Interaction Detector (Ink/  │ │ UNIFIED LLM PROVIDERS (M5b)          │
│       Pointing / Before-After)    │ │ • Gemini Provider (google-genai)     │
│ • M1: Suffer Point Synchronizer   │ │ • Ollama Client (Local Mistral)      │
│   (Fusion of M1 + M7)             │ │ • Structured Output Schemas          │
└──────┬────────────────────────────┘ └──────────────────┬───────────────────┘
       │                                                 │
       ▼                                                 │
┌───────────────────────────────────┐                    │
│ ANALYSIS & SPEECH GENERATION      │                    │
│ • M2: VLM Engine (Qwen2-VL)       │◄───────────────────┘
│ • M3: LLM Synthesizer (Fusion)    │
│ • M6: Dual TTS Engine (Edge-TTS   │
│       + SpeechT5 Fallback)        │
└──────┬────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             PERSISTENCE LAYER                                   │
│  ┌───────────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │   SQLITE DATABASE (synapse.db)    │  │     FILE SYSTEM (data/<video_id>/) │  │
│  │   • videos                        │  │     • keyframes/*.jpg              │  │
│  │   • transcripts                   │  │     • audio_segments/*.mp3         │  │
│  │   • interventions                 │  │     • interventions/*/             │  │
│  │   • explanations                  │  │     • manifest.json                │  │
│  └───────────────────────────────────┘  └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibility Matrix

| Module | Primary Responsibility | Key Technologies | Data Input | Data Output |
|---------|---------------------|------------------|-------------|--------------|
| **M0: Frame Extraction** | Extract pedagogical keyframes | OpenCV, CLIP, PyTorch | YouTube URL | keyframes/*.jpg, keyframes.json |
| **M0b: Transcription** | Convert speech to text | Whisper, YouTube API | YouTube URL | transcript.json |
| **M1: Synchronizer** | Detect suffer points | Pure Python, Regex | transcript.json, keyframes.json | intervention_points |
| **M7: Interaction Detection** | Detect writing, pointing & gestures | OpenCV, MediaPipe, NumPy | Video stream, transcript | Fused intervention_points |
| **M5a: VLM Interface** | VLM wrapper interface | Transformers, Qwen2-VL | Image, prompt | VLMResponse |
| **M2: VLM Engine** | Analyze visual content | Qwen2-VL, llama-cpp-python | Frame, content_type | vlm_snapshot.json |
| **M5b: LLM Interface & Providers** | Unified LLM abstraction | Ollama, Gemini SDK, Requests | Prompt, GenerationSettings | GenerationResult / OllamaResponse |
| **M3: Synthesizer** | Fuse transcript + VLM | Ollama, Mistral, NumPy | Transcript, VLM data | audio_script |
| **M6: TTS Engine** | Generate audio files | edge-tts, SpeechT5, PyTorch | Text script | *.mp3 files |
| **M4: Orchestrator** | Coordinate pipeline lifecycle | Threading, PyTorch | YouTube URL | manifest.json |
| **M8: Q&A Subsystem** | Transcript-grounded student Q&A | BM25, Gemini/Ollama, Regex | video_id, question, moment | AskResponse with citations |
| **M9: Quiz Subsystem** | Screened formative practice quizzes | Pydantic, Gemini/Ollama | video_id, question_count | QuizResponse with screened MCQs |
| **M10: FastAPI Backend** | REST API, SQLite persistence & CORS | FastAPI, SQLAlchemy, Pydantic | HTTP requests | JSON responses, SQLite records, disk files |

### Architecture Principles

#### 1. Separation of Concerns

Each module has a single, well-defined responsibility:
- **Extraction**: M0 and M0b handle data extraction from videos
- **Detection**: M1 identifies when interventions are needed
- **Analysis**: M2 analyzes visual content
- **Fusion**: M3 combines transcript with visual analysis
- **Generation**: M6 creates audio files
- **Coordination**: M4 orchestrates the entire pipeline

#### 2. Interface Abstraction

Interface modules (M5a, M5b) provide abstraction layers:
- Hide implementation details from calling modules
- Enable easy swapping of underlying models (e.g., GGUF vs. Transformers)
- Provide consistent error handling and retry logic
- Simplify testing (mock interfaces possible)

#### 3. Pipeline Flow

Data flows unidirectionally through the pipeline:
```
Extraction → Detection → Analysis → Fusion → Generation → Manifest
```

Each stage consumes the output of the previous stage and produces input for the next.

#### 4. Parallel Execution

Independent operations run in parallel:
- Audio transcription (M0b) and video extraction (M0) run simultaneously
- Reduces total processing time by 10-20s for typical videos
- No data dependencies between the two operations

#### 5. Resource Management

GPU memory is carefully managed:
- Clear memory after Whisper (before VLM load)
- Clear memory after VLM (before TTS)
- 4-bit quantization to reduce memory footprint
- CPU offloading options for limited GPU systems

### System Layers

#### Presentation Layer (Frontend)

- **Purpose**: User interface and API endpoints
- **Components**: Web UI, REST API
- **Communication**: JSON-based with manifest.json

#### Application Layer (Orchestration)

- **Purpose**: Business logic and workflow coordination
- **Module**: Orchestrator (M4)
- **Responsibilities**:
  - Pipeline lifecycle management
  - Workflow selection (full/intervention-only/detection-only)
  - Error handling and recovery
  - Resource management

#### Domain Layer (Core Logic)

- **Purpose**: Core algorithms and domain-specific logic
- **Modules**: M0, M0b, M1, M2, M3, M6
- **Responsibilities**:
  - Frame extraction and analysis
  - Suffer point detection
  - VLM-based visual analysis
  - Text synthesis and fusion
  - Audio generation

#### Infrastructure Layer (Interfaces)

- **Purpose**: External service integration and model management
- **Modules**: M5a (VLM Interface), M5b (LLM Interface)
- **Responsibilities**:
  - Model loading and inference
  - API communication (Ollama, Edge-TTS)
  - Error handling and retries
  - Connection management

### Data Flow Patterns

#### 1. Sequential Pipeline

Primary flow for full video processing:
```
YouTube URL
  → [Parallel: Audio + Video Extraction]
  → [Merge: Transcript + Keyframes]
  → [Synchronizer: Detect Interventions]
  → [VLM Engine: Analyze Interventions]
  → [Synthesizer: Generate Scripts]
  → [TTS Engine: Generate Audio]
  → [Manifest: Create JSON]
```

#### 2. Intervention-Only Flow

Alternative flow for generating only interventions:
```
YouTube URL
  → [Parallel: Audio + Video Extraction]
  → [Synchronizer: Detect Interventions]
  → [VLM Engine: Analyze Interventions]
  → [Synthesizer: Generate Scripts]
  → [TTS Engine: Generate Audio]
  → [Interventions Manifest]
```

#### 3. Detection-Only Flow

Fast flow for analyzing videos without full processing:
```
YouTube URL
  → [Parallel: Audio + Video Extraction]
  → [Synchronizer: Detect Interventions]
  → [Return: Intervention Points + Metadata]
```

#### 4. Single Intervention Flow

API endpoint flow for on-demand generation:
```
Intervention Request (frame_path, content_type, context)
  → [VLM Engine: Analyze Frame]
  → [Synthesizer: Generate Script]
  → [TTS Engine: Generate Audio]
  → [Return: Text + Audio Path]
```

### Error Handling Architecture

#### Error Hierarchy

```
ApplicationError (Base)
  ├── ValidationError
  │   ├── InvalidURLException
  │   └── MissingDataError
  ├── ProcessingError
  │   ├── VLMError
  │   ├── LLMError
  │   └── TTSError
  ├── ResourceError
  │   ├── GPUMemoryError
  │   ├── ModelNotFoundError
  │   └── ServiceUnavailableError
  └── NetworkError
      ├── OllamaConnectionError
      ├── EdgeTTSUnavailableError
      └── YouTubeAPIError
```

#### Recovery Strategies

1. **Retry with Exponential Backoff**: Network requests (Ollama, Edge-TTS)
2. **Fallback Chains**:
   - YouTube API → Whisper (transcription)
   - Edge-TTS → SpeechT5 (TTS)
   - GGUF → Transformers (VLM)
3. **Graceful Degradation**:
   - VLM failure → Transcript-only explanation
   - TTS failure → Return text only
   - Partial completion → Save intermediate results

### Deployment Architecture

#### Local Deployment

```
┌───────────────────────────────────────────┐
│          Local Machine                   │
├───────────────────────────────────────────┤
│  Python Environment                    │
│  ├── Synapse Modules                  │
│  ├── Models (VLM, LLM, TTS)        │
│  ├── Ollama Server                   │
│  └── Data Storage (Video frames, audio)│
├───────────────────────────────────────────┤
│  GPU (Optional)                       │
│  ├── VLM Model (5-8GB)              │
│  ├── Whisper Model (1-2GB)            │
│  └── SpeechT5 Model (0.5-1GB)       │
└───────────────────────────────────────────┘
```

#### Resource Requirements

| Component | CPU | RAM | GPU (VRAM) | Disk | Network |
|-----------|------|-----|-------------|-------|---------|
| Frame Extraction | Medium | 2-4GB | 4-6GB | 10-20GB/video | Required |
| Transcription | Medium | 2-4GB | 0-2GB | 100-200MB/video | Required |
| Synchronization | Low | <100MB | 0GB | <1MB | None |
| VLM Analysis | Medium | 4-8GB | 5-8GB | 5-10GB/model | None |
| Synthesis | Low | 2-4GB | 0GB | <100MB | Local |
| TTS Generation | Low | 2-4GB | 0-4GB | 10-50MB/audio | Optional |

---

## Module Integration

This section explains how modules integrate with each other, defining data contracts, dependencies, and interaction patterns.

### Module Dependency Graph

```
Orchestrator (M4)
    ├── Frame Extraction (M0)
    ├── Transcription (M0b)
    ├── Synchronizer (M1)
    │   ├── [Depends on: M0 output]
    │   └── [Depends on: M0b output]
    ├── VLM Engine (M2)
    │   ├── [Depends on: M1 output]
    │   └── [Uses: VLM Interface (M5a)]
    ├── Synthesizer (M3)
    │   ├── [Depends on: M1 output]
    │   ├── [Depends on: M2 output]
    │   └── [Uses: LLM Interface (M5b)]
    └── TTS Engine (M6)
        └── [Depends on: M3 output]

Interface Modules:
    ├── VLM Interface (M5a)
    │   └── [Used by: M2]
    └── LLM Interface (M5b)
        └── [Used by: M3]
```

### Data Contracts

#### Contract 1: Frame Extraction → Orchestrator

**Output from M0**: `List[ExtractedKeyframe]`

```python
@dataclass
class ExtractedKeyframe:
    timestamp: str              # "MM:SS" format
    timestamp_seconds: float     # Exact seconds
    frame_path: str            # Path to JPEG file
    content_type: str          # CLIP classification
    complexity_score: float    # 0.0 - 1.0
    clip_confidence: float     # 0.0 - 1.0
```

**File Outputs**:
- `keyframes/*.jpg` - Frame images
- `keyframes.json` - Serialized keyframes
- `video_meta.json` - Video metadata

**Integration Point**:
```python
# In Orchestrator._execute_parallel_extraction()
from frame_extraction.pedagogical_extractor_v2 import extract_pedagogical_keyframes

keyframes_result = extract_pedagogical_keyframes(
    video_id=video_id,
    output_dir=f"{video_dir}/keyframes",
    verbose=False
)
```

#### Contract 2: Transcription → Orchestrator

**Output from M0b**: `Dict` with transcript entries

```python
@dataclass
class TranscriptEntry:
    start: float       # Start time in seconds
    end: float         # End time in seconds
    duration: float    # Duration in seconds
    text: str         # Spoken content
```

**File Output**:
- `transcript.json` - Serialized transcript entries

```json
[
  {
    "start": 15.5,
    "end": 18.0,
    "duration": 2.5,
    "text": "Look at this graph on the screen."
  }
]
```

**Integration Point**:
```python
# In Orchestrator._execute_parallel_extraction()
# Subprocess call to transcription module
cmd = [
    python_exe,
    transcription_main,
    youtube_url,
    "-o", f"{video_dir}/transcript.json",
    "-m", "base",
    "-q"
]

result = subprocess.run(cmd, capture_output=True, text=True)
```

#### Contract 3: Frame Extraction + Transcription → Synchronizer

**Input to M1**: Two JSON files

1. `transcript.json`:
```python
class Synchronizer:
    def __init__(self, transcript_path: str, keyframes_path: str):
        # Load transcript
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
            self.transcripts = [
                TranscriptEntry(**entry) for entry in transcript_data
            ]

        # Load keyframes
        with open(keyframes_path, "r") as f:
            self.keyframes = json.load(f)
```

2. `keyframes.json`:
```json
[
  {
    "timestamp": "02:15",
    "timestamp_seconds": 135.0,
    "frame_path": "data/video_id/keyframes/frame_0001_02_15.jpg",
    "content_type": "graph",
    "complexity_score": 0.4523,
    "clip_confidence": 0.3891
  }
]
```

**Output from M1**: `List[InterventionPoint]`

```python
@dataclass
class InterventionPoint:
    timestamp: float              # When intervention is needed
    frame_path: str               # Path to frame image
    content_type: str             # Type of visual content
    complexity_score: float       # Visual complexity
    clip_confidence: float        # CLIP confidence
    transcript_context: str       # What teacher was saying
    trigger_reason: str           # Why triggered
    confidence: str = "medium"    # "high", "medium", or "low"
```

**Integration Point**:
```python
# In Orchestrator._synchronize()
from synchronizer import Synchronizer

synchronizer = Synchronizer(
    transcript_path=f"{video_dir}/transcript.json",
    keyframes_path=f"{video_dir}/keyframes/keyframes.json"
)

intervention_points = synchronizer.detect_suffer_points()
```

#### Contract 4: Synchronizer → VLM Engine

**Input to M2**: `InterventionPoint` objects

```python
# In Orchestrator._analyze_vlm_snapshots()
for intervention in intervention_points:
    vlm_data = vlm_engine.analyze_frame(
        intervention.frame_path,
        intervention.content_type
    )
    vlm_snapshots[intervention.timestamp] = vlm_data
```

**Output from M2**: `vlm_snapshot` (unified JSON format)

```json
{
  "content_type": "graph",
  "detected_types": ["graph", "text"],
  "visual_analysis": {
    "graph": {
      "axes": {...},
      "curves": [...],
      "relationship": "...",
      "legend": null,
      "annotations": []
    }
  },
  "structural_description": "A line graph on white background...",
  "reading_order": [
    "Read graph title: Velocity vs Time",
    "Identify X-axis: Time (s)",
    "Identify Y-axis: Velocity (m/s)",
    "Follow blue curve from origin upward"
  ],
  "conceptual_hints": "This graph demonstrates terminal velocity...",
  "layout": "Line graph on white background...",
  "text_readout": "Velocity vs Time graph...",
  "spatial_map": "2D coordinate system...",
  "colors_styles": "White background, black axes...",
  "missing_elements": null
}
```

#### Contract 5: VLM + Transcript → Synthesizer

**Input to M3**: VLM snapshots and transcript entries

```python
# In Orchestrator._synthesize_and_generate_audio()
synthesizer = LLMSynthesizer(
    ollama_url="http://localhost:11434",
    model="mistral",
    timeout=120,
    max_retries=3,
    transcripts=transcript_entries,  # All entries for context
    output_mode=output_mode
)

# For each intervention
synthesized_text = synthesizer.weave(
    transcript_context=segment.text,
    vlm_snapshot=vlm_match,
    content_type=vlm_match.get("content_type", "text"),
    intervention_reason=vlm_match.get("intervention_reason", "Unknown"),
    output_mode=output_mode
)
```

**Output from M3**: `audio_script` (string)

```python
# Example explanatory mode output
"""
On the screen, you'll find a line graph titled "Velocity vs Time."

The x-axis at the bottom shows time in seconds, ranging from 0 to 10.

The y-axis on the left shows velocity in meters per second, ranging from 0 to 50.

A blue curve starts at the bottom-left corner, at the origin point where both time and velocity are zero. The curve rises steeply for the first six seconds, then gradually levels off as it approaches 50 meters per second on the right side of the graph.

This graph shows how an object's velocity increases over time due to gravity, but eventually levels off as air resistance balances the gravitational force. This is known as terminal velocity in fluid dynamics.
"""
```

#### Contract 6: Synthesizer → TTS Engine

**Input to M6**: Text script string

```python
# In Orchestrator._synthesize_and_generate_audio()
tts_engine = TTSEngine()

audio_file = f"intervention_{idx:04d}.mp3"
audio_path = f"{video_dir}/audio_segments/{audio_file}"

tts_engine.generate(synthesized_text, audio_path)
```

**Output from M6**: MP3 audio file

- File path: `audio_segments/intervention_000X.mp3`
- Format: MP3, 16000 Hz (SpeechT5) or variable (Edge-TTS)
- Size: ~50-500 KB depending on length

#### Contract 7: All Modules → Manifest

**Input to Manifest Generation**: List of segment dictionaries

```python
segments = [
    {
        "chunk_id": idx,
        "timestamp": segment.start,
        "text": synthesized_text,  # or original transcript for non-interventions
        "is_intervention": True/False,
        "audio_file": f"audio_segments/{audio_file}" or None,
        "original_transcript": segment.text,
        "vlm_data": vlm_match or None
    }
    for idx, segment in enumerate(transcript_entries)
]
```

**Output**: `manifest.json`

```json
{
  "video_id": "abc123",
  "total_segments": 150,
  "intervention_count": 25,
  "created_at": "2026-05-01T12:34:56.789",
  "segments": [...]
}
```

### Interface Layer Integration

#### VLM Interface (M5a) Integration

**Purpose**: Abstract VLM model access

**Used by**: VLM Engine (M2)

**Interface**:
```python
class VLMInterface(ABC):
    @abstractmethod
    def analyze(self, image_path_or_pil, prompt, generation_params=None):
        """Analyze image and return VLMResponse."""
        pass

    @abstractmethod
    def cleanup(self):
        """Release resources."""
        pass
```

**Implementations**:
- `Qwen2VLInterface`: Transformers-based implementation
- `GGUFVLMEngine`: llama-cpp-python implementation

**Usage in M2**:
```python
# Auto-select backend
if use_gguf and GGUF_AVAILABLE:
    self.vlm_interface = GGUFVLMEngine(...)
else:
    self.vlm_interface = Qwen2VLInterface(...)

# Use via unified interface
response = self.vlm_interface.analyze(
    image_path_or_pil=frame_path,
    prompt=prompt,
    generation_params={"max_new_tokens": 384, "temperature": 0.1}
)
```

#### LLM Interface (M5b) Integration

**Purpose**: Abstract LLM model access

**Used by**: Synthesizer (M3)

**Interface**:
```python
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt, **params):
        """Generate text from prompt."""
        pass

    @abstractmethod
    def health_check(self):
        """Check if service is available."""
        pass
```

**Implementation**: `OllamaClient`

**Usage in M3**:
```python
self.client = OllamaClient(
    base_url=ollama_url,
    model=model,
    timeout=timeout,
    max_retries=max_retries
)

# Use via unified interface
explanation = self.client.generate(
    prompt=prompt,
    num_predict=num_predict,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p
)
```

### Error Propagation

Modules propagate errors to the Orchestrator, which handles them:

```
M0 → Error (ffmpeg failure)
    → Orchestrator: Log and skip video

M0b → Error (YouTube API)
    → M0b: Retry with Whisper
    → If still fails: Orchestrator: Log and use empty transcript

M1 → Error (JSON parse)
    → Orchestrator: Log and use empty interventions list

M2 → Error (GPU OOM)
    → M2: Try with CPU offloading
    → Orchestrator: Log and continue with transcript-only

M3 → Error (Ollama timeout)
    → M3: Retry with exponential backoff
    → If still fails: Orchestrator: Fall back to transcript-only

M6 → Error (Edge-TTS)
    → M6: Try SpeechT5 fallback
    → If still fails: Orchestrator: Return text without audio
```

### Resource Cleanup Sequence

Modules must clean up resources in reverse order of initialization:

```
1. Orchestrator starts pipeline
2. → Initialize and run M0, M0b
3. → Run M1 (no resources)
4. → Initialize M2 (load VLM model)
5. → Run M2 (analyze frames)
6. → M2.cleanup() (unload VLM, clear GPU)
7. → Initialize M3, M6
8. → Run M3, M6
9. → M6.cleanup() (unload TTS models)
10. → Orchestrator generates manifest
```

Critical cleanup points:
- After Whisper: Clear GPU before VLM load
- After VLM: Clear GPU before TTS
- After TTS: Release model references



#### Contract 8: Interaction Detection → Synchronizer (M1 Fusion)

**Module**: `synapse_interaction` → `synchronizer`

**Input to Interaction Fusion**:
- Raw interaction episodes (`List[Interaction]`: `WRITING_DRAWING`, `MOUSE_POINTING`, start/end times, confidence)
- M1 Synchronizer suffer points (`List[InterventionPoint]`)
- Transcript entries (`transcript.json`)

**Output**: Fused and enriched `List[InterventionPoint]` with visual interaction triggers:
```python
@dataclass
class InterventionPoint:
    timestamp: float              # Trigger timestamp in seconds
    frame_path: str               # Representative keyframe image
    content_type: str             # Visual category
    complexity_score: float       # Visual complexity
    clip_confidence: float        # Classification confidence
    transcript_context: str       # Teacher's spoken words
    trigger_reason: str           # "Silent Writing", "Deictic Phrase + Interaction"
    confidence: str = "high"      # Boosted confidence when visual + speech coincide
```

#### Contract 9: FastAPI Backend → Orchestrator & SQLite Database

**Module**: `api/routers/video.py` → `orchestrator` & `api/models.py`

**Input to Endpoint**: `ProcessVideoRequest(youtube_url="https://youtube.com/...")`

**Backend Execution**:
1. Creates or updates `Video` record in SQLite `videos` table.
2. Invokes `PipelineOrchestrator.process_detection_only()` or `process_video()`.
3. Inserts all transcript entries into `transcripts` table (`start_time`, `end_time`, `text`).
4. Inserts all detected suffer points into `interventions` table (`id` UUID, `timestamp`, `frame_path`, `content_type`, etc.).
5. Persists files to `data/{video_id}/`.
6. Commits database transaction.

**Output to Client**: `ProcessVideoResponse` containing video metadata, category counts, and full intervention list.

#### Contract 10: FastAPI Backend → Q&A Subsystem

**Module**: `api/routers/video.py` → `qa/service.py`

**Input to `AskService.answer()`**:
- `question`: User question string (1 to 500 chars)
- `video_id`: 11-character YouTube video ID
- `data_dir`: Path to storage directory
- `current_time`: Optional video playback position in seconds
- `moment`: Optional `MomentContext` (active intervention frame OCR, visual summary, and nearby captions)

**Output from `AskService`**: `AnswerOutcome` converted to `AskResponse`:
```json
{
  "status": "answered",
  "scope": "lecture",
  "answer_text": "The lecture explains that energy is conserved in closed systems...",
  "sources": [
    {
      "start_seconds": 120.0,
      "end_seconds": 145.0,
      "intervention_id": null
    }
  ]
}
```

#### Contract 11: FastAPI Backend → Screened Quiz Subsystem

**Module**: `api/routers/quiz.py` → `quiz/service.py`

**Input to `QuizService.generate_quiz()`**:
- `video_id`: 11-character YouTube video ID
- `question_count`: Requested questions (5 to 10, default 6)
- `db`: Active SQLAlchemy session (loads from `transcripts` table)
- `data_dir`: Storage directory path

**Output from `QuizService`**: `QuizResponse` with strictly screened questions, server-verified citations, and duplicate suppression:
```json
{
  "contract_version": 1,
  "quiz_id": "c4b3a120-8e12-4211-9a3b-9a8421b01234",
  "video_id": "3zLQfuMZMF4",
  "generated_at": "2026-10-21T05:00:00Z",
  "prompt_version": "v1",
  "model": "gemini-2.5-flash",
  "requested_count": 5,
  "returned_count": 5,
  "status": "complete",
  "reason": null,
  "questions": [...]
}
```

---
## Step-by-Step Implementation Process

This section documents the development journey from initial concept to working Synapse pipeline, highlighting key milestones, challenges faced, and solutions implemented.

### Phase 1: Problem Analysis and Requirements Definition

**Duration**: Weeks 1-2

**Activities**:
1. User interviews with blind STEM students
2. Literature review of accessibility solutions
3. Gap analysis of existing tools
4. Requirement specification

**Key Findings**:
- Blind students miss 60-80% of visual content in STEM lectures
- Existing solutions (captions, screen readers) don't describe visual content
- Human describers are expensive and not scalable
- AI models exist but no integrated solution

**Requirements Identified**:
- Automatic detection of moments needing description
- Accurate visual content analysis for STEM subjects
- Natural-sounding audio that integrates with teacher's speech
- Multiple explanation depth options
- Offline capability for accessibility in areas with poor internet

**Deliverable**: `REQUIREMENTS.md` document

### Phase 2: Architecture Design and Technology Selection

**Duration**: Weeks 3-4

**Activities**:
1. System architecture design
2. Technology stack evaluation
3. Proof-of-concept testing
4. Module boundary definition

**Technology Choices**:

| Component | Options Evaluated | Selected | Rationale |
|-----------|------------------|-----------|------------|
| VLM | GPT-4V, Claude 3.5 Sonnet, Qwen2-VL | Qwen2-VL-7B-Instruct | Local deployment, cost-effective, good STEM understanding |
| LLM | GPT-4, Claude, Llama 2, Mistral | Mistral via Ollama | Local deployment, balance of quality/performance |
| TTS | ElevenLabs, Azure Speech, Edge-TTS, SpeechT5 | Edge-TTS + SpeechT5 | Free, high quality, offline fallback |
| Frame Extraction | FFmpeg, OpenCV | OpenCV with CLIP | Pedagogical filtering, educational content detection |
| Transcription | AWS Transcribe, Whisper, YouTube API | YouTube API + Whisper | Hybrid approach, comprehensive coverage |

**Architecture Decisions**:
- Modular design for maintainability
- Interface abstraction for model swapping
- Pipeline-based data flow
- Parallel extraction for performance

**Deliverable**: `ARCHITECTURE.md` document and system diagrams

### Phase 3: Core Module Development

**Duration**: Weeks 5-12

#### Week 5-6: Frame Extraction (M0)

**Development Process**:
1. Initial implementation using FFmpeg for frame extraction
2. CLIP integration for educational content classification
3. Complexity scoring algorithm development
4. Scene segmentation using CLIP similarities
5. Semantic deduplication logic

**Challenges Faced**:
- CLIP model was too conservative (filtered too many frames)
- Complexity scores didn't correlate well with educational value
- Scene detection was too sensitive (too many scenes)

**Solutions Implemented**:
- Reduced classification thresholds from 0.7 to 0.4 similarity
- Developed 5-feature complexity scoring (edge, gradient, contrast, color, text)
- Implemented adaptive scene thresholding with minimum gap enforcement
- Added "scene guarantee" to always save best frame from each scene

**Testing Results**:
- Before: 5-10 frames per 10-minute video
- After: 15-25 frames per 10-minute video
- Quality: Manual review showed 85%+ accuracy for pedagogical frames

#### Week 7-8: Transcription (M0b)

**Development Process**:
1. YouTube Transcript API integration
2. Whisper AI fallback implementation
3. Multiple model support (tiny to large-v3)
4. Streaming vs. download mode
5. Progress callback implementation

**Challenges Faced**:
- YouTube API unavailable for many videos
- Whisper slow on CPU
- Audio extraction from YouTube streams was unreliable

**Solutions Implemented**:
- Hybrid approach: API first, Whisper fallback
- Added GPU support for Whisper
- Implemented yt-dlp for reliable audio streaming
- Added progress tracking for user feedback

**Testing Results**:
- YouTube API success rate: ~60%
- Whisper fallback accuracy: 95%+ on STEM content
- Processing time: 20-40s for 10-minute video (with GPU)

#### Week 9: Synchronizer (M1)

**Development Process**:
1. Transcript-keyframe matching logic
2. Scoring system implementation
3. Deictic phrase detection
4. Redundancy checking
5. Confidence level assignment

**Challenges Faced**:
- Initial scoring was too aggressive (too many interventions)
- Negative deictic phrases weren't detected
- Temporal matching failed for gaps in transcript

**Solutions Implemented**:
- Adjusted scoring thresholds (SUFFER_POINT_THRESHOLD from 40 to 50)
- Added negative deictic phrase detection with penalty
- Implemented nearest-endpoint matching for inter-segment gaps
- Added redundancy window (5 seconds minimum)

**Testing Results**:
- False positive rate reduced from 35% to 12%
- Coverage maintained at 85%+ for true suffer points
- Redundant interventions eliminated

#### Week 10-11: VLM Engine (M2)

**Development Process**:
1. Qwen2-VL model integration
2. System prompt engineering
3. Category-specific prompt development
4. Unified JSON output format
5. GGUF backend implementation

**Challenges Faced**:
- VLM missed small elements (labels, arrows)
- JSON output format inconsistent
- OCR quality poor for handwritten text
- VRAM requirements too high (12-14GB)

**Solutions Implemented**:
- Added Chain-of-Regions scanning (2x2 grid) in system prompt
- Developed robust JSON parser with fallback
- Implemented [ILLEGIBLE] marker for OCR failures
- Added GGUF backend with 4-bit quantization
- Enabled partial GPU offloading

**Testing Results**:
- Small element detection improved from 40% to 85%
- JSON parsing success rate: 95%+
- VRAM reduced to 5-6GB (60% reduction)
- Processing time: 2-4s per frame (GGUF)

#### Week 12: VLM and LLM Interfaces (M5a, M5b)

**Development Process**:
1. Interface abstraction layer design
2. Qwen2VL wrapper implementation
3. GGUF engine implementation
4. Ollama client development
5. Error handling and retry logic

**Challenges Faced**:
- Different interfaces had inconsistent return formats
- Network timeouts were common
- Model loading failures occurred without clear error messages

**Solutions Implemented**:
- Unified VLMResponse and OllamaResponse structures
- Exponential backoff retry (2s, 4s, 8s)
- Comprehensive error hierarchy with specific error types
- Health check methods for service availability

### Phase 4: Synthesis and Audio Generation

**Duration**: Weeks 13-16

#### Week 13-14: Synthesizer (M3)

**Development Process**:
1. LLM integration via Ollama
2. Prompt engineering for 3-mode system
3. Transcript+VLM fusion logic
4. Math linearization implementation
5. Hallucination detection

**Challenges Faced**:
- Explanations were too generic
- Math notation wasn't spoken correctly
- VLM hallucinations propagated to output
- Context wasn't maintained across segments

**Solutions Implemented**:
- Category-specific prompt development (equation, graph, circuit, etc.)
- MathLinearizer with symbol mappings (² → "squared", √ → "square root of")
- Cross-domain hallucination detection with keyword lists
- ContextManager for maintaining previous concepts and topic timeline
- 3-mode system (brief/explanatory/detailed)

**Testing Results**:
- Explanation quality (blind student review): 4.2/5.0 average
- Math accuracy: 95%+ for standard notation
- Hallucination detection: 90%+ accuracy
- Context coherence: Maintained across 85%+ of segments

#### Week 15-16: TTS Engine (M6)

**Development Process**:
1. Edge-TTS integration
2. SpeechT5 implementation
3. Dual-provider fallback architecture
4. Audio quality optimization
5. GPU memory management

**Challenges Faced**:
- Edge-TTS required internet (not always available)
- SpeechT5 was slow on CPU
- Audio quality varied by provider
- GPU memory conflicts with VLM

**Solutions Implemented**:
- Dual-provider system with automatic fallback
- GPU-only SpeechT5 with memory cleanup
- Voice selection options
- Model unloading after generation

**Testing Results**:
- Edge-TTS success rate: 90%+
- SpeechT5 fallback success: 95%+
- Audio quality: 4.5/5.0 average rating
- Processing time: 0.5-2s per audio file

### Phase 5: Orchestration and Integration

**Duration**: Weeks 17-20

**Development Process**:
1. Orchestrator module development
2. Parallel extraction implementation
3. Pipeline flow coordination
4. GPU memory management
5. Manifest generation
6. Multiple workflow modes

**Challenges Faced**:
- GPU memory conflicts between modules
- Pipeline was serial (slow)
- Error handling wasn't comprehensive
- No recovery from partial failures

**Solutions Implemented**:
- GPU memory manager with explicit cleanup points
- Threading for parallel audio/video extraction
- Comprehensive error hierarchy with recovery strategies
- Partial result saving for resume capability
- Three workflow modes (full, intervention-only, detection-only)

**Testing Results**:
- Parallel extraction: 20-30% time savings
- GPU memory: No OOM errors in testing
- Recovery: 95%+ success rate with error handling
- End-to-end: 2-8 minutes for 10-minute video

### Phase 6: Testing and Optimization

**Duration**: Weeks 21-24

**Testing Activities**:

1. **Unit Testing**
   - Individual module testing
   - Mock interfaces for dependencies
   - Coverage: 80%+ for core modules

2. **Integration Testing**
   - End-to-end pipeline testing
   - Real YouTube videos (50+ test cases)
   - Multiple subjects (math, physics, chemistry, biology)

3. **User Testing**
   - 10 blind STEM student participants
   - Feedback collection on:
     - Accuracy of descriptions
     - Naturalness of audio
     - Preference for explanation depth
     - Overall usefulness

4. **Performance Optimization**
   - Profiling bottlenecks
   - Implementing batch processing for CLIP
   - Reducing VLM processing to intervention points only
   - Caching frequently used data

**Results**:
- **Accuracy**: 85%+ of suffer points detected
- **Quality**: 4.2/5.0 user rating for descriptions
- **Performance**: 90-95% reduction in VLM processing time
- **Reliability**: 95%+ success rate across diverse video types

**Key Optimizations**:
- Suffer point detection → 90-95% VLM reduction
- Batch CLIP embeddings → 4-6x speedup
- Parallel extraction → 10-20s savings
- GGUF quantization → 60% VRAM reduction

### Phase 7: Documentation and Deployment

**Duration**: Weeks 25-26

**Activities**:
1. Module documentation (9 detailed docs)
2. API reference documentation
3. Installation and usage guides
4. Troubleshooting documentation
5. System documentation (this file)

**Deliverables**:
- Complete module documentation (docs/modules/)
- API reference (docs/API.md)
- Installation guide (docs/INSTALLATION.md)
- System documentation (docs/SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md)

### Development Statistics

**Total Development Time**: ~6 months

**Lines of Code**: ~15,000+

**Modules**: 7 core + 2 interface

**Test Cases**: 200+ unit tests + 50+ integration tests

**Test Videos**: 50+ YouTube STEM lectures covering:
- Mathematics (algebra, calculus, statistics)
- Physics (mechanics, thermodynamics, electromagnetism)
- Chemistry (organic, inorganic, reactions)
- Biology (genetics, ecology, anatomy)

**Models Integrated**:
- VLM: Qwen2-VL-7B-Instruct (Transformers + GGUF)
- LLM: Mistral 7B (via Ollama)
- TTS: Edge-TTS + SpeechT5
- Vision: CLIP ViT-B/32
- Audio: Whisper AI (tiny, base, small, medium, large-v3)

### Key Milestones

| Date | Milestone | Significance |
|-------|-----------|-------------|
| Week 4 | Architecture Complete | System design finalized |
| Week 8 | Extraction Working | Audio + video extraction pipeline functional |
| Week 10 | Detection Working | Suffer point detection operational |
| Week 12 | VLM Working | Visual analysis pipeline complete |
| Week 14 | Synthesis Working | Text generation with fusion operational |
| Week 16 | TTS Working | Audio generation pipeline complete |
| Week 20 | Integration Complete | Full end-to-end pipeline functional |
| Week 24 | Testing Complete | All testing phases finished |
| Week 26 | Documentation Complete | All documentation delivered |

### Lessons Learned

1. **Model Selection**: Local models (Qwen2-VL, Mistral) provide good quality without API costs or dependencies
2. **Performance Optimization**: Focusing on intervention points provided the biggest performance win (90-95% VLM reduction)
3. **User-Centered Design**: Regular feedback from blind students was critical for quality improvements
4. **Modularity**: Interface abstraction allowed easy model swapping and testing
5. **Error Handling**: Comprehensive error recovery is essential for reliability
6. **Resource Management**: GPU memory management is critical for multi-model pipelines
7. **Prompt Engineering**: Category-specific prompts significantly improved VLM accuracy
8. **Testing Diverse Content**: Testing across multiple subjects exposed edge cases and improved robustness

---

## Module Descriptions

### M0: Frame Extraction Module

**Purpose**: Extract pedagogically important keyframes from video lectures

**Tools & Technologies**:
- OpenCV (video processing)
- CLIP (content classification)
- yt-dlp (YouTube streaming)
- PyTorch (CLIP inference)

**Implementation Highlights**:
- 7-phase extraction pipeline
- CLIP-based educational content classification
- Adaptive scene segmentation
- Semantic deduplication
- Temporal coverage guarantee

**Output**:
- `keyframes/*.jpg` - Extracted frame images
- `keyframes.json` - Frame metadata with timestamps
- `video_meta.json` - Video information

**Documentation**: [01_frame_extraction_module.md](modules/01_frame_extraction_module.md)

---

### M0b: Transcription Module

**Purpose**: Convert spoken content to text with precise timing

**Tools & Technologies**:
- YouTube Transcript API (fast fallback)
- Whisper AI (local transcription)
- yt-dlp (audio extraction)
- FFmpeg (audio processing)

**Implementation Highlights**:
- Hybrid approach: API first, Whisper fallback
- Multiple Whisper models (tiny to large-v3)
- Streaming and download modes
- Progress callbacks
- Error handling with retries

**Output**:
- `transcript.json` - Transcript segments with timing

**Documentation**: [02_transcription_module.md](modules/02_transcription_module.md)

---

### M1: Synchronizer Module

**Purpose**: Detect when blind students lose visual access

**Tools & Technologies**:
- Pure Python (no external ML dependencies)
- Regex (pattern matching)
- JSON (data handling)

**Implementation Highlights**:
- Multi-factor scoring system:
  - Deictic phrases (+50)
  - Silent drawing (+80)
  - High complexity (+50)
  - Negative deictic (-50)
- Redundancy checking (5-second window)
- Confidence levels (high/medium/low)
- Spatiotemporal matching

**Output**:
- `List[InterventionPoint]` - Detected suffer points

**Documentation**: [03_synchronizer_module.md](modules/03_synchronizer_module.md)

---

### M5a: VLM Interface Module

**Purpose**: Provide unified interface for vision-language models

**Tools & Technologies**:
- Qwen2-VL-7B-Instruct
- Transformers (Hugging Face)
- PyTorch (deep learning)
- BitsAndBytes (4-bit quantization)

**Implementation Highlights**:
- Device detection (CPU/CUDA)
- Memory management
- Image preprocessing
- Chat template formatting
- Response parsing with fallback
- Async/sync bridging

**Output**:
- `VLMResponse` - Generated text with metadata

**Documentation**: [04_vlm_interface_module.md](modules/04_vlm_interface_module.md)

---

### M2: VLM Engine Module

**Purpose**: Analyze STEM lecture frames with category-specific prompts

**Tools & Technologies**:
- Qwen2-VL-7B-Instruct (primary)
- llama-cpp-python (GGUF fallback)
- Chain-of-Regions scanning
- 10 content type categories

**Implementation Highlights**:
- Category-specific analysis prompts
- Multiple content type detection
- Unified JSON output format
- GGUF backend with CPU offloading
- [ILLEGIBLE] text detection
- Cross-domain validation

**Output**:
- `vlm_snapshot.json` - Structured visual analysis

**Documentation**: [05_vlm_engine_module.md](modules/05_vlm_engine_module.md)

---

### M5b: LLM Interface Module

**Purpose**: Provide unified interface for local LLM operations

**Tools & Technologies**:
- Ollama (LLM runtime)
- Mistral (default model)
- Requests (HTTP client)

**Implementation Highlights**:
- Retry logic with exponential backoff
- Error hierarchy (connection, timeout, server, client)
- Response parsing
- Health checking
- Generation parameter management

**Output**:
- `OllamaResponse` - Generated text with metadata

**Documentation**: [06_llm_interface_module.md](modules/06_llm_interface_module.md)

---

### M3: Synthesizer Module

**Purpose**: Fuse transcript and VLM data into seamless audio scripts

**Tools & Technologies**:
- Ollama (LLM runtime)
- Mistral (default model)
- NumPy (numerical operations)
- Regex (math linearization)

**Implementation Highlights**:
- 3-mode explanation system (brief, explanatory, detailed)
- Context management across segments
- Math linearization for speech
- Cross-domain hallucination detection
- Safety checks (missing elements)
- Knowledge base integration

**Output**:
- `audio_script` - Text ready for TTS

**Documentation**: [07_synthesizer_module.md](modules/07_synthesizer_module.md)

---

### M6: TTS Engine Module

**Purpose**: Convert text scripts to high-fidelity audio

**Tools & Technologies**:
- edge-tts (Microsoft Edge-TTS)
- Transformers (SpeechT5)
- PyTorch (SpeechT5 inference)
- Soundfile (audio I/O)

**Implementation Highlights**:
- Dual-provider system (Edge-TTS + SpeechT5)
- Automatic fallback
- Multiple voice options
- Async/sync bridging
- GPU memory management
- Model unloading

**Output**:
- `*.mp3` - Audio files

**Documentation**: [08_tts_engine_module.md](modules/08_tts_engine_module.md)

---

### M4: Orchestrator Module

**Purpose**: Coordinate entire pipeline lifecycle

**Tools & Technologies**:
- Threading (parallel execution)
- Subprocess (external processes)
- PyTorch (GPU memory management)
- Ollama (process management)

**Implementation Highlights**:
- Parallel extraction (audio + video)
- GPU memory management
- Multiple workflow options:
  - Full pipeline
  - Intervention-only
  - Detection-only
  - Single intervention
- Ollama process lifecycle
- Manifest generation

**Output**:
- `manifest.json` - Master manifest
- `interventions_manifest.json` - Interventions metadata

**Documentation**: [09_orchestrator_module.md](modules/09_orchestrator_module.md)



### M7: Visual Interaction Detection & Fusion (synapse_interaction)

**Purpose**: Detect visual teacher actions—including chalkboard/whiteboard writing, drawing episodes, mouse cursor pointing, and hand gestures—and fuse them into pedagogical suffer points.

**Tools & Technologies**:
- OpenCV (video frame analysis and stroke contour extraction)
- MediaPipe / Heuristic Hand Tracker (index finger pointing detection)
- NumPy (spatial novelty and stroke containment)
- Subprocess & JSON (interaction serialization)

**Submodules & Implementation Highlights**:
1. **`classify.py`**: Classifies optical motion episodes into `WRITING_DRAWING` vs. `MOUSE_POINTING`. Tracks ink accumulation density over time and filters out transient cursor movements.
2. **`hand.py`**: Detects hand presence and verifies whether the index finger is extended toward a diagram or board element (`index_pointing` rule).
3. **`novelty.py`**: Evaluates stroke containment (`contained_fraction`). Distinguishes between ongoing progressive writing (which should be coalesced) and distinct page turns / slide transitions.
4. **`stream.py`**: `InteractionStream` receives the frames the frame extractor decodes (`frame_tap`), runs the motion scan and the hand scan on them, saves a picture for each event and writes `interactions.json`. The module never downloads or streams video itself.
5. **`fusion.py`**: Fuses raw interaction episodes with M1 Synchronizer suffer points (`fuse_interactions`). Identifies "Silent Writing" moments where teacher writes without speech, and boosts confidence for deictic speech that coincides with physical pointing.

**Output**:
- `interactions.json` - Episode list with timestamps, bounding boxes, and interaction classes.
- Fused `InterventionPoint` objects with enriched triggers (`Silent Writing`, `Deictic Phrase + Interaction`).

---

### M5b: LLM Interface & Provider Abstraction (llm_interface)

**Purpose**: Provide a robust, provider-neutral abstraction layer for text generation across local open-weights models and hosted API providers.

**Tools & Technologies**:
- Ollama (local Mistral 7B inference)
- Google GenAI SDK (`google-genai` for Gemini 2.5 Flash / Flash Lite)
- Pydantic & Python Dataclasses (settings, results, token usage)
- Requests / HTTP Client (Ollama REST communication)

**Architecture & Implementation Highlights**:
- **`providers.py`**: Defines `TextGenerationProvider` base class with:
  - `GenerationSettings`: Temperature (0.0 to 2.0), max tokens, system instructions.
  - `GenerationResult`: Generated text, provider name, model name, latency in milliseconds, and token usage breakdown (`prompt_tokens`, `output_tokens`, `total_tokens`).
  - Sanitized Error Hierarchy: `LLMProviderError`, `LLMUnavailableError` (503), `LLMAuthenticationError` (401), `LLMRateLimitError` (429), `LLMTimeoutError` (504), `LLMConfigurationError`. Never leaks API keys or prompts in log messages.
- **`gemini_provider.py`**: Implements `GeminiProvider` and `QuizGeminiProvider`. Supports structured JSON output enforcement (`response_json_schema`) ensuring strict compliance with Pydantic schemas.
- **`ollama_client.py`**: Backward-compatible local Ollama client used for offline synthesis and TTS script generation.

**Output**:
- `GenerationResult` - Standardized text output with metadata.

---

### M8: Transcript-Grounded Q&A Subsystem (qa)

**Purpose**: Provide real-time, student-facing question answering strictly grounded in lecture transcripts and visual intervention moments, returning precise timestamp citations.

**Tools & Technologies**:
- BM25 / Lexical Retrieval (`qa/retrieval.py`)
- LLM Provider Abstraction (`llm_interface/providers.py`)
- Python Dataclasses & Regex (`qa/tokens.py`, `qa/moment.py`)

**Architecture & Implementation Highlights**:
1. **`transcript.py`**: Loads transcript segments. Queries SQLite `transcripts` table first; if unavailable, falls back safely to `data/{video_id}/transcript.json`. Validates video IDs to prevent path traversal.
2. **`retrieval.py`**: Selects transcript passages within token budget:
   - *Whole transcript*: Used for short lectures fitting context limits.
   - *Time-window retrieval*: Used when `current_time_seconds` is supplied to prioritize nearby context.
   - *Lexical / BM25 ranking*: Used for focused student questions.
3. **`moment.py`**: Builds `MomentContext` when the student is viewing an active intervention point, injecting frame OCR, visual summaries, and adjacent captions into the prompt.
4. **`tokens.py`**: Estimates token usage before invoking the LLM to prevent context overflow.
5. **`service.py`**: `AskService` orchestrates answering: constructs prompts with strict anti-hallucination instructions, invokes provider, parses status (`answered`, `not_in_lecture`, `insufficient_context`), and formats `AskResponse`.
6. **`faithfulness.py`**: Post-generation verification checking that model citations map to actual transcript intervals.

**Output**:
- `AskResponse`: Status, scope, answer text, and source intervals (`start_seconds`, `end_seconds`, `intervention_id`).

---

### M9: Screened Formative Quiz Subsystem (quiz)

**Purpose**: Generate high-quality formative practice questions grounded in lecture transcripts, strictly screened against distractor conflicts, duplicate facts, and suspect speech-recognition tokens.

**Tools & Technologies**:
- Gemini / Ollama with Structured JSON Output
- Pydantic v2 (Strict schema validation)
- Pure Python Screening Algorithms (Jaccard token similarity, lexical window containment)

**Architecture & Implementation Highlights**:
1. **`passages.py`**: Deterministic passage segmentation (`S1`, `S2`, ...). Passage boundaries and IDs are stable pure functions of transcript timings.
2. **`service.py`**: `QuizService` coordinates generation:
   - Verifies transcript duration and passage count; returns `insufficient_material` (`too_short` or `no_transcript`) without calling LLM if material is inadequate.
   - Requests candidate questions via structured JSON schema (`candidate_extra` buffer).
   - Runs two-tier screening filter.
3. **`screening.py`**: Enforces contract invariants (choice counts 3-4 for MCQ, exactly True/False for TF, valid option IDs `a`, `b`, `c`, `d`, explanation bounds).
4. **`support_screening.py`**:
   - *Lexical support check*: Verifies key terms appear in cited evidence passage.
   - *Distractor margin check*: Rejects questions where distractors are better supported than the answer key.
   - *Duplicate suppression*: Flags questions testing the same fact using token-set Jaccard overlap and supporting window overlap.
   - *Suspect token screening*: Rejects questions containing garbled ASR artifacts.
5. **`config.py`**: Safe deployment toggles: enabled by default in development, disabled in production unless `QUIZ_ENABLED=1`.

**Output**:
- `QuizResponse`: Formative quiz questions with server-owned quotes, timestamps, hints, and explanations.

---

### M10: FastAPI Backend & Security (api)

**Purpose**: Serve production REST endpoints for video processing, on-demand explanations, bookmarking, Q&A, and quizzes, with SQLite persistence and CORS security.

**Tools & Technologies**:
- FastAPI (ASGI Web Framework)
- SQLAlchemy (ORM with SQLite)
- Pydantic v2 (Request/Response validation)
- Starlette & Uvicorn (ASGI server & middleware)

**Architecture & Implementation Highlights**:
1. **`main.py`**: App initialization, lifespan event handler creating SQLite database tables (`Base.metadata.create_all(bind=engine)`), route inclusion, and healthcheck endpoints.
2. **`routers/video.py`**: Video processing (`/video/process`), metadata (`/video/meta/{video_id}`), on-demand explanations (`/video/intervention/explain`), bookmarking (`/video/intervention/bookmark/{id}`), intervention queries, and Q&A (`/video/ask`).
3. **`routers/quiz.py`**: Quiz generation (`/video/quiz`) with pre-validation feature flag checks and worker-thread execution.
4. **`security.py`**: Strict CORS configuration. Validates allowed origins via `SYNAPSE_CORS_ORIGINS`, permits development tooling (`localhost:5173`), rejects untrusted origins with 403 Forbidden, and safely handles Electron null origins.
5. **`logging_setup.py`**: Standardized logging with prompt and API key redaction.

**Output**:
- REST API responses, SQLite `synapse.db` records, and persisted disk files.

---
## Data Flow, Database & File Persistence

### Complete Pipeline Flow

```mermaid
graph TD
    Client[REST Client / Frontend] -->|POST /video/process| Gateway[FastAPI Backend Gateway]
    Gateway --> DB[(SQLite: synapse.db)]
    Gateway --> Orch[Pipeline Orchestrator]

    Orch --> Extraction[Parallel Extraction]
    Extraction --> TransModule[Transcription Engine]
    Extraction --> FrameModule[Frame Extraction Module]
    Extraction --> InterModule[Interaction Detector: synapse_interaction]

    TransModule --> TranscriptFile[data/<id>/transcript.json]
    FrameModule --> KeyframesFile[data/<id>/keyframes/keyframes.json]
    InterModule --> InteractionsFile[data/<id>/interactions.json]

    TranscriptFile --> Sync[Synchronizer + Fusion]
    KeyframesFile --> Sync
    InteractionsFile --> Sync

    Sync --> InterventionsList[intervention_points]
    InterventionsList -->|Save Suffer Points| DB

    InterventionsList --> VLM[VLM Engine: Qwen2-VL]
    VLM --> Snapshots[vlm_snapshots.json]

    Snapshots --> Synth[LLM Synthesizer]
    TranscriptFile --> Synth
    Synth --> AudioScripts[audio_scripts]

    AudioScripts --> TTS[TTS Engine: Edge-TTS / SpeechT5]
    TTS --> MP3Files[data/<id>/audio_segments/*.mp3]

    MP3Files --> ManifestGen[Manifest Generator]
    ManifestGen --> ManifestFile[data/<id>/manifest.json]
    ManifestGen -->|Save Explanations| DB

    Client -->|POST /video/ask| QA[Q&A Service]
    DB -->|Read Transcripts/Moments| QA
    QA --> AskAnswer[AskResponse with Citations]

    Client -->|POST /video/quiz| Quiz[Quiz Service]
    DB -->|Read Transcripts| Quiz
    Quiz --> ScreenedQuiz[QuizResponse with MCQs]
```

### Database Schema & SQLAlchemy Models

The Synapse API uses SQLite with SQLAlchemy for local metadata storage (`synapse.db`). The database maintains relational integrity and allows instant retrieval of videos, transcripts, interventions, and cached explanations without re-reading the filesystem.

```mermaid
erDiagram
    VIDEOS ||--o{ TRANSCRIPTS : "has"
    VIDEOS ||--o{ INTERVENTIONS : "contains"
    INTERVENTIONS ||--o{ EXPLANATIONS : "generates"

    VIDEOS {
        string id PK "YouTube Video ID (11 chars)"
        string title "Video Title"
        int duration_seconds "Total Duration in Seconds"
        string duration_formatted "e.g. 05:30"
        int total_keyframes "Number of extracted keyframes"
        string subject "Academic subject classification"
        string source_url "YouTube URL"
        datetime date_processed "Timestamp processed"
        string thumbnail_url "Thumbnail image URL"
    }

    TRANSCRIPTS {
        int id PK "Auto-increment ID"
        string video_id FK "References videos.id"
        float start_time "Segment start in seconds"
        float end_time "Segment end in seconds"
        text text "Spoken transcription text"
    }

    INTERVENTIONS {
        string id PK "UUID"
        string video_id FK "References videos.id"
        float timestamp "Suffer point timestamp (seconds)"
        string timestamp_formatted "e.g. 02:15"
        string frame_path "Relative path to keyframe JPEG"
        string content_type "graph, equation, diagram, etc."
        float complexity_score "Visual complexity (0.0 to 1.0)"
        float clip_confidence "CLIP classification confidence"
        string trigger_reason "Deictic, Silent Writing, etc."
        string confidence "high, medium, low"
        text transcript_context "Nearby spoken captions"
        boolean is_bookmarked "User bookmark flag"
        datetime created_at "Creation timestamp"
    }

    EXPLANATIONS {
        int id PK "Auto-increment ID"
        string intervention_id FK "References interventions.id"
        text text_explanation "Woven audio script"
        string audio_file_path "Path to generated MP3 file"
        string text_file_path "Path to generated TXT file"
        string vlm_snapshot_path "Path to VLM snapshot JSON"
        string output_mode "brief, explanatory, detailed"
        text vlm_snapshot "Full VLM analysis JSON string"
        datetime created_at "Generation timestamp"
    }
```

#### Detailed Table Specifications

1. **`videos` Table**:
   - `id` (String, PK, indexed): 11-character YouTube video ID.
   - `title` (String, nullable=False): Video title.
   - `duration_seconds` (Integer): Video length in seconds.
   - `duration_formatted` (String): Human-readable duration (`"MM:SS"` or `"HH:MM:SS"`).
   - `total_keyframes` (Integer): Count of extracted pedagogical keyframes.
   - `subject` (String): Inferred academic subject (e.g., Mathematics, Physics, Chemistry).
   - `source_url` (String, nullable=False): Original YouTube URL.
   - `date_processed` (DateTime): Server timestamp when processed.
   - `thumbnail_url` (String): YouTube video thumbnail URL.

2. **`transcripts` Table**:
   - `id` (Integer, PK, autoincrement): Unique segment record ID.
   - `video_id` (String, FK `videos.id`, indexed): Parent video ID.
   - `start_time` (Float, indexed): Caption start time in seconds.
   - `end_time` (Float): Caption end time in seconds.
   - `text` (Text, nullable=False): Spoken words during the segment.

3. **`interventions` Table**:
   - `id` (String, PK, indexed): Unique UUID for the intervention point.
   - `video_id` (String, FK `videos.id`, indexed): Parent video ID.
   - `timestamp` (Float, indexed): Exact timestamp in seconds where intervention occurs.
   - `timestamp_formatted` (String): Formatted timestamp (e.g., `"03:45"`).
   - `frame_path` (String, nullable=False): Relative filesystem path to the keyframe JPEG.
   - `content_type` (String): Primary visual type (`equation`, `graph`, `circuit`, etc.).
   - `complexity_score` (Float): Edge-density and visual complexity metric.
   - `clip_confidence` (Float): CLIP classifier confidence score.
   - `trigger_reason` (String): Why the point was triggered (`Deictic Phrase`, `Silent Writing`, etc.).
   - `confidence` (String): Confidence rating (`"high"`, `"medium"`, `"low"`).
   - `transcript_context` (Text): Spoken context surrounding the trigger.
   - `is_bookmarked` (Boolean, default=False, indexed): Student bookmark flag.
   - `created_at` (DateTime): Record creation timestamp.

4. **`explanations` Table**:
   - `id` (Integer, PK, autoincrement): Unique explanation ID.
   - `intervention_id` (String, FK `interventions.id`, indexed): Target intervention.
   - `output_mode` (String): Depth mode (`"brief"`, `"explanatory"`, `"detailed"`).
   - `text_explanation` (Text, nullable=False): Synthesized description text.
   - `audio_file_path` (String): Path to synthesized MP3 file on disk.
   - `text_file_path` (String): Path to plain text description on disk.
   - `vlm_snapshot_path` (String): Path to VLM snapshot JSON file on disk.
   - `vlm_snapshot` (Text): Complete serialized VLM analysis JSON.
   - `created_at` (DateTime): Timestamp of explanation generation.

---

### File System Hierarchy & On-Disk Saves

In addition to SQLite metadata, Synapse stores all heavy binary assets (images, audio) and JSON manifests on the filesystem under the `data/` directory partitioned by `video_id`.

```
data/
└── {video_id}/                               # 11-character YouTube video ID
    ├── keyframes/                            # Extracted pedagogical keyframe images
    │   ├── frame_0000_00_00.jpg              # Frame image at 00:00
    │   ├── frame_0001_02_15.jpg              # Frame image at 02:15
    │   └── keyframes.json                    # Metadata for all keyframes
    ├── audio_segments/                       # Audio description MP3 files
    │   ├── intervention_0000.mp3             # Synthesized audio for intervention 0
    │   ├── intervention_0001.mp3             # Synthesized audio for intervention 1
    │   └── segment_0005.mp3                  # Optional stitched/continuous audio
    ├── interventions/                        # Self-contained intervention bundles
    │   ├── intervention_0000/
    │   │   ├── frame.jpg                     # Keyframe image copy
    │   │   ├── description.txt               # Plain text audio description
    │   │   ├── audio.mp3                     # Synthesized MP3 narration
    │   │   ├── vlm_snapshot.json             # VLM visual analysis JSON
    │   │   └── metadata.json                 # Intervention metadata & trigger info
    │   └── intervention_0001/
    │       └── ...
    ├── transcript.json                       # Normalized transcript segments
    ├── video_meta.json                       # YouTube metadata & download info
    ├── manifest.json                         # Master playback and timeline manifest
    ├── interventions_manifest.json           # Intervention-only timeline manifest
    └── interactions.json                     # Visual interaction detection output
```

---

### Data Save Lifecycle & Transaction Boundaries

To maintain data integrity and prevent partial state corruption, Synapse defines strict save points and transaction boundaries across each phase of execution:

#### 1. Video Processing Lifecycle (`POST /video/process`)
1. **Initial Record Creation**: The API checks if `Video(id=video_id)` exists in SQLite. If not, a record is created and committed with `processed=False`.
2. **Parallel Extraction Phase**:
   - Thread A runs transcription; outputs `data/{video_id}/transcript.json`.
   - Thread B runs keyframe extraction; outputs images to `data/{video_id}/keyframes/` and writes `keyframes.json`.
3. **Database Transcript Commit**: The API reads `transcript.json` and inserts each caption line into the `transcripts` table within a single atomic database transaction.
4. **Suffer Point Synchronization & Interaction Fusion**:
   - `synchronizer.py` and `synapse_interaction/fusion.py` compute suffer points.
   - Suffer points are transformed into `Intervention` ORM objects and bulk-inserted into the `interventions` table.
5. **Video Status Finalization**: The `Video` record is updated with `duration_seconds`, `total_keyframes`, `total_interventions`, and committed.

#### 2. On-Demand Explanation Lifecycle (`POST /video/intervention/explain`)
1. **Cache Lookup**: Before invoking heavy models, the API queries `explanations` table for `(intervention_id, output_mode)`:
   ```python
   explanation = db.query(Explanation).filter(
       Explanation.intervention_id == intervention_id,
       Explanation.output_mode == output_mode
   ).first()
   if explanation and os.path.exists(explanation.audio_file_path):
       return cached_response
   ```
2. **Model Invocation (On Cache Miss)**:
   - VLM analyzes keyframe image (`vlm_engine.analyze_frame()`).
   - LLM Synthesizer weaves transcript context with VLM data (`synthesizer.weave()`).
   - TTS Engine generates speech audio (`tts_engine.generate()`).
3. **Disk Persistence**: Files are written to disk:
   - `data/{video_id}/interventions/{intervention_id}/description.txt`
   - `data/{video_id}/interventions/{intervention_id}/audio.mp3`
   - `data/{video_id}/interventions/{intervention_id}/vlm_snapshot.json`
4. **Database Transaction**: A new `Explanation` record is created, linked to the `Intervention`, and committed to SQLite.

#### 3. Storage Fallback Hierarchy
Modules requiring transcript access (such as Q&A and Quiz) employ an automatic fallback hierarchy:
- **Tier 1 (Database)**: Query SQLite `transcripts` table by `video_id` ordered by `start_time`.
- **Tier 2 (Disk File)**: If the database returns 0 rows, load `data/{video_id}/transcript.json` from disk and normalize segments.
- **Tier 3 (Degraded)**: If no file exists, return clean empty segment lists (`source="none"`) and respond with informative student messages without raising unhandled 500 server crashes.

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10+, Linux, macOS | Windows 11, Ubuntu 22.04, macOS 13+ |
| Python | 3.9+ | 3.10+ |
| RAM | 16GB | 32GB |
| GPU (VRAM) | 6GB | 12GB+ |
| Disk Space | 30GB | 50GB+ |
| Internet | Required (Edge-TTS) | Optional (SpeechT5 fallback) |

### Prerequisites

1. **Python 3.9+**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

2. **Git**
   ```bash
   git --version  # Should be installed
   ```

3. **FFmpeg**
   - Windows: Bundled in `frame_extraction/ffmpeg/bin/`
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

4. **Ollama**
   ```bash
   # Download from https://ollama.ai/download
   # Install and run:
   ollama serve

   # Pull model:
   ollama pull mistral
   ```

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/synapse.git
   cd synapse
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv synapse-env

   # Windows
   synapse-env\Scripts\activate

   # Linux/macOS
   source synapse-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download VLM Models**

   **Option A: GGUF (Recommended)**
   ```bash
   # Qwen2-VL-7B-Instruct-GGUF
   # Download from Hugging Face:
   # - Qwen2-VL-7B-Instruct-Q4_K_M.gguf
   # - mmproj-Qwen2-VL-7B-Instruct-f16.gguf

   # Place in: models/Qwen2-VL-7B-Instruct-GGUF/
   ```

   **Option B: Transformers**
   ```bash
   # Auto-download on first use
   # Or manually download to: models/Qwen2-VL-7B-Instruct/
   ```

5. **Verify Installation**
   ```bash
   # Check Python packages
   pip list | grep -E "torch|transformers|edge-tts|ollama"

   # Check Ollama
   ollama list  # Should show mistral

   # Check GPU (if available)
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## Usage Guide

### Basic Usage

#### Full Pipeline

```python
from orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(base_dir="data")

# Process a YouTube video
manifest = orchestrator.process_video(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID",
    output_mode="explanatory"  # brief, explanatory, or detailed
)

# Results
print(f"Video ID: {manifest['video_id']}")
print(f"Total segments: {manifest['total_segments']}")
print(f"Interventions: {manifest['intervention_count']}")
```

#### Intervention-Only Mode

```python
# Generate only interventions
result = orchestrator.process_video_intervention_only(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID",
    output_mode="detailed"
)

print(f"Total interventions: {len(result['interventions'])}")
print(f"Output directory: data/{result['video_id']}/interventions/")
```

#### Detection-Only Mode

```python
# Detect intervention points without processing
result = orchestrator.process_detection_only(
    youtube_url="https://www.youtube.com/watch?v=VIDEO_ID"
)

print(f"Keyframes: {result['video_metadata']['total_keyframes']}")
print(f"Interventions: {result['video_metadata']['total_interventions']}")
```

### Command Line Interface

```bash
# Activate virtual environment
# Windows
synapse-env\Scripts\activate

# Linux/macOS
source synapse-env/bin/activate

# Run pipeline
python -c "
from orchestrator import PipelineOrchestrator
orchestrator = PipelineOrchestrator()
manifest = orchestrator.process_video('https://www.youtube.com/watch?v=VIDEO_ID')
print(f'Complete: {manifest[\"video_id\"]}')
"
```

### Output Structure

```
data/
└── {video_id}/
    ├── keyframes/
    │   ├── frame_0000_00_00.jpg
    │   ├── frame_0001_02_15.jpg
    │   └── ...
    ├── audio_segments/
    │   ├── intervention_0000.mp3
    │   ├── intervention_0001.mp3
    │   └── ...
    ├── interventions/
    │   ├── intervention_0000/
    │   │   ├── audio.mp3
    │   │   ├── description.txt
    │   │   ├── frame.jpg
    │   │   ├── metadata.json
    │   │   └── vlm_snapshot.json
    │   └── ...
    ├── transcript.json
    ├── keyframes.json
    ├── video_meta.json
    ├── manifest.json
    └── interventions_manifest.json
```

### Output Modes Explained

| Mode | Depth | Token Count | Audio Length | Best For |
|------|-------|-------------|---------------|----------|
| brief | Surface level | 150-250 | 30-60s | Quick overviews, rapid understanding |
| explanatory | Standard depth | 300-500 | 60-120s | Complete coverage, normal teaching depth |
| detailed | Deep exploration | 500-900 | 120-240s | Thorough understanding, tutoring sessions |

---

## API Reference

### FastAPI REST API Reference

The Synapse backend provides a RESTful HTTP API hosted with FastAPI and Uvicorn. By default, the API binds to `http://127.0.0.1:8000`.

#### CORS Security Policy

Cross-Origin Resource Sharing (CORS) is governed by `api/security.py`:
- **Allowed Origins**: Configured via `SYNAPSE_CORS_ORIGINS` environment variable (comma-separated).
- **Default Origins**: `http://localhost:5173`, `http://127.0.0.1:5173` (Vite development servers).
- **Electron Origin (`null`)**: Packaged desktop Electron windows transmitting `Origin: null` are rejected by default unless explicitly configured in `SYNAPSE_CORS_ORIGINS`.
- **Untrusted Origins**: Any non-whitelisted origin receives `403 Forbidden` (`{"detail": "Origin not allowed"}`).

---

#### Endpoint 1: Health Check

##### `GET /health` & `GET /`
Basic service health and readiness check.

- **Request**: No parameters or body required.
- **Response `200 OK`**:
  ```json
  {
    "status": "healthy"
  }
  ```

---

#### Endpoint 2: Process Video

##### `POST /video/process`
Initiates complete video analysis, pedagogical keyframe extraction, transcript synchronization, and visual interaction detection.

- **Request Headers**: `Content-Type: application/json`
- **Request Body (`ProcessVideoRequest`)**:
  ```json
  {
    "youtube_url": "https://www.youtube.com/watch?v=3zLQfuMZMF4"
  }
  ```
- **Response `200 OK` (`ProcessVideoResponse`)**:
  ```json
  {
    "video_id": "3zLQfuMZMF4",
    "title": "Integration by Parts - Calculus Lecture",
    "duration_seconds": 273,
    "duration_formatted": "04:33",
    "total_keyframes": 12,
    "total_interventions": 8,
    "interventions": [
      {
        "id": "e8d7a312-4211-4a9b-8134-123456789abc",
        "video_id": "3zLQfuMZMF4",
        "timestamp": 105.4,
        "timestamp_formatted": "01:45",
        "frame_path": "data/3zLQfuMZMF4/keyframes/frame_0004_01_45.jpg",
        "content_type": "equation",
        "complexity_score": 0.521,
        "clip_confidence": 0.892,
        "trigger_reason": "Deictic Phrase",
        "confidence": "high",
        "transcript_context": "notice how we choose u according to the formula",
        "is_bookmarked": false,
        "created_at": "2026-10-21T05:00:00"
      }
    ],
    "category_counts": {
      "text": 1,
      "equation": 4,
      "graph": 2,
      "circuit": 0,
      "diagram": 1,
      "code": 0,
      "handwritten_notes": 0,
      "biology": 0,
      "chemistry": 0,
      "physics": 0,
      "slide": 0,
      "table": 0,
      "unknown": 0
    }
  }
  ```
- **Error Codes**:
  - `400 Bad Request`: Invalid YouTube URL or video unavailable.
  - `500 Internal Server Error`: Processing pipeline failure.

---

#### Endpoint 3: Video Metadata

##### `GET /video/meta/{video_id}`
Checks whether a video exists and whether processing has completed.

- **Path Parameters**:
  - `video_id` (string): 11-character YouTube ID.
- **Response `200 OK` (`VideoMetaResponse`)**:
  ```json
  {
    "video_id": "3zLQfuMZMF4",
    "title": "Integration by Parts - Calculus Lecture",
    "duration_seconds": 273,
    "duration_formatted": "04:33",
    "thumbnail_url": "https://img.youtube.com/vi/3zLQfuMZMF4/maxresdefault.jpg",
    "exists": true,
    "processed": true
  }
  ```
- **Response `404 Not Found`**: Returned when video ID has not been registered.

---

#### Endpoint 4: Generate Intervention Explanation

##### `POST /video/intervention/explain`
Generates (or retrieves from SQLite cache) an AI audio description for a specific intervention point at the requested depth.

- **Request Body (`GenerateExplanationRequest`)**:
  ```json
  {
    "video_id": "3zLQfuMZMF4",
    "intervention_id": "e8d7a312-4211-4a9b-8134-123456789abc",
    "output_mode": "explanatory"
  }
  ```
- **Response `200 OK` (`GenerateExplanationResponse`)**:
  ```json
  {
    "intervention_id": "e8d7a312-4211-4a9b-8134-123456789abc",
    "text_explanation": "On the chalkboard, the instructor writes the integration by parts formula: the integral of u d v equals u times v minus the integral of v d u.",
    "audio_file_path": "data/3zLQfuMZMF4/interventions/e8d7a312-4211-4a9b-8134-123456789abc/audio.mp3",
    "text_file_path": "data/3zLQfuMZMF4/interventions/e8d7a312-4211-4a9b-8134-123456789abc/description.txt",
    "vlm_snapshot_path": "data/3zLQfuMZMF4/interventions/e8d7a312-4211-4a9b-8134-123456789abc/vlm_snapshot.json",
    "output_mode": "explanatory"
  }
  ```
- **Error Codes**:
  - `400 Bad Request`: Unknown output mode (must be `brief`, `explanatory`, or `detailed`).
  - `404 Not Found`: Intervention ID or parent video not found.
  - `500 Internal Server Error`: VLM analysis or TTS generation failure.

---

#### Endpoint 5: Bookmark Intervention

##### `POST /video/intervention/bookmark/{intervention_id}`
Toggles or sets the student bookmark flag on an intervention point.

- **Path Parameters**: `intervention_id` (string UUID).
- **Request Body (`BookmarkRequest`)**:
  ```json
  {
    "bookmarked": true
  }
  ```
- **Response `200 OK` (`BookmarkResponse`)**:
  ```json
  {
    "intervention_id": "e8d7a312-4211-4a9b-8134-123456789abc",
    "is_bookmarked": true
  }
  ```
- **Error Codes**: `404 Not Found` if intervention does not exist.

---

#### Endpoint 6: List Video Interventions

##### `GET /video/{video_id}/interventions`
Returns all interventions for a video, ordered reverse-chronologically, along with category counts.

- **Path Parameters**: `video_id` (string).
- **Response `200 OK` (`ProcessVideoResponse`)**: Same schema as Endpoint 2.
- **Error Codes**: `404 Not Found` if video does not exist.

---

#### Endpoint 7: Ask a Question

##### `POST /video/ask`
Answers student questions strictly grounded in lecture transcripts and visual intervention moments, returning verified timestamped sources.

- **Request Body (`AskRequest`)**:
  ```json
  {
    "video_id": "3zLQfuMZMF4",
    "question": "What is the acronym LIATE used for?",
    "current_time_seconds": 120.0,
    "intervention_id": "e8d7a312-4211-4a9b-8134-123456789abc"
  }
  ```
- **Validation Constraints**:
  - `video_id`: 11 ASCII characters matching `^[A-Za-z0-9_-]{11}$`.
  - `question`: 1 to 500 characters, whitespace stripped.
  - `current_time_seconds`: Optional float between 0 and 172800 (48 hours).
  - `intervention_id`: Optional UUID string.
- **Response `200 OK` (`AskResponse`)**:
  ```json
  {
    "status": "answered",
    "scope": "lecture_with_moment",
    "answer_text": "The acronym LIATE is a rule of thumb for choosing which function to set as u in integration by parts. L stands for logarithmic, I for inverse trigonometric, A for algebraic, T for trigonometric, and E for exponential functions.",
    "sources": [
      {
        "start_seconds": 103.96,
        "end_seconds": 139.48,
        "intervention_id": "e8d7a312-4211-4a9b-8134-123456789abc"
      }
    ]
  }
  ```
- **Degraded Status Values**:
  - `not_in_lecture`: When the topic is completely absent from the lecture transcript.
  - `insufficient_context`: When the transcript is missing or too short to answer reliably.
- **Error Codes**:
  - `404 Not Found`: Video ID does not exist in database or filesystem.
  - `422 Unprocessable Entity`: Malformed video ID or empty question text.
  - `503 Service Unavailable`: Upstream LLM provider is unavailable or experiencing rate limits.

---

#### Endpoint 8: Generate Formative Quiz

##### `POST /video/quiz`
Generates auto-scored practice questions (multiple-choice and true/false) grounded in the lecture transcript, with two-tier screening against distractors, duplicates, and suspect ASR tokens.

- **Request Body (`QuizRequest`)**:
  ```json
  {
    "video_id": "3zLQfuMZMF4",
    "question_count": 5
  }
  ```
- **Validation Constraints**:
  - `video_id`: 11 ASCII characters.
  - `question_count`: Integer between 5 and 10 (default 6).
- **Response `200 OK` (`QuizResponse`)**:
  ```json
  {
    "contract_version": 1,
    "quiz_id": "c4b3a120-8e12-4211-9a3b-9a8421b01234",
    "video_id": "3zLQfuMZMF4",
    "generated_at": "2026-10-21T05:00:00Z",
    "prompt_version": "v1",
    "model": "gemini-2.5-flash",
    "requested_count": 5,
    "returned_count": 5,
    "status": "complete",
    "reason": null,
    "notice": null,
    "omitted": [],
    "questions": [
      {
        "id": "q1",
        "type": "multiple_choice",
        "prompt": "In the LIATE acronym, which type of function takes priority when choosing u?",
        "choices": [
          {"id": "a", "text": "Logarithmic functions"},
          {"id": "b", "text": "Exponential functions"},
          {"id": "c", "text": "Algebraic functions"},
          {"id": "d", "text": "Trigonometric functions"}
        ],
        "correct_choice_id": "a",
        "evidence": [
          {
            "start_seconds": 103.96,
            "end_seconds": 125.0,
            "quote": "L stands for logarithmic functions... pick u as the function that comes first in LIATE"
          }
        ],
        "concept": {
          "id": "liate-priority",
          "label": "LIATE Rule"
        },
        "hints": [
          "Look at the first letter of the acronym."
        ],
        "explanation": "L comes first in LIATE, representing logarithmic functions."
      }
    ]
  }
  ```
- **Insufficient Material Response**:
  If the lecture has fewer than 300 words or too few distinct passages:
  ```json
  {
    "contract_version": 1,
    "quiz_id": "...",
    "video_id": "...",
    "status": "insufficient_material",
    "reason": "too_short",
    "returned_count": 0,
    "questions": []
  }
  ```
- **Error Codes**:
  - `404 Not Found`: Returned when `QUIZ_ENABLED=0` (or production without explicit flag) or unknown video.
  - `422 Unprocessable Entity`: Question count outside range [5, 10].
  - `502 Bad Gateway`: Model output failed screening repeatedly.
  - `503 Service Unavailable`: Upstream LLM provider unavailable.

---

### PipelineOrchestrator Python API

## Performance

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

### Optimization Techniques

1. **Parallel Extraction**: Saves 10-20s by running audio and video extraction simultaneously
2. **VLM Optimization**: Analyzing only intervention points saves 900-1900s (90-95% reduction)
3. **4-bit Quantization**: Reduces VLM VRAM from 12GB to 5-6GB
4. **GPU Memory Management**: Prevents fragmentation and OOM errors
5. **Provider Fallback**: Ensures reliable TTS generation

---

## Troubleshooting

### Common Issues

#### "Ollama connection failed"

**Symptoms**: LLM synthesis fails with connection error

**Solutions**:
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Ensure model is available
ollama list
ollama pull mistral
```

#### "CUDA out of memory"

**Symptoms**: VLM loading or processing fails with OOM error

**Solutions**:
- Enable 4-bit quantization in VLM Engine
- Reduce `max_pixels` parameter
- Use CPU offloading with `device_map="balanced"`
- Process fewer interventions at once

#### "YouTube API unavailable"

**Symptoms**: Transcription falls back to Whisper

**Solutions**:
- This is normal behavior
- Whisper will handle the transcription
- No action required

#### "TTS generation failing"

**Symptoms**: Audio files not being created

**Solutions**:
- Check internet connection (Edge-TTS)
- Verify SpeechT5 models are downloaded
- Try different provider
- Check output directory permissions

### Debugging

#### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check System Information

```python
from orchestrator.gpu_memory_manager import print_system_info
print_system_info()
```

#### Test Individual Modules

```python
# Test frame extraction
from frame_extraction.pedagogical_extractor_v2 import extract_pedagogical_keyframes
keyframes = extract_pedagogical_keyframes("VIDEO_ID", "output")

# Test transcription
from transcriber import TranscriptionEngine
engine = TranscriptionEngine()
result = engine.transcribe("YouTube URL")

# Test synchronizer
from synchronizer import Synchronizer
sync = Synchronizer("transcript.json", "keyframes.json")
interventions = sync.detect_suffer_points()
```

---

## Future Enhancements

### Planned Features

1. **Semantic Retrieval**
   - Use embeddings instead of timestamps for VLM-transcript matching
   - Captures meaning instead of just timing
   - Robust to timing drift

2. **Vector Persistence**
   - Cache transcript embeddings for faster subsequent runs
   - First run: ~5-10s
   - Subsequent runs: ~100ms

3. **Progress Tracking**
   - Emit progress events for UI feedback
   - Real-time status updates
   - Estimated time remaining

4. **Resume Capability**
   - Check for partial results and resume
   - Skip completed phases
   - Handle interruptions gracefully

5. **Batch VLM Processing**
   - Process multiple frames at once
   - Better GPU utilization
   - Reduced overhead

6. **Distributed Processing**
   - Spread across multiple machines
   - Handle large batches efficiently
   - Horizontal scaling

### Recently Implemented

1. **FastAPI REST API & SQLite Persistence**
   - Full REST gateway with health checks, video processing, explanation, bookmarking, Q&A, and quiz endpoints.
   - SQLite relational schema (`videos`, `transcripts`, `interventions`, `explanations`) for fast query execution and deduplication.
   - Configurable CORS policy rejecting untrusted origins and safeguarding desktop Electron environments.

2. **Visual Interaction Detection & Fusion (`synapse_interaction`)**
   - Optical motion and stroke accumulation distinguishing writing/drawing from pointer movements.
   - MediaPipe / heuristic hand tracking for index finger pointing gestures.
   - Novelty containment filtering and before/after frame pair extraction.
   - Automatic fusion with M1 Synchronizer suffer points.

3. **Transcript-Grounded Q&A Subsystem (`qa`)**
   - Real-time student question answering grounded in timestamped lecture captions.
   - Dynamic passage retrieval combining temporal proximity, whole-lecture sampling, and BM25 lexical ranking.
   - Active visual moment context injection (frame OCR and visual summary).
   - Strict token budgeting and post-generation citation faithfulness verification.

4. **Screened Formative Quiz Engine (`quiz`)**
   - Deterministic passage segmentation (`S1..SN`).
   - Two-tier screening filter checking lexical support, distractor margins, same-fact duplicates, and suspect ASR tokens.
   - Server-owned quotes and timestamp attribution preventing hallucinated citations.

5. **Unified Multi-Provider LLM Abstraction (`llm_interface`)**
   - Unified `TextGenerationProvider` supporting local Ollama (Mistral) and hosted Google GenAI (Gemini 2.5 Flash / Flash Lite).
   - Structured JSON schema enforcement (`QuizGeminiProvider`).
   - Sanitized exception hierarchy with automatic HTTP status mapping.

6. **3-Mode Explanation System**
   - Brief, Explanatory, and Detailed output modes with token range enforcement.

7. **TTS Provider Fallback**
   - Edge-TTS primary with automatic SpeechT5 offline fallback.

### Under Consideration

1. **Multi-Language Support**
   - Transcription in multiple languages
   - VLM prompts for international content
   - TTS voices for different languages

2. **Custom Vocabularies**
   - Domain-specific terminology
   - Subject-specific dictionaries
   - Improved accuracy for specialized fields

3. **User Feedback Integration**
   - Learn from corrections
   - Improve quality over time
   - Personalized explanations

4. **Real-Time Processing**
   - Stream processing of live lectures
   - Low-latency interventions
   - Live captioning integration

---

## Summary

The Synapse pipeline is a comprehensive, AI-powered accessibility solution for blind students in STEM education. It automatically processes educational videos, identifies critical moments where visual content is inaccessible, and generates seamless audio descriptions that integrate naturally with the original lecture.

### Key Features

- ✅ **Intelligent Detection**: Multi-factor scoring identifies exactly when descriptions are needed
- ✅ **Context-Aware**: Explanations weave naturally with teacher's speech
- ✅ **Multiple Depth Options**: Brief, explanatory, or detailed explanations
- ✅ **Comprehensive Coverage**: Handles equations, graphs, circuits, diagrams, code, and more
- ✅ **Reliable Generation**: Provider fallback ensures audio is always created
- ✅ **Efficient Processing**: 90-95% VLM reduction through smart optimization
- ✅ **Flexible Workflows**: Full pipeline, intervention-only, or detection-only modes

### Technology Stack

- **Vision**: CLIP, Qwen2-VL-7B-Instruct
- **Language**: Whisper AI, Mistral (via Ollama)
- **Audio**: Edge-TTS, SpeechT5
- **Infrastructure**: PyTorch, Transformers, Python 3.9+

### Impact

Synapse makes STEM education truly accessible to blind students by:
- Eliminating the need for manual description
- Providing consistent, high-quality explanations
- Reducing social exclusion and cognitive load
- Enabling independent learning and participation


---

*Last Updated: April 30, 2026*



