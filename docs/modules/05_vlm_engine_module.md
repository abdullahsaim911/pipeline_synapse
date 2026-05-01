# VLM Engine Module (M2) Documentation

## Overview

The VLM Engine Module serves as the "Universal Eye" of the Synapse pipeline, analyzing complete video frames from STEM lectures. It uses Qwen2-VL-7B-Instruct (with GGUF fallback) to extract structured visual information that makes content accessible to blind students.

## Purpose

- Analyze STEM lecture frames with category-specific prompts
- Extract structured visual information (equations, graphs, circuits, diagrams, etc.)
- Detect multiple content types in single frames
- Provide spatial and conceptual descriptions
- Support both Transformers and GGUF backends

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **Qwen2-VL-7B-Instruct**: Primary vision-language model
- **llama-cpp-python**: GGUF model support (optional)
- **Transformers**: Hugging Face model loading
- **PyTorch**: Deep learning framework
- **qwen-vl-utils**: Qwen-specific vision processing

### Backend Options

| Backend | VRAM | Speed | Quality | Notes |
|---------|------|-------|---------|-------|
| Transformers (4-bit) | 5-6GB | Medium | Best | Recommended |
| Transformers (full) | 12-14GB | Slow | Best | Maximum quality |
| GGUF (quantized) | 4-5GB | Fast | Good | CPU offloading |

## Implementation Details

### Module Structure

```
vlm_engine/
├── __init__.py
├── snapshot_engine.py         # Main VLM engine
├── image_preprocessor.py      # Image preprocessing utilities
└── gguf_vlm_engine.py         # GGUF backend support
```

### Core Components

#### 1. SnapshotEngine Class

**Main Engine Class**: Orchestrates VLM analysis with category-specific prompts

```python
class SnapshotEngine:
    """
    VLM Engine for analyzing STEM lecture frames.

    Loads Qwen2-VL-7B-Instruct model once and keeps it in memory
    for the duration of the pipeline loop.
    """
```

**Initialization Options**:

```python
def __init__(
    self,
    model_path: str = "f:\\Prototyping\\Synapse\\models\\Qwen2-VL-7B-Instruct-GGUF",
    device_map: str = "auto",
    use_4bit: bool = True,
    use_flash_attention: bool = False,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    use_gguf: bool = True,          # NEW: Use GGUF by default
    n_gpu_layers: int = 24,          # NEW: For GGUF tuning
    n_ctx: int = 4096,              # NEW: Context window
    n_threads: int = 6              # NEW: CPU threads
):
```

**Backend Selection**:

```python
# Try GGUF backend first if enabled
if use_gguf and GGUF_AVAILABLE:
    try:
        gguf_model_path = os.path.join(model_path, "Qwen2-VL-7B-Instruct-Q4_K_M.gguf")
        mmproj_path = os.path.join(model_path, "mmproj-Qwen2-VL-7B-Instruct-f16.gguf")

        if os.path.exists(gguf_model_path) and os.path.exists(mmproj_path):
            self.vlm_interface = GGUFVLMEngine(
                model_path=gguf_model_path,
                mmproj_path=mmproj_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads
            )
            return  # Success - use GGUF

    except Exception as e:
        print(f"GGUF initialization failed: {e}")
        use_gguf = False

# Fallback to Transformers backend
self.vlm_interface = Qwen2VLInterface(
    model_name=model_path,
    device_map=device_map,
    use_4bit=use_4bit,
    use_flash_attention=use_flash_attention,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)
```

#### 2. System Prompt

**Enhanced Instructions**: Guides comprehensive visual analysis

```python
SYSTEM_PROMPT = """
You are an expert Vision-Language Model analyst specializing in STEM educational content.
Your task is to analyze video frames from lectures and extract structured information
that makes visual content accessible to blind students.

CORE RULES:
1. Scan ENTIRE frame (not just center) using Chain-of-Regions:
   Partition screen into distinct zones (2x2 grid: top-left, top-right, bottom-left, bottom-right)
   to ensure small details like labels and arrows are not missed.

2. MULTIPLE CONTENT TYPE DETECTION:
   Even if one type is specified, DETECT AND DESCRIBE ALL content types present:
   - equations: Extract ALL equations with their positions and reading order
   - graphs: Extract data, curves, movement/flow, and relationships
   - circuits: Components, connections, flow, and topology
   - diagrams: Complete structural description with element links
   - code: OCR with layout preservation
   - handwritten: OCR with layout and legibility notes
   - biology/chemistry/physics: Structural description with element links
   - text: OCR with hierarchy and formatting

3. Distraction Filtering:
   - Ignore teacher face (unless actively pointing at visual)
   - Ignore decorative borders, unrelated content

4. Conflict Resolution:
   - Prioritize handwritten notes over printed text
   - If both exist, describe handwritten first

5. STRUCTURAL DESCRIPTION REQUIREMENTS:
   - Provide spatial descriptions that build mental models for blind students
   - Include reading order for navigation (left→right, top→bottom)
   - Describe movement/flow for graphs (not just data points)
   - Link all elements to show relationships

6. CONCEPTUAL EXTRACTION:
   - Identify what the visual represents conceptually
   - Provide hints about the underlying principle or law

7. STRICT JSON OUTPUT:
   - Output ONLY valid JSON (no markdown, no filler text, no explanations)
   - Do NOT use ```json ... ``` formatting
   - Use plain JSON object only
"""
```

#### 3. Category-Specific Directives

**Equation Analysis**:

```python
"equation": """
CRITICAL: Extract ALL equations with complete structural information.

For EACH equation provide in visual_analysis.equations array:

1. text (Linearized Spoken Form):
   - Read left-to-right, top-to-bottom
   - Convert ALL notation to spoken form
   - Examples: x² → "x squared", ∫ → "integral", π → "pi"
   - For integrals: "integral of [function] from [lower] to [upper]"
   - For fractions: "[numerator] over [denominator]"

2. description (Spatial Layout):
   - Position: where in frame (top-left, center, bottom, etc.)
   - Size relative to other elements
   - Relationship to surrounding content

3. position (Exact Coordinates):
   - "at top of frame", "below graph", "right of text"

4. reading_order (Mental Navigation):
   - Step-by-step: "Start with integral symbol, then lower limit 0, then upper limit π, then sin(x)dx"
   - For multi-line equations: line-by-line order

5. conceptual_meaning (What it represents):
   - If equation has a label, include it
   - What it models or represents

ADDITIONAL: If multiple equations exist, describe their relationship.
"""
```

**Graph Analysis**:

```python
"graph": """
CRITICAL: Extract complete graph information focusing on CURVE MOVEMENT.

Provide in visual_analysis.graph object:

1. axes (Complete Setup):
   - x_axis: label, range (min-max), units, position
   - y_axis: label, range (min-max), units, position
   - grid_lines: present? color? purpose?

2. curves (MOVEMENT DESCRIPTION - MOST IMPORTANT):
   For EACH curve in array:
   a. description: Shape and Movement - describe HOW the curve looks and moves
      - "starts flat at origin, rises steeply, then gradually levels off"
      - "oscillates with decreasing amplitude around x-axis"
      - "forms a bell shape, rising then falling symmetrically"
      - NOT: "points at (0,0), (4,40), (10,50)"
   b. shape: geometric type (linear, exponential, asymptotic, bell curve, etc.)
   c. key_points: meaningful points with coordinates, what they represent, significance
   d. color/style: what distinguishes this curve

3. relationship (What the Graph Shows):
   - Mathematical relationship
   - Physical meaning
   - Causal relationship
   - This is CRITICAL for LLM to explain the concept

4. legend (If Present): what each color/line represents and position

5. annotations: Labels, equations, notes with positions

MOVEMENT DESCRIPTION GUIDELINES:
- Use active verbs: "rises", "falls", "levels off", "oscillates"
- Describe rate: "steeply", "gradually", "rapidly"
- Describe changes: "bends", "curves", "flattens"
"""
```

**Other Categories**: circuit, diagram, code, handwritten_notes, biology, chemistry, physics, text

#### 4. Prompt Building

```python
def _build_prompt(self, primary_type: str) -> str:
    """
    Build complete prompt for VLM analysis with unified output format.

    Args:
        primary_type: Content classification (equation, graph, circuit, etc.)

    Returns:
        Complete prompt string with system, user, and category directives
    """
    category_directive = self.CATEGORY_DIRECTIVES.get(
        primary_type.lower(),
        ""
    )

    user_prompt = f"""
Analyze this STEM lecture frame and provide a structured description.

PRIMARY CONTENT TYPE: {primary_type}

IMPORTANT: Even though {primary_type} is specified as the primary type,
YOU MUST DETECT AND DESCRIBE ALL content types present in the frame.

CATEGORY-SPECIFIC DIRECTIVES for {primary_type}:
{category_directive}

OUTPUT REQUIREMENTS:
- Scan ENTIRE frame using Chain-of-Regions (2x2 grid approach)
- Detect ALL content types (equations, graphs, text, diagrams, etc.)
- Provide unified JSON output format with visual_analysis object
- Include structural_description, reading_order, and conceptual_hints
- Output ONLY valid JSON (no markdown, no filler)
"""
    return self.SYSTEM_PROMPT + user_prompt
```

#### 5. Frame Analysis Method

```python
def analyze_frame(self, frame_path: str, primary_type: str) -> Dict:
    """
    Analyze a STEM lecture frame and extract structured information.

    Args:
        frame_path: Absolute local path to JPEG image
        primary_type: Dominant content classification
                    (equation, graph, circuit, diagram, code, handwritten_notes,
                     biology, chemistry, text)

    Returns:
        JSON dictionary with structural visual analysis
    """
    # Handle GGUF backend (has analyze_frame method directly)
    if self.use_gguf and hasattr(self.vlm_interface, 'analyze_frame'):
        return self.vlm_interface.analyze_frame(frame_path, primary_type)

    # Transformers backend
    # 1. Build prompt with category directive
    prompt = self._build_prompt(primary_type)

    # 2. Generate via VLM Interface
    response = self.vlm_interface.analyze(
        image_path_or_pil=frame_path,
        prompt=prompt,
        generation_params={
            "max_new_tokens": 384,   # 512 peak VRAM not needed; 384 fits 6GB comfortably
            "temperature": 0.1
        }
    )

    # 3. Parse into structured JSON
    result = self._parse_json_response(response.text)

    # 4. Check for [ILLEGIBLE] text (OCR fallback placeholder)
    text_readout = result.get("text_readout", "")
    illegible_count = text_readout.count("[ILLEGIBLE]")
    text_length = len(text_readout.replace("[ILLEGIBLE]", "").strip())

    if text_length > 0 and (illegible_count / text_length) > 0.2:
        if primary_type.lower() in ["equation", "handwritten_notes"]:
            result["missing_elements"] = "OCR fallback required"
            print(f"[Snapshot Engine] High [ILLEGIBLE] content detected for {primary_type}")

    return result
```

#### 6. JSON Response Parsing

```python
def _parse_json_response(self, text: str) -> Dict:
    """
    Parse VLM output into structured JSON with new unified format.

    Handles:
    - Markdown blocks (```json ... ```)
    - Extra whitespace
    - Trailing commas
    - Missing keys (uses defaults)
    - New unified format fields
    """
    # Strip markdown blocks
    text = re.sub(r'```(?:json)?\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)

    # Find JSON object
    text = text.strip()

    # Remove trailing commas before closing brace
    text = re.sub(r',\s*}', '}', text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        # Return fallback structure with new format
        return {
            "content_type": "unknown",
            "detected_types": ["unknown"],
            "visual_analysis": {},
            "structural_description": text,
            "reading_order": [],
            "conceptual_hints": "",
            "layout": text,
            "text_readout": text,
            "spatial_map": text,
            "colors_styles": text,
            "missing_elements": "Parse error"
        }

    # Ensure new unified format fields exist
    new_required_fields = [
        "content_type", "detected_types", "visual_analysis",
        "structural_description", "reading_order", "conceptual_hints"
    ]

    # Set defaults for missing fields
    if "content_type" not in result:
        detected = result.get("detected_types", [])
        result["content_type"] = detected[0] if detected else "unknown"

    if "detected_types" not in result or not result["detected_types"]:
        result["detected_types"] = [result.get("content_type", "unknown")]

    if "visual_analysis" not in result:
        result["visual_analysis"] = {}

    if "structural_description" not in result:
        result["structural_description"] = result.get("layout", "")

    if "reading_order" not in result:
        result["reading_order"] = []

    if "conceptual_hints" not in result:
        result["conceptual_hints"] = ""

    # Ensure backward compatibility
    old_required_keys = ["layout", "text_readout", "spatial_map", "colors_styles", "missing_elements"]
    for key in old_required_keys:
        if key not in result:
            if key == "missing_elements":
                result[key] = None
            else:
                result[key] = ""

    if result.get("missing_elements") == "null":
        result["missing_elements"] = None

    return result
```

#### 7. Cleanup Method

```python
def cleanup(self):
    """
    Clean up VLM Engine resources.

    Releases GPU memory and deletes model references.
    """
    print("[Snapshot Engine] Cleaning up...")
    self.vlm_interface.cleanup()
    print("[Snapshot Engine] Cleanup complete")
```

## Output Format

### Unified JSON Structure

```json
{
  "content_type": "graph",
  "detected_types": ["graph", "text"],
  "visual_analysis": {
    "graph": {
      "axes": {
        "x_axis": {
          "label": "Time (s)",
          "range": [0, 10],
          "units": "seconds",
          "position": "bottom"
        },
        "y_axis": {
          "label": "Velocity (m/s)",
          "range": [0, 50],
          "units": "meters per second",
          "position": "left"
        },
        "grid_lines": {
          "present": true,
          "color": "light gray",
          "purpose": "readability aid"
        }
      },
      "curves": [
        {
          "description": "starts flat at origin, rises steeply between 2-6 seconds, then gradually levels off approaching 50 m/s",
          "shape": "asymptotic curve",
          "key_points": [
            {
              "coordinates": [0, 0],
              "what": "origin, starting point",
              "significance": "initial condition: zero velocity at time zero"
            },
            {
              "coordinates": [6, 40],
              "what": "point of maximum growth rate",
              "significance": "transition from rapid acceleration to leveling off"
            }
          ],
          "color_style": "blue line, solid, 2px thickness"
        }
      ],
      "relationship": "The graph shows velocity approaching terminal velocity asymptotically over time, following an exponential decay pattern for acceleration.",
      "legend": null,
      "annotations": []
    }
  },
  "structural_description": "A line graph on white background with blue curve showing velocity increasing over time. X-axis shows time in seconds from 0 to 10. Y-axis shows velocity in meters per second from 0 to 50. Grid lines are present for readability.",
  "reading_order": [
    "Read graph title: Velocity vs Time",
    "Identify X-axis: Time (s) with range 0-10",
    "Identify Y-axis: Velocity (m/s) with range 0-50",
    "Follow blue curve from origin upward and rightward",
    "Note curve levels off near 50 m/s"
  ],
  "conceptual_hints": "This graph demonstrates terminal velocity in fluid dynamics, where drag force balances gravitational force, resulting in constant velocity.",
  "layout": "Line graph on white background. Title at top. Axes on left and bottom. Blue curve in center.",
  "text_readout": "Velocity vs Time graph. X-axis: Time (s). Y-axis: Velocity (m/s). Blue curve shows velocity increasing from 0 to 50 m/s over 10 seconds.",
  "spatial_map": "2D coordinate system with blue curve starting at bottom-left corner and rising toward top-right quadrant, leveling off before reaching top edge.",
  "colors_styles": "White background, black axes and grid lines, blue curve, black text labels",
  "missing_elements": null
}
```

## Configuration Parameters

### Engine Configuration

```python
SnapshotEngine(
    model_path="models/Qwen2-VL-7B-Instruct-GGUF",
    device_map="auto",              # Device allocation strategy
    use_4bit=True,                  # Enable 4-bit quantization
    use_flash_attention=False,      # Flash Attention 2
    min_pixels=256*22*22,           # Minimum resolution
    max_pixels=896*22*22,           # Maximum resolution
    use_gguf=True,                  # Use GGUF backend
    n_gpu_layers=24,                # GGUF GPU layers
    n_ctx=4096,                    # Context window size
    n_threads=6                    # CPU threads
)
```

### Content Types

| Type | Focus | Output Fields |
|------|-------|---------------|
| equation | Mathematical notation | equations array |
| graph | Data visualization | graph object |
| circuit | Electronic components | circuit object |
| diagram | Structural relationships | diagram object |
| code | Programming syntax | code object |
| handwritten_notes | Handwritten content | handwritten_notes object |
| biology | Biological structures | biology object |
| chemistry | Molecular structure | chemistry object |
| physics | Physical phenomena | physics object |
| text | Text hierarchy | text object |

## Usage Examples

### Basic Usage

```python
from vlm_engine import SnapshotEngine

# Initialize engine with GGUF backend
engine = SnapshotEngine(
    model_path="models/Qwen2-VL-7B-Instruct-GGUF",
    use_gguf=True,
    n_gpu_layers=24
)

# Analyze a frame
result = engine.analyze_frame(
    frame_path="data/video_id/keyframes/frame_0001.jpg",
    primary_type="graph"
)

print(f"Content Type: {result['content_type']}")
print(f"Description: {result['structural_description']}")
print(f"Concept: {result['conceptual_hints']}")

# Cleanup when done
engine.cleanup()
```

### With Transformers Backend

```python
# Use Transformers instead of GGUF
engine = SnapshotEngine(
    model_path="Qwen/Qwen2-VL-7B-Instruct",
    use_gguf=False,
    use_4bit=True,
    device_map="auto"
)

result = engine.analyze_frame("frame.jpg", "equation")
```

### Batch Processing

```python
engine = SnapshotEngine(use_gguf=True)

frames = [
    ("frame1.jpg", "graph"),
    ("frame2.jpg", "equation"),
    ("frame3.jpg", "diagram")
]

results = []
for frame_path, content_type in frames:
    result = engine.analyze_frame(frame_path, content_type)
    results.append(result)
    print(f"Processed {frame_path}")

engine.cleanup()
```

### Error Handling

```python
engine = SnapshotEngine(use_gguf=True)

try:
    result = engine.analyze_frame("frame.jpg", "graph")

    # Check for missing elements
    if result.get("missing_elements"):
        print(f"Warning: {result['missing_elements']}")
        # Handle missing visual content
    else:
        # Process valid result
        print(f"Detected: {result['detected_types']}")

except Exception as e:
    print(f"Error analyzing frame: {e}")
    # Use fallback

finally:
    engine.cleanup()
```

## Performance Characteristics

### Processing Time (per frame)

| Backend | Configuration | Time | VRAM |
|---------|---------------|------|------|
| GGUF | Q4_K_M, 24 GPU layers | 2-4s | 4-5GB |
| Transformers | 4-bit, balanced | 3-5s | 5-6GB |
| Transformers | Full precision | 5-8s | 12-14GB |
| CPU-only | Any | 15-30s | 0GB |

### Memory Management

**GGUF Backend**:
- Model: 4-5GB (Q4_K_M quantization)
- MMProj: 0.5GB
- Activations: 0.5-1GB
- Total: 5-6.5GB

**Transformers Backend (4-bit)**:
- Model: 5-6GB
- Processor: 0.5GB
- Activations: 1-2GB
- Total: 6.5-8.5GB

### Optimization Techniques

1. **GGUF with Partial GPU Offloading**: Best balance of speed and memory
2. **4-bit Quantization**: 50% VRAM reduction
3. **Resolution Capping**: Lower memory for image processing
4. **Context Window Limiting**: Reduce n_ctx for smaller memory footprint
5. **Cleanup Between Frames**: Prevent memory fragmentation

## Troubleshooting

### Common Issues

**Issue**: "GGUF model not found"
- **Solution**: Verify model path and file names
  - Expected: `Qwen2-VL-7B-Instruct-Q4_K_M.gguf`
  - Expected: `mmproj-Qwen2-VL-7B-Instruct-f16.gguf`

**Issue**: "CUDA out of memory"
- **Solutions**:
  - Reduce `n_gpu_layers` (GGUF)
  - Enable 4-bit quantization
  - Reduce `max_pixels`
  - Use `device_map="balanced"`

**Issue**: "JSON parsing failed"
- **Solution**: Check system prompt and response format
  - Ensure "ONLY valid JSON" instruction
  - Check for markdown blocks
  - Verify no conversational filler

**Issue**: "Poor content detection"
- **Solution**: Adjust category directives
  - Enhance prompts for specific content types
  - Add Chain-of-Regions instruction
  - Check image quality and resolution

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check backend selection
engine = SnapshotEngine(use_gguf=True)
print(f"Using GGUF: {engine.use_gguf}")
print(f"Backend: {type(engine.vlm_interface).__name__}")

# Test single frame
result = engine.analyze_frame("test.jpg", "graph")
print(f"Detected types: {result.get('detected_types')}")
print(f"Missing: {result.get('missing_elements')}")

# Check memory
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
```

## Best Practices

### 1. Memory Management

```python
# Always cleanup
engine = SnapshotEngine(use_gguf=True)
try:
    result = engine.analyze_frame("frame.jpg", "graph")
finally:
    engine.cleanup()
```

### 2. Batch Processing

```python
# Process in small batches to manage memory
def process_in_batches(engine, frames, batch_size=5):
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_results = [engine.analyze_frame(f, t) for f, t in batch]
        results.extend(batch_results)

        # Optional: cleanup between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
```

### 3. Error Recovery

```python
# Fallback to simpler content type
def analyze_with_fallback(engine, frame_path, primary_type):
    try:
        return engine.analyze_frame(frame_path, primary_type)
    except Exception as e:
        print(f"Primary analysis failed: {e}")
        # Try with "unknown" type
        return engine.analyze_frame(frame_path, "unknown")
```

## Future Enhancements

1. **Multi-Frame Context**: Analyze sequences of frames together
2. **Adaptive Prompts**: Dynamically adjust based on detected content
3. **Confidence Scoring**: Per-field confidence metrics
4. **Caching**: Cache embeddings and intermediate results
5. **Progressive Refinement**: Iterative improvement of analysis
6. **Custom Models**: Support for other VLM architectures
7. **Real-Time Processing**: Streaming frame analysis
