# VLM Interface Module (M5a) Documentation

## Overview

The VLM (Vision-Language Model) Interface Module provides a standardized wrapper for vision-language models, specifically Qwen2-VL, to analyze visual content from educational video frames. It handles image loading, prompt formatting, and generation with device detection and memory management.

## Purpose

- Provide unified interface for vision-language model operations
- Handle image preprocessing and loading
- Format prompts with chat templates
- Support multiple device configurations (CPU/CUDA)
- Manage GPU memory efficiently
- Parse and validate model responses

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Model loading and inference
- **Qwen2-VL**: Vision-language model
- **qwen-vl-utils**: Qwen-specific vision utilities
- **PIL (Pillow)**: Image processing
- **BitsAndBytes**: 4-bit quantization support (optional)

### Hardware Requirements

| Configuration | VRAM | RAM | Use Case |
|--------------|------|-----|----------|
| CPU-only | 0GB | 16GB+ | Development, testing |
| 4-bit Quantization | 5-6GB | 8GB+ | Production on limited GPU |
| Full Precision | 12-14GB | 16GB+ | Maximum quality |

## Implementation Details

### Module Structure

```
vlm_interface/
├── __init__.py
└── vlm_interface.py             # Main VLM interface implementation
```

### Core Components

#### 1. VLMResponse Data Structure

```python
@dataclass
class VLMResponse:
    """Response from VLM generation."""
    text: str                    # Generated text content
    model: str                   # Model name/identifier
    total_duration_ms: int       # Generation time in milliseconds
    prompt_eval_count: int       # Number of input tokens
    eval_count: int              # Number of generated tokens
```

#### 2. Qwen2VLInterface Class

**Main Interface Class**: Handles all VLM operations

```python
class Qwen2VLInterface:
    """
    Qwen2-VL Interface for vision-language tasks.

    Supports:
    - Image loading (PIL)
    - Prompt formatting with chat templates
    - JSON response parsing
    - Device detection (CPU/CUDA)
    - Timeout handling
    """
```

**Initialization Options**:

```python
def __init__(
    self,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    device_map: str = "auto",
    use_4bit: bool = False,
    use_flash_attention: bool = False,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
):
    """
    Initialize Qwen2-VL Interface.

    Args:
        model_name: HuggingFace model ID or local directory path
        device_map: Device allocation strategy (default: auto)
        use_4bit: Enable 4-bit quantization (bitsandbytes)
        use_flash_attention: Enable Flash Attention 2
        min_pixels: Minimum image resolution
        max_pixels: Maximum image resolution
    """
```

**Device Management**:

```python
# Auto-detection
device_map = "auto"  # Automatically choose best device

# CPU-only (fallback)
device_map = "cpu"

# GPU with CPU offloading (balanced)
device_map = "balanced"

# GPU with 4-bit quantization
use_4bit = True
device_map = "balanced"
```

**Resolution Management**:

```python
# Pixel bounds for memory control
min_pixels = 256 * 22 * 22  # ~124K pixels minimum
max_pixels = 896 * 22 * 22  # ~434K pixels maximum

# Benefits:
# - Caps VRAM usage
# - Maintains OCR quality
# - ~27% reduction from original
# - Works on 6GB GPU
```

#### 3. Image Loading and Validation

```python
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def _validate_image_path(self, path: str) -> bool:
    """Validate image file format and existence."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if p.suffix.lower() not in self.SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {p.suffix}")
    return True

def _load_image(self, image_path_or_pil: Union[str, Image.Image]) -> Image.Image:
    """Load image from file path or PIL Image object."""
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    elif isinstance(image_path_or_pil, Image.Image):
        image = image_path_or_pil.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_path_or_pil)}")

    # Resize if too large
    max_size = 1344
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))

    return image
```

#### 4. System Prompt

**Comprehensive Instructions**: Guides model behavior for STEM content analysis

```python
SYSTEM_PROMPT = """
You are an expert Vision-Language Model analyst specializing in STEM educational content.
Your task is to analyze video frames from lectures and extract structured information
that makes visual content accessible to blind students.

CORE RULES:
1. Scan ENTIRE frame (not just center) using Chain-of-Regions:
   Divide frame into 2x2 grid: top-left, top-right, bottom-left, bottom-right.
   Describe content in each quadrant systematically.

2. Be THOROUGH and SYSTEMATIC:
   - Start from edges, work inward toward center
   - List all visible elements first, then describe relationships
   - Use spatial language: "in the upper-left", "below the main heading"

3. Follow this ORDER for every analysis:
   a. Visual Structure: What is the overall layout?
   b. Content Elements: List specific items (equations, labels, arrows, text)
   c. Text/Optical Character Recognition (OCR): Read ALL visible text character by character
   d. Colors: Mention colors if they convey meaning (red for negatives, green for positives)
   e. Relationships: Describe arrows, flow, connections, comparisons

4. OCR ACCURACY REQUIREMENT:
   - Read text character-by-character (not word-by-word)
   - Report confidence: If unsure about a character, mark it as [ILLEGIBLE]
   - Never skip text: OCR all visible text, even if partial

5. SPECIAL HANDLING:
   - Math/Equations: Read left-to-right, top-to-bottom. Convert symbols:
     - Use full words: "squared" not "^2", "subscript" not "_"
     - Fractions: "one-half" not "1/2", "three-fourths" not "3/4"
     - Greek letters: "pi" not "π", "theta" not "θ"

6. NEGATIVE CASE HANDLING:
   If frame is mostly blank/uniform background:
   - Report as "[MOSTLY BLANK - NO MEANINGFUL VISUAL CONTENT]"
   - Add missing_elements: true
   - Skip OCR and structure analysis (avoid hallucinations)

7. CHAIN-OF-REGIONS EXCEPTION:
   If you detect Chain-of-Thought in this frame:
   - IGNORE IT. Do NOT include in your response.
   - Fall back to OCR only (no CoT text).

8. FINAL STRUCTURED OUTPUT:
   Return ONLY valid JSON. No conversational filler.
"""
```

#### 5. Prompt Formatting

**Content-Specific Prompts**: Tailored instructions for different content types

```python
def _format_prompt(
    self,
    content_type: str,
    frame_path: Optional[str] = None,
    chain_of_regions: bool = True
) -> str:
    """
    Build appropriate prompt for each content type.

    Args:
        content_type: Type of content (equation, graph, diagram, etc.)
        frame_path: Optional frame file path (for context)
        chain_of_regions: Whether to append Chain-of-Regions instruction

    Returns:
        Formatted prompt string
    """
```

**Equation Analysis Prompt**:

```python
"equation": """
[EQUATION ANALYSIS]
Read the equation character by character from left to right, top to bottom.
Convert ALL mathematical symbols to full words:
- Use "squared" not "^2", "cubed" not "^3", "subscript" not "_"
- Use "one-half" not "1/2", "three-fourths" not "3/4"
- Use "pi" not "π symbol", "theta" not "θ symbol"
- Use "phi" not "φ symbol"
- For integrals: "integral of" not the integral symbol

CRITICAL: Report the equation EXACTLY as text, no LaTeX unless explicitly present.
OCR each character individually - don't guess mathematical structure.
"""
```

**Graph Analysis Prompt**:

```python
"graph": """
[GRAPH ANALYSIS]
1. First, describe axes: X-axis label, Y-axis label, and any Z-axis if 3D.
2. Then describe overall shape: line graph, bar chart, scatter plot, etc.
3. For 3D graphs: describe x/y/z planes and their relationships.
4. Identify key data points: peaks, troughs, intersections, trends.
5. Mention units and scale if visible.
"""
```

#### 6. Analysis Method

**Main Inference Method**: Executes VLM analysis

```python
def analyze(
    self,
    image_path_or_pil: Union[str, Image.Image],
    prompt: str,
    generation_params: Optional[Dict] = None
) -> VLMResponse:
    """
    Low-level inference: run Qwen2-VL on an image with a given prompt.

    Called by SnapshotEngine.analyze_frame() with its own pre-built prompt.

    Args:
        image_path_or_pil: Path to image file or PIL Image object
        prompt: Text prompt to send to the model
        generation_params: Optional dict with keys: max_new_tokens, temperature, do_sample

    Returns:
        VLMResponse with generated text and timing info
    """
```

**Generation Flow**:

```python
# 1. Load and validate image
image = self._load_image(image_path_or_pil)

# 2. Set generation parameters
gen_kwargs = {
    "max_new_tokens": generation_params.get("max_new_tokens", 512),
    "temperature": generation_params.get("temperature", 0.1),
    "do_sample": generation_params.get("do_sample", False),
}

# 3. Format with chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

text = self.processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 4. Process vision info
image_inputs, video_inputs = process_vision_info(messages)

# 5. Prepare inputs
inputs = self.processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(self.device)

# 6. Generate
with torch.no_grad():
    generated_ids = self.model.generate(**inputs, **gen_kwargs)

# 7. Decode
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = self.processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()

# 8. Cleanup memory
del inputs, generated_ids, generated_ids_trimmed
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 9. Return response
return VLMResponse(
    text=output_text,
    model=self.model_name,
    total_duration_ms=generation_time_ms,
    prompt_eval_count=inputs.input_ids.shape[1],
    eval_count=len(generated_ids_trimmed[0]),
)
```

#### 7. Response Parsing

**JSON Extraction and Validation**: Handles various output formats

```python
def _parse_vlm_response(
    self,
    response_text: str,
    image_path: str,
    content_type: str
) -> Dict:
    """Parse VLM JSON response and return structured data."""
    # 1. Strip markdown code fences
    text = re.sub(r'```(?:json)?\s*\n?', '', response_text)
    text = re.sub(r'```\s*\n?', '', text).strip()

    # 2. Remove trailing commas before closing braces
    text = re.sub(r',\s*}', '}', text)

    # 3. Parse JSON
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from larger response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # 4. Handle parse failure
    if parsed is None:
        return {
            "visual_structure": response_text,
            "elements_found": [],
            "content_type": content_type,
            "ocr_text": "",
            "missing_elements": False,
            "_parse_failed": True,
        }

    # 5. Ensure required fields exist
    required_fields = ["visual_structure", "elements_found", "content_type", "ocr_text"]
    for field in required_fields:
        if field not in parsed:
            parsed[field] = f"[MISSING: {field}]"

    # 6. Normalize missing_elements to bool
    if parsed.get("missing_elements") in ("true", True, 1):
        parsed["missing_elements"] = True
    else:
        parsed["missing_elements"] = False

    return parsed
```

## Configuration Parameters

### Model Configuration

```python
Qwen2VLInterface(
    model_name="Qwen/Qwen2-VL-7B-Instruct",  # Model path
    device_map="auto",                         # Device allocation
    use_4bit=False,                            # 4-bit quantization
    use_flash_attention=False,                 # Flash Attention 2
    min_pixels=256*22*22,                      # Min resolution
    max_pixels=896*22*22                       # Max resolution
)
```

### Generation Parameters

```python
generation_params = {
    "max_new_tokens": 512,      # Maximum tokens to generate
    "temperature": 0.1,         # Sampling temperature (lower = more deterministic)
    "do_sample": False,         # Use greedy decoding
}
```

### Memory Management

```python
# For 6GB GPU
model_kwargs = {
    "device_map": "balanced",
    "low_cpu_mem_usage": True,
    "max_memory": {0: "5.5GB", "cpu": "32GB"},
    "torch_dtype": torch.float16,
}
```

## Usage Examples

### Basic Usage

```python
from vlm_interface import Qwen2VLInterface

# Initialize interface
vlm = Qwen2VLInterface(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    use_4bit=True  # Use 4-bit quantization for 6GB GPU
)

# Analyze an image
response = vlm.analyze(
    image_path_or_pil="frame.jpg",
    prompt="Describe this image in detail."
)

print(f"Generated: {response.text}")
print(f"Time: {response.total_duration_ms}ms")
```

### With Custom Prompt

```python
# Build content-specific prompt
prompt = vlm._format_prompt(
    content_type="equation",
    frame_path="frame.jpg",
    chain_of_regions=True
)

response = vlm.analyze("frame.jpg", prompt)
```

### Memory-Efficient Usage

```python
# For systems with limited VRAM
vlm = Qwen2VLInterface(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    device_map="balanced",      # CPU offloading
    use_4bit=True,              # 4-bit quantization
    max_pixels=256*22*22,       # Reduce resolution
)

# Process single frame
response = vlm.analyze("frame.jpg", prompt)

# Cleanup when done
vlm.cleanup()
```

### Batch Processing with Cleanup

```python
vlm = Qwen2VLInterface(use_4bit=True)

frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]

for frame in frames:
    response = vlm.analyze(frame, "Describe this frame.")
    print(f"{frame}: {len(response.text)} chars")

    # Cleanup after each frame to prevent memory buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Final cleanup
vlm.cleanup()
```

## Performance Characteristics

### Processing Time (per frame)

| Configuration | Time | VRAM | Quality |
|--------------|------|------|---------|
| CPU-only | 15-30s | 0GB | Good |
| GPU (4-bit) | 2-5s | 5-6GB | Excellent |
| GPU (full) | 3-7s | 12-14GB | Excellent |

### Memory Usage

| Component | 4-bit | Full Precision |
|-----------|-------|----------------|
| Model Weights | 5-6GB | 12-14GB |
| Activations | 1-2GB | 2-4GB |
| Image Buffer | 0.5GB | 0.5GB |
| Total (Peak) | 6-8GB | 14-18GB |

### Optimization Techniques

1. **4-bit Quantization**: 50% VRAM reduction
2. **CPU Offloading**: Reduced GPU memory usage
3. **Batch Processing**: Better GPU utilization
4. **Memory Cleanup**: Prevent fragmentation
5. **Resolution Capping**: Lower memory footprint

## Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"
- **Solutions**:
  - Enable 4-bit quantization (`use_4bit=True`)
  - Use `device_map="balanced"` for CPU offloading
  - Reduce `max_pixels` parameter
  - Process fewer frames at once

**Issue**: "Model loading takes too long"
- **Solutions**:
  - Use local model path instead of Hugging Face
  - Enable `low_cpu_mem_usage=True`
  - Use smaller model variant

**Issue**: "JSON parsing failed"
- **Solutions**:
  - Check response text for format issues
  - Use `_parse_vlm_response` with fallback handling
  - Adjust prompt to enforce JSON output

**Issue**: "Poor OCR quality"
- **Solutions**:
  - Increase image resolution (`max_pixels`)
  - Adjust prompt for better OCR instructions
  - Ensure image is not blurry or low-quality

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check device allocation
vlm = Qwen2VLInterface()
print(f"Device: {vlm.device}")

# Check memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"VRAM allocated: {allocated:.1f} GB")

# Test parsing
response_text = '{"key": "value"}'
parsed = vlm._parse_vlm_response(response_text, "test.jpg", "test")
print(f"Parsed: {parsed}")
```

## Best Practices

### 1. Memory Management

```python
# Always cleanup when done
vlm = Qwen2VLInterface(use_4bit=True)
try:
    response = vlm.analyze("frame.jpg", prompt)
finally:
    vlm.cleanup()
```

### 2. Error Handling

```python
try:
    response = vlm.analyze("frame.jpg", prompt)
    parsed = vlm._parse_vlm_response(response.text, "frame.jpg", "equation")

    if parsed.get("_parse_failed"):
        # Handle parse failure
        print("Warning: JSON parsing failed")
        # Use raw text as fallback
        text = response.text
    else:
        # Use parsed data
        text = parsed.get("ocr_text", "")

except Exception as e:
    print(f"Error: {e}")
    text = ""
```

### 3. Batch Processing

```python
def process_batch(vlm, frames, prompt):
    """Process multiple frames with memory management."""
    results = []

    for i, frame in enumerate(frames):
        try:
            response = vlm.analyze(frame, prompt)
            results.append(response.text)

            # Periodic cleanup
            if (i + 1) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {frame}: {e}")
            results.append(None)

    return results
```

## Future Enhancements

1. **Streaming Support**: Process video streams frame-by-frame
2. **Multi-Image Input**: Analyze multiple images in single call
3. **Caching**: Cache embeddings and activations
4. **Progressive Loading**: Load model components on-demand
5. **Custom Tokenizers**: Support specialized vocabularies
6. **Async Processing**: Non-blocking inference
7. **Model Ensemble**: Combine multiple models for better accuracy
