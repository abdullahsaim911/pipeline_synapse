# VLM Snapshot Engine Implementation Plan

## Module Overview
- **File**: `snapshot_engine.py`
- **Purpose**: "Universal Eye" for pipeline - analyzes complete video frames from STEM lectures
- **Technology**: Qwen2-VL-7B-Instruct (Local Inference via transformers/torch)

---

## Input/Output Specification

### Input
- `frame_path` (str): Absolute local path to JPEG image
- `primary_type` (str): Dominant content classification (equation, graph, circuit, diagram, code, handwritten_notes, biology, chemistry, text)

### Output (JSON Dictionary)
```json
{
  "layout": "Physical arrangement of all regions (e.g., 'Graph on right, text on left')",
  "text_readout": "Linear text from ALL regions, separated by ' ||| ' delimiter",
  "spatial_map": "Geometry/relationships in ALL regions, ' ||| ' separated",
  "colors_styles": "Visual descriptors (colors, line styles) for all regions",
  "missing_elements": "null if successful, 'No clear STEM visual' if irrelevant/blurry/empty"
}
```

---

## Core 5-Step Workflow

### 1. Initialization
- Load `Qwen/Qwen2-VL-7B-Instruct` using `AutoModelForVision2Seq` and `AutoProcessor`
- **Optimizations**: 4-bit quantization (bitsandbytes), Flash Attention 2, torch.float16/bfloat16
- Keep model in VRAM for pipeline loop duration

### 2. Prompt Construction
- Select **Category Directive** based on `primary_type`
- Inject into **Universal User Prompt**
- Combine with **Universal System Prompt**

### 3. Inference ("Universal Analyst")
- Pass complete image + constructed prompt to model
- Model performs **Regional Segmentation** internally (text, diagram, overlay zones)

### 4. Robust Parsing
- Strip markdown blocks (```json ... ```)
- Parse string to Python dictionary
- Validate all required keys exist

### 5. Edge Case Handling (OCR Fallback)
- Check `text_readout` for `[ILLEGIBLE]` substring
- If >20% of text is `[ILLEGIBLE]` for equation/handwritten_notes:
  - Trigger **PaddleOCR** on raw image
  - Repair `text_readout` field with OCR result

---

## Prompt Architecture

### System Prompt (Constant)
```
You are an expert Vision-Language Model analyst specializing in STEM educational content.
Your task is to analyze video frames from lectures and extract structured information
that makes visual content accessible to blind students.

CORE RULES:
1. Scan ENTIRE frame (not just center) using Chain-of-Regions:
   Partition screen into distinct zones (2x2 grid: top-left, top-right, bottom-left, bottom-right)
   to ensure small details like labels and arrows are not missed.

2. Distraction Filtering:
   - Ignore teacher face (unless actively pointing at visual)
   - Ignore decorative borders, unrelated content

3. Conflict Resolution:
   - Prioritize handwritten notes over printed text
   - If both exist, describe handwritten first

4. STRICT JSON OUTPUT:
   - Output ONLY valid JSON (no markdown, no filler text, no explanations)
   - Do NOT use ```json ... ``` formatting
   - Use plain JSON object only

5. REGION DELIMITER:
   - Use ||| to separate multiple visual zones in output

OUTPUT FORMAT (JSON only):
{
  "layout": "Physical arrangement of all regions (e.g., 'Graph on right, text on left')",
  "text_readout": "Linear text from ALL regions, separated by ' ||| ' delimiter",
  "spatial_map": "Geometry/relationships in ALL regions, ' ||| ' separated",
  "colors_styles": "Visual descriptors (colors, line styles) for all regions",
  "missing_elements": "null if successful, 'No clear STEM visual' if irrelevant/blurry/empty"
}

If the frame contains no clear STEM visual content (blurry, empty, irrelevant),
set "missing_elements": "No clear STEM visual" and return immediately.
```

### User Prompt Template
```
Analyze this STEM lecture frame and provide a structured description.

CONTENT TYPE: {primary_type}

CATEGORY-SPECIFIC DIRECTIVES:
{CATEGORY_DIRECTIVE}

OUTPUT REQUIREMENTS:
- Scan ENTIRE frame using Chain-of-Regions (2x2 grid approach)
- Ignore teacher face unless actively pointing at visual
- Use ||| delimiter to separate multiple visual zones
- Output ONLY valid JSON (no markdown, no filler)
```

### Category Directives (9 types) - EXACT PROMPTS

**equation:**
```
Read equations linearly, left-to-right, top-to-bottom.
Convert mathematical notation to spoken form:
- Superscripts: "x^2" → "x squared", "x^3" → "x cubed"
- Subscripts: "x_2" → "x sub 2", "x^{-1}" → "x to the power of negative one"
- Fractions: "a/b" → "a over b" or "a divided by b"
- Greek letters: "α" → "alpha", "β" → "beta", "π" → "pi"
- Operators: "∫" → "integral of", "∑" → "summation of", "∂" → "partial derivative"
```

**graph:**
```
Identify graph type (line, bar, scatter, pie, etc.).
Extract axis information:
- X-axis: label, range/scale, units
- Y-axis: label, range/scale, units
- Legend: what each color/line represents
- Key data points: specific values if labeled
- Trend/shape: describe overall pattern (increasing, decreasing, curved, etc.)
```

**circuit:**
```
Trace the circuit flow and identify:
- Power source: battery, voltage source, etc.
- Components: resistors, capacitors, inductors, diodes, transistors
- Connections: series, parallel, or mixed
- Values: resistor values (in ohms), capacitor values (in farads), etc.
- Circuit topology: describe overall arrangement
```

**diagram:**
```
List all labeled parts and their relationships:
- Component names and labels
- Arrows: what they connect or indicate (flow, causality, etc.)
- Spatial positions: left/right/center/top/bottom
- Physical layout: vertical/horizontal arrangement
```

**code:**
```
Read code literally (do NOT explain logic):
- Syntax: preserve indentation, brackets, symbols
- Language: identify programming language
- Key functions: list function names if visible
- Code structure: describe overall organization (classes, functions, imports)
- Do NOT explain what code does, only what it looks like
```

**handwritten_notes:**
```
Describe handwritten content:
- Mark messy/unreadable words as [ILLEGIBLE]
- Overall layout: title, bullet points, equations position
- Content type: lecture notes, equations, diagrams in notes
- If majority is [ILLEGIBLE], note this in text_readout
```

**biology:**
```
Identify biological structure:
- Main structure: name of organism, organ, or system
- Key parts: labeled components with their locations
- Scale/size: relative proportions if visible
- Labels: all visible text labels on diagram
```

**chemistry:**
```
Describe chemical structure or reaction:
- Molecules: name compounds shown
- Bonds: single, double, triple bonds
- Geometry: shape of molecule (linear, bent, tetrahedral, etc.)
- Text labels: chemical formulas, atom labels
- Read chemical names phonetically: "H2O" → "H two O", not "H two O"
```

**text:**
```
Describe textual content:
- Hierarchy: title, headings, bullet points, sub-bullets
- Text formatting: bold, italic, colored text (note these)
- Structure: preserve organization (lists, numbered items)
- Key terms: important vocabulary highlighted
```

---

## File Structure
```
vlm_engine/
├── __init__.py
├── snapshot_engine.py    # Main VLM orchestration class
└── VLM_IMPLEMENTATION_PLAN.md  # This file
```

---

## Dependencies
```bash
# CLIP is already used in frame-extraction module (no install needed)
pip install transformers qwen-vl-utils torch
```

**Already installed** (from requirements.txt):
- transformers>=2.0.0
- torch>=2.0.0
- openai-clip (CLIP used in frame-extraction)

**To install:**
```bash
pip install qwen-vl-utils
```

**Note:** Ollama with Mistral is already installed locally.

---

## Key Implementation Details

### Class: `SnapshotEngine`

**`__init__()` Parameters:**
- `model_name`: Default "Qwen/Qwen2-VL-7B-Instruct"
- `device_map`: "auto" (CUDA > MPS > CPU)
- `use_4bit`: Enable 4-bit quantization
- `use_flash_attention`: Enable Flash Attention 2
- `min_pixels`/`max_pixels`: Control image resolution

**`analyze_frame(frame_path, primary_type)` Method:**
1. Validate file exists
2. Build prompts with category injector
3. Construct message for Qwen2-VL
4. Process vision info with `process_vision_info()`
5. Generate with `max_new_tokens=512`, `temperature=0.1`
6. Trim input tokens, decode output
7. Parse into structured JSON

**`cleanup()` Method:**
- Release GPU memory with `torch.cuda.empty_cache()`
- Delete model references

---

## Performance Targets
- **Model Load**: ~30s (first run, cached thereafter)
- **Per Frame**: 2-5s (GPU), 10-20s (CPU)
- **VRAM**: ~13GB (float16), ~7GB (4-bit quantized)

---

## Error Handling
- `FileNotFoundError`: Frame path invalid
- `JSONDecodeError`: Model output not valid JSON → fallback structure
- `[ILLEGIBLE]` >20%: Trigger OCR fallback (PaddleOCR)

---

*Approved Plan - Ready for Implementation*
