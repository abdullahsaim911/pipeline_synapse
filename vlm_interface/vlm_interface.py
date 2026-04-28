"""
VLM Interface Module (M5a)

Transformers wrapper for Qwen2-VL-7B-Instruct.
Handles image loading, prompt formatting, and generation.
"""

import json
import re
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, Optional

from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# AutoAWQ removed - using standard transformers quantization instead


@dataclass
class VLMResponse:
    """Response from VLM generation."""
    text: str
    model: str
    total_duration_ms: int
    prompt_eval_count: int
    eval_count: int


class VLMInterfaceError(Exception):
    """Base exception for VLM Interface errors."""
    pass


class VLMTimeoutError(VLMInterfaceError):
    """Timeout during VLM generation."""
    pass


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

    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

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
     - Greek letters: "pi" not "pi", "theta" not "theta"

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
        self.model_name = model_name
        self.device = None
        self.dtype = torch.float16

        is_local_model = os.path.isdir(model_name)

        print(f"[VLM Interface] Loading model: {model_name}")
        print(f"[VLM Interface] Device map: {device_map}, mode: {'float16+CPU offload' if use_4bit else 'float16'}")

        model_kwargs: Dict = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "max_memory": {0: "5.5GB", "cpu": "32GB"},
            "torch_dtype": torch.float16,
        }

        # Use balanced device map for CPU offloading (more stable than 4-bit quantization)
        if use_4bit and torch.cuda.is_available():
            model_kwargs["device_map"] = "balanced"
            print("[VLM Interface] Using balanced device map for CPU offloading (GPU + CPU hybrid)")

        if is_local_model:
            model_kwargs["local_files_only"] = True
            print(f"[VLM Interface] Using local model from: {model_name}")
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Processor: set pixel bounds to cap image memory usage.
        # Balanced resolution for 6GB GPU - maintains OCR quality
        # 256*22*22=124K min_pixels, 896*22*22=434K max_pixels
        # ~27% reduction from original but maintains fine detail
        processor_kwargs: Dict = {
            "min_pixels": min_pixels if min_pixels is not None else 256 * 22 * 22,
            "max_pixels": max_pixels if max_pixels is not None else 896 * 22 * 22,
        }
        if is_local_model:
            processor_kwargs["local_files_only"] = True

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        self.device = next(self.model.parameters()).device

        print(f"[VLM Interface] Model loaded on: {self.device}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[VLM Interface] VRAM used after load: {allocated:.1f} GB")

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

        # FIX: was `max(image.size) > max_size * max_size` (1,806,336) — never triggered
        max_size = 1344
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))

        return image

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
        # FIX: this method was entirely missing — snapshot_engine.py calls vlm_interface.analyze()
        image = self._load_image(image_path_or_pil)

        if generation_params is None:
            generation_params = {}

        gen_kwargs = {
            "max_new_tokens": generation_params.get("max_new_tokens", 512),
            "temperature": generation_params.get("temperature", 0.1),
            "do_sample": generation_params.get("do_sample", False),
        }

        # Qwen2-VL multimodal message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info via qwen_vl_utils (handles image tensors correctly)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim input tokens so we only decode newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generation_time_ms = int((time.perf_counter() - start_time) * 1000)
        generated_text = output_text[0].strip()

        # Free intermediate activation tensors after each frame.
        # On 6GB GPU this prevents fragmentation across sequential analyze() calls.
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return VLMResponse(
            text=generated_text,
            model=self.model_name,
            total_duration_ms=generation_time_ms,
            prompt_eval_count=inputs.input_ids.shape[1],
            eval_count=len(generated_ids_trimmed[0]),
        )

    def _format_prompt(
        self,
        content_type: str,
        frame_path: Optional[str] = None,
        chain_of_regions: bool = True
    ) -> str:
        """
        Build appropriate prompt for each content type.

        Used by the standalone analyze_frame() convenience method.

        Args:
            content_type: Type of content (equation, graph, diagram, etc.)
            frame_path: Optional frame file path (for context)
            chain_of_regions: Whether to append Chain-of-Regions instruction

        Returns:
            Formatted prompt string
        """
        content_prompts = {
            "equation": """
[EQUATION ANALYSIS]
Read the equation character by character from left to right, top to bottom.
Convert ALL mathematical symbols to full words:
- Use "squared" not "^2", "cubed" not "^3", "subscript" not "_"
- Use "one-half" not "1/2", "three-fourths" not "3/4"
- Use "pi" not "pi symbol", "theta" not "theta symbol"
- Use "phi" not "phi symbol"
- For integrals: "integral of" not the integral symbol

CRITICAL: Report the equation EXACTLY as text, no LaTeX unless explicitly present.
OCR each character individually - don't guess mathematical structure.
""",

            "graph": """
[GRAPH ANALYSIS]
1. First, describe axes: X-axis label, Y-axis label, and any Z-axis if 3D.
2. Then describe overall shape: line graph, bar chart, scatter plot, etc.
3. For 3D graphs: describe x/y/z planes and their relationships.
4. Identify key data points: peaks, troughs, intersections, trends.
5. Mention units and scale if visible.
""",

            "diagram": """
[DIAGRAM ANALYSIS]
1. Describe overall layout and spatial arrangement.
2. List ALL components in a logical order (e.g., top-left to bottom-right).
3. Identify and describe ALL arrows/connections between components.
4. Read ALL labels and text within the diagram.
5. OCR and read any numeric values with precision.
""",

            "circuit": """
[CIRCUIT ANALYSIS]
1. Identify ALL components: resistors, capacitors, inductors, sources, grounds.
2. Trace ALL connections/wires between components.
3. Read ALL labels and values with units (e.g., "10kOhm resistor").
4. Describe circuit topology (series, parallel, bridge).
""",

            "code": """
[CODE ANALYSIS]
1. Read code line-by-line. Preserve exact indentation.
2. Identify language: Python, Java, C++, MATLAB, etc.
3. Describe logic flow and algorithms.
4. Read ALL function names and variable names.
5. OCR any comments for additional context.
""",

            "table": """
[TABLE ANALYSIS]
1. Describe overall table structure (rows, columns, headers).
2. List ALL headers with their data.
3. Identify data patterns across rows and columns.
4. Read ALL cell contents with values.
5. Mention trends or comparisons if evident.
""",

            "chemistry": """
[CHEMISTRY ANALYSIS]
1. Identify ALL chemical structures: molecules, bonds, reactions.
2. Read ALL chemical formulas with proper formatting.
3. Describe processes and experimental setup if visible.
4. Mention states (solid, liquid, gas) and phase changes.
""",

            "biology": """
[BIOLOGY ANALYSIS]
1. Identify ALL biological structures: cells, organelles, organisms.
2. Describe structures and spatial relationships.
3. Read ALL labels and annotations within the image.
4. Describe biological processes (mitosis, photosynthesis) if visible.
""",

            "slide": """
[SLIDE ANALYSIS]
1. Describe overall slide layout: title, sections, lists.
2. List ALL bullet points and numbered items.
3. Read ALL text in hierarchical order.
4. Identify emphasis markers (bold, colors, highlighting).
5. Describe any images, charts, or diagrams within the slide.
""",

            "text": """
[TEXT ANALYSIS]
1. Read ALL visible text character by character.
2. Preserve line breaks and paragraphs.
3. Identify any headings or emphasis.
""",

            "unknown": """
[GENERIC ANALYSIS]
Describe all visible elements in the frame in a systematic manner.
""",
        }

        # FIX: was named `system_prompt` and never referenced when building full_prompt
        content_directive = content_prompts.get(content_type, content_prompts["unknown"])

        user_prompt = f"""
Analyze this STEM educational frame.

Frame type: {content_type}
{f'Frame path: {frame_path}' if frame_path else ''}

{content_directive}

Analysis Requirements:
1. Provide structured JSON output with these fields:
   - visual_structure: Overall layout and organization
   - elements_found: List of all specific items (labels, arrows, text, values)
   - content_type: Best fit for the main content
   - ocr_text: ALL visible text character-by-character
   - missing_elements: Only if frame is blank/uniform (set to true)

2. For equations/mathematics: Read EVERY character left-to-right, top-to-bottom.
   Convert symbols to full words (not LaTeX).
   Report exact equation text.

3. OCR Requirement: Read text character-by-character.

Output Format (strict JSON):
{{
    "visual_structure": "...",
    "elements_found": [...],
    "content_type": "...",
    "ocr_text": "...",
    "missing_elements": false
}}
"""

        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}\n\nRespond with ONLY the JSON."

        if chain_of_regions:
            full_prompt += "\n\nIMPORTANT: Use Chain-of-Regions scanning method:"
            full_prompt += "\nDivide frame into 2x2 grid (top-left, top-right, bottom-left, bottom-right)."
            full_prompt += "\nSystematically analyze each quadrant."
            full_prompt += "\nStart from edges, work inward toward center."
            full_prompt += "\nReport content in each quadrant separately."

        return full_prompt

    def analyze_frame(
        self,
        image_path: str,
        content_type: str = "unknown"
    ) -> Dict:
        """
        Convenience wrapper: format prompt for content_type, run inference, parse result.

        Note: SnapshotEngine.analyze_frame() calls analyze() directly with its own prompt.
        This method is for standalone use of the interface.

        Args:
            image_path: Path to frame image
            content_type: Type of content (equation, graph, etc.)

        Returns:
            Parsed VLM response as dictionary
        """
        # FIX: entire old body was broken — used undefined vars (content_prompts, frame_path,
        # chain_of_regions, full_prompt) and had dead unreachable inference code after return.
        self._validate_image_path(image_path)
        prompt = self._format_prompt(content_type, image_path, chain_of_regions=True)
        response = self.analyze(image_path, prompt)
        return self._parse_vlm_response(response.text, image_path, content_type)

    def _parse_vlm_response(
        self,
        response_text: str,
        image_path: str,
        content_type: str
    ) -> Dict:
        """Parse VLM JSON response and return structured data."""
        # FIX: json and re are now imported at module level (were missing)
        # FIX: removed `import re` and `import json as stdlib` from inside this method

        # Strip markdown code fences
        text = re.sub(r'```(?:json)?\s*\n?', '', response_text)
        text = re.sub(r'```\s*\n?', '', text).strip()

        # Remove trailing commas before closing braces (common model output issue)
        text = re.sub(r',\s*}', '}', text)

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract a JSON object from a larger response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            # FIX: old code used `str(e)` but `e` was not bound in bare `except json.JSONDecodeError:`
            return {
                "visual_structure": response_text,
                "elements_found": [],
                "content_type": content_type,
                "ocr_text": "",
                "missing_elements": False,
                "_parse_failed": True,
            }

        required_fields = ["visual_structure", "elements_found", "content_type", "ocr_text"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = f"[MISSING: {field}]"

        # Normalize missing_elements to bool
        if parsed.get("missing_elements") in ("true", True, 1):
            parsed["missing_elements"] = True
        else:
            parsed["missing_elements"] = False

        return parsed

    def cleanup(self):
        """
        Release GPU memory and delete model references.

        Called by SnapshotEngine.cleanup() at end of pipeline.
        """
        # FIX: method was entirely missing — snapshot_engine.py calls vlm_interface.cleanup()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VLM Interface] Cleanup complete")


if __name__ == "__main__":
    vlm = Qwen2VLInterface()

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"Device Count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print("\n*** GPU IS AVAILABLE ***")
        print("VLM will run on GPU for much faster inference.")
    else:
        print("\n*** CPU ONLY - NO GPU ***")
        print("VLM will run on CPU (slower).")
