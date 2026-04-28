"""
GGUF VLM Engine using llama-cpp-python.

Optimized for 6GB VRAM with:
- Flash Attention
- Controlled n_gpu_layers
- Limited context window
- P-core only threading
- Pre-scaled images
"""

import os
import time
import json
import re
from typing import Dict, Optional, Union
from pathlib import Path

from PIL import Image

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

from .image_preprocessor import preprocess_image


class GGUFVLMDetectionError(Exception):
    """Exception raised when llama-cpp-python is not available."""
    pass


class GGUFVLMEngine:
    """
    GGUF-based VLM Engine for Qwen2-VL with 6GB VRAM optimization.

    Uses llama-cpp-python with flash attention, controlled GPU layers,
    and pre-scaled images for maximum performance.
    """

    # System prompt - same as existing SnapshotEngine for consistency
    SYSTEM_PROMPT = """
You are an expert Vision-Language Model analyst specializing in accessible STEM education.
Your ONLY job is to act as the objective "eyes" for a blind student. You must extract exact text, geometric shapes, spatial coordinates, and visual flow from the provided image.

CORE RULES:
1. SINGLE CATEGORY FOCUS: You will be given a TARGET CONTENT TYPE. You must ONLY extract information relevant to this specific category. Ignore all other unrelated diagrams to maintain strict focus.
2. CHAIN-OF-REGIONS: Scan the ENTIRE frame systematically (top-left, top-right, bottom-left, bottom-right).
3. STRUCTURAL DESCRIPTION REQUIREMENTS:
   - Provide spatial descriptions that build mental models for blind students.
   - Include reading order for navigation (left→right, top→bottom).
   - Describe movement/flow for graphs (not just data points).
   - Link all physical elements to show their spatial relationships.
4. DISTRACTION FILTERING: Ignore the teacher's face (unless actively pointing at a visual). Ignore decorative borders.
5. NO CONCEPTUAL REASONING: DO NOT guess the underlying scientific concepts. Just report what is physically drawn and written.
6. STRICT JSON OUTPUT: Output ONLY valid JSON. No markdown, no conversational filler.

OUTPUT FORMAT (JSON only):
{
  "content_type": "string (echo the requested target type)",
  "visual_analysis": {"details": "Extract components, curves, or elements as a simple, flat dictionary of strings"},
  "structural_description": "string (A complete spatial narrative adhering to the structural rules above)",
  "reading_order": ["array", "of", "steps to navigate the visual top-to-bottom, left-to-right"],
  "layout": "string (Overall physical arrangement)",
  "text_readout": "string (Full OCR text with absolute accuracy)",
  "spatial_map": "string (Where things are relative to each other)",
  "colors_styles": "string (Colors, line thicknesses, shading)",
  "missing_elements": null
}

If the frame contains no clear visual for the requested category (blurry, empty, irrelevant), set "missing_elements": "No clear STEM visual" and return immediately.
"""

    # Category-specific directives (from original SnapshotEngine)
    CATEGORY_DIRECTIVES = {
        "equation": """
CRITICAL: Extract the equation spatially and phonetically.
1. text: Read left-to-right, top-to-bottom. Convert notation to spoken form (x² → "x squared"). For fractions use "[numerator] over [denominator]".
2. description: Position in frame (e.g., "centered", "top right") and size relative to other elements.
3. reading_order: Step-by-step mental navigation (e.g., "Start with integral symbol, then lower limit...").
        """,

        "graph": """
CRITICAL: Extract axes and CURVE MOVEMENT physically.
1. axes: Extract exact text labels, ranges (min-max), units, and positions for x-axis and y-axis.
2. curves: Describe HOW the curve visually moves. Use active verbs ("rises steeply", "oscillates", "levels off"). Describe the geometric shape.
3. key_points: Extract exact (x,y) coordinates for intersections, peaks, or labeled points.
4. annotations: Note any legends, text, or arrows pointing to specific parts.
        """,

        "circuit": """
CRITICAL: Extract physical components and wiring topology.
1. components: List every visible component (battery, resistor, switch), its label (R1), and numerical value.
2. flow: Describe the physical path of the wires from start to finish, including branches.
3. topology: Describe the visual arrangement ("series loop", "parallel branches").
4. connections: Note where components connect to each other and any junction points.
        """,

        "diagram": """
CRITICAL: Extract structural nodes and connecting arrows.
1. elements: List every distinct shape/box, its text content, and its physical position.
2. relationships: List every arrow/connection. State exactly where it starts, where it points, and its label.
3. spatial_layout: Describe the visual hierarchy and grouping (e.g., "Flows from left to right", "Circular").
        """,

        "code": """
CRITICAL: Extract code with OCR preserving exact structure.
1. code_text: Transcribe exactly, preserving all indentation, brackets, and line breaks.
2. layout: Note visual organization (single file, split view, panel arrangement).
3. formatting: Note any syntax highlighting or bolded lines.
        """,

        "handwritten_notes": """
CRITICAL: OCR handwritten content with physical structure.
1. text: Transcribe all readable content. Mark [ILLEGIBLE] for unreadable portions.
2. layout: Describe the organization (title, sections, bullet points, margins).
3. reading_order: Explain how to mentally navigate the notes top-to-bottom.
        """,

        "biology": """
CRITICAL: Extract physical biological structures and callout lines.
1. main_structure: Describe the central organism/system drawn.
2. components: List all parts, their shapes, and spatial positions.
3. labels: List ALL callout text and trace exactly what physical part the line points to.
4. scale: Note if there is a scale bar or relative size differences.
        """,

        "chemistry": """
CRITICAL: Extract molecular structure with atomic connections.
1. atoms: List every element symbol and its relative physical position.
2. bonds: Describe the lines connecting symbols (single, double, solid wedge, dashed wedge).
3. molecular_geometry: Describe the overall 2D or 3D shape drawn on screen (e.g., "hexagonal ring").
        """,

        "physics": """
CRITICAL: Extract physical objects and vector arrows.
1. elements: Describe the physical objects drawn (e.g., "A square block resting on a diagonal ramp").
2. vectors: Describe every arrow. Provide origin point, direction (angle/up/down), and text label (e.g., "Arrow pointing straight down labeled mg").
3. measurements: Note any explicitly labeled distances, angles, velocities, or coordinates.
4. spatial_relationships: Describe contact points and how objects are positioned relative to each other.
        """,

        "text": """
CRITICAL: OCR text with structural organization.
1. text: Actual content with preserved line breaks and punctuation.
2. layout: Note the hierarchy (title, headings, body text, lists).
3. formatting: Note bold, italic, or colored text.
        """,

        "slide": """
CRITICAL: Extract slide content with hierarchical structure.
1. layout: Overall organization (title, sections, lists).
2. text: Transcribe all bullet points with their hierarchy, numbered lists, and footnotes.
3. tables_images: Briefly note the position of any tables or sub-images on the slide.
        """,

        "table": """
CRITICAL: Extract tabular data strictly by row and column.
1. headers: List all column and row headers.
2. structure: Describe the grid layout, noting any merged cells.
3. rows: Read the data out row by row, left to right.
        """,

        "unknown": """
CRITICAL: Describe the geometric and textual layout of all visible elements from top-left to bottom-right in a systematic manner.
"""
    }

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_gpu_layers: int = 24,    # CHANGED: 15 is safe for 6GB VRAM
        n_ctx: int = 4096,         # CHANGED: Fit the image tokens
        n_threads: int = 6,
        flash_attn: bool = False,  # CHANGED: Disabled for Windows stability
        verbose: bool = True
    ):
        """
        Initialize GGUF VLM Engine with 6GB VRAM optimizations.

        Args:
            model_path: Path to Qwen2-VL GGUF model file (.gguf)
            mmproj_path: Path to vision projector mmproj file (.gguf)
            n_gpu_layers: Number of layers to offload to GPU (15 for 6GB VRAM)
            n_ctx: Context window size (4096 to fit image tokens)
            n_threads: Number of CPU threads (6 for i5 P-cores)
            flash_attn: Enable Flash Attention 2 (disabled for stability)
            verbose: Enable verbose logging

        Raises:
            GGUFVLMDetectionError: If llama-cpp-python is not installed
        """
        if not LLAMA_CPP_AVAILABLE:
            raise GGUFVLMDetectionError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python --upgrade --no-cache-dir --force-reinstall "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        self.verbose = verbose

        # ADDED: Initialize the specialized Qwen Vision Chat Handler
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=self.mmproj_path,
            verbose=self.verbose
        )

        print(f"[GGUF VLM Engine] Initializing...")
        print(f"  Model: {os.path.basename(model_path)}")
        print(f"  MMProj: {os.path.basename(mmproj_path)}")
        print(f"  n_gpu_layers: {n_gpu_layers}")
        print(f"  n_ctx: {n_ctx}")
        print(f"  n_threads: {n_threads}")
        print(f"  flash_attn: {flash_attn}")

        # Load model with optimizations and the chat handler
        self.model = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,  # ADDED: This routes the image correctly
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            flash_attn=flash_attn,
            verbose=verbose,
            use_mmap=True,
            use_mlock=False,
            embedding=False,
            n_batch=512,
        )

        print(f"[GGUF VLM Engine] Model loaded successfully")

    def analyze_frame(
        self,
        image_path: str,
        content_type: str = "unknown",
        max_tokens: int = 2048,  # CHANGED: Increased to prevent JSON cutoff
        temperature: float = 0.1
    ) -> Dict:
        """
        Analyze a frame using GGUF VLM with pre-scaling.
        """
        import base64
        import io  # ADDED: Required for the image buffer

        # 1. Preprocess and pre-scale image (1024x1024 max for better OCR)
        try:
            image = preprocess_image(image_path, max_resolution=(1024, 1024))
        except Exception as e:
            print(f"[GGUF VLM Engine] Error preprocessing image: {e}")
            return self._error_response(str(e), content_type)

        # 2. Convert image to base64 data URL
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_base64}"

        # 3. Build prompt text
        prompt_text = self._build_prompt(content_type)

        # 4. Package into OpenAI-style messages list
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # 5. Generate response using create_chat_completion
        try:
            start_time = time.perf_counter()

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=2048,        # CHANGED: Increased from 768 to prevent JSON cutoff
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.0,     # CHANGED: Set to 1.0 so JSON keys can repeat safely
                stop=["###", "[END]"],  # CHANGED: Removed \n\n to prevent early cutoff
                response_format={"type": "json_object"}
            )

            generation_time = time.perf_counter() - start_time

            text = response["choices"][0]["message"]["content"].strip()
            print(f"[GGUF VLM Engine] Generated in {generation_time:.2f}s ({max_tokens} tokens)")
            print(f"[GGUF VLM Engine] Raw response (first 500 chars): {text[:500]}...")

            # 6. Parse response
            return self._parse_response(text, content_type)

        except Exception as e:
            # FIXED: Cleaned up the weird typo in the print statement
            print(f"[GGUF VLM Engine] Error generating response: {e}")
            return self._error_response(str(e), content_type)

    def _build_prompt(self, content_type: str) -> str:
        """Build prompt for VLM analysis with category-specific directives."""
        # Get category directive (from original SnapshotEngine)
        category_directive = self.CATEGORY_DIRECTIVES.get(
            content_type.lower(),
            self.CATEGORY_DIRECTIVES.get("unknown", "")
        )

        user_prompt = f"""
Analyze this STEM lecture frame and provide a structured description.

PRIMARY CONTENT TYPE: {content_type}

CATEGORY-SPECIFIC DIRECTIVES for {content_type}:
{category_directive}

OUTPUT REQUIREMENTS:
- Scan ENTIRE frame using Chain-of-Regions (2x2 grid approach)
- Detect ALL content types (equations, graphs, text, diagrams, etc.)
- Provide unified JSON output format with visual_analysis object
- Include structural_description, reading_order, and conceptual_hints
- Output ONLY valid JSON (no markdown, no filler)
"""
        return self.SYSTEM_PROMPT + user_prompt

    def _parse_response(self, text: str, content_type: str) -> Dict:
        """
        Bulletproof JSON parser for VLM output.
        Finds the actual JSON block, fixes minor syntax errors, and ensures 
        all required fields exist for the pipeline.
        """
        import json
        import re

        # 1. Extract JSON block from potentially chatty LLM text
        # Finds the first '{' and the last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
        else:
            json_str = text # Fallback to original text if no brackets found

        # 2. Fix common LLM syntax hallucinations (trailing commas)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[GGUF VLM Engine] Parse error: {e}")
            return self._error_response("JSON parse error", content_type, raw_text=text)

        # 3. Ensure required fields exist (Auto-fill missing data to prevent pipeline crashes)
        required_fields = [
            "content_type", "detected_types", "visual_analysis",
            "structural_description", "reading_order", "conceptual_hints"
        ]

        for field in required_fields:
            if field not in result:
                if field == "content_type":
                    result[field] = content_type
                elif field == "detected_types":
                    result[field] = [content_type]
                elif field == "visual_analysis":
                    result[field] = {}
                elif field == "reading_order":
                    result[field] = []
                else:
                    result[field] = ""

        # 4. Ensure backward compatibility for the rest of the SnapshotEngine fields
        old_fields = ["layout", "text_readout", "spatial_map", "colors_styles", "missing_elements"]
        for field in old_fields:
            if field not in result:
                if field == "missing_elements":
                    result[field] = None 
                else:
                    result[field] = ""

        # Normalize string "null" to actual Python None
        if result.get("missing_elements") == "null":
            result["missing_elements"] = None

        return result

    def _error_response(
        self,
        error_msg: str,
        content_type: str,
        raw_text: str = ""
    ) -> Dict:
        """Return error response structure."""
        print(f"[GGUF VLM Engine] Creating error response: {error_msg}")
        if raw_text:
            print(f"[GGUF VLM Engine] Raw text (first 200 chars): {raw_text[:200]}...")
        return {
            "content_type": content_type,
            "detected_types": [content_type],
            "visual_analysis": {},
            "structural_description": f"Error: {error_msg}",
            "reading_order": [],
            "conceptual_hints": "",
            "layout": raw_text or error_msg,
            "text_readout": raw_text or error_msg,
            "spatial_map": "",
            "colors_styles": "",
            "missing_elements": None,  # Don't set as error string, use None instead
            "_error": error_msg
        }

    def cleanup(self):
        """Release resources and free GPU memory."""
        # Step 1: Reset model context if available (llama-cpp-python method)
        if hasattr(self, 'model') and self.model is not None:
            try:
                if hasattr(self.model, 'reset'):
                    self.model.reset()
            except Exception as e:
                print(f"[GGUF VLM Engine] Warning during model reset: {e}")

            # Step 2: Delete chat handler if it exists
            if hasattr(self, 'chat_handler') and self.chat_handler is not None:
                try:
                    del self.chat_handler
                except Exception as e:
                    print(f"[GGUF VLM Engine] Warning deleting chat_handler: {e}")

            # Step 3: Delete the model reference
            del self.model

        # Step 4: Force garbage collection to release Python object references
        import gc
        gc.collect()

        # Step 5: Clear CUDA cache if torch is available (releases VRAM blocks)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception as e:
            print(f"[GGUF VLM Engine] Warning during CUDA cleanup: {e}")

        print("[GGUF VLM Engine] Cleanup complete")
