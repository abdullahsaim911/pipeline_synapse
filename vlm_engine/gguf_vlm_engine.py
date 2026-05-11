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
Your ONLY job is to act as the objective, literal "eyes" for a blind student.

CORE RULES:
1. LITERAL EXTRACTION ONLY: You are an optical sensor. Describe exactly what is physically drawn or written. DO NOT guess, infer, or invent conceptual meaning. DO NOT invent coordinates if they are not written.
2. PRIMARY + PERIPHERAL FOCUS: Focus on the TARGET CONTENT TYPE requested, but DO NOT ignore standalone text, equations, or labels floating around it. Capture all relevant on-screen information.
3. NO PARROTING: Use your own descriptive words based solely on the visual evidence.
4. DISTRACTION FILTERING: Ignore the teacher's face/body (unless actively pointing) and decorative video UI elements.
5. CHAIN-OF-REGIONS: Scan the frame systematically (top-left to bottom-right) so you don't miss peripheral text.
6. STRICT JSON OUTPUT: Output ONLY valid JSON. No markdown, no conversational filler.

OUTPUT FORMAT (JSON only):
{
  "content_type": "string (echo the requested target type)",
  "visual_analysis": {"details": "Extract all physical features, text, and elements based on the directives as a flat dictionary of strings"},
  "structural_description": "string (A complete spatial narrative of what is on the screen)",
  "reading_order": ["array", "of", "steps to navigate the visual top-to-bottom, left-to-right"],
  "layout": "string (Overall physical arrangement)",
  "text_readout": "string (Full OCR text with absolute accuracy)",
  "spatial_map": "string (Where things are relative to each other)",
  "colors_styles": "string (Colors, line thicknesses, shading)",
  "missing_elements": null
}

If the frame contains no clear visual for the requested category, set "missing_elements": "No clear STEM visual" and return immediately.
"""

    # Category-specific directives (from original SnapshotEngine)
    CATEGORY_DIRECTIVES = {
        "graph": """
CRITICAL FOCUS: Coordinate systems, vectors, and curves.
PERIPHERAL CAPTURE: Extract any equations or text written near the graph.
1. axes: Describe grid lines, axis lines, and extract exact numeric labels. If no numbers exist, explicitly state "Axes are unnumbered".
2. geometric_shapes: Describe the physical appearance of curves, straight lines, or vectors (arrows). State their color and direction. DO NOT invent coordinates.
3. associated_math: Transcribe any mathematical equations written inside or next to the graph exactly as they appear.
4. floating_text: OCR all standalone text/words on the screen. State exactly where this text is located relative to the geometric shapes.
        """,

        "equation": """
CRITICAL FOCUS: Mathematical formulas and notation.
PERIPHERAL CAPTURE: Extract any small diagrams or text notes pointing to the math.
1. math_text: Transcribe the math left-to-right, top-to-bottom. Convert notation to spoken English words.
2. layout: Describe how multiple lines of math are physically arranged.
3. annotations: If there are arrows pointing to parts of the equation, describe the arrow and transcribe the text it points to.
4. peripheral_drawings: If there is a small graph or shape next to the math, briefly describe its basic physical shape.
        """,

        "physics": """
CRITICAL FOCUS: Physical objects and force vectors.
PERIPHERAL CAPTURE: Extract kinematic/dynamic equations written around the drawing.
1. objects: Describe the physical items drawn.
2. force_vectors: Describe every arrow. State its origin point, direction, and exactly what text/symbol is written next to it. DO NOT guess what force it represents if unlabeled.
3. measurements: Extract explicitly written angles, masses, or distances. If none are written, state "No measurements provided".
4. associated_math: Transcribe any physics formulas or equations written anywhere on the screen.
        """,

        "circuit": """
CRITICAL FOCUS: Electrical components and wiring.
PERIPHERAL CAPTURE: Extract math/values written next to components.
1. topology: Describe the physical layout of the wires.
2. components: List every visible symbol and transcribe the exact text/value written next to it.
3. peripheral_text: Transcribe any standalone text or equations written on the board near the circuit.
        """,

        "diagram": """
CRITICAL FOCUS: Nodes, boxes, and connecting flow.
PERIPHERAL CAPTURE: Extract any code snippets or math inside the nodes.
1. nodes: List every distinct shape/box and accurately transcribe ALL text or math written inside it.
2. connections: Describe every line or arrow connecting the nodes. State exactly where it starts, where it ends, and any text written on the line itself.
3. overall_flow: Describe the visual direction of the diagram.
        """,

        "code": """
CRITICAL FOCUS: Programming syntax and IDE structure.
PERIPHERAL CAPTURE: Extract terminal outputs or architectural sketches.
1. code_block: Transcribe the code exactly, preserving indentation and brackets.
2. highlights: Note if specific lines are highlighted, bolded, or colored differently.
3. peripheral_visuals: If there is a terminal output window or a small architectural drawing next to the code, describe its contents exactly.
        """,

        "biology": """
CRITICAL FOCUS: Organic structures and callout lines.
PERIPHERAL CAPTURE: Extract text blocks explaining the structures.
1. main_shape: Describe the physical appearance of the biological drawing.
2. callouts: Trace every line/arrow. State what part of the drawing it points to, and transcribe the exact text label at the other end.
3. peripheral_text: Transcribe any paragraphs, lists, or standalone text written around the diagram.
        """,

        "chemistry": """
CRITICAL FOCUS: Molecular structures and bonds.
PERIPHERAL CAPTURE: Extract chemical reaction equations.
1. molecules: Describe the layout of element symbols and the types of lines connecting them.
2. reaction_math: Transcribe any chemical equations written on the screen.
3. annotations: Note any floating text, charges, or labels associated with the molecules.
        """,

        "text_and_slides": """
CRITICAL FOCUS: Textual hierarchy and bullet points.
PERIPHERAL CAPTURE: Note the presence of any decorative or supporting images.
1. main_text: OCR all text, maintaining the hierarchy of titles, subtitles, and bullet points.
2. emphasis: Note which words are underlined, bolded, or circled.
3. supporting_visuals: If there is a small chart or picture on the slide, provide a 1-sentence physical description of it.
        """,

        "unknown": """
CRITICAL FOCUS: Multiple distinct STEM elements sharing the same screen.
PERIPHERAL CAPTURE: The spatial boundaries and connections between these different elements.
1. zone_mapping: Scan the frame and divide it into physical spatial zones (e.g., left side, right side, top right). State exactly what type of content (equation, graph, physical drawing, text block) occupies each zone.
2. isolated_extraction: Go through each zone one by one. Describe the physical shapes, vectors, or axes in that specific zone exactly as they appear. DO NOT mix the descriptions of different zones together.
3. zoned_ocr: Transcribe all text, numbers, and mathematical equations. You MUST state exactly which zone the text belongs to. 
4. cross_connections: Describe any lines, arrows, or drawn visual cues that connect the content in one zone to the content in another zone.
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

        print(f"[GGUF VLM Engine] Initializing...")
        print(f"  Model: {os.path.basename(model_path)}")
        print(f"  MMProj: {os.path.basename(mmproj_path)}")
        print(f"  n_gpu_layers: {n_gpu_layers}")
        print(f"  n_ctx: {n_ctx}")
        print(f"  n_threads: {n_threads}")
        print(f"  flash_attn: {flash_attn}")

        # ==========================================
        # ADDED 1: The VRAM Sweeper (Kill Ollama)
        # ==========================================
        
        if os.name == 'nt':
            print("[GGUF VLM Engine] Sweeping VRAM to evict Mistral...")
            import subprocess
            import time
            subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Give Windows exactly 1.5 seconds to flush the physical memory
            time.sleep(1.5)
        # ==========================================

        # Initialize the specialized Qwen Vision Chat Handler
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=self.mmproj_path,
            verbose=self.verbose
        )

        # Load model with optimizations and the chat handler
        self.model = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,  # This routes the image correctly
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            flash_attn=flash_attn,
            verbose=verbose,
            use_mmap=False,
            use_mlock=False,
            embedding=False,
            n_batch=512,
            split_mode=2,  # ADDED 2: Forces strict loading to prevent "0 MiB Free" crashes
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
        """Release resources and free GPU memory completely."""
        print("[GGUF VLM Engine] Commencing aggressive cleanup...")
        
        # Step 1: Patch the llama-cpp-python library bug before closing
        if hasattr(self, 'model') and self.model is not None:
            try:
                # THE FIX: Inject the missing 'sampler' attribute to prevent the C++ crash
                if hasattr(self.model, '_model') and not hasattr(self.model._model, 'sampler'):
                    self.model._model.sampler = None
                    
                # Now it is safe to close the C++ model and dump the weights
                if hasattr(self.model, 'close'):
                    self.model.close()
            except Exception as e:
                print(f"[GGUF VLM Engine] Warning during model close: {e}")
                
            self.model = None

        # Step 2: Sever the Vision Handler to drop the 1.3 GB Vision Encoder
        if hasattr(self, 'chat_handler') and self.chat_handler is not None:
            self.chat_handler = None

        # Step 3: Double-Tap Garbage Collection
        import gc
        gc.collect()
        gc.collect()

        # Step 4: Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        print("[GGUF VLM Engine] Cleanup complete")