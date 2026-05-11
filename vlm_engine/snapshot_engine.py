"""
VLM Snapshot Engine (M2)

Universal Eye" for pipeline - analyzes complete video frames from STEM lectures.
Uses Qwen2-VL-7B-Instruct for visual structure analysis.
Supports both transformers and GGUF backends.
"""

import json
import os
import re
from typing import Dict, Optional

# Import Qwen2VLInterface which handles transformers
from vlm_interface import Qwen2VLInterface

# Try to import GGUF engine for llama-cpp-python support
try:
    from .gguf_vlm_engine import GGUFVLMEngine, GGUFVLMDetectionError
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    GGUFVLMEngine = None
    GGUFVLMDetectionError = Exception


class SnapshotEngine:
    """
    VLM Engine for analyzing STEM lecture frames.

    Loads Qwen2-VL-7B-Instruct model once and keeps it in memory
    for the duration of the pipeline loop.
    """

    # System prompt (constant) - Enhanced for multiple content type detection
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

OUTPUT FORMAT (JSON only):
{
  "content_type": "dominant content type",
  "detected_types": ["all", "types", "present"],
  "visual_analysis": {
    "graph": { ... },
    "equation": [ ... ],
    "circuit": { ... },
    "diagram": { ... },
    "code": { ... },
    "handwritten_notes": { ... },
    "biology": { ... },
    "chemistry": { ... },
    "physics": { ... },
    "text": { ... }
  },
  "structural_description": "Complete spatial narrative for mental model building",
  "reading_order": ["Step 1", "Step 2", "Step 3"],
  "conceptual_hints": "What this visual demonstrates conceptually",
  "layout": "Overall physical arrangement",
  "text_readout": "Linear text from ALL regions",
  "spatial_map": "Geometry/relationships in ALL regions",
  "colors_styles": "Visual descriptors (colors, line styles)",
  "missing_elements": "null if successful, 'No clear STEM visual' if irrelevant/blurry/empty"
}

If the frame contains no clear STEM visual content (blurry, empty, irrelevant),
set "missing_elements": "No clear STEM visual" and return immediately.
"""

    # Category-specific directives (enhanced for unified output format)
    CATEGORY_DIRECTIVES = {
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
        """,

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
        """,

        "circuit": """
CRITICAL: Extract complete circuit information focusing on FLOW and TOPOLOGY.

Provide in visual_analysis.circuit object:

1. components (Complete Inventory):
   For EACH component in array:
   a. type: battery, resistor, capacitor, inductor, diode, transistor, switch, etc.
   b. label: Component identifier (R1, C1, etc.)
   c. value: Numerical value with units if shown
   d. position: Where in the diagram
   e. state: If applicable (switch: open/closed)

2. flow (Current Path):
   - Start point, path description through components, direction, branches, end point

3. topology (Circuit Structure):
   - Overall arrangement: "series", "parallel", "series-parallel", "bridge", "ladder"
   - How components are grouped

4. connections (Wiring): How components connect to each other, junctions, grounds

5. purpose: What the circuit does or demonstrates
        """,

        "diagram": """
CRITICAL: Extract complete structural information focusing on RELATIONSHIPS.

Provide in visual_analysis.diagram object:

1. main_structure: Overall subject and position

2. elements (Complete Inventory):
   For EACH element in array:
   a. name/label, type, position, description
   b. text_content: any text inside or associated
   c. relationships: how it connects to other elements

3. relationships (Connections and Arrows):
   For EACH connection in array: from, to, direction, meaning, label

4. spatial_layout: Organization, grouping, reading order

5. process_flow (If Applicable): Start point, steps, decision points, end point

6. conceptual_meaning: What phenomenon/system it models, key principle
        """,

        "code": """
CRITICAL: Extract code with OCR preserving structure.

Provide in visual_analysis.code object:

1. code_text: Actual code with PRESERVED indentation and line breaks

2. language: Programming language identified from syntax

3. layout: Visual organization (single file, split view, panel arrangement)

4. structure: High-level organization (imports, classes, functions visible)

5. formatting: Syntax highlighting colors, special formatting

IMPORTANT CONSTRAINTS:
- Do NOT explain what code does
- Do NOT describe execution flow
- Just extract what is visible
        """,

        "handwritten_notes": """
CRITICAL: OCR handwritten content with structure.

Provide in visual_analysis.handwritten_notes object:

1. text: Readable content, mark [ILLEGIBLE] for unreadable portions

2. layout: Organization (title, sections, bullet points, diagrams, equations)

3. legibility_note: Overall legibility, percentage [ILLEGIBLE], handwriting style

4. content_types: What's mixed in (text, equations, diagrams, formulas, examples)

5. reading_order: How to mentally navigate the notes
        """,

        "biology": """
CRITICAL: Extract structural biological information with connections.

Provide in visual_analysis.biology object:

1. main_structure: Organism/system name and type

2. components (All Parts with Locations):
   For EACH component: name/label, type, position, description, function, relationships

3. element_links: How parts connect spatially and functionally

4. scale_relationships: Relative sizes, scale bar if present

5. labels: All callouts with text and what they point to

6. conceptual_meaning: Biological principle being illustrated
        """,

        "chemistry": """
CRITICAL: Extract molecular structure with atomic connections.

Provide in visual_analysis.chemistry object:

1. molecule: Name, formula, position

2. atoms (All Atoms with Positions):
   For EACH atom: element, symbol, position, size, charge

3. bonds (Connections):
   For EACH bond: from, to, type (single/double/triple), representation, angle

4. molecular_geometry: Shape, bond angles, 3D orientation, description

5. element_links: How atoms are connected in sequence, central atoms, ring structures

6. labels: Element symbols, bond angles, charges, additional text
        """,

        "physics": """
CRITICAL: Extract physics diagram with physical relationships.

Provide in visual_analysis.physics object:

1. main_diagram: Type, subject, position

2. elements (All Physical Objects):
   For EACH element: name/label, type, position, description, physical_properties, state

3. forces/arrows (All Vectors):
   For EACH force: name, label, position, direction, magnitude, meaning

4. spatial_relationships: Positions relative to each other, contact points, distances, angles

5. element_links: Physical connections, constraints, how forces transmit

6. conceptual_meaning: Physical law or principle, key variables, what it demonstrates

7. measurements: Distances, lengths, dimensions, angles, velocities, accelerations
        """,

        "text": """
CRITICAL: OCR text with structural organization.

Provide in visual_analysis.text object:

1. text: Actual content with preserved line breaks and punctuation

2. layout: Hierarchy (title, headings, body text), lists (bullet/numbered)

3. formatting: Bold, italic, colored text and their meanings, font sizes

4. structure: Logical organization, section breaks, indentation levels, visual flow

5. key_terms: Important vocabulary that's emphasized or defined
        """
    }

    def __init__(
        self,
        model_path: str = "f:\\Prototyping\\Synapse\\models\\Qwen2-VL-7B-Instruct-GGUF",
        device_map: str = "auto",
        use_4bit: bool = True,
        use_flash_attention: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_gguf: bool = True,  # NEW: Use GGUF by default
        n_gpu_layers: int = 24,  # NEW: For GGUF tuning
        n_ctx: int = 4096,  # NEW: Context window
        n_threads: int = 6  # NEW: CPU threads
    ):
        """
        Initialize VLM Engine.

        Args:
            model_path: Local path to Qwen2-VL model
            device_map: Device allocation strategy (default: auto)
            use_4bit: Enable 4-bit quantization
            use_flash_attention: Enable Flash Attention 2
            min_pixels: Minimum image resolution
            use_gguf: Try GGUF backend first (default: True)
            n_gpu_layers: Number of layers to offload to GPU for GGUF
            n_ctx: Context window size for GGUF
            n_threads: Number of CPU threads for GGUF
        """
        self.model_name = model_path  # Store for logging
        self.model_path = model_path
        self.use_gguf = use_gguf

        # Try GGUF backend first if enabled
        if use_gguf and GGUF_AVAILABLE:
            try:
                # GGUF files should be directly in model_path directory
                gguf_model_path = os.path.join(model_path, "Qwen2-VL-7B-Instruct-Q4_K_M.gguf")
                mmproj_path = os.path.join(model_path, "mmproj-Qwen2-VL-7B-Instruct-f16.gguf")

                if os.path.exists(gguf_model_path) and os.path.exists(mmproj_path):
                    print("[Snapshot Engine] Using GGUF backend")
                    self.vlm_interface = GGUFVLMEngine(
                        model_path=gguf_model_path,
                        mmproj_path=mmproj_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        n_threads=n_threads
                        # REMOVED: flash_attn=True (Let the engine use its safe default of False)
                    )
                    print("[Snapshot Engine] VLM Engine initialized (GGUF)")
                    return
                else:
                    print("[Snapshot Engine] GGUF files not found, falling back to transformers")
                    raise FileNotFoundError("GGUF files not found")
            except Exception as e:
                print(f"[Snapshot Engine] 7B GGUF initialization failed: {e}")
                print("[Snapshot Engine] Falling back to 2B GGUF backend")

                # Fallback: 2B GGUF model
                fallback_dir = os.path.normpath(os.path.join(
                    os.path.dirname(model_path), "Qwen2-VL-2B-Instruct-GGUF"
                ))
                fallback_model = os.path.join(fallback_dir, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
                fallback_mmproj = os.path.join(fallback_dir, "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")

                try:
                    if os.path.exists(fallback_model) and os.path.exists(fallback_mmproj):
                        self.vlm_interface = GGUFVLMEngine(
                            model_path=fallback_model,
                            mmproj_path=fallback_mmproj,
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=n_ctx,
                            n_threads=n_threads
                        )
                        self.model_name = fallback_dir
                        print("[Snapshot Engine] VLM Engine initialized (2B GGUF fallback)")
                        return
                    else:
                        print(f"[Snapshot Engine] 2B GGUF files not found at {fallback_dir}")
                except Exception as fallback_e:
                    print(f"[Snapshot Engine] 2B GGUF fallback also failed: {fallback_e}")

                raise RuntimeError("Both 7B and 2B GGUF backends failed to initialize")
        print(f"[Snapshot Engine] 4-bit quantization: {use_4bit}")
        print(f"[Snapshot Engine] Flash Attention 2: {use_flash_attention}")

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

    def _parse_json_response(self, text: str) -> Dict:
        """
        Parse VLM output into structured JSON with new unified format.

        Handles:
        - Markdown blocks (```json ... ```)
        - Extra whitespace
        - Trailing commas
        - Missing keys (uses defaults)
        - New unified format fields

        Args:
            text: Raw output from VLM

        Returns:
            Parsed dictionary with required keys (old + new format)
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
            print(f"[Snapshot Engine] Failed to parse JSON: {e}")
            print(f"[Snapshot Engine] Raw output: {text}")
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

        # Set content_type if not present (use first detected type or unknown)
        if "content_type" not in result:
            detected = result.get("detected_types", [])
            result["content_type"] = detected[0] if detected else "unknown"

        # Set detected_types if not present (single type from content_type)
        if "detected_types" not in result or not result["detected_types"]:
            result["detected_types"] = [result.get("content_type", "unknown")]

        # Set visual_analysis if not present (empty object)
        if "visual_analysis" not in result:
            result["visual_analysis"] = {}

        # Set structural_description if not present (use layout)
        if "structural_description" not in result:
            result["structural_description"] = result.get("layout", "")

        # Set reading_order if not present (empty array)
        if "reading_order" not in result:
            result["reading_order"] = []

        # Set conceptual_hints if not present (empty string)
        if "conceptual_hints" not in result:
            result["conceptual_hints"] = ""

        # Ensure backward compatibility fields exist
        old_required_keys = ["layout", "text_readout", "spatial_map", "colors_styles", "missing_elements"]
        for key in old_required_keys:
            if key not in result:
                if key == "missing_elements":
                    result[key] = None  # Use actual None, not empty string
                else:
                    result[key] = ""  # Default to empty string

        # Also convert string "null" to actual None for missing_elements
        if result.get("missing_elements") == "null":
            result["missing_elements"] = None

        return result

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
            # GGUF engine has its own analyze_frame that returns parsed result
            return self.vlm_interface.analyze_frame(frame_path, primary_type)

        # Transformers backend (uses vlm_interface.analyze method)
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

        # Check if >20% of text is [ILLEGIBLE] for equation/handwritten_notes
        illegible_count = text_readout.count("[ILLEGIBLE]")
        text_length = len(text_readout.replace("[ILLEGIBLE]", "").strip())

        if text_length > 0 and (illegible_count / text_length) > 0.2:
            if primary_type.lower() in ["equation", "handwritten_notes"]:
                result["missing_elements"] = "OCR fallback required"
                print(f"[Snapshot Engine] High [ILLEGIBLE] content detected for {primary_type}")

        return result

    def cleanup(self):
        """
        Clean up VLM Engine resources.

        Releases GPU memory and deletes model references.
        """
        print("[Snapshot Engine] Cleaning up...")
        self.vlm_interface.cleanup()
        print("[Snapshot Engine] Cleanup complete")
