"""
VLM Snapshot Engine (M2)

Universal Eye" for pipeline - analyzes complete video frames from STEM lectures.
Uses Qwen2-VL-7B-Instruct for visual structure analysis.
Supports both transformers and GGUF backends.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from category_normalization import normalize_category_labels, prompt_category_for
from modal_inference.config import (
    DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from visual_evidence import (
    COMPACT_OBSERVATION_SCHEMA,
    adapt_legacy_snapshot,
    assemble_compact_qwen_observation,
    normalize_compact_relations,
)
from stage_tracing import trace_stage

# Lightweight provider constants and response types (no PyTorch/Transformers eager load)
from vlm_interface import (
    PROVIDER_ENV_VAR,
    PROVIDER_LOCAL,
    PROVIDER_MODAL,
    VALID_PROVIDERS,
    VLMResponse,
    VLMInterfaceError,
    VLMAuthenticationError,
    VLMContextBudgetError,
    VLMTimeoutError,
    VLMRemoteError,
)


logger = logging.getLogger(__name__)


def print(*values, **_kwargs):  # noqa: A001
    """Keep legacy VLM diagnostics available only at DEBUG level."""
    logger.debug(" ".join(str(value) for value in values))


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

    # One compact observation contract is used for every Task 2B category.
    # These assignments replace the active legacy prompt while local GGUF code
    # remains isolated in gguf_vlm_engine.py.
    SYSTEM_PROMPT = f"""
Analyze ONLY the supplied single frame of an educational video. Return ONLY JSON
conforming to the existing {COMPACT_OBSERVATION_SCHEMA} contract. First identify
distinct meaningful visual regions across the entire image and their reading order;
do not force the image into arbitrary quadrants. Extract visible facts, not a
learner-facing explanation. For each region capture its applicable CT categories,
important elements, exact readable labels, symbols, numbers and units, positions,
and supported relations. Describe structural organization with a short overview,
orientation, containment, hierarchy, topology and directional relations. Use short
local IDs consistently, including for critical relation or arrow labels and exact
associations.

Do not assert a connection from proximity or causation from co-location. Report
unreadable or ambiguous labels, intersections and attachments explicitly and
preserve partial evidence. Do not claim previous or future states, motion, or an
action's success from one frame. Never fabricate ticks, measurements, words,
operators, anatomical labels, teacher intent, hierarchy, junctions, arrows or
connections. Use [ILLEGIBLE] only when text is visible but unreadable. Avoid
duplicate prose when structured fields already encode a fact. Put the most
educationally consequential supported evidence first; do not pad decorative facts.

OUTPUT FORMAT (JSON only): return exactly one object on a single line as minified
JSON: no indentation, no newlines between tokens and no spaces after ":" or ",".
Required keys are shown below. Omit every other key whose value would be null, an
empty array or an empty object; an omitted key means null or empty.
{{"schema":"{COMPACT_OBSERVATION_SCHEMA}","status":"complete|partial|failed","evidence_status":"supported|partially_supported|ambiguous_or_conflicting|unavailable_or_unreadable","categories":["CT-01"],"regions":[{{"ref":"r1","categories":["CT-01"],"evidence_status":"supported","elements":[{{"ref":"e1","kind":"specific visible kind","evidence_status":"supported"}},{{"ref":"e2","kind":"specific visible kind","evidence_status":"supported"}}]}}],"relations":[{{"ref":"q1","source":"e1","target":"e2","type":"left_of|right_of|above|below|contains|inside|grouped_with|connected_to|points_to|leads_to|branches_to|depends_on|corresponds_to|parent_of|child_of|precedes|follows|intersects|parallel_to|perpendicular_to|bonded_to|labels|custom:specific_semantic","evidence_status":"supported"}}],"structure":{{"overview":"grounded structural overview"}},"exact":[{{"text":"exact visible content","status":"supported","notation":"plain|math|code|chemical|other","source_ref":"e1"}}]}}

Optional keys, included only when they carry a value:
root: layout, details, uncertainties[{{target:"e1|r1|scene",reason,impact}}], missing[].
region: position, bbox, reading_order_index, description.
element: categories, label, text, description, position, bbox, source_ref, attributes,
parent_ref, child_refs. relation: label, directed, cross_category, source_refs.
structure: orientation, components, relations, reading_order, hierarchy, topology,
uncertainties.

Rules: refs are short local references, not backend IDs. Every relation endpoint,
structure reference, parent/child reference, exact source_ref and uncertainty target
must name a declared ref (or "scene" where allowed). Put each grounded relationship
in root relations once; structure.relations contains only its relation refs. Use bbox
only for grounded normalized [x_min,y_min,x_max,y_max] values in [0,1], otherwise
null. Keep exact math, code, labels, values, units, signs, indentation, branch labels,
and connector direction. Exact text must point to the element or region that visibly
contains it. Preserve partial evidence instead of guessing. Under output pressure
prioritize: valid JSON/status, essential regions/elements, exact critical content,
essential topology/spatial relations, category details, then secondary context.
Never call truncated or materially incomplete evidence complete. Do not repeat a
fact across fields. Output minified JSON only, with no markdown.
"""

    CANONICAL_CATEGORY_DIRECTIVES = {
        "CT-01": (
            "In details.chart capture chart type; every meaningful x/y axis label and unit; "
            "only visible tick scales or ranges; series-to-legend associations; visible "
            "points, intersections, thresholds, annotations and trends. Do not infer values "
            "without readable scales; line direction is spatial, not temporal motion."
        ),
        "CT-02": (
            "In details.mathematics preserve exact notation (=, ≠, ≤, ≥, −, "
            "superscripts/subscripts), grouping, fraction numerator/denominator, integral or "
            "sum bounds, matrix rows, every visible derivation line and its order. Do not "
            "silently correct suspicious mathematics."
        ),
        "CT-03": (
            "In details.scientific_structure inventory labeled components and orientation, "
            "leader-line label targets, containment and only confirmed contact/connectivity. "
            "For circuits preserve terminals, wires, junctions versus crossings, branches, "
            "switch state and visible values; for chemistry preserve atoms, bond types and "
            "charges; for biology preserve compartments, parts and label associations. "
            "Distinguish adjacency, custom:part_of, diagram arrows and electrical connections. Do "
            "not invent biological function or current direction."
        ),
        "CT-04": (
            "In details.flow_network capture node labels/shapes, start/end, each directed "
            "edge source and target, exact Yes/No or other branch labels, merges, cycles and "
            "grouping. If arrow orientation or attachment is unclear, record uncertainty "
            "instead of an edge."
        ),
        "CT-05": (
            "In details.table capture headers and multi-level headers, row/column indexes and "
            "units, and important cells with correct header associations. Preserve merged "
            "cells and distinguish blank or illegible cells from zero; prioritize association "
            "correctness over row quantity."
        ),
        "CT-06": (
            "CRITICAL: OCR handwritten content and structured text exactly. In "
            "details.structured_text preserve exact headings, bullets or steps and their "
            "hierarchy/order, highlighted terms, formulas, warnings, negation, quotes and "
            "meaningful punctuation."
        ),
        "CT-07": (
            "In details.geometry capture labeled points, segments, lines and vertices; explicit "
            "right-angle, parallel and equality marks; lengths, angle values, containment and "
            "incidence. Never infer exact measures from apparent proportions."
        ),
        "CT-08": (
            "In details.programming preserve exact displayed code, line breaks, indentation, "
            "names, punctuation and operator variants such as > versus >= and = versus ==; "
            "record the displayed language and terminal output separately. Do not assert that "
            "the code caused the output without supporting evidence."
        ),
        "CT-09": (
            "In details.interface capture visible controls, labels, panels, selections, status "
            "badges, dialogs, typed values and toggle states. Do not fabricate click history, "
            "save success or network results."
        ),
        "CT-10": (
            "In details.map capture labeled places, marked compass orientation, legend-to-symbol "
            "associations, routes, adjacency, scale labels and qualitative relative positions. "
            "Do not invent distances or coordinates."
        ),
        "CT-11": (
            "In details.observational_image capture observable form, texture, color, distribution "
            "and comparisons between co-present regions, while marking identity uncertainty. "
            "Do not add diagnoses or inferred mechanisms."
        ),
        "CT-12": (
            "In details.physical_scene capture objects, supports and contacts, positions, arrows "
            "or force labels, orientations, and distinguish a depicted action symbol from an "
            "observed change. Do not invent dynamics from a static frame."
        ),
    }

    CATEGORY_DIRECTIVES = {
        "graph": CANONICAL_CATEGORY_DIRECTIVES["CT-01"],
        "equation": CANONICAL_CATEGORY_DIRECTIVES["CT-02"],
        "circuit": CANONICAL_CATEGORY_DIRECTIVES["CT-03"],
        "biology": CANONICAL_CATEGORY_DIRECTIVES["CT-03"],
        "chemistry": CANONICAL_CATEGORY_DIRECTIVES["CT-03"],
        "physics": CANONICAL_CATEGORY_DIRECTIVES["CT-03"],
        "diagram": "Use the generic contract; do not force an ambiguous diagram into one category.",
        "flowchart": CANONICAL_CATEGORY_DIRECTIVES["CT-04"],
        "table": CANONICAL_CATEGORY_DIRECTIVES["CT-05"],
        "text": CANONICAL_CATEGORY_DIRECTIVES["CT-06"],
        "handwritten_notes": CANONICAL_CATEGORY_DIRECTIVES["CT-06"],
        "geometry": CANONICAL_CATEGORY_DIRECTIVES["CT-07"],
        "code": CANONICAL_CATEGORY_DIRECTIVES["CT-08"],
        "software_interface": CANONICAL_CATEGORY_DIRECTIVES["CT-09"],
        "map": CANONICAL_CATEGORY_DIRECTIVES["CT-10"],
        "observational_image": CANONICAL_CATEGORY_DIRECTIVES["CT-11"],
        "physical_illustration": CANONICAL_CATEGORY_DIRECTIVES["CT-12"],
    }

    def _create_failed_snapshot(self, error_msg: str, primary_type: str = "unknown") -> Dict:
        """
        Return a normalized failure snapshot dictionary.

        Guarantees all required fields are present with correct types so that
        downstream consumers (synthesizer, QA, orchestrator) can detect the failure
        reliably without crashing or generating hallucinated visual evidence.
        """
        clean_type = primary_type if primary_type else "unknown"
        snapshot = {
            "_error": True,
            "_error_detail": str(error_msg),
            "content_type": clean_type,
            "detected_types": [clean_type],
            "visual_analysis": {},
            "structural_description": "",
            "reading_order": [],
            "conceptual_hints": "",
            "layout": "",
            "text_readout": "",
            "spatial_map": "",
            "colors_styles": "",
            "missing_elements": f"VLM analysis failed: {error_msg}",
        }
        snapshot["visual_evidence"] = adapt_legacy_snapshot(snapshot).to_schema_dict()
        return snapshot

    def __init__(
        self,
        model_path: str = "f:\\Prototyping\\Synapse\\models\\Qwen2-VL-7B-Instruct-GGUF",
        device_map: str = "auto",
        use_4bit: bool = True,
        use_flash_attention: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_gguf: bool = True,
        n_gpu_layers: int = 24,
        n_ctx: int = 4096,
        n_threads: int = 6,
        provider: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize VLM Engine.

        Supports:
        - Remote Modal VLM backend (provider="modal")
        - Local GGUF / Transformers backend (provider="local")
        """
        if provider is None:
            provider = os.environ.get(PROVIDER_ENV_VAR, PROVIDER_MODAL).strip().lower()
        else:
            provider = provider.strip().lower()

        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported VLM provider '{provider}'. Must be one of: {sorted(VALID_PROVIDERS)}"
            )

        self.provider = provider

        if self.provider == PROVIDER_MODAL:
            from vlm_interface.modal_vlm_interface import ModalVLMInterface
            print("[Snapshot Engine] Initializing remote Modal VLM backend...")
            configured_tokens = max_tokens
            if configured_tokens is None:
                configured_tokens = int(
                    os.environ.get(
                        "SYNAPSE_VLM_STRUCTURED_MAX_TOKENS",
                        os.environ.get(
                            "SYNAPSE_MODAL_MAX_TOKENS",
                            str(DEFAULT_STRUCTURED_MAX_TOKENS),
                        ),
                    )
                )
            if not 1 <= configured_tokens <= MAX_OUTPUT_TOKENS:
                raise ValueError(
                    "Structured VLM output token limit must be between 1 and "
                    f"{MAX_OUTPUT_TOKENS}; received {configured_tokens}."
                )
            self.structured_max_tokens = configured_tokens
            self.vlm_interface = ModalVLMInterface(
                endpoint_url=endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_tokens=self.structured_max_tokens,
            )
            self.model_name = self.vlm_interface.model_name
            self.model_path = "modal"
            self.use_gguf = False
            print(f"[Snapshot Engine] VLM Engine initialized (Modal: {self.model_name})")
            return

        # Local provider initialization (lazy imports)
        print("[Snapshot Engine] Initializing local VLM backend...")
        try:
            from .gguf_vlm_engine import GGUFVLMEngine, GGUFVLMDetectionError
            gguf_available = True
        except ImportError:
            gguf_available = False
            GGUFVLMEngine = None
            GGUFVLMDetectionError = Exception

        self.model_name = model_path
        self.model_path = model_path
        self.use_gguf = use_gguf

        # Try GGUF backend first if enabled
        if use_gguf and gguf_available:
            try:
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
                    )
                    print("[Snapshot Engine] VLM Engine initialized (GGUF)")
                    return
                else:
                    print("[Snapshot Engine] GGUF files not found, falling back to transformers")
                    raise FileNotFoundError("GGUF files not found")
            except Exception as e:
                print(f"[Snapshot Engine] 7B GGUF initialization failed: {e}")
                print("[Snapshot Engine] Falling back to 2B GGUF backend")

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

        # Transformers backend fallback for local provider
        from vlm_interface import Qwen2VLInterface
        print(f"[Snapshot Engine] 4-bit quantization: {use_4bit}")
        print(f"[Snapshot Engine] Flash Attention 2: {use_flash_attention}")
        self.vlm_interface = Qwen2VLInterface(
            model_name=model_path,
            device_map=device_map,
            use_4bit=use_4bit,
            use_flash_attention=use_flash_attention,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

    def _build_prompt(
        self,
        primary_type: str,
        category_uncertainty: Optional[str] = None,
        canonical_category_ids: Optional[list[str]] = None,
    ) -> str:
        """Build the common compact contract plus only relevant category rules."""
        category_ids = list(canonical_category_ids or [])
        if not category_ids:
            category_ids = list(
                normalize_category_labels(primary_type).canonical_categories
            )
        category_ids = [
            category for category in category_ids
            if category in self.CANONICAL_CATEGORY_DIRECTIVES
        ]
        category_directives = "\n".join(
            f"- {category}: {self.CANONICAL_CATEGORY_DIRECTIVES[category]}"
            for category in category_ids
        )
        if not category_directives:
            category_directives = (
                "- No unambiguous canonical hint is available. Use the generic "
                "contract and classify only what is clearly visible."
            )

        uncertainty_note = (
            f"\nCATEGORY HINT UNCERTAINTY:\n{category_uncertainty}\n"
            if category_uncertainty
            else ""
        )

        user_prompt = f"""
Analyze this educational frame and return the compact structured observation.

RAW CATEGORY HINT: {primary_type}
PRIMARY CONTENT TYPE: {primary_type}
CANONICAL CATEGORY HINTS: {', '.join(category_ids) if category_ids else 'none'}
{uncertainty_note}

Hints are not facts. Verify them from the frame and include additional clearly
visible categories. Do not force an ambiguous hint into a CT category.

CATEGORY-SPECIFIC REQUIREMENTS:
{category_directives}

OUTPUT REQUIREMENTS:
- Scan the entire frame and create meaningful regions rather than arbitrary quadrants.
- Preserve exact visible content and its source association.
- Encode supported containment, hierarchy, topology, label targets, spatial arrangement,
  connector direction, reading order and cross-category relationships with local refs.
- Treat intersections, wire crossings, arrow attachments and label targets as uncertain
  unless their visual attachment is clear.
- A partial or uncertain result must be labeled partial or ambiguous.
- Output only the single JSON object defined above.
"""
        self._last_prompt_budget_parts = {
            "common_prompt": self.SYSTEM_PROMPT + user_prompt.replace(
                category_directives, "", 1
            ),
            "category_directives": category_directives,
        }
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
        parse_started = time.monotonic()
        text = text.strip()
        trace_stage("vlm_json_parsing", "start", output_chars=len(text))
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Repair only documented wrapper noise, and only after a strict
            # parse failed, so exact code or math inside valid JSON is untouched.
            repaired = re.sub(r"^```(?:json)?\s*", "", text)
            repaired = re.sub(r"\s*```$", "", repaired)
            repaired = re.sub(
                r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])',
                lambda match: match.group(1) or match.group(2),
                repaired,
            )
            try:
                result = json.loads(repaired)
            except json.JSONDecodeError as exc:
                print(
                    f"[Snapshot Engine] Failed to parse JSON: {exc} "
                    f"(output length {len(text)} characters)"
                )
                # The raw output is kept once, for debugging only. It must never
                # reach the fields downstream consumers read as visual facts.
                trace_stage(
                    "vlm_json_parsing", "failed", started_at=parse_started,
                    output_chars=len(text), error_type=type(exc).__name__,
                    error_position=exc.pos,
                )
                return {
                    "_error": True,
                    "_error_detail": f"Invalid JSON: {exc}",
                    "_raw_response": text,
                    "content_type": "unknown",
                    "detected_types": ["unknown"],
                    "visual_analysis": {},
                    "structural_description": "",
                    "reading_order": [],
                    "conceptual_hints": "",
                    "layout": "",
                    "text_readout": "",
                    "spatial_map": "",
                    "colors_styles": "",
                    "missing_elements": "Parse error",
                }

        if not isinstance(result, dict):
            trace_stage(
                "vlm_json_parsing", "failed", started_at=parse_started,
                output_chars=len(text), error_type="non_object",
            )
            return {
                "_error": True,
                "_error_detail": "Model output was JSON but not an object",
                "_raw_response": text,
                "content_type": "unknown",
                "detected_types": ["unknown"],
                "visual_analysis": {},
                "structural_description": "",
                "reading_order": [],
                "conceptual_hints": "",
                "layout": "",
                "text_readout": "",
                "spatial_map": "",
                "colors_styles": "",
                "missing_elements": "Parse error",
            }

        # Task 2B responses are normalized after deterministic canonical
        # assembly so their compact structure is not polluted with legacy keys.
        if result.get("schema") == COMPACT_OBSERVATION_SCHEMA:
            trace_stage(
                "vlm_json_parsing", "complete", started_at=parse_started,
                output_chars=len(text), schema=COMPACT_OBSERVATION_SCHEMA,
            )
            return result

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

        trace_stage(
            "vlm_json_parsing", "complete", started_at=parse_started,
            output_chars=len(text), schema="legacy",
        )
        return result

    @staticmethod
    def _assess_truncation(
        response: Optional[VLMResponse], token_limit: Optional[int]
    ) -> tuple[bool, str, str | None]:
        """Prefer server completion metadata and explicitly classify fallback use."""

        if response is None:
            return False, "no_response", None
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason == "length":
            return True, "finish_reason", finish_reason
        if finish_reason == "stop":
            return False, "finish_reason", finish_reason
        if finish_reason is not None:
            # This extraction path does not request tools or content filters.
            # Unknown terminal states must not upgrade evidence to complete.
            return True, "unknown_finish_reason", finish_reason

        token_count_at_limit = bool(
            token_limit
            and getattr(response, "eval_count", 0) > 0
            and getattr(response, "eval_count", 0) >= token_limit
        )
        return token_count_at_limit, "token_count_fallback", None

    @staticmethod
    def _failure_from_exception(exc: Exception) -> Dict[str, Any]:
        """Classify a request failure; ``retryable`` means a retry may help."""

        if isinstance(exc, VLMAuthenticationError):
            return {"code": "vlm_authentication_failed", "retryable": False}
        if isinstance(exc, VLMContextBudgetError):
            return {"code": "vlm_context_budget_exceeded", "retryable": False}
        if isinstance(exc, VLMRemoteError):
            return {"code": "vlm_remote_rejected", "retryable": False}
        if isinstance(exc, (VLMTimeoutError, VLMInterfaceError)):
            return {"code": "vlm_remote_unavailable", "retryable": True}
        return {"code": "vlm_request_failed", "retryable": True}

    def _apply_parse_failure(
        self,
        snapshot: Dict[str, Any],
        response: VLMResponse,
        token_limit: Optional[int],
    ) -> None:
        """Separate genuine truncation from malformed output using metadata."""

        truncated, detection, finish_reason = self._assess_truncation(
            response, token_limit
        )
        tokens = getattr(response, "eval_count", 0)
        if truncated and detection != "unknown_finish_reason":
            code = "vlm_output_truncated"
            summary = (
                f"Qwen output was cut off at {tokens} of {token_limit} tokens "
                f"(finish_reason={finish_reason})"
            )
        else:
            code = "vlm_output_malformed"
            summary = (
                f"Qwen returned malformed JSON after {tokens} tokens "
                f"(finish_reason={finish_reason})"
            )
        snapshot["_error_detail"] = f"{summary}: {snapshot.get('_error_detail')}"
        snapshot["_failure"] = {
            "code": code,
            "retryable": True,
            "finish_reason": finish_reason,
            "completion_tokens": tokens,
        }

    def _attach_visual_evidence(
        self,
        snapshot: Dict[str, Any],
        *,
        primary_type: str,
        frame_path: str,
        frame_id: Optional[str],
        timestamp_seconds: Optional[float],
        raw_category_labels: Optional[list[str]],
        canonical_category_ids: Optional[list[str]],
        classification_confidence: Optional[float],
        confidence_semantics: str,
        response: Optional[VLMResponse] = None,
        token_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach canonical evidence while retaining the legacy snapshot API."""

        context_budget = getattr(
            getattr(self, "vlm_interface", None),
            "last_context_budget",
            None,
        )
        effective_token_limit = (
            context_budget.get("effective_output_tokens", token_limit)
            if isinstance(context_budget, dict) else token_limit
        )
        metrics = {
            "model": getattr(response, "model", getattr(self, "model_name", None)),
            "latency_ms": getattr(response, "total_duration_ms", None),
            "prompt_tokens": getattr(response, "prompt_eval_count", 0),
            "completion_tokens": getattr(response, "eval_count", 0),
            "configured_output_token_limit": token_limit,
            "output_token_limit": effective_token_limit,
            "context_budget": context_budget,
        }
        possible_truncation, truncation_detection, finish_reason = (
            self._assess_truncation(response, effective_token_limit)
        )
        metrics["possible_truncation"] = possible_truncation
        metrics["finish_reason"] = finish_reason
        metrics["truncation_detection"] = truncation_detection
        provenance = {
            "provider": self.provider,
            "model": metrics["model"],
            "frame_path": frame_path,
            "classification_confidence": classification_confidence,
            "confidence_semantics": confidence_semantics,
            "latency_ms": metrics["latency_ms"],
            "prompt_tokens": metrics["prompt_tokens"],
            "completion_tokens": metrics["completion_tokens"],
            "configured_output_token_limit": token_limit,
            "output_token_limit": effective_token_limit,
            "context_budget": metrics["context_budget"],
            "possible_truncation": possible_truncation,
            "finish_reason": finish_reason,
            "truncation_detection": truncation_detection,
        }

        if snapshot.get("schema") == COMPACT_OBSERVATION_SCHEMA:
            compact, relation_normalization_count = normalize_compact_relations(snapshot)
            metrics["relation_normalization_count"] = relation_normalization_count
            try:
                evidence = assemble_compact_qwen_observation(
                    compact,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    category_ids=canonical_category_ids or [],
                    raw_category_labels=raw_category_labels or [primary_type],
                    provenance=provenance,
                    possible_truncation=possible_truncation,
                )
            except Exception as exc:
                error_desc = f"{type(exc).__name__}: {exc}"
                fallback = {
                    "_error": True,
                    "_error_detail": error_desc,
                    "content_type": primary_type,
                    "detected_types": [primary_type],
                    "visual_analysis": {},
                    "structural_description": "",
                    "reading_order": [],
                    "conceptual_hints": "",
                    "layout": "",
                    "text_readout": "",
                    "spatial_map": "",
                    "colors_styles": "",
                    "missing_elements": f"Structured evidence validation failed: {error_desc}",
                    "_failure": {
                        "code": "vlm_contract_invalid",
                        "retryable": True,
                        "finish_reason": finish_reason,
                    },
                    "_structured_observation": compact,
                    "_vlm_metrics": metrics,
                }
                evidence = adapt_legacy_snapshot(
                    fallback,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    category_ids=canonical_category_ids or [],
                    raw_category_labels=raw_category_labels or [primary_type],
                    provenance=provenance,
                )
                fallback["visual_evidence"] = evidence.to_schema_dict()
                return fallback

            evidence_dict = evidence.to_schema_dict()
            structure = evidence_dict["structural_description"]
            exact_text = "\n".join(
                item["text"] for item in evidence_dict.get("exact_content", [])
                if item.get("text")
            )
            visual_analysis = {
                "regions": evidence_dict.get("regions", []),
                "relations": evidence_dict.get("relations", []),
                "category_details": evidence_dict.get("category_details", {}),
                "uncertainties": evidence_dict.get("uncertainties", []),
            }
            unavailable = evidence_dict["extraction_status"] in {"failed", "not_attempted"}
            failure_fields: Dict[str, Any] = {}
            if unavailable:
                failure_fields = {
                    "_error_detail": "The model reported this frame as unreadable",
                    "_failure": {
                        "code": "evidence_reported_unavailable",
                        "retryable": False,
                        "finish_reason": finish_reason,
                    },
                }
            return {
                **failure_fields,
                "content_type": primary_type,
                "detected_types": evidence_dict.get("raw_category_labels", [])
                or evidence_dict.get("category_ids", []),
                "visual_analysis": visual_analysis,
                "structural_description": structure.get("overview") or "",
                "reading_order": structure.get("reading_order", []),
                "conceptual_hints": "",
                "layout": evidence_dict.get("global_layout") or "",
                "text_readout": exact_text,
                "spatial_map": structure.get("orientation") or "",
                "colors_styles": "",
                "missing_elements": "Structured visual evidence unavailable" if unavailable else None,
                "_error": unavailable,
                "_structured_observation": compact,
                "_vlm_metrics": metrics,
                "visual_evidence": evidence_dict,
            }

        legacy_source = dict(snapshot)
        legacy_source.pop("visual_evidence", None)
        legacy_source.pop("_vlm_metrics", None)
        evidence = adapt_legacy_snapshot(
            legacy_source,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            category_ids=canonical_category_ids or [],
            raw_category_labels=raw_category_labels or [primary_type],
            provenance=provenance,
        )
        snapshot["visual_evidence"] = evidence.to_schema_dict()
        snapshot["_vlm_metrics"] = metrics
        return snapshot

    def analyze_frame(
        self,
        frame_path: str,
        primary_type: str,
        classification_confidence: Optional[float] = None,
        confidence_semantics: str = "unknown",
        frame_id: Optional[str] = None,
        timestamp_seconds: Optional[float] = None,
        raw_category_labels: Optional[list[str]] = None,
        canonical_category_ids: Optional[list[str]] = None,
    ) -> Dict:
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

        # Transformers or Modal backend
        # 1. Build prompt with category directive
        # Keep the existing directive vocabulary while normalizing known
        # legacy aliases (for example ``notes`` -> ``handwritten_notes``).
        # Ambiguous labels remain unchanged so the model still receives the
        # original hint rather than an invented category.
        prompt_type = prompt_category_for(primary_type)
        uncertainty = None
        if (
            confidence_semantics == "classifier"
            and classification_confidence is not None
            and classification_confidence < 0.40
        ):
            uncertainty = (
                f"The supplied '{prompt_type}' category is tentative; classifier "
                f"confidence is {classification_confidence:.3f}. Verify the frame "
                "and describe other applicable categories if present."
            )
        prompt = self._build_prompt(
            prompt_type,
            uncertainty,
            canonical_category_ids=canonical_category_ids,
        )

        # 2. Generate via VLM Interface
        token_limit = getattr(self, "structured_max_tokens", 384)
        try:
            response = self.vlm_interface.analyze(
                image_path_or_pil=frame_path,
                prompt=prompt,
                generation_params={
                    "max_new_tokens": token_limit,
                    "temperature": DEFAULT_TEMPERATURE,
                    "_context_budget_parts": getattr(
                        self, "_last_prompt_budget_parts", {}
                    ),
                }
            )
            # 3. Parse into structured JSON
            result = self._parse_json_response(response.text)
        except Exception as exc:
            error_desc = f"{type(exc).__name__}: {exc}"
            print(f"[Snapshot Engine] Frame analysis failed for {frame_path}: {error_desc}")
            if self.provider == PROVIDER_MODAL:
                # In Modal mode: NEVER fall back to local VLM!
                failed = self._create_failed_snapshot(error_desc, primary_type)
                failed["_failure"] = self._failure_from_exception(exc)
                return self._attach_visual_evidence(
                    failed,
                    primary_type=primary_type,
                    frame_path=frame_path,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    raw_category_labels=raw_category_labels,
                    canonical_category_ids=canonical_category_ids,
                    classification_confidence=classification_confidence,
                    confidence_semantics=confidence_semantics,
                    token_limit=token_limit,
                )
            raise

        if result.get("_error"):
            self._apply_parse_failure(result, response, token_limit)

        # 4. Check for [ILLEGIBLE] text (OCR fallback placeholder)
        text_readout = result.get("text_readout", "")

        # Check if >20% of text is [ILLEGIBLE] for equation/handwritten_notes
        illegible_count = text_readout.count("[ILLEGIBLE]")
        text_length = len(text_readout.replace("[ILLEGIBLE]", "").strip())

        if text_length > 0 and (illegible_count / text_length) > 0.2:
            if primary_type.lower() in ["equation", "handwritten_notes"]:
                result["missing_elements"] = "OCR fallback required"
                print(f"[Snapshot Engine] High [ILLEGIBLE] content detected for {primary_type}")

        attached = self._attach_visual_evidence(
            result,
            primary_type=primary_type,
            frame_path=frame_path,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            raw_category_labels=raw_category_labels,
            canonical_category_ids=canonical_category_ids,
            classification_confidence=classification_confidence,
            confidence_semantics=confidence_semantics,
            response=response,
            token_limit=token_limit,
        )
        failure_code = (attached.get("_failure") or {}).get("code")
        if (
            self.provider == PROVIDER_MODAL
            and failure_code in {"vlm_contract_invalid", "vlm_output_malformed"}
        ):
            corrective_prompt = (
                prompt
                + "\nCORRECTION: Return one JSON object matching the supplied schema "
                  "exactly. Use only listed relation types; prefix any other "
                  "relation with 'custom:'."
            )
            try:
                corrective_response = self.vlm_interface.analyze(
                    image_path_or_pil=frame_path,
                    prompt=corrective_prompt,
                    generation_params={
                        "max_new_tokens": token_limit,
                        "temperature": DEFAULT_TEMPERATURE,
                        "_context_budget_parts": getattr(
                            self, "_last_prompt_budget_parts", {}
                        ),
                    },
                )
                corrective_result = self._parse_json_response(corrective_response.text)
                if corrective_result.get("_error"):
                    self._apply_parse_failure(
                        corrective_result, corrective_response, token_limit
                    )
                corrected = self._attach_visual_evidence(
                    corrective_result,
                    primary_type=primary_type,
                    frame_path=frame_path,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    raw_category_labels=raw_category_labels,
                    canonical_category_ids=canonical_category_ids,
                    classification_confidence=classification_confidence,
                    confidence_semantics=confidence_semantics,
                    response=corrective_response,
                    token_limit=token_limit,
                )
                corrected.setdefault("_vlm_metrics", {})[
                    "corrective_retry_count"
                ] = 1
                return corrected
            except Exception as exc:
                attached.setdefault("_vlm_metrics", {})[
                    "corrective_retry_count"
                ] = 1
                attached["_vlm_metrics"]["corrective_retry_error_type"] = (
                    type(exc).__name__
                )
        return attached

    def cleanup(self):
        """
        Clean up VLM Engine resources.

        Releases GPU memory and deletes model references.
        """
        print("[Snapshot Engine] Cleaning up...")
        if hasattr(self, "vlm_interface") and self.vlm_interface is not None:
            self.vlm_interface.cleanup()
        print("[Snapshot Engine] Cleanup complete")
