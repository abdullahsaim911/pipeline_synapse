# VLM and LLM Module Implementation Documentation

This document details the implementation of enhanced VLM (Vision-Language Model) and LLM (Large Language Model) modules for the Synapse pipeline to generate accessible explanations for blind students.

---

## Table of Contents
1. [VLM Module Implementation](#vlm-module-implementation)
2. [LLM Module Implementation](#llm-module-implementation)
3. [Context Manager Enhancements](#context-manager-enhancements)

---

## VLM Module Implementation

### File: `vlm_engine/snapshot_engine.py`

### Overview
The VLM module (M2) serves as the "Universal Eye" for the pipeline, analyzing complete video frames from STEM lectures using Qwen2-VL-7B-Instruct. It extracts structured visual information that makes content accessible to blind students.

### Key Features Implemented

#### 1. Enhanced System Prompt for Multiple Content Type Detection

The system prompt now enforces detection of ALL content types present in a frame, not just the primary type.

```
SYSTEM PROMPT:
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
```

#### 2. Unified Output Format

New JSON structure returned by VLM:

```json
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
```

#### 3. Category-Specific Directives

##### Equation Directive
```
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
```

##### Graph Directive (Most Important - Movement Description)
```
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
```

##### Circuit Directive
```
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
```

##### Diagram Directive
```
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
```

##### Code Directive
```
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
```

##### Handwritten Notes Directive
```
CRITICAL: OCR handwritten content with structure.

Provide in visual_analysis.handwritten_notes object:

1. text: Readable content, mark [ILLEGIBLE] for unreadable portions

2. layout: Organization (title, sections, bullet points, diagrams, equations)

3. legibility_note: Overall legibility, percentage [ILLEGIBLE], handwriting style

4. content_types: What's mixed in (text, equations, diagrams, formulas, examples)

5. reading_order: How to mentally navigate the notes
```

##### Biology Directive
```
CRITICAL: Extract structural biological information with connections.

Provide in visual_analysis.biology object:

1. main_structure: Organism/system name and type

2. components (All Parts with Locations):
   For EACH component: name/label, type, position, description, function, relationships

3. element_links: How parts connect spatially and functionally

4. scale_relationships: Relative sizes, scale bar if present

5. labels: All callouts with text and what they point to

6. conceptual_meaning: Biological principle being illustrated
```

##### Chemistry Directive
```
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
```

##### Physics Directive
```
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
```

##### Text Directive
```
CRITICAL: OCR text with structural organization.

Provide in visual_analysis.text object:

1. text: Actual content with preserved line breaks and punctuation

2. layout: Hierarchy (title, headings, body text), lists (bullet/numbered)

3. formatting: Bold, italic, colored text and their meanings, font sizes

4. structure: Logical organization, section breaks, indentation levels, visual flow

5. key_terms: Important vocabulary that's emphasized or defined
```

#### 4. Enhanced JSON Parser

The `_parse_json_response` method now:
- Strips markdown blocks (```json ... ```)
- Removes trailing commas before closing braces
- Ensures all new format fields exist with defaults
- Maintains backward compatibility with old format fields

---

## LLM Module Implementation

### File: `synthesizer/llm_synthesizer.py`

### Overview
The LLM module (M3), also called "Audio Weaver", fuses teacher's spoken words with VLM's structural data into a single, seamless TTS script for blind students.

### Key Features Implemented

#### 1. Three-Mode Explanation System

The LLM now generates explanations at three depth levels, differing in detail (not length):

- **Brief Mode**: Surface level, essentials only, 2-3 sentences equivalent depth
- **Explanatory Mode**: Standard depth, complete coverage, 4-6 sentences equivalent depth
- **Detailed Mode**: Deep exploration, exhaustive, 8-12 sentences equivalent depth

#### 2. All Prompts Follow 4-Layer Structure

Each prompt enforces the following structure:
1. **Visual Description** - What's on the screen
2. **Teacher's Words** - What is being said
3. **Concept** - What is being explained
4. **Visual-Concept Link** - How visuals link to concept

#### 3. Brief Mode Prompt

```
YOU ARE: Expert teacher for blind students. Your student cannot see the screen.

TASK: Generate a BRIEF, surface-level accessible explanation for this visual content.

EXPLANATION DEPTH: Surface level only. Cover essentials. Quick overview.

INPUT DATA:

1. CONTENT TYPES DETECTED: {detected_types}

2. VISUAL ANALYSIS (from VLM):
{visual_analysis_json}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. CONCEPTUAL HINTS:
{conceptual_hints}

5. WHAT TEACHER IS SAYING:
"{transcript_context}"

6. PREVIOUS CONCEPTS:
{previous_concepts}

YOUR EXPLANATION - FOCUS ON THESE ELEMENTS:

1. QUICK VISUAL SNAPSHOT (Surface Level):
- Describe ONLY the most important visual elements
- Use the structural_description for the essential layout
- Use spatial language: "On the screen, you'll find..."
- Build a basic mental image
- Don't go into detail about every element

2. WHAT'S BEING SHOWN (Core Concept):
- State the main concept or principle in simple terms
- What is the teacher demonstrating with this visual?
- Use the conceptual_hints as your guide

3. KEY CONNECTION (The Main Link):
- Make ONE clear connection between the visual and the concept
- How does the visual help understand the concept?
- Keep it simple and direct

ACCESSIBILITY:
- Use "you" and direct address
- Quick mental model building
- Clear, straightforward language
- No lengthy explanations or analogies

DEPTH GUIDELINE:
- Provide just enough information to understand what's happening
- Focus on the most important aspects
- Don't elaborate on details, exceptions, or related concepts
- Quick and to the point

OUTPUT: Brief, surface-level explanation. No minimum or maximum length - let depth guide you naturally.
```

#### 4. Explanatory Mode Prompt

```
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate a STANDARD-DEPTH accessible explanation for this visual content.

EXPLANATION DEPTH: Complete coverage. Cover all necessary aspects thoroughly but concisely.

INPUT DATA:

1. CONTENT TYPES DETECTED: {detected_types}

2. VISUAL ANALYSIS (from VLM):
{visual_analysis_json}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{reading_order}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. SURROUNDING CONTEXT:
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{previous_concepts}

YOUR EXPLANATION - FOCUS ON THESE ELEMENTS:

1. COMPLETE VISUAL DESCRIPTION:
- Use the structural_description and reading_order thoroughly
- Build a complete mental model for your student
- Describe ALL important visual elements
- Use spatial language consistently
- For graphs: describe curve movement, axes, key points, relationship
- For equations: reading order, element positions, structure
- For diagrams: all elements, positions, relationships
- "To visualize this, imagine..." when helpful
- Don't skip important details

2. WHAT THE TEACHER IS EXPLAINING:
- Summarize what the teacher is saying about this visual
- Connect teacher's words to the visual elements
- Explain the context in the lecture
- "Your teacher is showing this to explain..."

3. THE CONCEPT BEING EXPLAINED:
- Explain the underlying principle, law, or formula clearly
- Use the conceptual_hints as your starting point
- Expand with your knowledge appropriately
- Connect to previous concepts if relevant
- Include the essential meaning and significance

4. HOW THE VISUAL LINKS TO THE CONCEPT:
- Make explicit connections between visual elements and concept aspects
- Explain WHY this visual helps understand the concept
- Use specific details from the visual_analysis
- "The [visual element] shows [behavior], which represents [concept aspect]"
- Show cause-and-effect relationships
- Make the connection clear and understandable

ACCESSIBILITY TECHNIQUES:
- "To visualize this, imagine..."
- "Think of it like..."
- Use analogies when they help understanding
- Speak directly to "you"
- Build mental models step by step

DEPTH GUIDELINE:
- Cover all necessary aspects completely
- Be thorough but don't over-explain
- Include important details
- Don't go into edge cases, exceptions, or tangential topics
- Standard teaching depth - what you'd explain in a lecture

OUTPUT: Standard-depth explanation. Let the content determine the length naturally.
```

#### 5. Detailed Mode Prompt

```
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate an EXHAUSTIVE, DEEP accessible explanation for this visual content.

EXPLANATION DEPTH: Comprehensive and deep. Explore fully, make connections, provide rich context.

INPUT DATA:

1. CONTENT TYPES DETECTED: {detected_types}

2. VISUAL ANALYSIS (from VLM):
{visual_analysis_json}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{reading_order}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. FULL LECTURE CONTEXT (±2 minutes):
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{previous_concepts}

9. RELATED PRINCIPLES:
{related_principles}

10. TOPIC TIMELINE:
{topic_timeline}

11. CROSS-DOMAIN CONNECTIONS:
{cross_domain}

YOUR EXPLANATION - FOCUS ON THESE ELEMENTS:

1. EXHAUSTIVE VISUAL DESCRIPTION:
- Use the structural_description and reading_order extensively
- Build a complete, detailed mental model
- Describe EVERYTHING that matters - no important element left out
- Use spatial language systematically throughout
- For graphs: complete curve description, all axes details, every key point, full relationship
- For equations: complete reading order, every element's position, full structural layout
- For diagrams: every element, all positions, all relationships, all connections
- "To visualize this, imagine..." - build the mental image completely and vividly
- Include subtle but important visual details

2. WHAT THE TEACHER IS EXPLAINING (Full Context):
- Summarize what the teacher is saying in detail
- Connect teacher's words to specific visual elements
- Explain the full context in the lecture flow
- Why is this being shown at this specific point?
- How does it connect to what came before and what comes after?
- "Your teacher is showing this at this point because..."

3. DEEP CONCEPTUAL FRAMEWORK:
- Explain the underlying principle, law, or formula in depth
- Use the conceptual_hints as foundation, then expand significantly
- Include mathematical/physical meaning thoroughly
- Explain historical context if relevant
- Discuss boundary conditions, limitations, special cases
- Explain why this concept matters
- Connect to related principles
- Discuss the significance and applications
- "The fundamental principle being demonstrated is..."
- "This concept is important because..."

4. HOW THE VISUAL LINKS TO THE CONCEPT (Multiple Connections):
- Make MULTIPLE explicit connections between visual and concept
- Explain how each key visual element represents different concept aspects
- Use specific details from visual_analysis thoroughly
- Show cause-and-effect relationships in detail
- Explain the "why" behind each connection
- "The [visual element] shows [behavior], which represents [concept aspect] because..."
- "You can see this in how the [visual feature] corresponds to [concept element] by..."
- "This visual demonstrates the principle by showing [connection], which illustrates [principle aspect]"
- Make each connection clear and understandable

5. CROSS-DOMAIN CONNECTIONS:
- Connect to previous concepts
- Use cross-domain connections if relevant
- Explain relationships to other domains (math, physics, chemistry, etc.)
- "This connects to what we discussed earlier about [previous concept] because..."
- "This same principle appears in [other domain] as..."
- Discuss parallels and analogies

6. RICH ACCESSIBILITY TECHNIQUES:
- Multiple analogies when helpful for different perspectives
- "To visualize this, imagine..." with vivid, detailed descriptions
- "Think of it like..." with relatable, varied comparisons
- Use rhetorical questions to engage: "Have you ever...?" "Can you picture...?"
- Speak directly to "you" throughout
- Build and reinforce mental models repeatedly
- Use sensory language when appropriate
- Check understanding: "Does that make sense?" "Can you see how...?"
- Provide alternative ways to conceptualize

DEPTH GUIDELINE:
- Explore thoroughly and comprehensively
- Include important details, nuances, and connections
- Discuss related concepts and principles
- Provide rich context and multiple perspectives
- Go into depth on the why and how, not just the what
- This is like a one-on-one tutoring session - explore fully
- No detail is too small if it helps understanding

OUTPUT: Comprehensive, deep explanation. Let the depth of content determine the length naturally.
```

#### 6. Sentence Limit Removal

The `_apply_output_mode()` method now returns the full LLM output without any truncation:

```python
def _apply_output_mode(self, audio_script: str, output_mode: str) -> str:
    """
    Return audio_script as-is.

    No truncation - prompts control depth through guidance, not sentence limits.
    Modes differ in explanation depth, not length.
    """
    if not audio_script:
        return ""

    # No sentence limits - return full explanation as generated
    # Prompts control depth, not length
    return audio_script
```

#### 7. Enhanced Explanation Generation Flow

```
1. Safety Check: VLM missing_elements?
   → If yes, return transcript only

2. Safety Check: Cross-domain hallucination?
   → If yes, return transcript only

3. Find current transcript index for context

4. Gather context based on output_mode:
   - brief: 30 seconds window
   - explanatory: 60 seconds window
   - detailed: 120 seconds window

5. Linearize math in VLM snapshot

6. Build mode-specific prompt:
   - brief → _build_brief_prompt()
   - explanatory → _build_explanatory_prompt()
   - detailed → _build_detailed_prompt()

7. Call LLM with prompt

8. Guard: if LLM returned empty, fall back to transcript

9. Post-process: math linearization (double-check)

10. Return full explanation (no truncation)
```

---

## Context Manager Enhancements

### File: `synthesizer/context_manager.py`

### Enhanced Context Windows

Context windows were expanded to provide better context for different explanation depths:

```python
CONTEXT_WINDOWS = {
    "brief": 30,      # Increased from 10s
    "explanatory": 60, # Increased from 30s
    "detailed": 120,  # Increased from 50s
    "standard": 30
}
```

### Context Gathering Features

The ContextManager now provides:
- **Context Text**: Combined text from within the time window
- **Previous Concepts**: List of concepts mentioned before current point
- **Related Principles**: Related mathematical/physical principles
- **Topic Timeline**: Timeline of topics discussed

### Concept and Principle Mappings

Extensive mappings for concept extraction and principle connection:
- 65+ concept keywords (derivative, integral, velocity, etc.)
- Concept-to-principle mappings for connecting related ideas
- Topic detection keywords for 9 domains (equations, graphs, theorems, functions, calculus, geometry, probability, linear algebra, statistics, physics)

---

## Summary of Changes

### VLM Module (snapshot_engine.py)
- ✅ Enhanced SYSTEM_PROMPT for multiple content type detection
- ✅ Added Chain-of-Regions scanning (2x2 grid)
- ✅ Created detailed CATEGORY_DIRECTIVES for 9 content types
- ✅ Implemented unified JSON output format with 10+ new fields
- ✅ Enhanced JSON parser for backward compatibility
- ✅ Added structural description, reading order, conceptual hints

### LLM Module (llm_synthesizer.py)
- ✅ Created _build_brief_prompt() for surface-level explanations
- ✅ Created _build_explanatory_prompt() for standard-depth explanations
- ✅ Created _build_detailed_prompt() for deep exploration explanations
- ✅ All prompts follow 4-layer structure (Visual → Teacher → Concept → Link)
- ✅ Removed sentence limits from _apply_output_mode()
- ✅ Enhanced context gathering for better concept connections
- ✅ Added cross-domain explanation support

### Context Manager (context_manager.py)
- ✅ Expanded context windows (30s/60s/120s)
- ✅ Enhanced concept extraction
- ✅ Added related principles mapping
- ✅ Implemented topic timeline tracking

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FRAME EXTRACTION (M0)                       │
│                - Extract pedagogical keyframes                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSCRIPTION (M0b)                        │
│                  - Transcribe audio to text                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SYNCHRONIZER (M1)                         │
│            - Detect intervention points (suffer points)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       VLM ENGINE (M2)                           │
│  - Analyze frames with enhanced prompts                        │
│  - Detect ALL content types (not just primary)                 │
│  - Extract structural descriptions                             │
│  - Provide reading order and conceptual hints                  │
│  - Return unified JSON format                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONTEXT MANAGER (part of M3)                    │
│  - Gather context based on output_mode                         │
│  - Extract previous concepts                                   │
│  - Find related principles                                     │
│  - Build topic timeline                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LLM SYNTHESIZER (M3)                            │
│  - Build mode-specific prompt (brief/explanatory/detailed)     │
│  - 4-layer structure: Visual → Teacher → Concept → Link        │
│  - Generate accessible explanation                             │
│  - No sentence limits (depth-based)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TTS ENGINE (M6)                            │
│                - Generate MP3 from explanation                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ACCESSIBLE OUTPUT FOR BLIND STUDENTS           │
│  - Visual description (what's on screen)                       │
│  - Teacher's words (what is being said)                        │
│  - Concept explanation (what is being explained)               │
│  - Visual-concept link (how visuals link to concept)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Date
2026-04-27

## Implementation Notes
- All changes were designed to not break the existing pipeline
- Backward compatibility maintained through JSON parser enhancements
- Sentence limits removed in favor of depth-based prompting
- Context windows expanded for better concept connections
- Multiple content type detection enforced at VLM level
