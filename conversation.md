Complete Solution - No Pipeline Breaking
My Recommendations
Q2 (VLM Output Format): Option B (Unified Format)

Why: Easier for LLM to process, more flexible for mixed content, cleaner code
Q4 (Visual-Concept Link): LLM generates it

Why: LLM has transcript context + prior concepts = better explanations
Q6 (Physics Category): Keep as "diagram"

Why: No time to retrain classifier, VLM can detect physics-specific content
Final Solution Architecture
VLM Output Format (Unified)

{
  "content_type": "graph",
  "detected_types": ["graph", "text", "equation"],
  
  "visual_analysis": {
    "graph": {
      "data": {
        "x_axis": {"label": "Time (s)", "range": "0-10"},
        "y_axis": {"label": "Velocity (m/s)", "range": "0-50"}
      },
      "curves": [{
        "description": "Blue curve starting at origin, rising steeply then leveling off",
        "shape": "asymptotic",
        "key_points": [{"x": 0, "y": 0}, {"x": 4, "y": 40}, {"x": 10, "y": 50}]
      }],
      "relationship": "Velocity increases with time until reaching terminal velocity when air resistance equals gravity"
    },
    "equation": [{
      "text": "v = u + at",
      "description": "velocity equals initial velocity plus acceleration times time",
      "position": "at bottom of frame"
    }],
    "text": "Annotation explaining acceleration phase and terminal velocity"
  },
  
  "structural_description": "Complete spatial narrative for mental model building...",
  
  "reading_order": ["Start with graph axes...", "Trace the curve...", "Read annotations...", "See equation..."],
  
  "conceptual_hints": "This visual demonstrates terminal velocity - when air resistance balances gravitational force",
  
  "layout": "Graph on right 60%, text on left 20%, equation at bottom 20%",
  
  "missing_elements": null
}
Implementation Plan
Step 1: Update VLM System Prompt
File: vlm_engine/snapshot_engine.py

Add to the system prompt:


MULTIPLE CONTENT TYPE DETECTION:
Even if one type is specified, DETECT AND DESCRIBE ALL content types present:
- equations: Extract ALL equations with their positions
- graphs: Extract data, curves, and relationships
- circuits: Components, connections, flow
- diagrams: Complete structural description
- code: OCR + layout
- handwritten: OCR + layout
- biology/chemistry/physics: Structural description with element links
- text: OCR + layout

OUTPUT REQUIREMENTS:
1. Use unified format (see below)
2. Provide conceptual_hints linking visual to concept
3. Describe relationships between multiple content types if present
Step 2: Redesign Category Directives
File: vlm_engine/snapshot_engine.py

Replace CATEGORY_DIRECTIVES with enhanced versions:


CATEGORY_DIRECTIVES = {
    "equation": """
    Extract ALL equations present in the frame.

    For each equation provide:
    - text: The equation in spoken form (linearized)
    - description: Structural description of equation layout
    - position: Where in the frame (top, bottom, left, right, center)
    - reading_order: How to read mentally (left→right, top→bottom)

    Convert ALL notation: x^2 → "x squared", ∫ → "integral", etc.
    """,

    "graph": """
    Extract graph data comprehensively.

    Provide:
    - data: Axes information (labels, ranges, units)
    - curves: Array of curves with:
      * description: Shape and movement description (not just points)
      * shape: Type (linear, exponential, asymptotic, etc.)
      * key_points: Important labeled points
    - relationship: What the graph shows mathematically/physically
    
    For movement: "starts flat, rises steeply, levels off" not "points at (0,0), (4,40), (10,50)"
    """,

    "circuit": """
    Extract complete circuit information.

    Provide:
    - components: Array of all components with:
      * type: resistor, capacitor, battery, etc.
      * label: Component identifier (R1, C1, etc.)
      * value: Component value if shown
      * position: Where in the diagram
    - flow: Current flow description (start point, path through components)
    - topology: Circuit arrangement (series, parallel, mixed)
    - connections: How components are connected
    """,

    "diagram": """
    Provide complete structural description.

    For ALL components:
    - Name/label and position
    - Spatial relationships to other components
    - Arrows/connections and what they represent

    Describe:
    - Overall layout (vertical, horizontal, circular, etc.)
    - How elements connect or relate
    - Reading order for mental navigation
    """,

    "code": """
    Perform OCR on code content.

    Provide:
    - code_text: The actual code with proper indentation
    - layout: Basic layout (single file, split view, panel arrangement)
    - structure: High-level organization (imports, classes, functions visible)
    
    Do NOT explain what code does.
    """,

    "handwritten_notes": """
    Perform OCR on handwritten content.

    Provide:
    - text: The readable text
    - layout: Overall arrangement (title, sections, bullet points)
    - legibility_note: Note if [ILLEGIBLE] portions exist
    """,

    "biology": """
    Provide structural biological description.

    Describe:
    - Main structure and its position
    - All labeled parts with their locations
    - How parts connect or relate spatially
    - Scale/size relationships if visible
    - Element links: How each part connects to the whole structure
    """,

    "chemistry": """
    Provide structural chemical description.

    Describe:
    - Molecule structure and geometry (linear, bent, tetrahedral)
    - All atoms with their positions
    - Bonds: type (single/double/triple) and connections
    - Molecular shape and spatial arrangement
    - Element links: How atoms connect to form the structure
    """,

    "physics": """
    Provide structural physics diagram description.

    Describe:
    - Main diagram elements and their positions
    - Physical objects (blocks, pulleys, forces, etc.)
    - Forces/arrows and what they represent
    - Spatial relationships between elements
    - Element links: How components connect physically
    """,

    "text": """
    Perform OCR on text content.

    Provide:
    - text: The actual text content
    - layout: Hierarchy (title, headings, bullets, paragraphs)
    - formatting: Bold, italic, highlighted text and their meanings
    """,
}
Step 3: New VLM Output Parser
File: vlm_engine/snapshot_engine.py

Update _parse_json_response to handle the new unified format:


def _parse_json_response(self, text: str) -> Dict:
    """Parse VLM output into structured JSON with new format."""
    
    # Strip markdown
    text = re.sub(r'```(?:json)?\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()
    
    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback for parsing errors
        return {
            "content_type": "unknown",
            "detected_types": [],
            "visual_analysis": {},
            "structural_description": text,
            "reading_order": [],
            "conceptual_hints": "",
            "layout": text,
            "missing_elements": "Parse error"
        }
    
    # Ensure required fields exist
    required_fields = [
        "content_type", "detected_types", "visual_analysis",
        "structural_description", "reading_order", "conceptual_hints",
        "layout", "missing_elements"
    ]
    
    for field in required_fields:
        if field not in result:
            if field == "detected_types":
                result[field] = [result.get("content_type", "unknown")]
            elif field == "visual_analysis":
                result[field] = {result.get("content_type", "unknown"): {}}
            elif field == "reading_order":
                result[field] = []
            elif field == "conceptual_hints":
                result[field] = ""
            elif field == "structural_description":
                result[field] = result.get("layout", "")
            else:
                result[field] = ""
    
    return result
Step 4: Expand Context Windows
File: synthesizer/context_manager.py


CONTEXT_WINDOWS = {
    "brief": 30,      # Was 10s
    "explanatory": 60, # Was 30s
    "detailed": 120,  # Was 50s
    "standard": 30
}
Step 5: Remove Sentence Limits from LLM
File: synthesizer/llm_synthesizer.py

In _apply_output_mode(), remove all truncation logic:


def _apply_output_mode(self, audio_script: str, output_mode: str) -> str:
    """Return audio_script as-is - no truncation."""
    if not audio_script:
        return ""
    
    # No sentence limits - let prompts control length
    return audio_script
Step 6: Redesign LLM Prompts
File: synthesizer/llm_synthesizer.py

Unified LLM Prompt Template (All Modes)

def _build_unified_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict,
    content_type: str,
    mode: str
) -> str:
    """
    Build unified prompt that works for all content types.
    
    The LLM receives:
    1. VLM's unified analysis (visual_analysis, structural_description, etc.)
    2. Transcript context (what teacher is saying)
    3. Conceptual hints from VLM
    4. Previous concepts from context
    """
    
    detected_types = vlm_snapshot.get("detected_types", [content_type])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    reading_order = vlm_snapshot.get("reading_order", [])
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    context_text = context.get("context_text", "")
    
    # Mode-specific depth
    if mode == "brief":
        depth_instruction = """
        Provide a concise 2-3 sentence explanation.
        Focus on: key visual elements + main concept + why shown now.
        """
    elif mode == "explanatory":
        depth_instruction = """
        Provide a 4-6 sentence comprehensive explanation.
        Include: visual description + concept explanation + connections.
        """
    else:  # detailed
        depth_instruction = """
        Provide an exhaustive 8-12 sentence explanation.
        Include: complete visual description + deep conceptual framework + 
        connections to prior concepts + accessibility techniques.
        """
    
    prompt = f"""
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Create a complete, accessible explanation for this visual content.

INPUT DATA:
1. WHAT'S ON SCREEN (VLM Analysis):
   - Content types detected: {', '.join(detected_types)}
   
   Visual Analysis:
   {json.dumps(visual_analysis, indent=2)}
   
   Structural Description:
   {structural_description}
   
   Reading Order:
   {'; '.join(reading_order) if reading_order else 'Not specified'}
   
   Conceptual Hints from Visual:
   {conceptual_hints}

2. WHAT TEACHER IS SAYING:
   "{transcript_context}"

3. SURROUNDING CONTEXT:
   "{context_text}"
   
4. PREVIOUS CONCEPTS MENTIONED:
   {', '.join(previous_concepts) if previous_concepts else 'None'}

{depth_instruction}

YOUR EXPLANATION MUST INCLUDE:

LAYER 1: VISUAL DESCRIPTION
- Use the structural_description and reading_order
- Build a mental model for your blind student
- Use spatial language: "On the left/right...", "Above/below..."
- For graphs: describe CURVE MOVEMENT (not just points)
- For equations: read left-to-right with linearization
- For diagrams: describe all elements and their relationships

LAYER 2: WHAT THE TEACHER IS EXPLAINING
- Summarize what the teacher is saying about this visual
- Connect the teacher's words to the visual elements

LAYER 3: THE CONCEPT BEING EXPLAINED
- Explain the underlying principle/law/formula
- Use the conceptual_hints as a starting point
- Expand with your knowledge
- Connect to previous concepts if relevant: {', '.join(previous_concepts) if previous_concepts else 'none'}

LAYER 4: HOW THE VISUAL LINKS TO THE CONCEPT
- This is CRITICAL: explain WHY this visual helps understand the concept
- Example: "The graph shows velocity leveling off, which represents terminal velocity 
  - the point where air resistance equals gravity"
- Make the connection explicit

LAYER 5: ACCESSIBILITY TECHNIQUES
- "To visualize this, imagine..."
- "Think of it like..."
- Use analogies when helpful
- Speak directly to "you"

OUTPUT: Natural, conversational explanation for your blind student.
No length limits - let the explanation be as complete as needed.
"""
    
    return prompt
Step 7: Update weave() Method
File: synthesizer/llm_synthesizer.py


def weave(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    content_type: str,
    intervention_reason: str,
    output_mode: Optional[str] = None
) -> str:
    """Generate complete explanation for blind students."""
    
    mode = output_mode or self.output_mode
    
    # Safety checks
    missing_elements = vlm_snapshot.get("missing_elements")
    if missing_elements:
        return transcript_context  # Fallback
    
    # Get context
    current_index = self._find_current_transcript_index(transcript_context)
    if self.context_manager:
        context = self.context_manager.get_context(current_index, mode)
    else:
        context = {
            "context_text": "",
            "previous_concepts": [],
            "related_principles": [],
            "topic_timeline": {}
        }
    
    # Build unified prompt
    prompt = self._build_unified_prompt(
        transcript_context=transcript_context,
        vlm_snapshot=vlm_snapshot,
        context=context,
        content_type=content_type,
        mode=mode
    )
    
    # Generate explanation
    try:
        response = self._call_llm(prompt)
        # No truncation - return full explanation
        return response
    except Exception as e:
        print(f"[Synthesizer] Error: {e}")
        return transcript_context
Summary of Changes
File	Changes	Lines to Change
snapshot_engine.py	New system prompt, new category directives, new output format	~100 lines
snapshot_engine.py	Updated _parse_json_response()	~50 lines
context_manager.py	Larger context windows	~5 lines
llm_synthesizer.py	Remove truncation from _apply_output_mode()	~30 lines
llm_synthesizer.py	New _build_unified_prompt() method	~100 lines
llm_synthesizer.py	Updated weave() method	~20 lines
Total: ~305 lines of changes

Zero breaking changes - all existing functionality preserved, new fields additive.

Expected Final Output Example
Input:

VLM: Graph showing velocity vs time leveling off
Transcript: "Now notice how velocity stops increasing..."
Context: Previous concepts (gravity, air resistance)
Output:


On the screen, you'll see a velocity-time graph. The y-axis on the left shows Velocity 
in meters per second, ranging from 0 to 50. The x-axis at the bottom shows Time in seconds, 
ranging from 0 to 10. A blue curve starts at the origin, rises steeply, then gradually 
levels off approaching a horizontal red line at 50.

Your teacher is showing this graph to demonstrate terminal velocity. The concept being 
explained is that falling objects don't accelerate forever - they reach a maximum speed 
when air resistance balances gravity.

Here's how the visual links to the concept: The curve's steep rise at the beginning shows 
the object accelerating due to gravity. As it speeds up, air resistance increases. When 
the curve levels off at the red line, that's terminal velocity - the point where the 
upward force of air resistance exactly equals the downward force of gravity, so there's 
no net force and no further acceleration.

To visualize this, imagine a skydiver falling. At first, they accelerate quickly. But as 
they gain speed, air pushes back harder. Eventually, they stop speeding up and fall at a 
constant rate - that's terminal velocity, shown by the flat part of the curve.
Ready to Implement?
This solution:

✅ Doesn't break your pipeline
✅ Handles multiple content types
✅ Provides visual + conceptual explanations
✅ Links visuals to concepts
✅ Accessible for blind students
✅ Can be implemented quickly


Critical Prompt Design for Each Content Type
Let me think carefully about what VLM must extract for each type to enable the LLM to generate your desired output format.

EQUATIONS
What a blind student needs:

Reading order (mental navigation)
Each symbol's position
Mathematical structure
What the equation represents
VLM Must Extract:


"equation": """
CRITICAL: Extract ALL equations with complete structural information.

For EACH equation provide:

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
   - Example: "Large equation centered, spanning 60% width"

3. position (Exact Coordinates):
   - "at top of frame", "below graph", "right of text"
   - This helps LLM integrate multiple content types

4. reading_order (Mental Navigation):
   - Step-by-step: "Start with integral symbol, then lower limit 0, 
     then upper limit π, then sin(x)dx"
   - For multi-line equations: line-by-line order

5. conceptual_meaning (What it represents):
   - If equation has a label (e.g., "Newton's Second Law"), include it
   - If embedded in context, describe what it models
   - Example: "This is the kinematic equation relating velocity, 
     acceleration, and time"

ADDITIONAL REQUIREMENTS:
- If multiple equations exist, describe their relationship
  - "Two equations stacked vertically, top is velocity, bottom is position"
- If equation has parts (derivatives, limits), describe structure
  - "The derivative is written as a fraction with numerator and denominator"
"""
VLM Output Example:


{
  "visual_analysis": {
    "equation": [{
      "text": "the integral from 0 to pi of sine of x dx equals 2",
      "description": "A large integral symbol spans vertically through the center. The numeral 0 sits at the symbol's bottom edge. The Greek letter pi floats at the top. To the immediate right, 'sin(x)dx' is written. An equals sign follows with the number 2 at the far right.",
      "position": "centered, occupies 70% of frame width",
      "reading_order": [
        "Start with the tall integral symbol in the center",
        "Read the lower limit 0 written in small print at the bottom",
        "Read the upper limit π floating at the top",
        "Move right to read 'sin(x)dx'",
        "End with the equals sign and result 2"
      ],
      "conceptual_meaning": "Definite integral calculating the area under sine curve from 0 to π, fundamental result in calculus"
    }]
  }
}
GRAPHS
What a blind student needs:

Coordinate system (axes setup)
Curve SHAPE and MOVEMENT (not just points)
Key points (what they mean)
What relationship is shown
VLM Must Extract:


"graph": """
CRITICAL: Extract complete graph information focusing on CURVE MOVEMENT.

1. axes (Complete Setup):
   - x_axis: label, range (min-max), units, position
   - y_axis: label, range (min-max), units, position
   - grid_lines: present? color? purpose?
   - origin: where it's positioned

2. curves (MOVEMENT DESCRIPTION - MOST IMPORTANT):
   For EACH curve/line:
   
   a. description (Shape and Movement):
      - Describe HOW the curve looks and moves
      - "starts flat at origin, rises steeply, then gradually levels off"
      - "oscillates with decreasing amplitude around x-axis"
      - "forms a bell shape, rising then falling symmetrically"
      - "straight line with constant positive slope"
      - NOT: "points at (0,0), (4,40), (10,50)" - this doesn't build mental image
      
   b. shape (Geometric Type):
      - linear, exponential, logarithmic, sinusoidal, asymptotic, bell curve, etc.
      
   c. key_points (Meaningful Points):
      - Only include points that are LABELED or have MEANING
      - intersections (with axes or other curves)
      - maxima/minima (peaks/valleys)
      - inflection points
      - intercepts
      - Each with: coordinates, what it represents, significance
      
   d. color/style: what distinguishes this curve

3. relationship (What the Graph Shows):
   - Mathematical relationship: "velocity increases with time"
   - Physical meaning: "object accelerates until reaching terminal velocity"
   - Causal relationship: "as temperature increases, reaction rate increases exponentially"
   - This is CRITICAL for LLM to explain the concept

4. legend (If Present):
   - What each color/line represents
   - Position relative to graph

5. annotations (Additional Text):
   - Labels, equations, notes
   - Their positions and what they explain

MOVEMENT DESCRIPTION GUIDELINES:
- Use active verbs: "rises", "falls", "levels off", "oscillates"
- Describe rate: "steeply", "gradually", "rapidly"
- Describe changes: "bends", "curves", "flattens"
- Describe overall behavior: "approaches but never reaches", "crosses at"
- Think: How would you describe this to someone drawing it blindly?
"""
VLM Output Example:


{
  "visual_analysis": {
    "graph": {
      "data": {
        "x_axis": {
          "label": "Time (s)",
          "range": "0-10",
          "units": "seconds",
          "position": "bottom edge of frame"
        },
        "y_axis": {
          "label": "Velocity (m/s)",
          "range": "0-50",
          "units": "meters per second",
          "position": "left edge of frame"
        },
        "grid_lines": "present in light gray, help read values"
      },
      "curves": [{
        "description": "The blue curve begins at the origin (0,0) and moves upward with increasing steepness, representing rapid acceleration. Around x=4, the curve starts to bend, becoming less steep. By x=8, the curve is nearly horizontal, gradually approaching but never reaching the red dashed line at y=50.",
        "shape": "asymptotic (approaches a horizontal line)",
        "key_points": [
          {
            "coordinates": "(0, 0)",
            "what": "starting point - object at rest with zero velocity",
            "significance": "initial condition"
          },
          {
            "coordinates": "(4, 40)",
            "what": "inflection point where acceleration decreases",
            "significance": "transition from dominant gravity to dominant air resistance"
          },
          {
            "coordinates": "(8, 48)",
            "what": "near terminal velocity",
            "significance": "air resistance nearly balancing gravity"
          }
        ],
        "color": "blue solid line"
      }, {
        "description": "A red dashed horizontal line at y=50 extends across the graph, representing the theoretical maximum velocity",
        "shape": "horizontal asymptote",
        "key_points": [],
        "color": "red dashed line",
        "meaning": "terminal velocity - 50 m/s"
      }],
      "relationship": "Velocity increases with time due to gravity, but as velocity increases, air resistance also increases. Eventually, air resistance balances gravitational force, and velocity stops increasing. The curve shows this transition from acceleration to constant velocity - terminal velocity.",
      "legend": "Blue curve: actual velocity, Red dashed line: terminal velocity",
      "annotations": [
        {
          "text": "acceleration phase",
          "position": "above the steep part of the curve"
        },
        {
          "text": "v = u + at",
          "position": "at bottom of frame"
        }
      ]
    }
  }
}
CIRCUIT
What a blind student needs:

All components and their values
How they're connected (topology)
Current flow direction
What the circuit does
VLM Must Extract:


"circuit": """
CRITICAL: Extract complete circuit information focusing on FLOW and TOPOLOGY.

1. components (Complete Inventory):
   For EACH component:
   
   a. type: What component is it?
      - battery, voltage source, resistor, capacitor, inductor, 
        diode, transistor, switch, LED, ground, etc.
      
   b. label: Component identifier if shown
      - R1, R2, C1, V1, etc.
      
   c. value: Numerical value with units if shown
      - "100 Ω", "10 μF", "12 V", "5 kΩ"
      
   d. position: Where in the diagram
      - "on the left", "in the center", "at the top"
      
   e. state: If applicable (switch: open/closed)
   
2. flow (Current Path):
   - Start point: where current begins (usually positive terminal)
   - Path description: step-by-step through components
   - Direction: clockwise, counterclockwise
   - Branches: where current splits or rejoins
   - End point: where current returns (usually negative terminal/ground)
   
   Example: "Current flows from the positive terminal upward through the 
   switch, then rightward through resistor R1, then splits - one path 
   through R2, another through R3, then rejoins before returning to 
   the negative terminal."

3. topology (Circuit Structure):
   - Overall arrangement: "series", "parallel", "series-parallel", 
     "bridge", "ladder", etc.
   - How components are grouped
   - Key relationships: "R2 and R3 in parallel, in series with R1"

4. connections (Wiring):
   - How components connect to each other
   - Junction points where wires branch
   - Ground connections and their positions

5. purpose (What the Circuit Does):
   - If label/description exists, include it
   - Based on components: "voltage divider", "filter", "amplifier", etc.
   - What phenomenon it demonstrates

6. additional_elements:
   - Current direction arrows and their meaning
   - Voltage labels at points
   - Measurement points
"""
VLM Output Example:


{
  "visual_analysis": {
    "circuit": {
      "components": [
        {
          "type": "DC voltage source (battery)",
          "label": "V1",
          "value": "12 V",
          "position": "left side of frame",
          "polarity": "positive terminal pointing up"
        },
        {
          "type": "switch",
          "label": "S1",
          "value": null,
          "position": "top wire, between battery and resistor",
          "state": "closed (current can flow)"
        },
        {
          "type": "resistor",
          "label": "R1",
          "value": "100 Ω",
          "position": "center of top wire"
        },
        {
          "type": "capacitor",
          "label": "C1",
          "value": "10 μF",
          "position": "right side"
        },
        {
          "type": "ground symbol",
          "label": null,
          "value": null,
          "position": "bottom wire, center"
        }
      ],
      "flow": "Conventional current flows from the battery's positive terminal (top of battery) upward along the left wire, through the closed switch S1, continues rightward through resistor R1, reaches capacitor C1, flows through it, then travels leftward along the bottom wire back to the battery's negative terminal. The ground symbol is connected to the bottom wire indicating the reference point.",
      "topology": "Series circuit - all components connected in a single loop. Current is the same through all components.",
      "connections": "Top wire: positive terminal → switch → resistor → capacitor. Bottom wire: capacitor → ground → negative terminal. Single continuous loop.",
      "purpose": "RC circuit showing capacitor charging through a resistor from a voltage source. Demonstrates time-dependent charging behavior.",
      "additional_elements": [
        {
          "type": "current direction arrow",
          "position": "above top wire",
          "label": "I",
          "meaning": "indicates clockwise current flow direction"
        }
      ]
    }
  }
}
DIAGRAM
What a blind student needs:

All elements with labels
Spatial relationships
How elements connect
Process flow (if applicable)
What the diagram represents
VLM Must Extract:


"diagram": """
CRITICAL: Extract complete structural information focusing on RELATIONSHIPS.

1. main_structure (What the Diagram Shows):
   - Overall subject: "free body diagram", "process flowchart", 
     "system architecture", "organizational chart", etc.
   - Main object/subject: what's being diagrammed
   - Position: where the main subject is located

2. elements (Complete Inventory):
   For EACH element:
   
   a. name/label: What is it called?
   b. type: What kind of element is it?
      - box, circle, arrow, person, object, etc.
   c. position: Exact location
      - "in the center", "top-left corner", "below the main title"
   d. description: What it looks like
   e. text content: Any text inside or associated
   f. relationships: How it connects to other elements
      - "points to", "contains", "connected to", "surrounds"

3. relationships (Connections and Arrows):
   For EACH arrow/connection:
   
   a. from: Which element does it start from?
   b. to: Which element does it point to?
   c. direction: arrow direction
   d. meaning: What does this connection represent?
      - "causes", "leads to", "contains", "flows to", "represents"
   e. label: If arrow has a label

4. spatial_layout (Overall Arrangement):
   - Organization: vertical, horizontal, circular, hierarchical, clustered
   - How elements are grouped
   - Reading order (if applicable)
   - Visual hierarchy (what's central, what's peripheral)

5. process_flow (If Applicable):
   - Start point: where process begins
   - Steps: sequence of elements
   - Decision points: branches, conditions
   - End point: where process concludes
   - Describe the journey through the diagram

6. conceptual_meaning (What it Represents):
   - If diagram has a title or purpose, include it
   - What phenomenon/system it models
   - Key principle being illustrated

SPATIAL LANGUAGE REQUIREMENTS:
- Use precise directional terms
- Describe relative positions clearly
- Explain how elements relate spatially
- Help build mental map
"""
VLM Output Example (Free Body Diagram):


{
  "visual_analysis": {
    "diagram": {
      "main_structure": "Free body diagram showing forces acting on an object",
      "elements": [
        {
          "name": "box/object",
          "type": "rectangle",
          "position": "center of frame",
          "description": "A rectangular box representing the object under analysis",
          "text_content": null,
          "relationships": ["has upward force", "has downward force", "has rightward force", "has leftward force"]
        },
        {
          "name": "normal force",
          "type": "arrow pointing upward",
          "label": "N = 50 N",
          "position": "extending upward from center of box",
          "description": "Blue arrow pointing straight up from box center",
          "text_content": "N = 50 N",
          "relationships": ["acts on box", "opposes weight"]
        },
        {
          "name": "weight/gravity",
          "type": "arrow pointing downward",
          "label": "mg = 50 N",
          "position": "extending downward from center of box",
          "description": "Blue arrow pointing straight down from box center",
          "text_content": "mg = 50 N",
          "relationships": ["acts on box", "points toward ground"]
        },
        {
          "name": "applied force",
          "type": "arrow pointing right and slightly up",
          "label": "F_app = 20 N",
          "position": "from box center, angled 30° above horizontal",
          "description": "Blue arrow pointing rightward at 30° angle",
          "text_content": "F_app = 20 N",
          "relationships": ["acts on box", "causes motion"]
        },
        {
          "name": "friction force",
          "type": "arrow pointing left",
          "label": "f = 10 N",
          "position": "from box center, pointing horizontally left",
          "description": "Blue arrow pointing straight left from box center",
          "text_content": "f = 10 N",
          "relationships": ["acts on box", "opposes motion"]
        },
        {
          "name": "angle marking",
          "type": "arc showing 30 degrees",
          "label": "30°",
          "position": "between applied force and horizontal",
          "description": "Small arc marking the 30-degree angle",
          "text_content": "30°",
          "relationships": ["shows angle of applied force"]
        },
        {
          "name": "coordinate system",
          "type": "x-y axes",
          "label": "x, y",
          "position": "bottom-right corner",
          "description": "Small coordinate system showing x and y axes",
          "text_content": "x, y",
          "relationships": ["reference frame for analysis"]
        }
      ],
      "relationships": [
        {
          "from": "normal force",
          "to": "box",
          "direction": "points toward",
          "meaning": "exerts upward force on box",
          "label": null
        },
        {
          "from": "weight",
          "to": "box",
          "direction": "points toward",
          "meaning": "gravitational force pulling down",
          "label": null
        },
        {
          "from": "applied force",
          "to": "box",
          "direction": "points toward",
          "meaning": "external push/pull causing motion",
          "label": null
        },
        {
          "from": "friction",
          "to": "box",
          "direction": "points toward",
          "meaning": "opposes motion, acts opposite to velocity",
          "label": null
        }
      ],
      "spatial_layout": "Box centered with four forces radiating outward: up (normal force), down (weight), right-up (applied force), left (friction). Forces are balanced vertically but not horizontally. Coordinate system in corner provides reference.",
      "process_flow": null,
      "conceptual_meaning": "Newton's laws of motion applied to an object on a surface. Shows all forces acting simultaneously. Vertical forces are balanced (N = mg), horizontal forces are unbalanced (F_app > f), causing acceleration. Demonstrates force diagram method for physics problems."
    }
  }
}
CODE
What a blind student needs:

The actual code
Language identification
Basic structure
Layout
VLM Must Extract:


"code": """
CRITICAL: Extract code with OCR preserving structure.

1. code_text (The Actual Code):
   - OCR all visible code
   - PRESERVE indentation (critical for Python)
   - PRESERVE line breaks
   - Include all symbols: brackets, parentheses, braces, operators
   - Include comments if visible
   - If code spans multiple files/panels, separate clearly
   
2. language (Programming Language):
   - Identify from syntax: Python, Java, C++, JavaScript, etc.
   - If uncertain, note as "unknown programming language"

3. layout (Visual Organization):
   - Single file or multiple files?
   - Split view (code on left, output on right)?
   - Panel arrangement if multiple
   - Line numbers present?

4. structure (High-Level Organization):
   - Imports and their positions
   - Class definitions (names visible)
   - Function definitions (names and parameters if visible)
   - Main execution block
   - Control structures visible (if, for, while)

5. formatting (Visual Styling):
   - Syntax highlighting colors (describe which parts have which colors)
   - Any special formatting (bold, italic)

IMPORTANT CONSTRAINTS:
- Do NOT explain what the code does
- Do NOT describe execution flow
- Do NOT provide algorithmic analysis
- Just extract what is visible
"""
VLM Output Example:


{
  "visual_analysis": {
    "code": {
      "code_text": "def calculate_fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\n\nprint(calculate_fibonacci(10))",
      "language": "Python",
      "layout": "Single file displayed in code editor with dark background. Line numbers 1-7 on left edge.",
      "structure": [
        "Line 1: Function definition 'def calculate_fibonacci(n):'",
        "Lines 2-5: Function body with conditional logic",
        "Line 7: Function call to print result"
      ],
      "formatting": "Dark gray background, blue function names, orange numbers, green comments (none present), white code text, indentation preserved with 4 spaces"
    }
  }
}
HANDWRITTEN NOTES
What a blind student needs:

Readable text
Layout organization
Any equations or diagrams
Legibility notes
VLM Must Extract:


"handwritten_notes": """
CRITICAL: OCR handwritten content with structure.

1. text (Readable Content):
   - OCR all readable handwriting
   - Mark [ILLEGIBLE] for unreadable portions
   - Preserve structure: line breaks, paragraphs
   - If equations are present, linearize them

2. layout (Organization):
   - Title: position and content
   - Sections: how content is organized
   - Bullet points or numbered lists
   - Diagrams or sketches: describe their positions
   - Equations: their positions

3. legibility_note (Quality Assessment):
   - Overall legibility: "mostly legible", "partially legible", "mostly illegible"
   - What percentage is [ILLEGIBLE]?
   - Handwriting style: cursive, print, mixed
   - Any cross-outs or corrections

4. content_types (What's Mixed In):
   - Text explanations
   - Mathematical equations
   - Diagrams or drawings
   - Formulas or derivations
   - Examples or problems

5. reading_order (Navigation):
   - How to mentally navigate the notes
   - Section breaks
   - Logical flow if visible
"""
VLM Output Example:


{
  "visual_analysis": {
    "handwritten_notes": {
      "text": "Derivatives\n\nDefinition: The derivative of a function at a point is the instantaneous rate of change.\n\nPower Rule: d/dx(x^n) = n·x^(n-1)\n\nExamples:\n- d/dx(x^2) = 2x\n- d/dx(x^3) = 3x^2",
      "layout": "Title 'Derivatives' at top in large letters. Below, two main sections vertically stacked: 'Definition' on left, 'Power Rule' on right. 'Examples' section at bottom.",
      "legibility_note": "Mostly legible. Some cross-outs visible. Formula notation is clear. Approximately 95% readable.",
      "handwriting_style": "Mixed: headings in cursive, content in print",
      "content_types": ["text explanations", "mathematical formulas", "examples"],
      "reading_order": [
        "Start with title 'Derivatives' at top",
        "Read definition section on left",
        "Read power rule section on right with formula",
        "Read examples at bottom showing rule applied"
      ]
    }
  }
}
BIOLOGY (Diagrams)
What a blind student needs:

Main structure
All parts with locations
How parts connect
What each part does
Scale/size relationships
VLM Must Extract:


"biology": """
CRITICAL: Extract structural biological information with connections.

1. main_structure (What Organism/System):
   - Name: cell, organ, system, organism
   - Type: animal cell, plant cell, heart, circulatory system, etc.
   - Position: where it's located in frame

2. components (All Parts with Locations):
   For EACH component:
   
   a. name/label: What is it called?
   b. type: organelle, tissue, organ, structure
   c. position: Exact location within main structure
      - "at the center", "surrounding the nucleus", "in the upper-left"
   d. description: What it looks like
   e. function (if labeled or inferable): What does it do?
   f. relationships: How it connects to other parts
      - "surrounded by", "contains", "connected to", "extends from"

3. element_links (How Parts Connect):
   - Describe spatial relationships between components
   - How parts relate structurally
   - Flow or process if shown (e.g., blood flow, neural signals)
   - Example: "Mitochondria are scattered throughout the cytoplasm, 
     providing energy to the cell. Endoplasmic reticulum surrounds 
     the nucleus, transporting materials."

4. scale_relationships (Relative Sizes):
   - Relative sizes of components
   - What's larger/smaller than what
   - If scale bar present, note it

5. labels (All Callouts):
   - Label text
   - Which component it points to
   - Position of label

6. conceptual_meaning (What's Being Illustrated):
   - Biological principle being shown
   - Key function or process
   - Importance to the organism
"""
VLM Output Example (Animal Cell):


{
  "visual_analysis": {
    "biology": {
      "main_structure": "Animal cell",
      "type": "eukaryotic cell",
      "position": "center of frame, circular membrane enclosing all components",
      "components": [
        {
          "name": "cell membrane",
          "type": "structure",
          "position": "outer boundary, enclosing all components",
          "description": "Pink circular membrane representing the cell boundary",
          "function": "controls what enters and exits the cell",
          "relationships": ["contains all organelles", "defines cell shape"]
        },
        {
          "name": "nucleus",
          "type": "organelle",
          "position": "center of cell",
          "description": "Large blue circle with smaller dark circle inside (nucleolus)",
          "function": "contains genetic material, controls cell activities",
          "relationships": ["surrounded by cytoplasm", "contains nucleolus"]
        },
        {
          "name": "nucleolus",
          "type": "organelle",
          "position": "inside the nucleus, slightly off-center",
          "description": "Smaller dark circle within nucleus",
          "function": "produces ribosomes",
          "relationships": ["contained within nucleus"]
        },
        {
          "name": "mitochondria",
          "type": "organelle",
          "position": "scattered throughout cytoplasm (3 visible)",
          "description": "Orange bean-shaped structures with internal folds",
          "function": "powerhouse of cell, produces ATP energy",
          "relationships": ["suspended in cytoplasm", "provide energy to cell"]
        },
        {
          "name": "endoplasmic reticulum",
          "type": "organelle",
          "position": "surrounding the nucleus",
          "description": "Purple wavy structures extending outward from nucleus",
          "function": "transports materials, synthesizes proteins",
          "relationships": ["connected to nucleus", "extends into cytoplasm"]
        },
        {
          "name": "Golgi apparatus",
          "type": "organelle",
          "position": "near nucleus, right side",
          "description": "Green stack of flattened sacs",
          "function": "packages and ships proteins",
          "relationships": ["receives from ER", "positioned near nucleus"]
        },
        {
          "name": "lysosomes",
          "type": "organelle",
          "position": "distributed around cell (2 visible)",
          "description": "Small red circles",
          "function": "break down waste and cellular debris",
          "relationships": ["scattered in cytoplasm"]
        }
      ],
      "element_links": "The cell membrane encloses all components. The nucleus sits at the center as the control center. The endoplasmic reticulum wraps around the nucleus, connecting it to the rest of the cell. Mitochondria are distributed throughout, providing energy everywhere. The Golgi apparatus receives materials from the ER and packages them. Lysosomes are scattered to handle waste cleanup. All organelles float in the cytoplasm.",
      "scale_relationships": "Nucleus is largest organelle, about 20% of cell diameter. Mitochondria are medium-sized, about 5% each. Lysosomes are smallest, about 2% each. Membrane encloses everything.",
      "labels": [
        {"text": "Cell membrane", "points_to": "outer pink circle"},
        {"text": "Nucleus", "points_to": "blue center circle"},
        {"text": "Mitochondria", "points_to": "orange bean shapes"},
        {"text": "Endoplasmic reticulum", "points_to": "purple wavy structures"},
        {"text": "Golgi apparatus", "points_to": "green stack"},
        {"text": "Lysosomes", "points_to": "small red circles"}
      ],
      "conceptual_meaning": "Illustrates the internal organization of an animal cell, showing key organelles and their relative positions. Demonstrates cellular compartmentalization and specialization of functions."
    }
  }
}
CHEMISTRY (Diagrams)
What a blind student needs:

Molecular structure
All atoms and positions
Bond types
Molecular geometry
How atoms connect
VLM Must Extract:


"chemistry": """
CRITICAL: Extract molecular structure with atomic connections.

1. molecule (What Molecule):
   - Name if labeled (e.g., "Water molecule", "Benzene")
   - Chemical formula if shown
   - Position in frame

2. atoms (All Atoms with Positions):
   For EACH atom:
   
   a. element: Which element? (H, O, C, N, etc.)
   b. position: Where is it located relative to other atoms?
      - "at the center", "upper-left", "rightmost"
   c. size: Relative size compared to other atoms
   d. charge: If indicated (+, -, 2+, etc.)

3. bonds (Connections):
   For EACH bond:
   
   a. from: Which atom?
   b. to: Which atom?
   c. type: single, double, triple
   d. representation: how is it shown (line, double line, dots)
   e. angle: if angle is marked, what is it?

4. molecular_geometry (3D Structure):
   - Overall shape: linear, bent, trigonal planar, tetrahedral, etc.
   - Bond angles if shown or inferable
   - 3D orientation if indicated (wedges, dashes)
   - Example: "Bent shape with 104.5° angle between hydrogen atoms"

5. element_links (How Atoms Connect):
   - Describe how atoms are connected in sequence
   - Central atom(s) and what connects to them
   - Ring structures if present
   - Example: "Oxygen atom at center with two hydrogen atoms bonded to it, 
     forming a V-shaped or bent structure"

6. labels (All Annotations):
   - Element symbols
   - Bond angles
   - Charges
   - Any additional text
"""
VLM Output Example (Water):


{
  "visual_analysis": {
    "chemistry": {
      "molecule": {
        "name": "Water molecule",
        "formula": "H₂O",
        "position": "center of frame"
      },
      "atoms": [
        {
          "element": "Oxygen",
          "symbol": "O",
          "position": "at the center, largest atom",
          "size": "largest, about 2x hydrogen atoms",
          "charge": null
        },
        {
          "element": "Hydrogen",
          "symbol": "H",
          "position": "upper-left of oxygen",
          "size": "smaller, about half oxygen size",
          "charge": null
        },
        {
          "element": "Hydrogen",
          "symbol": "H",
          "position": "upper-right of oxygen",
          "size": "smaller, about half oxygen size",
          "charge": null
        }
      ],
      "bonds": [
        {
          "from": "Oxygen",
          "to": "Hydrogen (left)",
          "type": "single covalent bond",
          "representation": "gray line connecting atoms",
          "angle": null
        },
        {
          "from": "Oxygen",
          "to": "Hydrogen (right)",
          "type": "single covalent bond",
          "representation": "gray line connecting atoms",
          "angle": null
        }
      ],
      "molecular_geometry": {
        "shape": "bent (V-shaped)",
        "bond_angle": "104.5 degrees between the two O-H bonds",
        "orientation": "in the plane of the diagram",
        "description": "The molecule is not linear but bent at approximately 104.5°, giving water its polar nature"
      },
      "element_links": "The oxygen atom sits at the center and forms single covalent bonds with two hydrogen atoms. The hydrogen atoms are positioned above the oxygen, one to the left and one to the right, forming a bent or V-shaped structure. The 104.5° angle between the hydrogen atoms is less than the 180° of a linear molecule, creating a polar molecule with a partial negative charge near oxygen and partial positive charges near the hydrogens.",
      "labels": [
        {"text": "H₂O", "position": "below molecule"},
        {"text": "104.5°", "position": "marked between the two O-H bonds"},
        {"text": "Polar molecule", "position": "below molecule"},
        {"text": "Bent geometry", "position": "below molecule"}
      ]
    }
  }
}
PHYSICS (Diagrams)
What a blind student needs:

Main diagram elements
Physical objects and their positions
Forces/arrows and what they represent
Spatial relationships
What phenomenon is being shown
VLM Must Extract:


"physics": """
CRITICAL: Extract physics diagram with physical relationships.

1. main_diagram (What's Shown):
   - Type: free body diagram, projectile motion, wave, field, etc.
   - Subject: what physical phenomenon is illustrated
   - Position in frame

2. elements (All Physical Objects):
   For EACH element:
   
   a. name/label: What is it?
   b. type: block, ball, ramp, pulley, spring, field lines, etc.
   c. position: Where is it located?
   d. description: What it looks like
   e. physical_properties: if shown (mass, length, angle, etc.)
   f. state: if applicable (moving, stationary, accelerating, etc.)

3. forces/arrows (All Vectors):
   For EACH force/vector:
   
   a. name: What force is it? (gravity, normal, friction, applied, etc.)
   b. label: Text label if present
   c. position: Where does it act? (from which point?)
   d. direction: Which way does it point?
   e. magnitude: If labeled or indicated
   f. meaning: What does this force represent?

4. spatial_relationships (How Elements Relate):
   - Describe positions relative to each other
   - Contact points (what touches what)
   - Distances or dimensions if shown
   - Angles if marked

5. element_links (Physical Connections):
   - How objects are connected physically
   - What constraints exist (rope, surface, pivot, etc.)
   - How forces transmit through connections
   - Example: "Block rests on ramp, connected by rope to hanging mass over pulley"

6. conceptual_meaning (What Principle is Shown):
   - Physical law or principle being illustrated
   - What phenomenon is being modeled
   - Key variables involved
   - What the diagram helps understand

7. measurements (If Present):
   - Distances, lengths, dimensions
   - Angles
   - Velocities, accelerations (if shown with vectors)
   - Times or durations
"""
VLM Output Example (Block on Inclined Plane):


{
  "visual_analysis": {
    "physics": {
      "main_diagram": "Block on inclined plane with forces",
      "subject": "Newton's laws on an inclined surface",
      "position": "center of frame",
      "elements": [
        {
          "name": "inclined plane/ramp",
          "type": "triangular surface",
          "position": "left side, sloping upward to the right at 30° angle",
          "description": "Triangle representing a ramp with the hypotenuse as the surface",
          "physical_properties": {
            "angle": "30 degrees from horizontal",
            "surface": "smooth (no friction shown)"
          },
          "state": "stationary"
        },
        {
          "name": "block",
          "type": "rectangular object",
          "position": "resting on the inclined surface",
          "description": "Rectangle representing a block or object",
          "physical_properties": {
            "mass": "labeled as 'm' on block",
            "size": "small compared to ramp"
          },
          "state": "sliding down the ramp (accelerating)"
        },
        {
          "name": "gravitational force",
          "type": "force vector (arrow)",
          "label": "mg",
          "position": "from center of block, pointing straight down",
          "direction": "vertically downward",
          "magnitude": "mg (mass times gravity)",
          "meaning": "force of gravity pulling the block downward"
        },
        {
          "name": "normal force",
          "type": "force vector (arrow)",
          "label": "N",
          "position": "from center of block, perpendicular to ramp surface",
          "direction": "perpendicular to ramp, pointing up and right",
          "magnitude": "N (normal force)",
          "meaning": "force exerted by ramp on block, perpendicular to surface"
        },
        {
          "name": "angle marking",
          "type": "arc",
          "label": "30°",
          "position": "at the bottom of the ramp, showing angle with horizontal",
          "description": "Small arc marking the 30-degree angle"
        }
      ],
      "spatial_relationships": "The ramp slopes upward to the right at 30° from horizontal. The block rests on the sloping surface. Gravity pulls straight down from the block's center. The normal force pushes perpendicular to the ramp surface from the block's center.",
      "element_links": "The block is in contact with the ramp surface along its bottom face. The normal force acts perpendicular to this contact surface. Gravity acts through the block's center of mass. The block is free to slide down the ramp (no friction shown). The ramp itself is fixed and doesn't move.",
      "conceptual_meaning": "Newton's Second Law applied to an object on an inclined plane. Demonstrates how gravity resolves into components parallel and perpendicular to the surface. The block accelerates down the ramp due to the component of gravity parallel to the surface. Shows that only the parallel component of gravity causes acceleration, while the perpendicular component is balanced by the normal force.",
      "measurements": [
        {"quantity": "ramp angle", "value": "30 degrees"},
        {"quantity": "block mass", "value": "m (variable)"}
      ]
    }
  }
}
TEXT
What a blind student needs:

The actual text content
Hierarchy and structure
Formatting that conveys meaning
Key terms
VLM Must Extract:


"text": """
CRITICAL: OCR text with structural organization.

1. text (The Actual Content):
   - OCR all visible text
   - Preserve line breaks and paragraphs
   - Include all punctuation

2. layout (Hierarchy and Organization):
   - Title: position, content, size
   - Headings: levels (H1, H2, H3), positions
   - Subheadings: positions, indentation
   - Body text: paragraphs, blocks
   - Lists: bullet points, numbered lists
   - Nesting: how items are grouped

3. formatting (Visual Styling):
   - Bold text: which words, what they emphasize
   - Italic text: which words, what they emphasize
   - Colored text: colors and their meanings
   - Highlighted or underlined text
   - Font sizes (relative): title larger than headings, etc.

4. structure (Logical Organization):
   - How information is grouped
   - Section breaks or separators
   - Indentation levels for nested content
   - Visual flow (top to bottom, left to right)

5. key_terms (Important Vocabulary):
   - Terms that are emphasized or defined
   - Vocabulary in bold or different colors
   - Terms that appear to be definitions
"""
VLM Output Example:


{
  "visual_analysis": {
    "text": {
      "text": "Newton's Laws of Motion\n\nFirst Law: An object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force. This is called inertia.\n\nSecond Law: The acceleration of an object as produced by a net force is directly proportional to the magnitude of the net force, in the same direction as the net force, and inversely proportional to the mass of the object.\n\nFormula: F = ma\n\nThird Law: For every action, there is an equal and opposite reaction.",
      "layout": "Title at top in large text. Three main sections vertically stacked. Each section has a bold heading followed by paragraph. Formula centered below Second Law section.",
      "formatting": [
        {"text": "Newton's Laws of Motion", "style": "large, bold, blue"},
        {"text": "First Law:", "style": "bold, white"},
        {"text": "inertia", "style": "yellow highlighted"},
        {"text": "Second Law:", "style": "bold, white"},
        {"text": "F = ma", "style": "larger, centered, blue"},
        {"text": "Third Law:", "style": "bold, white"}
      ],
      "structure": [
        "Section 1: First Law with definition and example (inertia)",
        "Section 2: Second Law with mathematical principle and formula",
        "Section 3: Third Law with principle"
      ],
      "key_terms": [
        {"term": "inertia", "context": "First Law", "emphasis": "yellow highlight"},
        {"term": "unbalanced force", "context": "First Law", "emphasis": "none"},
        {"term": "acceleration", "context": "Second Law", "emphasis": "none"},
        {"term": "net force", "context": "Second Law", "emphasis": "none"},
        {"term": "mass", "context": "Second Law", "emphasis": "none"},
        {"term": "action-reaction", "context": "Third Law", "emphasis": "none"}
      ]
    }
  }
}
Summary: Unified VLM Output Format
All content types use this unified format:


{
  "content_type": "detected_primary_type",
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
  
  "missing_elements": null
}
Ready to Implement?
These prompts are designed to extract EXACTLY what the LLM needs to generate your desired output format:

✅ Visual description (from structural_description + reading_order)

✅ What teacher is saying (transcript provided separately)

✅ Concept being explained (from conceptual_hints + LLM knowledge)

✅ How visuals link to concept (from visual_analysis relationships + conceptual_hints)

✅ Accessibility techniques (LLM adds these based on extracted data)

Should I now implement these prompts in the code?


LLM Prompts Design for Enhanced VLM Output
Let me design prompts that will transform the VLM's unified output into your desired explanation format for each mode.

Unified Prompt Strategy
All modes share the same structure but vary in depth and length.

LLM Input:


1. VLM's unified output (visual_analysis, structural_description, reading_order, conceptual_hints)
2. Transcript segment (what teacher is saying)
3. Context (previous concepts, surrounding text)
LLM Output (Your Desired Format):


Layer 1: Visual Description (what's on screen)
Layer 2: What Teacher is Saying (transcript context)
Layer 3: Concept Being Explained (underlying principle)
Layer 4: How Visual Links to Concept (the connection)
+ Accessibility techniques throughout
Brief Mode Prompt (2-3 sentences)
Goal: Quick, essential information for rapid understanding


def _build_brief_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Brief mode: 2-3 sentences covering essentials.
    
    Focus: Key visual + main concept + why shown now.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    
    prompt = f"""
YOU ARE: Expert teacher for blind students. Your student cannot see the screen.

TASK: Generate a BRIEF (2-3 sentences) accessible explanation for this visual content.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. CONCEPTUAL HINTS:
{conceptual_hints}

5. WHAT TEACHER IS SAYING:
"{transcript_context}"

6. PREVIOUS CONCEPTS:
{', '.join(previous_concepts) if previous_concepts else 'None'}

YOUR EXPLANATION (2-3 sentences maximum):

Must include these THREE elements in one smooth narrative:

ELEMENT 1 - VISUAL SNAPSHOT (1 sentence):
- Use the structural_description and visual_analysis
- Describe ONLY the most important visual elements
- Use spatial language: "On the screen, you'll find..."
- Keep it focused on what matters most

ELEMENT 2 - WHAT'S BEING EXPLAINED (1 sentence):
- Summarize what the teacher is showing
- State the main concept or principle

ELEMENT 3 - WHY IT MATTERS (1 sentence):
- How this visual helps understand the concept
- Connect visual to concept briefly

ACCESSIBILITY:
- Use "you" and direct address
- Build a quick mental image
- Keep sentences clear and focused

OUTPUT: 2-3 sentences total. No more. Make every word count.
"""
    
    return prompt
Explanatory Mode Prompt (4-6 sentences)
Goal: Comprehensive explanation with all 4 layers


def _build_explanatory_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Explanatory mode: 4-6 sentences with complete coverage.
    
    Focus: Visual description + transcript context + concept + visual-concept link.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    reading_order = vlm_snapshot.get("reading_order", [])
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    context_text = context.get("context_text", "")
    
    prompt = f"""
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate a COMPREHENSIVE (4-6 sentences) accessible explanation for this visual content.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{'; '.join(reading_order) if reading_order else 'Not specified'}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. SURROUNDING CONTEXT:
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

YOUR EXPLANATION (4-6 sentences):

Organize into these FOUR LAYERS in one smooth narrative:

LAYER 1 - VISUAL DESCRIPTION (2 sentences):
- Use the structural_description and reading_order
- Build a clear mental model for your student
- Use spatial language consistently: "On the left/right...", "Above/below..."
- For graphs: describe CURVE MOVEMENT, not just points
- For equations: read left-to-right with linearization
- For diagrams: describe elements and their relationships
- "To visualize this, imagine..." when helpful

LAYER 2 - WHAT THE TEACHER IS EXPLAINING (1 sentence):
- Summarize what the teacher is saying about this visual
- Connect teacher's words to the visual elements you just described
- "Your teacher is showing this to explain..."

LAYER 3 - THE CONCEPT BEING EXPLAINED (1-2 sentences):
- Explain the underlying principle, law, or formula
- Use the conceptual_hints as your starting point
- Expand with your knowledge
- Connect to previous concepts if relevant: {', '.join(previous_concepts) if previous_concepts else 'none'}
- "The concept being demonstrated is..." or "This illustrates the principle of..."

LAYER 4 - HOW THE VISUAL LINKS TO THE CONCEPT (1-2 sentences):
- CRITICAL: Explain WHY this visual helps understand the concept
- Make the connection explicit and clear
- Use specific details from the visual_analysis
- Example format: "The [visual element] shows [behavior], which represents [concept aspect] - this demonstrates [principle]"
- "You can see this connection in how the [visual feature] relates to [concept element]"

ACCESSIBILITY TECHNIQUES:
- "To visualize this, imagine..."
- "Think of it like..."
- Use analogies when helpful
- Speak directly to "you"
- Build mental models step by step

OUTPUT: 4-6 sentences total. Natural, conversational tone. Complete coverage.
"""
    
    return prompt
Detailed Mode Prompt (8-12 sentences)
Goal: Exhaustive, concept-rich explanation with deep connections


def _build_detailed_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Detailed mode: 8-12 sentences with deep exploration.
    
    Focus: Complete visual + deep concept + connections + rich accessibility.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    reading_order = vlm_snapshot.get("reading_order", [])
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    related_principles = context.get("related_principles", [])
    topic_timeline = context.get("topic_timeline", {})
    context_text = context.get("context_text", "")
    
    # Get cross-domain knowledge
    cross_domain = self._get_cross_domain_explanations(detected_types[0] if detected_types else "unknown", previous_concepts)
    
    prompt = f"""
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate an EXHAUSTIVE (8-12 sentences) concept-rich accessible explanation for this visual content.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{'; '.join(reading_order) if reading_order else 'Not specified'}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. FULL LECTURE CONTEXT (±2 minutes):
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

9. RELATED PRINCIPLES:
{', '.join(related_principles) if related_principles else 'None'}

10. TOPIC TIMELINE:
{topic_timeline}

11. CROSS-DOMAIN CONNECTIONS:
{cross_domain}

YOUR EXPLANATION (8-12 sentences):

Organize into these FIVE LAYERS in one smooth narrative:

LAYER 1 - COMPLETE VISUAL DESCRIPTION (3 sentences):
- Use the structural_description and reading_order extensively
- Build a complete, detailed mental model
- Describe EVERYTHING that matters
- Use spatial language systematically
- For graphs: complete curve description, axes, key points, relationship
- For equations: reading order, each element's position, structural layout
- For diagrams: all elements, their positions, relationships, connections
- "To visualize this, imagine..." - build the mental image completely

LAYER 2 - WHAT THE TEACHER IS EXPLAINING (1-2 sentences):
- Summarize what the teacher is saying
- Connect teacher's words to specific visual elements
- Explain the context in the lecture flow
- "Your teacher is showing this at this point to explain..."

LAYER 3 - DEEP CONCEPTUAL FRAMEWORK (2-3 sentences):
- Explain the underlying principle, law, or formula in depth
- Use conceptual_hints as foundation, then expand significantly
- Include mathematical/physical meaning, historical context if relevant
- Explain boundary conditions, limitations, or special cases
- Connect to related principles: {', '.join(related_principles) if related_principles else 'none'}
- "The fundamental principle being demonstrated is..."
- "This concept is important because..."

LAYER 4 - HOW THE VISUAL LINKS TO THE CONCEPT (2-3 sentences):
- CRITICAL: Make multiple explicit connections between visual and concept
- Explain how each key visual element represents a concept aspect
- Use specific details from visual_analysis
- Show cause-and-effect relationships visually
- "The [visual element] shows [behavior], which represents [concept aspect]"
- "You can see this in how the [visual feature] corresponds to [concept element]"
- "This visual demonstrates the principle by showing [connection]"

LAYER 5 - CROSS-DOMAIN CONNECTIONS & ACCESSIBILITY (1-2 sentences):
- Connect to previous concepts: {', '.join(previous_concepts) if previous_concepts else 'none'}
- Use cross-domain connections if relevant: {cross_domain}
- Provide analogies, mental models, rhetorical questions
- "Think of it like..." or "You can imagine..."
- "This connects to what we discussed earlier about [previous concept] because..."
- Check understanding: "Can you picture how [visual] relates to [concept]?"

ACCESSIBILITY TECHNIQUES (Rich):
- Multiple analogies when helpful
- "To visualize this, imagine..." with vivid descriptions
- "Think of it like..." with relatable comparisons
- Use rhetorical questions to engage: "Have you ever...?"
- Speak directly to "you" throughout
- Build and reinforce mental models
- Use sensory language when appropriate

OUTPUT: 8-12 sentences total. Comprehensive, conversational, conceptually rich.
"""
    
    return prompt
Updated weave() Method

def weave(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    content_type: str,
    intervention_reason: str,
    output_mode: Optional[str] = None
) -> str:
    """
    Generate complete explanation for blind students using enhanced VLM output.
    
    Uses unified prompt strategy with mode-specific depth.
    """
    
    mode = output_mode or self.output_mode
    
    # Safety checks
    missing_elements = vlm_snapshot.get("missing_elements")
    if missing_elements:
        print(f"[Synthesizer] VLM marked as missing - transcript-only fallback")
        return transcript_context
    
    # Cross-domain hallucination check
    if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
        print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
        return transcript_context
    
    # Get context
    current_index = self._find_current_transcript_index(transcript_context)
    if self.context_manager:
        context = self.context_manager.get_context(current_index, mode)
    else:
        context = {
            "context_text": "",
            "previous_concepts": [],
            "related_principles": [],
            "topic_timeline": {}
        }
    
    # Linearize VLM data for math
    linearized_snapshot = self._linearize_vlm_data(vlm_snapshot)
    
    # Build mode-specific prompt
    if mode == "brief":
        prompt = self._build_brief_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    elif mode == "detailed":
        prompt = self._build_detailed_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    else:  # explanatory (default)
        prompt = self._build_explanatory_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    
    # Generate explanation
    try:
        explanation = self._call_llm(prompt)
        
        # Post-process: math linearization (double-check)
        explanation = self.math_linearizer.linearize(explanation)
        
        return explanation
        
    except Exception as e:
        print(f"[Synthesizer] Error generating explanation: {e}", exc_info=True)
        return transcript_context
Remove _apply_output_mode() Truncation

def _apply_output_mode(
    self,
    audio_script: str,
    output_mode: str
) -> str:
    """
    Return audio_script as-is.
    
    No truncation - prompts control length through sentence count guidance.
    """
    if not audio_script:
        return ""
    
    # No sentence limits - return full explanation
    return audio_script
Example Outputs
Brief Mode Output (Graph)

On the screen, you'll see a velocity-time graph with the y-axis showing Velocity in meters per second and the x-axis showing Time in seconds. A blue curve starts at zero, rises steeply, then gradually levels off at 50 m/s. Your teacher is showing this to demonstrate terminal velocity - when air resistance balances gravity and the object stops accelerating.
Explanatory Mode Output (Graph)

On the screen, you'll see a velocity-time graph. The y-axis on the left shows Velocity in meters per second, ranging from 0 to 50. The x-axis at the bottom shows Time in seconds, ranging from 0 to 10. A blue curve starts at the origin, rises steeply, then gradually levels off approaching a red dashed line at 50. Your teacher is showing this graph to explain why falling objects don't accelerate forever. The concept being demonstrated is terminal velocity - the maximum speed an object reaches when the upward force of air resistance exactly equals the downward force of gravity. The curve's steep rise at the beginning shows the object accelerating due to gravity, while the flat section represents terminal velocity where air resistance has caught up to gravity, so there's no net force and no further acceleration. To visualize this, imagine a skydiver falling - they speed up quickly at first, but as they gain speed, air pushes back harder until they stop accelerating and fall at a constant rate.
Detailed Mode Output (Graph)

On the screen, you'll see a velocity-time graph with a complete coordinate system. The y-axis runs along the left edge, labeled "Velocity (m/s)" with tick marks from 0 to 50. The x-axis runs along the bottom edge, labeled "Time (s)" with tick marks from 0 to 10. Light gray grid lines help read values. A smooth blue curve starts at the origin (0,0), climbs rapidly through the middle section, and gradually flattens as it approaches y=50. A red dashed horizontal line at y=50 is labeled "Terminal velocity." To visualize this, imagine plotting speed on the vertical axis and time on the horizontal, then tracing how speed changes over time.

Your teacher is showing this graph at this point in the lecture to explain why falling objects reach a maximum speed, connecting our discussion of gravity and air resistance. The fundamental principle being demonstrated is Newton's Second Law applied with air resistance - the net force equals mass times acceleration. When the skydiver first jumps, gravity dominates and acceleration is high. As velocity increases, air resistance grows proportionally. The concept is important because it explains real-world phenomena like terminal velocity, which has applications from skydiving to particle physics.

Here's how the visual links to the concept: The curve's steep rise from the origin represents the initial phase where gravitational force (mg) greatly exceeds air resistance, resulting in large net force and high acceleration. Around x=4 seconds, the curve begins to bend - this is the inflection point where air resistance has grown significantly and is approaching the magnitude of gravitational force. The flat section approaching y=50 represents terminal velocity - the point where air resistance exactly equals gravity (F_air = mg), so net force is zero and acceleration stops. The red dashed line marks this theoretical maximum speed that can never be exceeded. You can see this connection in how the curve's shape directly mirrors the changing balance of forces - steep when forces are unbalanced, flat when balanced.

This connects to what we discussed earlier about Newton's laws and force diagrams. Remember how free body diagrams showed multiple forces acting simultaneously? This graph shows the result of those forces over time. Think of it like this: the curve's shape tells the story of a battle between two forces - gravity constantly pulling down, and air resistance growing stronger as speed increases. When they reach a stalemate (equal magnitudes), the curve goes flat. Can you picture how the changing slope of the curve represents the changing net force and acceleration?
Summary
Mode	Sentences	Focus	Layers
Brief	2-3	Essentials	Visual snapshot + main concept + why matters
Explanatory	4-6	Complete	Visual + transcript + concept + link
Detailed	8-12	Exhaustive	All layers + deep connections + cross-domain
All modes:

✅ Use unified VLM output format
✅ Generate your desired 4-layer output structure
✅ Include accessibility techniques
✅ No truncation (prompts control length)
Ready to implement these prompts in the code?

i do not want the sentence limit , just the modes differ in detial , and how deep should be explained
Revised LLM Prompts - Depth-Based (No Sentence Limits)
You're right. Let me redesign the prompts to focus on depth of explanation rather than sentence counts.

Brief Mode Prompt (Surface Level, Essential Information)
Focus: Quick mental image + core concept


def _build_brief_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Brief mode: Surface-level explanation.
    
    Focus: Essential visual elements + main concept + key connection.
    Quick overview for rapid understanding.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    
    prompt = f"""
YOU ARE: Expert teacher for blind students. Your student cannot see the screen.

TASK: Generate a BRIEF, surface-level accessible explanation for this visual content.

EXPLANATION DEPTH: Surface level only. Cover essentials. Quick overview.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. CONCEPTUAL HINTS:
{conceptual_hints}

5. WHAT TEACHER IS SAYING:
"{transcript_context}"

6. PREVIOUS CONCEPTS:
{', '.join(previous_concepts) if previous_concepts else 'None'}

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
"""
    
    return prompt
Explanatory Mode Prompt (Standard Depth)
Focus: Complete coverage, all necessary aspects


def _build_explanatory_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Explanatory mode: Standard depth explanation.
    
    Focus: Complete visual description + full concept explanation + clear connections.
    Comprehensive but not exhaustive.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    reading_order = vlm_snapshot.get("reading_order", [])
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    context_text = context.get("context_text", "")
    
    prompt = f"""
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate a STANDARD-DEPTH accessible explanation for this visual content.

EXPLANATION DEPTH: Complete coverage. Cover all necessary aspects thoroughly but concisely.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{'; '.join(reading_order) if reading_order else 'Not specified'}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. SURROUNDING CONTEXT:
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

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
- Connect to previous concepts if relevant: {', '.join(previous_concepts) if previous_concepts else 'none'}
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
"""
    
    return prompt
Detailed Mode Prompt (Deep Exploration)
Focus: Exhaustive coverage, deep connections, cross-domain


def _build_detailed_prompt(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    context: Dict
) -> str:
    """
    Detailed mode: Deep exploration explanation.
    
    Focus: Exhaustive visual description + deep conceptual framework + multiple connections + cross-domain.
    Comprehensive coverage for thorough understanding.
    """
    
    detected_types = vlm_snapshot.get("detected_types", [])
    visual_analysis = vlm_snapshot.get("visual_analysis", {})
    structural_description = vlm_snapshot.get("structural_description", "")
    reading_order = vlm_snapshot.get("reading_order", [])
    conceptual_hints = vlm_snapshot.get("conceptual_hints", "")
    
    previous_concepts = context.get("previous_concepts", [])
    related_principles = context.get("related_principles", [])
    topic_timeline = context.get("topic_timeline", {})
    context_text = context.get("context_text", "")
    
    # Get cross-domain knowledge
    cross_domain = self._get_cross_domain_explanations(detected_types[0] if detected_types else "unknown", previous_concepts)
    
    prompt = f"""
YOU ARE: Expert teacher for blind students with 20+ years experience.
YOUR STUDENT: Cannot see the screen. Builds mental models through sound.

TASK: Generate an EXHAUSTIVE, DEEP accessible explanation for this visual content.

EXPLANATION DEPTH: Comprehensive and deep. Explore fully, make connections, provide rich context.

INPUT DATA:

1. CONTENT TYPES DETECTED: {', '.join(detected_types)}

2. VISUAL ANALYSIS (from VLM):
{json.dumps(visual_analysis, indent=2)}

3. STRUCTURAL DESCRIPTION:
{structural_description}

4. READING ORDER (for mental navigation):
{'; '.join(reading_order) if reading_order else 'Not specified'}

5. CONCEPTUAL HINTS (from VLM):
{conceptual_hints}

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. FULL LECTURE CONTEXT (±2 minutes):
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

9. RELATED PRINCIPLES:
{', '.join(related_principles) if related_principles else 'None'}

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
- Connect to related principles: {', '.join(related_principles) if related_principles != 'None' else 'various principles'}
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
- Connect to previous concepts: {', '.join(previous_concepts) if previous_concepts else 'none'}
- Use cross-domain connections if relevant: {cross_domain}
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
"""
    
    return prompt
Updated weave() Method (No Truncation)

def weave(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    content_type: str,
    intervention_reason: str,
    output_mode: Optional[str] = None
) -> str:
    """
    Generate complete explanation for blind students using enhanced VLM output.
    
    Modes differ in depth of explanation, not sentence limits.
    """
    
    mode = output_mode or self.output_mode
    
    # Safety checks
    missing_elements = vlm_snapshot.get("missing_elements")
    if missing_elements:
        print(f"[Synthesizer] VLM marked as missing - transcript-only fallback")
        return transcript_context
    
    # Cross-domain hallucination check
    if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
        print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
        return transcript_context
    
    # Get context
    current_index = self._find_current_transcript_index(transcript_context)
    if self.context_manager:
        context = self.context_manager.get_context(current_index, mode)
    else:
        context = {
            "context_text": "",
            "previous_concepts": [],
            "related_principles": [],
            "topic_timeline": {}
        }
    
    # Linearize VLM data for math
    linearized_snapshot = self._linearize_vlm_data(vlm_snapshot)
    
    # Build mode-specific prompt
    if mode == "brief":
        prompt = self._build_brief_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    elif mode == "detailed":
        prompt = self._build_detailed_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    else:  # explanatory (default)
        prompt = self._build_explanatory_prompt(
            transcript_context=transcript_context,
            vlm_snapshot=linearized_snapshot,
            context=context
        )
    
    # Generate explanation
    try:
        explanation = self._call_llm(prompt)
        
        # Post-process: math linearization (double-check)
        explanation = self.math_linearizer.linearize(explanation)
        
        # NO TRUNCATION - return full explanation as generated
        return explanation
        
    except Exception as e:
        print(f"[Synthesizer] Error generating explanation: {e}", exc_info=True)
        return transcript_context
Mode Comparison
Aspect	Brief Mode	Explanatory Mode	Detailed Mode
Visual Description	Essentials only	Complete	Exhaustive, every detail
Concept Explanation	Core concept stated	Full explanation	Deep framework, meaning, significance
Visual-Concept Links	One key connection	Clear, explicit connections	Multiple connections, "why" explained
Context	Current moment only	+ surrounding context	+ full lecture flow, timeline
Previous Concepts	Not emphasized	Connected if relevant	Thoroughly connected and explained
Related Principles	Not included	Mentioned if needed	Discussed with connections
Cross-Domain	Not included	Not included	Full cross-domain connections
Analogies	Minimal or none	When helpful	Multiple, varied analogies
Depth	Surface	Standard	Deep exploration
Length	Short (naturally)	Medium (naturally)	Long (naturally)
Example: Graph Showing Terminal Velocity
Brief Mode Output

On the screen, you'll see a velocity-time graph with velocity on the y-axis and time on the x-axis. A blue curve starts at zero, rises steeply, then levels off at 50 m/s. Your teacher is showing this to demonstrate terminal velocity - when air resistance equals gravity and the object stops accelerating.
Explanatory Mode Output

On the screen, you'll see a velocity-time graph. The y-axis on the left shows Velocity in meters per second, ranging from 0 to 50. The x-axis at the bottom shows Time in seconds, ranging from 0 to 10. A blue curve starts at the origin, rises steeply, then gradually levels off approaching a red dashed line at 50. Your teacher is showing this graph to explain why falling objects don't accelerate forever. The concept being demonstrated is terminal velocity - the maximum speed an object reaches when the upward force of air resistance exactly equals the downward force of gravity. The curve's steep rise at the beginning shows the object accelerating due to gravity, while the flat section represents terminal velocity where air resistance has caught up to gravity, so there's no net force and no further acceleration. To visualize this, imagine a skydiver falling - they speed up quickly at first, but as they gain speed, air pushes back harder until they stop accelerating and fall at a constant rate.
Detailed Mode Output

On the screen, you'll see a velocity-time graph with a complete coordinate system. The y-axis runs along the left edge, labeled "Velocity (m/s)" with tick marks from 0 to 50. The x-axis runs along the bottom edge, labeled "Time (s)" with tick marks from 0 to 10. Light gray grid lines help read values. A smooth blue curve starts at the origin (0,0), climbs rapidly through the middle section, and gradually flattens as it approaches y=50. A red dashed horizontal line at y=50 is labeled "Terminal velocity." To visualize this, imagine plotting speed on the vertical axis and time on the horizontal, then tracing how speed changes over time.

Your teacher is showing this graph at this point in the lecture to explain why falling objects reach a maximum speed, connecting our discussion of gravity and air resistance. The fundamental principle being demonstrated is Newton's Second Law applied with air resistance - the net force equals mass times acceleration. When the skydiver first jumps, gravity dominates and acceleration is high. As velocity increases, air resistance grows proportionally. The concept is important because it explains real-world phenomena from skydiving to particle physics, and it's a classic example of forces reaching equilibrium.

Here's how the visual links to the concept: The curve's steep rise from the origin represents the initial phase where gravitational force (mg) greatly exceeds air resistance, resulting in large net force and high acceleration. Around x=4 seconds, the curve begins to bend - this is the inflection point where air resistance has grown significantly and is approaching the magnitude of gravitational force. The flat section approaching y=50 represents terminal velocity - the point where air resistance exactly equals gravity (F_air = mg), so net force is zero and acceleration stops. The red dashed line marks this theoretical maximum speed that can never be exceeded. You can see this connection in how the curve's shape directly mirrors the changing balance of forces - steep when forces are unbalanced, flat when balanced. This is a beautiful example of how mathematical graphs can tell physical stories.

This connects to what we discussed earlier about Newton's laws and force diagrams. Remember how free body diagrams showed multiple forces acting simultaneously? This graph shows the result of those forces over time. Think of it like this: the curve's shape tells the story of a battle between two forces - gravity constantly pulling down, and air resistance growing stronger as speed increases. When they reach a stalemate (equal magnitudes), the curve goes flat. This same principle of opposing forces reaching equilibrium appears in many other contexts - in chemistry as reaction equilibrium, in economics as market equilibrium, in biology as homeostasis. Can you picture how the changing slope of the curve represents the changing net force and acceleration? Have you ever felt air resistance when sticking your hand out a car window - that force increases with speed, just like this graph shows?
Summary
✅ No sentence limits - prompts guide depth, not length
✅ Modes differ in depth - surface vs. standard vs. deep exploration
✅ Natural length - let content determine how long the explanation should be
✅ Same 4-layer structure across all modes, just different depth levels
Ready to implement?
Implementation Plan - What I Will Do
Overview
I will implement the enhanced VLM and LLM prompts to generate your desired output format:

Visual description
What teacher is saying
Concept being explained
How visuals link to concept
Changes I Will Make
1. VLM Engine (vlm_engine/snapshot_engine.py)
Enhance system prompt - Add instructions to detect and describe ALL content types, not just the primary type
Redesign category directives - New detailed prompts for each content type (equation, graph, circuit, diagram, code, handwritten, biology, chemistry, physics, text)
Update output parser - Handle new unified format with fields: detected_types, visual_analysis, structural_description, reading_order, conceptual_hints
2. Context Manager (synthesizer/context_manager.py)
Expand context windows: brief=30s (was 10s), explanatory=60s (was 30s), detailed=120s (was 50s)
3. LLM Synthesizer (synthesizer/llm_synthesizer.py)
Remove sentence limits from _apply_output_mode() - no truncation, let prompts control depth
Add three new prompt builders:
_build_brief_prompt() - Surface level, essentials only
_build_explanatory_prompt() - Standard depth, complete coverage
_build_detailed_prompt() - Deep exploration, exhaustive
Update weave() method - Use new prompts, no truncation
What I Will NOT Change
✅ Existing output fields (layout, text_readout, spatial_map, colors_styles, missing_elements) - kept for backward compatibility
✅ API signatures - analyze_frame() still accepts primary_type
✅ Pipeline flow - no changes to how modules connect
✅ Existing methods that work - only enhancements, no breaking changes
Confirmation
Will this break the pipeline? NO - All changes are additive or enhance existing functionality. Existing fields remain. New fields are added. No API changes that would break calling code.

Will this give your desired output? YES - The prompts are designed to generate the 4-layer structure: visual description + teacher's words + concept + visual-concept link, with depth varying by mode.

Should I proceed with implementation?

Update Todos

Enhance VLM system prompt to detect and describe all content types

Redesign VLM category directives for each content type

Update VLM output parser for new unified format

Expand context windows in Context Manager

Remove sentence limits from LLM _apply_output_mode()

Add new LLM prompt builders (brief, explanatory, detailed)

Update LLM weave() method to use new prompts