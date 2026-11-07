"""
LLM Synthesizer Module (M3)

Audio Weaver" - Fuse teacher's spoken words with VLM's structural data into a single, seamless TTS script.
Enhanced with 3-mode intervention explanation system (brief, explanatory, detailed).
"""

import copy
import json
import logging
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from llm_interface import OllamaClient, OllamaError
from .math_linearizer import MathLinearizer
from .knowledge_base import KnowledgeBase
from .context_manager import ContextManager
from stage_tracing import trace_stage


logger = logging.getLogger(__name__)


def print(*values, **_kwargs):  # noqa: A001
    """Keep legacy synthesis diagnostics available only at DEBUG level."""
    logger.debug(" ".join(str(value) for value in values))


class ExplanationSynthesisError(RuntimeError):
    """A visual explanation could not be produced without a misleading fallback."""

    error_code = "generation_failed"


class VisualEvidenceUnavailableError(ExplanationSynthesisError):
    """The VLM did not provide usable visual evidence."""

    error_code = "visual_evidence_unavailable"


class TextSynthesisUnavailableError(ExplanationSynthesisError):
    """The explanation model did not produce usable text."""

    error_code = "text_synthesis_unavailable"


@dataclass
class SynthesizedTextResult:
    """Result of text synthesis process."""
    audio_script: str
    fallback_used: bool = False
    warning: Optional[str] = None


class LLMSynthesizer:
    """
    Audio Weaver that fuses transcript + VLM data via LLM.

    Implements Unified Injection Rule: VLM data never discarded for interventions
    unless marked as missing or cross-domain hallucination detected.
    """

    # Default generation parameters
    DEFAULT_GENERATION_PARAMS = {
        "num_predict": 256,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1
    }

    MAX_CANONICAL_PROMPT_CHARS = 18000
    MAX_COVERAGE_PROMPT_CHARS = 14000
    MAX_CONTEXT_PROMPT_CHARS = 6000
    MAX_TRANSCRIPT_PROMPT_CHARS = 6000

    CATEGORY_NARRATION_DIRECTIVES = {
        "CT-01": "Charts: identify chart kind; name series, axes, units, and scale; report supported values and visible trends. Do not turn approximate geometry into exact values.",
        "CT-02": "Mathematics: preserve exact operators and signs, bracket and fraction scope, superscripts, subscripts, bounds, and aligned transformations. Give a TTS-readable rendering without changing the expression.",
        "CT-03": "Science and technical structures: name components and supported containment, bonds, directions, leader-line targets, and connectivity. Distinguish circuit junctions from wire crossings and do not invent component functions.",
        "CT-04": "Flows, processes, and networks: identify node roles, directed edges, branch labels, loops, and reading order. Keep ambiguous arrow direction explicit.",
        "CT-05": "Tables: preserve row and column headers, cell-to-header associations, and units. State priority comparisons instead of reciting an irrelevant full grid.",
        "CT-06": "Instructional text: preserve exact headings, definitions, warning negation, list hierarchy, and meaningful emphasis; omit decorative text.",
        "CT-07": "Geometry: name points, shapes, axes, markings, angles, conditions, and units. Do not infer measurements or relationships merely because a drawing appears to scale.",
        "CT-08": "Programming: preserve exact tokens, operators, capitalization, and indentation; distinguish visible source code from displayed output and from predicted execution.",
        "CT-09": "User interfaces: name controls, positions, and visible enabled, disabled, or selected states and results. Do not predict what a click would do.",
        "CT-10": "Maps: state supported orientation, legend mappings, labeled places, and spatial relations. Do not invent distance or direction.",
        "CT-11": "Observational images: describe visible attributes, contrasts, patterns, and uncertainty. Do not infer identity, hidden events, causes, or chronology.",
        "CT-12": "Physical scenes: describe apparatus, objects, arrangement, arrows, and supported physical relations. A single frame does not establish motion or change over time.",
    }

    LEGACY_CATEGORY_IDS = {
        "graph": "CT-01", "chart": "CT-01", "equation": "CT-02", "math": "CT-02",
        "biology": "CT-03", "chemistry": "CT-03", "circuit": "CT-03", "diagram": "CT-03",
        "flowchart": "CT-04", "process": "CT-04", "network": "CT-04", "table": "CT-05",
        "text": "CT-06", "instructional_text": "CT-06", "geometry": "CT-07", "code": "CT-08",
        "programming": "CT-08", "software_interface": "CT-09", "interface": "CT-09",
        "map": "CT-10", "observational_image": "CT-11", "physical_illustration": "CT-12",
        "physical_scene": "CT-12",
    }

    # Cross-domain mismatch mappings for hallucination detection
    CROSS_DOMAIN_MISMATCHES = {
        "equation": ["person", "man", "woman", "face", "people", "teacher"],
        "graph": ["person", "man", "woman", "face", "people"],
        "circuit": ["person", "man", "woman", "face", "text", "paragraph"],
        "code": ["person", "face", "diagram", "graph"],
        "diagram": ["person", "man", "woman", "face", "text", "paragraph"],
        "biology": ["equation", "graph", "code"],
        "chemistry": ["equation", "graph", "code"],
        "text": ["circuit", "equation", "graph", "code"],
    }

    def __init__(
        self,
        ollama_url: str,
        model: str,
        timeout: int,
        max_retries: int,
        transcripts: Optional[List],
        output_mode: str
    ):
        """
        Initialize LLM Synthesizer.

        Args:
            ollama_url: Ollama API base URL
            model: Model name
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            transcripts: List of TranscriptEntry objects (for context gathering) - REQUIRED
            output_mode: "standard", "brief", "explanatory", or "detailed" - REQUIRED
        """
        self.client = OllamaClient(
            base_url=ollama_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries
        )

        # Enhanced features for 3-mode explanation system
        self.math_linearizer = MathLinearizer()
        self.knowledge_base = KnowledgeBase()
        self.context_manager = None
        self.transcripts = transcripts if transcripts else []
        self.output_mode = output_mode

        if self.transcripts:
            try:
                self.context_manager = ContextManager(self.transcripts)
            except Exception as e:
                print(f"[Synthesizer] Warning: Could not initialize ContextManager: {e}")

        print(f"[Synthesizer] Initialized with mode: {output_mode}")

    def _detect_cross_domain_hallucination(
        self,
        vlm_snapshot: Dict,
        content_type: str
    ) -> bool:
        """
        Check if VLM describes content that doesn't match expected content_type.

        Args:
            vlm_snapshot: VLM output from Qwen-VL
            content_type: Expected content type

        Returns:
            True if hallucination detected
        """
        evidence = vlm_snapshot.get("visual_evidence")
        if isinstance(evidence, dict) and evidence.get("category_ids"):
            # Canonical mixed-category evidence is stronger than the legacy
            # keyword heuristic.  For example, a circuit can legitimately
            # contain text labels and a code screenshot can include a graph.
            return False

        if content_type not in self.CROSS_DOMAIN_MISMATCHES:
            return False

        text_readout = vlm_snapshot.get("text_readout", "").lower()
        mismatch_keywords = self.CROSS_DOMAIN_MISMATCHES[content_type]

        return any(keyword in text_readout for keyword in mismatch_keywords)

    def weave(
        self,
        transcript_context: str,
        vlm_snapshot: Dict,
        content_type: str,
        intervention_reason: str,
        output_mode: Optional[str] = None,
        require_visual_explanation: bool = False,
    ) -> str:
        """
        Fuse teacher's words with VLM's structural data.

        Args:
            transcript_context: The raw text spoken by teacher
            vlm_snapshot: Structured JSON from Qwen-VL
            content_type: Category of visual (graph, equation, etc.)
            intervention_reason: Flag from Synchronizer
            output_mode: "brief", "explanatory", or "detailed"
                       (defaults to class output_mode if not provided)

        Returns:
            Audio script string ready for TTS
        """
        # Use provided output_mode or fall back to class output_mode
        mode = output_mode or self.output_mode

        if vlm_snapshot.get("_error"):
            if require_visual_explanation:
                raise VisualEvidenceUnavailableError("Visual evidence extraction failed")
            print("[Synthesizer] VLM extraction failed - transcript-only fallback")
            return self._apply_output_mode(transcript_context, mode)

        # Safety Check 1: VLM missing_elements
        missing_elements = vlm_snapshot.get("missing_elements")
        if missing_elements and missing_elements not in ["null", "None", ""]:
            if require_visual_explanation:
                raise VisualEvidenceUnavailableError("Visual evidence is unavailable")
            print(f"[Synthesizer] VLM marked as missing: {missing_elements} - transcript-only fallback")
            return self._apply_output_mode(transcript_context, mode)

        # Safety Check 2: Cross-domain hallucination
        if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
            if require_visual_explanation:
                raise VisualEvidenceUnavailableError("Visual evidence failed domain validation")
            print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
            return self._apply_output_mode(transcript_context, mode)

        # Generate enhanced explanation (always use new mode)
        try:
            return self._generate_enhanced_explanation(
                transcript_context=transcript_context,
                vlm_snapshot=vlm_snapshot,
                content_type=content_type,
                intervention_reason=intervention_reason,
                output_mode=mode,
                require_visual_explanation=require_visual_explanation,
            )
        except ExplanationSynthesisError:
            raise
        except Exception as e:
            logger.debug(
                "Error generating enhanced explanation (%s)", type(e).__name__
            )
            if require_visual_explanation:
                raise TextSynthesisUnavailableError(
                    "Text explanation generation failed"
                ) from e
            return self._apply_output_mode(transcript_context, mode)

    def _construct_prompt(
        self,
        transcript_context: str,
        vlm_snapshot_json: str,
        content_type: str,
        intervention_reason: str
    ) -> str:
        """
        Construct prompt for LLM based on intervention reason.

        Three Weaving Modes:
        - Deictic (Mode A): Teacher pointed at screen
        - Bridge (Mode B): Visual is complex, teacher ignored it
        - Narrator (Mode C): Teacher is silent/quiet
        """
        system_prompt = """You are an expert Audio Description Editor for blind students. Your goal is to make STEM video lectures fully accessible through sound.

THE GOLDEN RULE:
You MUST incorporate provided VISUAL DATA into final script. A blind student cannot see the screen. If visual data exists (and is not marked 'missing'), it MUST be spoken.

GENERAL GUIDELINES:
1. Natural Flow: Prioritize a conversational tone. Avoid robotic listing unless describing a complex list.
2. Math Linearization: Convert all math notation to spoken English.
3. Spatial Language: Use directional words (left, right, center, top, bottom) to build a mental map.
4. Brevity: Be concise. Do not overload sentence with too many clauses.
5. Region Delimiters: If visual data contains "|||", treat them as separate visual zones. Connect them with phrases like "...while on the right..." or "...below that..."
"""

        # Build case-specific instructions
        if "Deictic" in intervention_reason or "High Complexity" in intervention_reason:
            case_instruction = f"""
CASE 1: If Reason is "Deictic" (Teacher said "Look at this...")
- Integrate the visual data directly into the teacher's sentence.
- Replace vague references ("this", "here") with specific details from the visual data.
- Do NOT add "Meanwhile" or "On the screen". Just make the sentence complete.
"""
        elif "Silent" in intervention_reason or "Drawing" in intervention_reason:
            case_instruction = """
CASE 3: If Reason is "Silent" or "Drawing" (Teacher is quiet/low audio)
- IGNORE the teacher's words (they are just filler/silence).
- Construct a complete, standalone narration sentence describing ONLY the visual data.
- Start with "On the screen, ..." or "The diagram displays ...".
"""
        else:
            # Negative deictic or other (should not trigger intervention)
            case_instruction = """
CASE 2: If visual data is available, provide a helpful description.
Otherwise, output the polished transcript only.
"""

        # Build user prompt
        user_prompt = f"""
TASK: Weave Teacher's Words and Visual Data into a single TTS script.

CONTEXT:
- Content Type: {content_type}
- Intervention Reason: {intervention_reason}

INPUT DATA:
1. TEACHER'S WORDS: "{transcript_context}"
2. VISUAL DATA (JSON):
{vlm_snapshot_json}

{case_instruction}

OUTPUT RULES:
- Output ONLY final audio script string.
- Do NOT include explanations or metadata.
- Ensure all math is read linearly.

FINAL SCRIPT:
"""

        return system_prompt + user_prompt

    def _apply_output_mode(
        self,
        audio_script: str,
        output_mode: str
    ) -> str:
        """
        Return audio_script as-is.

        No truncation - prompts control depth through guidance, not sentence limits.
        Modes differ in explanation depth, not length.

        Args:
            audio_script: Raw LLM output
            output_mode: "standard", "brief", "explanatory", "detailed", or "short"

        Returns:
            Unfiltered audio script string
        """
        if not audio_script:
            return ""

        # No sentence limits - return full explanation as generated
        # Prompts control depth, not length
        return audio_script

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with basic punctuation handling.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text:
            return []

        try:
            # Handle common sentence endings
            # Split on . ! ? followed by space or end of string
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return sentences
        except Exception as e:
            print(f"[Synthesizer] Error splitting sentences: {e}")
            return text.split('.') if text else []

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via Ollama client.

        Args:
            prompt: Complete prompt string

        Returns:
            Generated text from LLM
        """
        start_time = time.perf_counter()

        try:
            response = self.client.generate(prompt)

            # Extract text from response
            audio_script = response.text.strip()

            print(f"[Synthesizer] LLM generated in {response.total_duration_ms}ms")
            trace_stage(
                "mistral_model", "response", started_at=start_time,
                prompt_chars=len(prompt), response_chars=len(audio_script),
                prompt_tokens=response.prompt_eval_count,
                completion_tokens=response.eval_count,
                model=response.model,
            )

            return audio_script

        except OllamaError as e:
            trace_stage(
                "mistral_model", "failed", started_at=start_time,
                error_type=type(e).__name__,
            )
            print(f"[Synthesizer] LLM error: {e}")
            # Return transcript-only fallback (empty script)
            return ""

    # ========================================================================
    # NEW METHODS: Enhanced 3-Mode Intervention Explanation System
    # ========================================================================

    def _find_current_transcript_index(self, transcript_context: str) -> int:
        """
        Find index of current transcript segment.

        Args:
            transcript_context: The transcript text to find

        Returns:
            Index of the matching transcript segment, or 0 if not found
        """
        if not self.transcripts:
            return 0

        try:
            cleaned_context = transcript_context.strip().lower()

            for idx, entry in enumerate(self.transcripts):
                if not hasattr(entry, 'text'):
                    continue

                entry_text = entry.text.strip().lower()

                # Exact match
                if entry_text == cleaned_context:
                    return idx

                # Partial match (context is substring of entry)
                if cleaned_context in entry_text:
                    return idx

            # If no match found, return 0 (first segment)
            return 0

        except Exception as e:
            print(f"[Synthesizer] Error finding transcript index: {e}")
            return 0

    def _linearize_vlm_data(self, vlm_snapshot: Dict) -> Dict:
        """
        Linearize math in VLM snapshot before sending to LLM.

        Args:
            vlm_snapshot: Original VLM snapshot

        Returns:
            VLM snapshot with math linearized
        """
        if not vlm_snapshot:
            return {}

        try:
            # Some legacy snapshots contain nested mutable lists.  A shallow
            # copy would rewrite the persisted source while preparing a prompt.
            linearized = copy.deepcopy(vlm_snapshot)

            # Linearize common text fields
            for field in ["text_readout", "layout", "spatial_map", "description"]:
                if field in linearized and isinstance(linearized[field], str):
                    linearized[field] = self.math_linearizer.linearize(linearized[field])

            # Handle nested structures
            if "elements" in linearized and isinstance(linearized["elements"], list):
                for element in linearized["elements"]:
                    if isinstance(element, dict):
                        for key, value in element.items():
                            if isinstance(value, str):
                                element[key] = self.math_linearizer.linearize(value)

            return linearized

        except Exception as e:
            print(f"[Synthesizer] Error linearizing VLM data: {e}")
            return vlm_snapshot

    def _format_canonical_evidence_legacy_baseline(self, vlm_snapshot: Dict) -> str:
        """Return the complete canonical block without altering exact content."""

        evidence = vlm_snapshot.get("visual_evidence")
        if not isinstance(evidence, dict):
            return "Not available; use the backward-compatible visual analysis."
        prompt_evidence = dict(evidence)
        # The raw model response is already retained in the persisted snapshot;
        # repeating it here would duplicate the complete canonical content.
        prompt_evidence.pop("legacy_snapshot", None)
        return json.dumps(prompt_evidence, ensure_ascii=False, separators=(",", ":"))

    def _canonical_evidence_rules_legacy_baseline(self) -> str:
        return """
CANONICAL EVIDENCE RULES:
- Treat canonical elements and relationships as the machine-readable source of truth.
- Preserve indispensable exact labels, values, units, symbols, code operators, grouping, branch labels, and connections.
- Use the structural overview, component placement, hierarchy, topology, and reading order to build a coherent mental model.
- State uncertainty specifically. Never upgrade partial, ambiguous, unreadable, or unavailable evidence to certainty.
- Never infer a connection, cause, result, teacher intent, or temporal change that is not supported by the evidence or transcript.
- Integrate five flexible functions where useful: orientation, structure, salient visual information, educational meaning, and lecture connection. They are not five compulsory paragraphs.
"""

    def _format_coverage_plan_legacy_baseline(self, vlm_snapshot: Dict) -> str:
        """Return the compact reconciliation plan without duplicating evidence."""

        plan = vlm_snapshot.get("coverage_plan")
        if not isinstance(plan, dict):
            return "Not available; compare transcript and evidence conservatively."
        return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))

    def _coverage_plan_rules_legacy_baseline(self) -> str:
        return """
TRANSCRIPT-VISUAL COVERAGE RULES:
- Prioritize critical unspoken facts. Supply only the missing part of spoken_partial facts.
- Avoid repeating spoken_explicit facts unless a short reference is needed for coherence.
- Treat uncertain_match as unresolved; do not omit essential information solely on that label.
- Never state not_usable facts as true. Preserve their uncertainty or omit them.
- The coverage plan selects content; canonical visual evidence remains the factual source of truth.
"""

    def _build_brief_prompt_legacy_baseline(
        self,
        transcript_context: str,
        context: Dict,
        vlm_snapshot: Dict
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
        canonical_evidence = self._format_canonical_evidence_legacy_baseline(vlm_snapshot)

        previous_concepts = context.get("previous_concepts", [])

        return f"""
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

5. CANONICAL VISUAL EVIDENCE:
{canonical_evidence}
{self._canonical_evidence_rules_legacy_baseline()}

6. TRANSCRIPT-VISUAL COVERAGE PLAN:
{self._format_coverage_plan_legacy_baseline(vlm_snapshot)}
{self._coverage_plan_rules_legacy_baseline()}

7. WHAT TEACHER IS SAYING:
"{transcript_context}"

8. PREVIOUS CONCEPTS:
{', '.join(previous_concepts) if previous_concepts else 'None'}

YOUR EXPLANATION - FOCUS ON THESE ELEMENTS:

1. QUICK VISUAL SNAPSHOT (Surface Level):
- Describe ONLY the most important visual elements
- Include every indispensable exact fact or relation even in Brief mode
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

    def _build_explanatory_prompt_legacy_baseline(
        self,
        transcript_context: str,
        context: Dict,
        vlm_snapshot: Dict
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
        canonical_evidence = self._format_canonical_evidence_legacy_baseline(vlm_snapshot)

        previous_concepts = context.get("previous_concepts", [])
        context_text = context.get("context_text", "")

        return f"""
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

6. CANONICAL VISUAL EVIDENCE:
{canonical_evidence}
{self._canonical_evidence_rules_legacy_baseline()}

7. TRANSCRIPT-VISUAL COVERAGE PLAN:
{self._format_coverage_plan_legacy_baseline(vlm_snapshot)}
{self._coverage_plan_rules_legacy_baseline()}

8. WHAT TEACHER IS SAYING:
"{transcript_context}"

9. SURROUNDING CONTEXT:
"{context_text}"

10. PREVIOUS CONCEPTS MENTIONED:
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

    def _build_detailed_prompt_legacy_baseline(
        self,
        transcript_context: str,
        context: Dict,
        vlm_snapshot: Dict
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
        canonical_evidence = self._format_canonical_evidence_legacy_baseline(vlm_snapshot)

        # Get cross-domain knowledge
        cross_domain = self._get_cross_domain_explanations(detected_types[0] if detected_types else "unknown", previous_concepts)

        return f"""
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

6. CANONICAL VISUAL EVIDENCE:
{canonical_evidence}
{self._canonical_evidence_rules_legacy_baseline()}

7. TRANSCRIPT-VISUAL COVERAGE PLAN:
{self._format_coverage_plan_legacy_baseline(vlm_snapshot)}
{self._coverage_plan_rules_legacy_baseline()}

8. WHAT TEACHER IS SAYING:
"{transcript_context}"

9. FULL LECTURE CONTEXT (±2 minutes):
"{context_text}"

10. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

11. RELATED PRINCIPLES:
{', '.join(related_principles) if related_principles != 'None' else 'various principles'}

12. TOPIC TIMELINE:
{topic_timeline}

13. CROSS-DOMAIN CONNECTIONS:
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
- Use conceptual hints only when supported by canonical evidence or transcript
- Include mathematical/physical meaning thoroughly
- Include historical context only when it appears in the supplied lecture context
- Discuss boundary conditions, limitations, or special cases only when supplied
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

    @staticmethod
    def _normalize_output_mode(output_mode: Optional[str]) -> str:
        """Normalize legacy aliases without changing public API values."""

        mode = (output_mode or "explanatory").strip().lower()
        if mode in {"brief", "short"}:
            return "brief"
        if mode == "detailed":
            return "detailed"
        return "explanatory"

    @classmethod
    def _prune_empty(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, item in value.items()
                if (cleaned := cls._prune_empty(item)) not in (None, "", [], {})
            }
        if isinstance(value, list):
            return [
                cleaned for item in value
                if (cleaned := cls._prune_empty(item)) not in (None, "", [], {})
            ]
        return value

    @classmethod
    def _limit_collections(cls, value: Any, item_limit: int) -> Any:
        if isinstance(value, dict):
            return {key: cls._limit_collections(item, item_limit) for key, item in value.items()}
        if isinstance(value, list):
            limited = [cls._limit_collections(item, item_limit) for item in value[:item_limit]]
            if len(value) > item_limit:
                limited.append({"_omitted_items": len(value) - item_limit})
            return limited
        if isinstance(value, str) and len(value) > 1200:
            return value[:1200] + f" [TRUNCATED {len(value) - 1200} CHARS]"
        return value

    @classmethod
    def _bounded_json(cls, value: Any, max_chars: int) -> str:
        """Serialize deterministic valid JSON with explicit bounded omissions."""

        cleaned = cls._prune_empty(value)
        for item_limit in (64, 32, 16, 8, 4, 2, 1):
            candidate = cls._limit_collections(cleaned, item_limit)
            serialized = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(serialized) <= max_chars:
                return serialized
        raw = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
        preview_size = max_chars // 2
        while preview_size > 0:
            fallback = json.dumps({
                "payload_truncated": True,
                "reason": "Prompt safety bound reached; do not claim exhaustive coverage.",
                "preview": raw[:preview_size],
            }, ensure_ascii=False, separators=(",", ":"))
            if len(fallback) <= max_chars:
                return fallback
            preview_size //= 2
        return '{"payload_truncated":true}'

    @staticmethod
    def _bounded_text(value: Any, max_chars: int) -> str:
        text = str(value or "")
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return text[:max_chars] + f" [TRANSCRIPT TRUNCATED: {omitted} CHARS OMITTED]"

    def _compact_canonical_payload(self, evidence: Dict) -> Dict:
        regions = []
        for region in evidence.get("regions") or []:
            if not isinstance(region, dict):
                continue
            elements = []
            for element in region.get("elements") or []:
                if not isinstance(element, dict):
                    continue
                elements.append({
                    key: copy.deepcopy(element.get(key))
                    for key in (
                        "id", "kind", "evidence_status", "category_ids", "label",
                        "exact_text", "description", "position", "source_ref", "attributes",
                        "parent_id", "child_ids",
                    )
                    if element.get(key) not in (None, "", [], {})
                })
            regions.append({
                "id": region.get("id"),
                "category_ids": copy.deepcopy(region.get("category_ids")),
                "position": region.get("position"),
                "evidence_status": region.get("evidence_status"),
                "attributes": copy.deepcopy(region.get("attributes")),
                "elements": elements,
            })

        return self._prune_empty({
            "schema_version": evidence.get("schema_version"),
            "extraction_status": evidence.get("extraction_status"),
            "evidence_status": evidence.get("evidence_status"),
            "category_ids": copy.deepcopy(evidence.get("category_ids")),
            "raw_category_labels": copy.deepcopy(evidence.get("raw_category_labels")),
            "global_layout": evidence.get("global_layout"),
            # Exact content and relations precede elements because they are the
            # highest-risk facts for educational narration.
            "exact_content": copy.deepcopy(evidence.get("exact_content")),
            "relations": copy.deepcopy(evidence.get("relations")),
            "structural_description": copy.deepcopy(evidence.get("structural_description")),
            "regions": regions,
            "category_details": copy.deepcopy(evidence.get("category_details")),
            "uncertainties": copy.deepcopy(evidence.get("uncertainties")),
            "validation_diagnostics": copy.deepcopy(evidence.get("validation_diagnostics")),
        })

    def _format_canonical_evidence(self, vlm_snapshot: Dict) -> str:
        """Return bounded canonical evidence without mutating exact content."""

        evidence = vlm_snapshot.get("visual_evidence")
        if not isinstance(evidence, dict):
            legacy = {
                key: copy.deepcopy(vlm_snapshot.get(key))
                for key in (
                    "detected_types", "text_readout", "visual_analysis", "layout",
                    "spatial_map", "structural_description", "reading_order",
                )
                if vlm_snapshot.get(key) not in (None, "", [], {})
            }
            return self._bounded_json({"legacy_visual_evidence": legacy}, self.MAX_CANONICAL_PROMPT_CHARS)
        return self._bounded_json(
            self._compact_canonical_payload(evidence), self.MAX_CANONICAL_PROMPT_CHARS
        )

    def _priority_coverage_payload(self, plan: Dict, prioritize: bool = True) -> Dict:
        importance_rank = {"critical": 0, "important": 1, "supporting": 2}
        coverage_rank = {
            "unspoken": 0, "spoken_partial": 1, "uncertain_match": 2,
            "spoken_explicit": 3, "not_usable": 4,
        }
        type_rank = {
            "exact_content": 0, "relation": 1, "hierarchy": 2, "spatial": 3,
            "structure": 4, "attribute": 5, "reading_order": 6, "component": 7,
            "category_detail": 8,
        }
        facts = [fact for fact in (plan.get("facts") or []) if isinstance(fact, dict)]
        if prioritize:
            facts.sort(key=lambda fact: (
                importance_rank.get(fact.get("importance"), 9),
                coverage_rank.get(fact.get("transcript_coverage"), 9),
                type_rank.get(fact.get("fact_type"), 9),
                str(fact.get("fact_id", "")),
            ))
        compact_facts = [{
            key: copy.deepcopy(fact.get(key))
            for key in (
                "fact_id", "fact_type", "description", "source_refs", "category_ids",
                "importance", "evidence_reliability", "evidence_status",
                "uncertainty_reasons", "transcript_coverage", "missing_information",
            )
            if fact.get(key) not in (None, "", [], {})
        } for fact in facts]
        return self._prune_empty({
            "schema": plan.get("schema"),
            "method": plan.get("method"),
            "coverage_counts": copy.deepcopy(plan.get("coverage_counts")),
            "truncated": plan.get("truncated"),
            "omitted_fact_count": plan.get("omitted_fact_count"),
            "total_fact_count": plan.get("total_fact_count"),
            "priority_order": (
                "critical missing/partial, exact and structural relations, supporting facts"
                if prioritize else "original Task 4 fact order"
            ),
            "facts": compact_facts,
        })

    def _format_coverage_plan(self, vlm_snapshot: Dict, prioritize: bool = True) -> str:
        """Return a bounded, priority-ordered Task 4 plan."""

        plan = vlm_snapshot.get("coverage_plan")
        if not isinstance(plan, dict):
            return json.dumps({
                "available": False,
                "instruction": "Compare transcript and evidence conservatively; do not assume coverage.",
            }, separators=(",", ":"))
        return self._bounded_json(
            self._priority_coverage_payload(plan, prioritize=prioritize),
            self.MAX_COVERAGE_PROMPT_CHARS,
        )

    def _resolve_category_ids(self, vlm_snapshot: Dict, content_type: str = "") -> List[str]:
        category_ids: List[str] = []
        evidence = vlm_snapshot.get("visual_evidence")
        if isinstance(evidence, dict):
            category_ids.extend(evidence.get("category_ids") or [])
        plan = vlm_snapshot.get("coverage_plan")
        if isinstance(plan, dict):
            for fact in plan.get("facts") or []:
                if isinstance(fact, dict):
                    category_ids.extend(fact.get("category_ids") or [])
        legacy_labels = list(vlm_snapshot.get("detected_types") or []) + [content_type]
        for label in legacy_labels:
            mapped = self.LEGACY_CATEGORY_IDS.get(str(label).strip().lower())
            if mapped:
                category_ids.append(mapped)
        return list(dict.fromkeys(
            category for category in category_ids if category in self.CATEGORY_NARRATION_DIRECTIVES
        ))

    def _category_directives(self, category_ids: List[str]) -> str:
        if not category_ids:
            return "- Unknown/legacy category: narrate only facts explicitly present in the supplied evidence."
        directives = [f"- {category}: {self.CATEGORY_NARRATION_DIRECTIVES[category]}" for category in category_ids]
        if len(category_ids) > 1:
            directives.append(
                "- Mixed frame: preserve each category and explicitly describe supported cross-category relations; do not collapse the frame to one type."
            )
        return "\n".join(directives)

    def _evidence_limitations(self, vlm_snapshot: Dict) -> str:
        limitations: List[str] = []
        evidence = vlm_snapshot.get("visual_evidence")
        if isinstance(evidence, dict):
            extraction_status = evidence.get("extraction_status")
            evidence_status = evidence.get("evidence_status")
            if extraction_status and extraction_status != "complete":
                limitations.append(f"extraction_status={extraction_status}")
            if evidence_status and evidence_status != "supported":
                limitations.append(f"evidence_status={evidence_status}")
            codes = [
                item.get("code") for item in (evidence.get("validation_diagnostics") or [])
                if isinstance(item, dict) and item.get("code")
            ]
            limitations.extend(codes)
        metrics = vlm_snapshot.get("_vlm_metrics")
        if isinstance(metrics, dict) and metrics.get("possible_truncation"):
            limitations.append("vlm_output_possible_truncation")
        plan = vlm_snapshot.get("coverage_plan")
        if isinstance(plan, dict) and plan.get("truncated"):
            limitations.append(
                f"coverage_plan_omitted_facts={int(plan.get('omitted_fact_count') or 0)}"
            )
        if not limitations:
            return "No recorded extraction limitation. This still does not permit unsupported inference."
        return "; ".join(dict.fromkeys(limitations)) + ". Do not claim the description is exhaustive."

    def _spoken_math_hints(self, vlm_snapshot: Dict) -> str:
        evidence = vlm_snapshot.get("visual_evidence")
        if not isinstance(evidence, dict):
            return "None supplied."
        hints = []
        for item in evidence.get("exact_content") or []:
            if not isinstance(item, dict) or item.get("notation") != "math" or not item.get("text"):
                continue
            exact = str(item["text"])
            hints.append({"exact_source": exact, "spoken_hint": MathLinearizer.linearize(exact)})
        return self._bounded_json(hints, 4000) if hints else "None supplied."

    def _canonical_evidence_rules(self) -> str:
        return """
SOURCE AND GROUNDING RULES:
- Canonical visual evidence is the authority for what was extracted from this frame. Teacher text is authority only for what the teacher said.
- Use only supported visual facts or clearly attributed teacher speech. General knowledge must not become a visual observation.
- Preserve exact labels, values, units, symbols, grouping, code tokens, indentation, branch labels, directions, hierarchy, and connectivity.
- Say what is uncertain or incomplete. Never upgrade partial, ambiguous, unreadable, failed, or truncated evidence to certainty.
- Do not infer causes, functions, teacher intentions, unseen states, execution results, or temporal change from a single frame.
"""

    def _coverage_plan_rules(self) -> str:
        return """
CONTENT-SELECTION RULES:
- Prioritize critical unspoken facts and supply only the missing portion of critical spoken_partial facts.
- Treat uncertain_match as unresolved. Include an essential fact cautiously when canonical evidence supports it.
- Do not repeat spoken_explicit content unless a short reference is indispensable for orientation or a relationship.
- Never assert not_usable facts. Omitted fact counts mean the plan is incomplete, not that omitted facts are unimportant.
- The plan ranks content; canonical evidence remains the factual source of truth.
"""

    def _mode_directive(self, mode: str) -> str:
        if mode == "brief":
            return (
                "BRIEF: lead with the immediate teaching point and include every indispensable exact symbolic, numeric, spatial, and structural fact. "
                "Omit optional detail, but never simplify into a contradiction."
            )
        if mode == "detailed":
            return (
                "DETAILED: retain the same core facts as Brief and Standard, then add useful supported secondary structure, comparisons, values, and clarification for review. "
                "Do not enumerate decoration or invent enrichment."
            )
        return (
            "STANDARD: explain the main concept with essential layout, labels, units, relationships, and relevant comparisons using clear speech-first pacing."
        )

    def _build_task5_prompt(
        self,
        transcript_context: str,
        context: Dict,
        vlm_snapshot: Dict,
        mode: str,
        content_type: str = "",
        intervention_reason: str = "",
        prioritize_coverage: bool = True,
    ) -> str:
        normalized_mode = self._normalize_output_mode(mode)
        category_ids = self._resolve_category_ids(vlm_snapshot, content_type)
        transcript = self._bounded_text(
            transcript_context, self.MAX_TRANSCRIPT_PROMPT_CHARS
        )
        surrounding = self._bounded_text(
            context.get("context_text"), self.MAX_CONTEXT_PROMPT_CHARS
        )
        canonical = self._format_canonical_evidence(vlm_snapshot)
        coverage = self._format_coverage_plan(vlm_snapshot, prioritize=prioritize_coverage)
        selection_rules = self._coverage_plan_rules() if prioritize_coverage else """
DIRECT SOURCE-GROUNDED SELECTION RULES:
- Use the Task 4 statuses to avoid unsupported claims and unnecessary repetition.
- Narrate supported essential facts directly without adding a separate priority-ranking policy.
- Never assert not_usable evidence; retain explicit uncertainty.
"""

        return f"""SYSTEM ROLE
You write accurate, conversational, speech-only educational descriptions for a learner who cannot see the current frame.

NON-NEGOTIABLE CONTRACT
{self._canonical_evidence_rules()}
{selection_rules}

Use five flexible functions only where useful; they are not five paragraphs and have no fixed order:
1. Orientation — identify the subject when needed, without generic filler.
2. Structure — convey essential parts, exact text or notation, values, spatial relations, hierarchy, connectivity, and reading order.
3. Salient visual information — state supported trends, contrasts, states, patterns, or visible features, never invented temporal change.
4. Educational meaning — interpret only when supported by the visual evidence or clearly attributed teacher speech.
5. Lecture connection — connect to the teacher's specific point only when grounded and useful; never invent intent.

Adapt the order to the teaching risk: for example, an operator, fraction scope, branch direction, or circuit connection may come first.

DETAIL LEVEL
{self._mode_directive(normalized_mode)}

CATEGORY-SPECIFIC NARRATION
{self._category_directives(category_ids)}

EVIDENCE LIMITATIONS
{self._evidence_limitations(vlm_snapshot)}

CANONICAL VISUAL EVIDENCE (bounded, source text remains exact)
{canonical}

TRANSCRIPT-VISUAL COVERAGE PLAN — TASK 4 PRIORITY PLAN (bounded and priority ordered)
{coverage}

OPTIONAL SPOKEN-MATH HINTS
{self._spoken_math_hints(vlm_snapshot)}
The exact source expression remains authoritative. State grouping explicitly, such as “the quantity ... over the quantity ...”, when needed.

LECTURE INPUT
Current teacher transcript: {json.dumps(transcript, ensure_ascii=False)}
Surrounding transcript context: {json.dumps(surrounding, ensure_ascii=False)}
Intervention reason: {json.dumps(intervention_reason or 'not supplied', ensure_ascii=False)}

OUTPUT REQUIREMENTS
- Return only the learner-facing audio script: no heading, markdown, JSON, source IDs, confidence labels, or planning commentary.
- Sound natural when spoken. Do not announce all five functions.
- Resolve vague words such as “this” only with supported visual details.
- Do not say “the image clearly shows” when evidence is partial or ambiguous.
- If no safe visual fact is available, return a concise transcript-based explanation without inventing visual content.
"""

    def _build_brief_prompt(
        self, transcript_context: str, context: Dict, vlm_snapshot: Dict,
        content_type: str = "", intervention_reason: str = ""
    ) -> str:
        return self._build_task5_prompt(
            transcript_context, context, vlm_snapshot, "brief", content_type, intervention_reason
        )

    def _build_explanatory_prompt(
        self, transcript_context: str, context: Dict, vlm_snapshot: Dict,
        content_type: str = "", intervention_reason: str = ""
    ) -> str:
        return self._build_task5_prompt(
            transcript_context, context, vlm_snapshot, "explanatory", content_type, intervention_reason
        )

    def _build_detailed_prompt(
        self, transcript_context: str, context: Dict, vlm_snapshot: Dict,
        content_type: str = "", intervention_reason: str = ""
    ) -> str:
        return self._build_task5_prompt(
            transcript_context, context, vlm_snapshot, "detailed", content_type, intervention_reason
        )

    def _get_cross_domain_explanations(self, content_type: str, previous_concepts: List[str]) -> str:
        """
        Get cross-domain explanations for concepts.

        Args:
            content_type: Type of visual content
            previous_concepts: List of previously mentioned concepts

        Returns:
            String with cross-domain explanations
        """
        if not previous_concepts:
            return "No direct connections"

        try:
            explanations = []

            for concept in previous_concepts:
                # Get explanation in multiple domains
                for domain in ["math", "physics", "chemistry", "economics", "biology"]:
                    domain_info = self.knowledge_base.get_explanation(concept, content_type, domain)
                    if domain_info and domain not in str(explanations):
                        explanations.append(f"In {domain}: {domain_info}")

            return "; ".join(explanations) if explanations else "No direct connections"

        except Exception as e:
            print(f"[Synthesizer] Error getting cross-domain explanations: {e}")
            return "No direct connections"

    def _generate_enhanced_explanation(
        self,
        transcript_context: str,
        vlm_snapshot: Dict,
        content_type: str,
        intervention_reason: str,
        output_mode: str,
        require_visual_explanation: bool = False,
    ) -> str:
        """
        Generate enhanced explanation for blind students using depth-based prompts.

        Args:
            transcript_context: Current transcript text
            vlm_snapshot: VLM analysis results
            content_type: Type of content (kept for API compatibility, not used in new prompts)
            intervention_reason: Reason for intervention (kept for API compatibility, not used in new prompts)
            output_mode: "brief", "explanatory", or "detailed"

        Returns:
            Enhanced explanation string
        """
        try:
            normalized_mode = self._normalize_output_mode(output_mode)

            # Safety checks
            if vlm_snapshot.get("_error"):
                if require_visual_explanation:
                    raise VisualEvidenceUnavailableError("Visual evidence extraction failed")
                print("[Synthesizer] VLM extraction failed - transcript-only fallback")
                return transcript_context

            missing_elements = vlm_snapshot.get("missing_elements")
            if missing_elements and missing_elements not in ["null", "None", ""]:
                if require_visual_explanation:
                    raise VisualEvidenceUnavailableError("Visual evidence is unavailable")
                print(f"[Synthesizer] VLM marked as missing: {missing_elements} - transcript-only fallback")
                return transcript_context

            if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
                if require_visual_explanation:
                    raise VisualEvidenceUnavailableError("Visual evidence failed domain validation")
                print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
                return transcript_context

            evidence = vlm_snapshot.get("visual_evidence")
            if isinstance(evidence, dict) and evidence.get("extraction_status") in {
                "failed", "not_attempted"
            }:
                if require_visual_explanation:
                    raise VisualEvidenceUnavailableError("Canonical visual evidence is unavailable")
                print("[Synthesizer] Canonical visual evidence unavailable - transcript-only fallback")
                return transcript_context

            # Find current transcript index for context
            current_index = self._find_current_transcript_index(transcript_context)

            # Gather context based on mode
            if self.context_manager:
                context = self.context_manager.get_context(current_index, normalized_mode)
            else:
                context = {
                    "context_text": "",
                    "previous_concepts": [],
                    "related_principles": [],
                    "topic_timeline": {}
                }

            # Linearize VLM data
            linearized_snapshot = self._linearize_vlm_data(vlm_snapshot)

            # Build mode-specific prompt (new signatures without content_type/intervention_reason)
            if normalized_mode == "brief":
                prompt = self._build_brief_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot,
                    content_type=content_type,
                    intervention_reason=intervention_reason,
                )
            elif normalized_mode == "detailed":
                prompt = self._build_detailed_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot,
                    content_type=content_type,
                    intervention_reason=intervention_reason,
                )
            else:  # explanatory
                prompt = self._build_explanatory_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot,
                    content_type=content_type,
                    intervention_reason=intervention_reason,
                )

            # Call LLM
            explanation = self._call_llm(prompt)

            # Guard: if LLM returned nothing, fall back to transcript
            if not explanation:
                if require_visual_explanation:
                    raise TextSynthesisUnavailableError("Text explanation model returned no content")
                print("[Synthesizer] LLM returned empty - transcript-only fallback")
                return transcript_context

            # Post-process: math linearization (double-check)
            #explanation = self.math_linearizer.linearize(explanation)

            # No truncation - return full explanation as generated
            return explanation

        except ExplanationSynthesisError:
            raise
        except Exception as e:
            logger.debug(
                "Error generating enhanced explanation (%s)", type(e).__name__
            )
            if require_visual_explanation:
                raise TextSynthesisUnavailableError(
                    "Text explanation generation failed"
                ) from e
            # Fallback to transcript-only on error
            return transcript_context

    def cleanup(self):
        """
        Cleanup LLMSynthesizer resources.

        Ollama runs as a separate process, but we clear any local caches
        to ensure the HTTP client doesn't hold references.
        """
        print("[Synthesizer] Cleanup complete - Ollama runs separately")
