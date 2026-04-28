"""
LLM Synthesizer Module (M3)

Audio Weaver" - Fuse teacher's spoken words with VLM's structural data into a single, seamless TTS script.
Enhanced with 3-mode intervention explanation system (brief, explanatory, detailed).
"""

import json
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

from llm_interface import OllamaClient, OllamaError
from .math_linearizer import MathLinearizer
from .knowledge_base import KnowledgeBase
from .context_manager import ContextManager


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
        output_mode: Optional[str] = None
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

        # Safety Check 1: VLM missing_elements
        missing_elements = vlm_snapshot.get("missing_elements")
        if missing_elements and missing_elements not in ["null", "None", ""]:
            print(f"[Synthesizer] VLM marked as missing: {missing_elements} - transcript-only fallback")
            return self._apply_output_mode(transcript_context, mode)

        # Safety Check 2: Cross-domain hallucination
        if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
            print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
            return self._apply_output_mode(transcript_context, mode)

        # Generate enhanced explanation (always use new mode)
        try:
            return self._generate_enhanced_explanation(
                transcript_context=transcript_context,
                vlm_snapshot=vlm_snapshot,
                content_type=content_type,
                intervention_reason=intervention_reason,
                output_mode=mode
            )
        except Exception as e:
            print(f"[Synthesizer] Error generating enhanced explanation: {e}", exc_info=True)
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

            return audio_script

        except OllamaError as e:
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
            linearized = vlm_snapshot.copy()

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

    def _build_brief_prompt(
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

    def _build_explanatory_prompt(
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

    def _build_detailed_prompt(
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

6. WHAT TEACHER IS SAYING:
"{transcript_context}"

7. FULL LECTURE CONTEXT (±2 minutes):
"{context_text}"

8. PREVIOUS CONCEPTS MENTIONED:
{', '.join(previous_concepts) if previous_concepts else 'None'}

9. RELATED PRINCIPLES:
{', '.join(related_principles) if related_principles != 'None' else 'various principles'}

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
        output_mode: str
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
            # Safety checks
            missing_elements = vlm_snapshot.get("missing_elements")
            if missing_elements and missing_elements not in ["null", "None", ""]:
                print(f"[Synthesizer] VLM marked as missing: {missing_elements} - transcript-only fallback")
                return transcript_context

            if self._detect_cross_domain_hallucination(vlm_snapshot, content_type):
                print(f"[Synthesizer] Cross-domain hallucination detected - transcript-only fallback")
                return transcript_context

            # Find current transcript index for context
            current_index = self._find_current_transcript_index(transcript_context)

            # Gather context based on mode
            if self.context_manager:
                context = self.context_manager.get_context(current_index, output_mode)
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
            if output_mode == "brief":
                prompt = self._build_brief_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot
                )
            elif output_mode == "detailed":
                prompt = self._build_detailed_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot
                )
            else:  # explanatory
                prompt = self._build_explanatory_prompt(
                    transcript_context=transcript_context,
                    context=context,
                    vlm_snapshot=linearized_snapshot
                )

            # Call LLM
            explanation = self._call_llm(prompt)

            # Guard: if LLM returned nothing, fall back to transcript
            if not explanation:
                print("[Synthesizer] LLM returned empty - transcript-only fallback")
                return transcript_context

            # Post-process: math linearization (double-check)
            #explanation = self.math_linearizer.linearize(explanation)

            # No truncation - return full explanation as generated
            return explanation

        except Exception as e:
            print(f"[Synthesizer] Error generating enhanced explanation: {e}", exc_info=True)
            # Fallback to transcript-only on error
            return transcript_context

    def cleanup(self):
        """
        Cleanup LLMSynthesizer resources.

        Ollama runs as a separate process, but we clear any local caches
        to ensure the HTTP client doesn't hold references.
        """
        print("[Synthesizer] Cleanup complete - Ollama runs separately")
