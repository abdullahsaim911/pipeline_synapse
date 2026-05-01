# Synthesizer Module (M3) Documentation

## Overview

The Synthesizer Module, known as the "Audio Weaver," fuses the teacher's spoken words with VLM-extracted visual data into seamless audio scripts ready for text-to-speech generation. It implements a 3-mode explanation system (brief, explanatory, detailed) with intelligent context management.

## Purpose

- Merge transcript text with VLM visual analysis
- Generate natural, accessible explanations for blind students
- Support multiple explanation depth modes
- Implement safety checks for hallucination prevention
- Manage context across transcript segments
- Linearize mathematical notation for speech

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **Ollama**: Local LLM for text generation
- **Mistral**: Default LLM model (via Ollama)
- **NumPy**: Numerical operations
- **Regex**: Pattern matching for math linearization

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| CPU | 4 cores | 8+ cores |
| Ollama | Latest | Latest |
| Model | mistral (7B) | mistral (7B) or larger |

## Implementation Details

### Module Structure

```
synthesizer/
├── __init__.py
├── llm_synthesizer.py         # Main synthesis logic
├── math_linearizer.py         # Math notation conversion
├── knowledge_base.py          # Domain knowledge storage
└── context_manager.py         # Transcript context management
```

### Core Components

#### 1. SynthesizedTextResult

```python
@dataclass
class SynthesizedTextResult:
    """Result of text synthesis process."""
    audio_script: str            # Final text ready for TTS
    fallback_used: bool = False  # Whether transcript-only fallback was used
    warning: Optional[str] = None  # Any warnings generated
```

#### 2. LLMSynthesizer Class

**Main Synthesizer Class**: Orchestrates text fusion and generation

```python
class LLMSynthesizer:
    """
    Audio Weaver that fuses transcript + VLM data via LLM.

    Implements Unified Injection Rule: VLM data never discarded for interventions
    unless marked as missing or cross-domain hallucination detected.
    """
```

**Default Generation Parameters**:

```python
DEFAULT_GENERATION_PARAMS = {
    "num_predict": 256,         # Maximum tokens to generate
    "temperature": 0.7,         # Sampling temperature
    "top_k": 40,                # Top-k sampling
    "top_p": 0.9,               # Nucleus sampling
    "repeat_penalty": 1.1       # Repetition penalty
}
```

**Cross-Domain Hallucination Detection**:

```python
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
```

#### 3. Initialization

```python
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
```

#### 4. Safety Checks

**Missing Elements Detection**:

```python
def weave(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    content_type: str,
    intervention_reason: str,
    output_mode: Optional[str] = None
) -> str:
    """Fuse teacher's words with VLM's structural data."""
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

    # Generate enhanced explanation
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
```

**Cross-Domain Hallucination Detection**:

```python
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
```

#### 5. Three-Mode Explanation System

**Brief Mode**: Surface-level explanation

```python
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
```

**Explanatory Mode**: Standard depth explanation

```python
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
    # Similar structure with more depth
    # Includes:
    # - Complete visual description
    # - What the teacher is explaining
    # - The concept being explained
    # - How the visual links to the concept
    # - Accessibility techniques
    # - Standard teaching depth
```

**Detailed Mode**: Deep exploration explanation

```python
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
    # Most comprehensive version
    # Includes:
    # - Exhaustive visual description
    # - Full context and teacher's explanation
    # - Deep conceptual framework
    # - Multiple explicit connections
    # - Cross-domain connections
    # - Rich accessibility techniques
    # - One-on-one tutoring depth
```

#### 6. Context Management

**Context Gathering**:

```python
def _generate_enhanced_explanation(
    self,
    transcript_context: str,
    vlm_snapshot: Dict,
    content_type: str,
    intervention_reason: str,
    output_mode: str
) -> str:
    """Generate enhanced explanation for blind students using depth-based prompts."""
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

    # Build mode-specific prompt
    if output_mode == "brief":
        prompt = self._build_brief_prompt(transcript_context, context, linearized_snapshot)
    elif output_mode == "detailed":
        prompt = self._build_detailed_prompt(transcript_context, context, linearized_snapshot)
    else:  # explanatory
        prompt = self._build_explanatory_prompt(transcript_context, context, linearized_snapshot)

    # Call LLM
    explanation = self._call_llm(prompt)

    # Guard: if LLM returned nothing, fall back to transcript
    if not explanation:
        print("[Synthesizer] LLM returned empty - transcript-only fallback")
        return transcript_context

    return explanation
```

#### 7. Math Linearization

**MathLinearizer Component**: Converts mathematical notation to spoken form

```python
class MathLinearizer:
    """Convert mathematical notation to spoken form for TTS."""

    # Symbol mappings
    SYMBOL_MAPPINGS = {
        "²": "squared",
        "³": "cubed",
        "^2": "squared",
        "^3": "cubed",
        "√": "square root of",
        "∫": "integral of",
        "π": "pi",
        "θ": "theta",
        "φ": "phi",
        "∑": "sum of",
        "∏": "product of",
        "∞": "infinity",
        "≠": "not equal to",
        "≤": "less than or equal to",
        "≥": "greater than or equal to",
        "→": "implies",
        "⇒": "implies that",
        "∴": "therefore",
        "∵": "because",
    }

    def linearize(self, text: str) -> str:
        """Convert mathematical notation to spoken form."""
        for symbol, spoken in self.SYMBOL_MAPPINGS.items():
            text = text.replace(symbol, spoken)
        return text
```

## Configuration Parameters

### Output Modes

| Mode | Depth | Best For | Token Count |
|------|-------|----------|-------------|
| brief | Surface level | Quick overviews | 100-200 |
| explanatory | Standard depth | Complete coverage | 200-400 |
| detailed | Deep exploration | Thorough understanding | 400-800 |

### Generation Parameters

```python
# Standard generation
DEFAULT_GENERATION_PARAMS = {
    "num_predict": 256,      # Maximum tokens to generate
    "temperature": 0.7,      # Sampling temperature
    "top_k": 40,             # Top-k sampling
    "top_p": 0.9,            # Nucleus sampling
    "repeat_penalty": 1.1    # Repetition penalty
}

# Mode-specific adjustments
MODE_PARAMS = {
    "brief": {"num_predict": 200, "temperature": 0.6},
    "explanatory": {"num_predict": 400, "temperature": 0.7},
    "detailed": {"num_predict": 800, "temperature": 0.8}
}
```

## Usage Examples

### Basic Usage

```python
from synthesizer import LLMSynthesizer

# Initialize synthesizer
synthesizer = LLMSynthesizer(
    ollama_url="http://localhost:11434",
    model="mistral",
    timeout=120,
    max_retries=3,
    transcripts=transcript_entries,
    output_mode="explanatory"  # brief, explanatory, or detailed
)

# Fuse transcript with VLM data
audio_script = synthesizer.weave(
    transcript_context="Look at this graph on the screen.",
    vlm_snapshot=vlm_data,
    content_type="graph",
    intervention_reason="Deictic Phrase",
    output_mode="explanatory"
)

print(f"Generated: {audio_script}")
```

### With Different Modes

```python
# Brief mode
brief_script = synthesizer.weave(
    transcript_context=transcript,
    vlm_snapshot=vlm_data,
    content_type="equation",
    intervention_reason="Deictic Phrase",
    output_mode="brief"
)

# Explanatory mode (default)
explanatory_script = synthesizer.weave(
    transcript_context=transcript,
    vlm_snapshot=vlm_data,
    content_type="equation",
    intervention_reason="Deictic Phrase",
    output_mode="explanatory"
)

# Detailed mode
detailed_script = synthesizer.weave(
    transcript_context=transcript,
    vlm_snapshot=vlm_data,
    content_type="equation",
    intervention_reason="Deictic Phrase",
    output_mode="detailed"
)
```

### Error Handling

```python
try:
    audio_script = synthesizer.weave(
        transcript_context=transcript,
        vlm_snapshot=vlm_data,
        content_type="graph",
        intervention_reason="Deictic Phrase"
    )

    if not audio_script:
        print("Empty script generated")
    else:
        print(f"Success: {len(audio_script)} characters")

except Exception as e:
    print(f"Synthesis failed: {e}")
    # Use fallback
    audio_script = transcript
```

## Performance Characteristics

### Processing Time (per intervention)

| Mode | Tokens | Time | Output Length |
|------|--------|------|---------------|
| brief | 150-250 | 2-4s | 100-200 words |
| explanatory | 300-500 | 4-8s | 200-400 words |
| detailed | 500-900 | 8-15s | 400-800 words |

### Memory Usage

| Component | RAM | Notes |
|-----------|-----|-------|
| Ollama (mistral) | 4-8GB | Depends on GPU usage |
| Context Manager | 100-500MB | Scales with transcript length |
| LLM Response | 10-50MB | Temporary during generation |

## Troubleshooting

### Common Issues

**Issue**: "Empty script generated"
- **Solutions**:
  - Check Ollama connection
  - Verify prompt is not empty
  - Check for timeout errors
  - Examine LLM logs

**Issue**: "Hallucination detected"
- **Solutions**:
  - Verify VLM output quality
  - Check content type classification
  - Review cross-domain mappings
  - Adjust VLM prompts

**Issue**: "Math not spoken correctly"
- **Solutions**:
  - Check MathLinearizer mappings
  - Add missing symbols
  - Verify VLM OCR quality
  - Use custom linearization

**Issue**: "Context not found"
- **Solutions**:
  - Verify transcript entries are provided
  - Check timestamp matching
  - Ensure transcripts are TranscriptEntry objects
  - Review context manager logic

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
synthesizer = LLMSynthesizer(...)

# Test math linearization
from synthesizer.math_linearizer import MathLinearizer
linearizer = MathLinearizer()
test = "x² + y² = r²"
print(f"Linearized: {linearizer.linearize(test)}")

# Test context manager
if synthesizer.context_manager:
    context = synthesizer.context_manager.get_context(0, "explanatory")
    print(f"Context: {context}")

# Test VLM linearization
linearized = synthesizer._linearize_vlm_data(vlm_snapshot)
print(f"Linearized VLM: {json.dumps(linearized, indent=2)}")
```

## Best Practices

### 1. Mode Selection

```python
# Choose appropriate mode based on use case
def select_mode(intervention_type, video_type):
    """Select output mode based on intervention type."""
    if intervention_type == "quick_reference":
        return "brief"
    elif intervention_type == "main_explanation":
        return "explanatory"
    elif intervention_type == "deep_dive":
        return "detailed"
    else:
        return "explanatory"  # Default
```

### 2. Fallback Handling

```python
# Robust fallback chain
def synthesize_with_fallback(synthesizer, transcript, vlm_data, content_type):
    """Synthesize with multiple fallback options."""
    modes = ["explanatory", "brief", "detailed"]

    for mode in modes:
        try:
            script = synthesizer.weave(
                transcript_context=transcript,
                vlm_snapshot=vlm_data,
                content_type=content_type,
                intervention_reason="Generated",
                output_mode=mode
            )
            if script and len(script) > 50:
                return script
        except Exception as e:
            print(f"Mode {mode} failed: {e}")
            continue

    # Final fallback: transcript only
    return transcript
```

### 3. Quality Assurance

```python
# Validate generated scripts
def validate_script(script: str, min_length: int = 50) -> bool:
    """Validate generated script meets quality criteria."""
    if not script or len(script) < min_length:
        return False

    # Check for common issues
    if script.count("[ILLEGIBLE]") / len(script) > 0.1:
        return False

    if "MISSING" in script:
        return False

    return True
```

## Future Enhancements

1. **Dynamic Mode Selection**: Automatically choose mode based on content complexity
2. **User Feedback Integration**: Learn from user preferences
3. **Multi-Language Support**: Generate explanations in different languages
4. **Style Customization**: Adjust tone, pacing, and vocabulary level
5. **Interactive Explanations**: Support for question-asking
6. **Progressive Disclosure**: Start brief, expand on request
7. **Personalization**: Adapt to individual learning styles
