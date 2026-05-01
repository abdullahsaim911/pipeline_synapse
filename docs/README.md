# Synapse Documentation Index

Welcome to the Synapse project documentation. This index provides an overview of all available documentation.

## Quick Start

- **[Complete Pipeline Documentation](SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md)** - Start here for a comprehensive overview of the entire Synapse pipeline, including problem statement, solution architecture, installation, usage guide, and troubleshooting.

## Module Documentation

Each module has its own detailed documentation covering tools, technologies, implementation details, and usage examples.

### Core Modules

1. **[Frame Extraction Module (M0)](modules/01_frame_extraction_module.md)**
   - Extracts pedagogically important keyframes from video lectures
   - Uses CLIP-based classification and complexity scoring
   - 7-phase extraction pipeline with semantic deduplication

2. **[Transcription Module (M0b)](modules/02_transcription_module.md)**
   - Converts spoken content to text with precise timing
   - Hybrid approach: YouTube API + Whisper AI fallback
   - Multiple model sizes and streaming support

3. **[Synchronizer Module (M1)](modules/03_synchronizer_module.md)**
   - Detects when blind students lose visual access
   - Multi-factor scoring system (deictic, silent drawing, complexity)
   - Confidence levels and redundancy checking

4. **[VLM Interface Module (M5a)](modules/04_vlm_interface_module.md)**
   - Unified interface for vision-language models
   - Qwen2-VL-7B-Instruct integration
   - Device detection and memory management

5. **[VLM Engine Module (M2)](modules/05_vlm_engine_module.md)**
   - Analyzes STEM lecture frames with category-specific prompts
   - 10 content type categories (equation, graph, circuit, etc.)
   - Chain-of-Regions scanning and unified JSON output

6. **[LLM Interface Module (M5b)](modules/06_llm_interface_module.md)**
   - Unified interface for local LLM operations
   - Ollama API client with retry logic
   - Error handling and response parsing

7. **[Synthesizer Module (M3)](modules/07_synthesizer_module.md)**
   - Fuses transcript and VLM data into seamless audio scripts
   - 3-mode explanation system (brief, explanatory, detailed)
   - Context management and math linearization

8. **[TTS Engine Module (M6)](modules/08_tts_engine_module.md)**
   - Converts text scripts to high-fidelity audio
   - Dual-provider system (Edge-TTS + SpeechT5)
   - Multiple voice options and automatic fallback

9. **[Orchestrator Module (M4)](modules/09_orchestrator_module.md)**
   - Coordinates entire pipeline lifecycle
   - Parallel extraction and GPU memory management
   - Multiple workflow options (full, intervention-only, detection-only)

## Documentation Structure

```
docs/
├── README.md                              # This file
├── SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md  # Complete pipeline overview
└── modules/
    ├── 01_frame_extraction_module.md
    ├── 02_transcription_module.md
    ├── 03_synchronizer_module.md
    ├── 04_vlm_interface_module.md
    ├── 05_vlm_engine_module.md
    ├── 06_llm_interface_module.md
    ├── 07_synthesizer_module.md
    ├── 08_tts_engine_module.md
    └── 09_orchestrator_module.md
```

## How to Use This Documentation

### For New Users

1. Start with [Complete Pipeline Documentation](SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md) to understand the overall system
2. Follow the installation guide to set up the environment
3. Use the usage guide to run your first pipeline

### For Developers

1. Read the [Complete Pipeline Documentation](SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md) for architecture overview
2. Review individual module documentation for implementation details
3. Refer to API reference for method signatures and parameters

### For Troubleshooting

1. Check the [Complete Pipeline Documentation](SYNAPSE_COMPLETE_PIPELINE_DOCUMENTATION.md) troubleshooting section
2. Review individual module documentation for specific issues
3. Use debugging techniques provided in each module

## Module Dependencies

```
Frame Extraction (M0) ─┐
                    ├─→ Synchronizer (M1) ─→ VLM Engine (M2) ─┐
Transcription (M0b) ─┘                                         ├─→ Synthesizer (M3) ─→ TTS Engine (M6)
                                                               │
VLM Interface (M5a) ───────────────────────────────────────────────┤
                                                               │
LLM Interface (M5b) ───────────────────────────────────────────────┘
                                                               │
Orchestrator (M4) ─────────────────────────────────────────────→ Manifest
```

## Key Concepts

### Suffer Points
Moments when blind students lose visual access due to:
- Deictic references ("look at this graph")
- Silent drawing on the board
- Complex visual content without description

### Intervention Points
Specific moments where audio descriptions must be inserted to restore accessibility

### 3-Mode Explanation System
- **Brief**: Surface-level explanation for quick understanding
- **Explanatory**: Standard-depth explanation for complete coverage
- **Detailed**: Deep exploration for thorough understanding

### Content Types
10 categories of visual content:
1. Equation
2. Graph
3. Circuit
4. Diagram
5. Code
6. Handwritten Notes
7. Biology
8. Chemistry
9. Physics
10. Text

## Support

For issues, questions, or contributions:
- Check individual module documentation
- Review troubleshooting sections
- Consult the complete pipeline documentation

---

*Documentation Version: 1.0*
*Last Updated: April 30, 2026*
