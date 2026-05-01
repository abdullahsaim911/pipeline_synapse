# TTS Engine Module (M6) Documentation

## Overview

The TTS Engine Module converts finalized text scripts (from the LLM Synthesizer) into high-fidelity MP3 audio files. It uses a dual-provider system with Microsoft Edge-TTS as the primary option and Microsoft SpeechT5 as a local fallback, ensuring reliable audio generation even without internet connectivity.

## Purpose

- Convert text scripts to natural-sounding audio
- Support multiple voices and languages
- Provide reliable generation with provider fallback
- Handle async/sync bridging for pipeline integration
- Manage GPU memory efficiently

## Tools & Technologies

### Core Dependencies
- **Python 3.9+**: Primary programming language
- **edge-tts**: Microsoft Edge-TTS (primary provider)
- **Transformers**: SpeechT5 model (fallback provider)
- **SpeechT5Processor**: Audio processing
- **Torch**: Deep learning framework for SpeechT5
- **Soundfile**: Audio file I/O

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB |
| GPU (for SpeechT5) | 0GB | 4GB |
| Internet (for Edge-TTS) | Required | Required |
| Disk Space | 100MB | 500MB |

## Implementation Details

### Module Structure

```
tts_engine/
├── __init__.py
├── tts_engine.py               # Main TTS engine
├── providers.py                # TTS provider implementations
└── exceptions.py               # Custom exceptions
```

### Core Components

#### 1. Exception Hierarchy

```python
class TTSEngineError(Exception):
    """Base exception for TTS Engine errors."""

class TTSConnectionError(TTSEngineError):
    """Network/connection error."""

class TTSTimeoutError(TTSEngineError):
    """Timeout during audio generation."""
```

#### 2. TTSProvider Interface

**Abstract Base Class**: Defines interface for all TTS providers

```python
class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    def generate(self, text: str, output_path: str) -> str:
        """Generate audio from text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
```

#### 3. EdgeTTSProvider

**Primary Provider**: Microsoft Edge-TTS (cloud-based)

```python
class EdgeTTSProvider(TTSProvider):
    """
    Microsoft Edge-TTS provider (primary).

    High-quality neural voices, requires internet connection.
    """

    def __init__(self, voice: str = "en-US-AndrewMultilingualNeural"):
        """
        Initialize Edge-TTS provider.

        Args:
            voice: Voice ID to use
        """
        self.voice = voice
        self._communicate = edge_tts.Communicate

    @property
    def name(self) -> str:
        return "Edge-TTS"

    async def generate(self, text: str, output_path: str) -> str:
        """Generate audio using Edge-TTS (async)."""
        communicate = edge_tts.Communicate(text, self.voice)

        await communicate.save(output_path)

        return output_path
```

**Available Voices**:

| Voice ID | Gender | Style | Use Case |
|----------|--------|-------|----------|
| `en-US-AndrewMultilingualNeural` | Male | Authoritative | Default, STEM content |
| `en-US-AriaNeural` | Female | Expressive | Narrative, storytelling |
| `en-US-GuyNeural` | Male | Conversational | Casual explanations |
| `en-US-JennyNeural` | Female | Friendly | Welcoming introductions |
| `en-GB-SoniaNeural` | Female | Professional | Academic content |
| `en-GB-RyanNeural` | Male | Professional | Lectures |

#### 4. SpeechT5Provider

**Fallback Provider**: Microsoft SpeechT5 (local)

```python
class SpeechT5Provider(TTSProvider):
    """
    Microsoft SpeechT5 provider (fallback).

    Good quality, works offline without internet connection.
    """

    def __init__(self):
        """Initialize SpeechT5 provider (lazy-loads models)."""
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embeddings = None

    @property
    def name(self) -> str:
        return "SpeechT5"

    def _load_models(self):
        """Lazy-load SpeechT5 models (only when needed)."""
        if self._model is not None:
            return

        print("[SpeechT5] Loading models...")
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        self._processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self._model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self._speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        print("[SpeechT5] Models loaded")

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio using SpeechT5."""
        self._load_models()

        # Process text
        inputs = self._processor(text=text, return_tensors="pt")

        # Generate speech
        with torch.no_grad():
            speech = self._model.generate_speech(
                inputs["input_ids"],
                self._speaker_embeddings,
                vocoder=self._vocoder
            )

        # Save audio
        sf.write(output_path, speech.numpy(), samplerate=16000)

        return output_path

    def unload_models(self):
        """Unload models to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._vocoder is not None:
            del self._vocoder
            self._vocoder = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[SpeechT5] Models unloaded")
```

#### 5. TTSEngine Class

**Main Engine Class**: Orchestrates provider selection and fallback

```python
class TTSEngine:
    """
    Text-to-Speech Engine with provider fallback.

    Primary: Microsoft Edge-TTS (high quality, requires internet)
    Fallback: Microsoft SpeechT5 (good quality, works offline)

    Handles async/sync bridging to work seamlessly with synchronous pipelines.
    """

    DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        fallback_enabled: bool = True,
        preferred_provider: Optional[str] = None
    ):
        """
        Initialize TTS Engine.

        Args:
            voice: Voice ID to use for Edge-TTS
            fallback_enabled: Enable fallback to SpeechT5 if Edge-TTS fails
            preferred_provider: Force specific provider ('edge' or 'speecht5'), or None for auto
        """
        self.voice = voice
        self.fallback_enabled = fallback_enabled
        self.preferred_provider = preferred_provider

        # Initialize providers
        self._providers: List[TTSProvider] = []

        if preferred_provider:
            # Use only the preferred provider
            self._providers.append(get_provider(preferred_provider, voice=voice))
        else:
            # Default: Edge-TTS first, then SpeechT5 if fallback enabled
            self._providers.append(EdgeTTSProvider(voice=voice))
            if fallback_enabled:
                self._providers.append(SpeechT5Provider())

        # Track which provider was used for last generation
        self._last_provider_used = None
```

#### 6. Generate Method with Fallback

```python
def generate(self, text: str, output_path: str) -> str:
    """
    Converts text to audio using configured providers with fallback.

    Args:
        text: The synthesized script ready for audio
        output_path: Full file path where audio should be saved

    Returns:
        Path to generated audio file

    Raises:
        TTSEngineError: If all providers fail
    """
    if not text or not text.strip():
        logger.warning("[TTS Engine] Empty text provided, skipping generation")
        return output_path

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    last_error = None

    for provider in self._providers:
        try:
            logger.info(f"[TTS Engine] Trying provider: {provider.name}")
            result = provider.generate(text, output_path)
            self._last_provider_used = provider.name
            logger.info(f"[TTS Engine] Successfully generated with {provider.name}")
            return result

        except (TTSConnectionError, TTSTimeoutError) as e:
            logger.warning(f"[TTS Engine] {provider.name} failed (network/timeout): {e}")
            last_error = e
            continue

        except Exception as e:
            logger.warning(f"[TTS Engine] {provider.name} failed: {e}")
            last_error = e
            continue

    # All providers failed
    error_msg = f"All TTS providers failed. Last error: {last_error}"
    logger.error(error_msg)
    raise TTSEngineError(error_msg)
```

#### 7. Async/Sync Bridging

```python
@staticmethod
def _run_async(coro):
    """Run coroutine, handling both async and sync contexts."""
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # nest_asyncio allows run_until_complete on running loop
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
```

#### 8. Cleanup Method

```python
def cleanup(self) -> None:
    """
    Unload all TTS provider models and free memory.

    Call this after all TTS generation is complete to ensure
    SpeechT5 models are removed from GPU/CPU memory.
    """
    logger.info("[TTS Engine] Cleaning up providers...")

    for provider in self._providers:
        try:
            # SpeechT5Provider has explicit unload method
            if hasattr(provider, 'unload_models'):
                provider.unload_models()
        except Exception as e:
            logger.warning(f"[TTS Engine] Error cleaning up {provider.name}: {e}")

    logger.info("[TTS Engine] Cleanup complete")
```

## Configuration Parameters

### Engine Configuration

```python
TTSEngine(
    voice="en-US-AndrewMultilingualNeural",  # Voice selection
    fallback_enabled=True,                     # Enable SpeechT5 fallback
    preferred_provider=None                    # Auto-select provider
)
```

### Provider Selection

```python
# Auto-select (default)
engine = TTSEngine()

# Force Edge-TTS only
engine = TTSEngine(preferred_provider="edge")

# Force SpeechT5 only
engine = TTSEngine(preferred_provider="speecht5")

# Disable fallback
engine = TTSEngine(fallback_enabled=False)
```

### Voice Selection

```python
# Male, authoritative (default)
engine = TTSEngine(voice="en-US-AndrewMultilingualNeural")

# Female, expressive
engine = TTSEngine(voice="en-US-AriaNeural")

# Male, conversational
engine = TTSEngine(voice="en-US-GuyNeural")

# Female, friendly
engine = TTSEngine(voice="en-US-JennyNeural")
```

## Usage Examples

### Basic Usage

```python
from tts_engine import TTSEngine

# Initialize engine (auto-select provider)
engine = TTSEngine()

# Generate audio
audio_path = engine.generate(
    text="The graph shows velocity increasing over time.",
    output_path="output.mp3"
)

print(f"Audio generated: {audio_path}")
print(f"Provider used: {engine.last_provider_used}")
```

### With Custom Voice

```python
# Use different voice
engine = TTSEngine(voice="en-US-AriaNeural")

audio_path = engine.generate(
    text="Welcome to today's lecture on calculus.",
    output_path="welcome.mp3"
)
```

### Error Handling

```python
try:
    audio_path = engine.generate(text, output_path)
    print(f"Success: {audio_path}")
except TTSEngineError as e:
    print(f"TTS failed: {e}")
    # Handle failure
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Batch Processing

```python
def generate_batch(engine, texts, output_dir):
    """Generate audio for multiple texts."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, text in enumerate(texts):
        try:
            output_path = os.path.join(output_dir, f"segment_{i:04d}.mp3")
            audio_path = engine.generate(text, output_path)
            results.append(audio_path)
            print(f"Generated {i+1}/{len(texts)}")

        except Exception as e:
            print(f"Error generating segment {i}: {e}")
            results.append(None)

    return results

texts = ["First segment", "Second segment", "Third segment"]
results = generate_batch(engine, texts, "audio_segments")
```

### List Available Voices

```python
# List all available Edge-TTS voices
TTSEngine.list_available_voices()
```

## Performance Characteristics

### Processing Time (per segment)

| Text Length | Words | Edge-TTS Time | SpeechT5 Time | Notes |
|-------------|-------|---------------|---------------|-------|
| Short | 10-30 | 0.3-0.8s | 1-2s | Quick responses |
| Medium | 30-100 | 0.8-2s | 3-6s | Standard segments |
| Long | 100-300 | 2-5s | 8-15s | Detailed explanations |
| Very Long | 300+ | 5-10s | 20-40s | Deep dives |

### Audio Quality

| Provider | Sample Rate | Quality | Voice Variety | Offline |
|----------|-------------|---------|---------------|---------|
| Edge-TTS | 24kHz | Excellent | 100+ | No |
| SpeechT5 | 16kHz | Good | 1 (default) | Yes |

### Memory Usage

| Component | RAM | VRAM | Notes |
|-----------|-----|------|-------|
| Edge-TTS | 100-200MB | 0GB | Streaming, no model load |
| SpeechT5 | 2-4GB | 2-4GB | Models loaded once |

## Troubleshooting

### Common Issues

**Issue**: "Edge-TTS connection failed"
- **Solutions**:
  - Check internet connection
  - Verify Edge-TTS service is available
  - Try fallback to SpeechT5
  - Check firewall settings

**Issue**: "SpeechT5 out of memory"
- **Solutions**:
  - Use CPU-only mode
  - Process shorter segments
  - Close other applications
  - Use Edge-TTS instead

**Issue**: "Audio quality is poor"
- **Solutions**:
  - Try different voice
  - Check text for pronunciation issues
  - Verify provider is working correctly
  - Adjust text formatting

**Issue**: "Generated audio is too fast/slow"
- **Solutions**:
  - Add SSML tags for rate control (Edge-TTS)
  - Adjust text length
  - Break long text into shorter segments

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check available providers
engine = TTSEngine()
engine.list_available_providers()

# Test with short text
try:
    result = engine.generate("Test", "test.mp3")
    print(f"Test successful: {result}")
    print(f"Provider: {engine.last_provider_used}")
except Exception as e:
    print(f"Test failed: {e}")
```

## Best Practices

### 1. Provider Selection

```python
# Choose based on requirements
def select_engine(requirements):
    """Select TTS engine based on requirements."""
    if requirements.get("offline"):
        # Must work offline
        return TTSEngine(preferred_provider="speecht5")
    elif requirements.get("highest_quality"):
        # Best quality, need internet
        return TTSEngine(preferred_provider="edge")
    else:
        # Auto-select with fallback
        return TTSEngine(fallback_enabled=True)
```

### 2. Text Preparation

```python
# Prepare text for better TTS
def prepare_text(text: str) -> str:
    """Prepare text for TTS generation."""
    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Add pauses for readability
    text = text.replace('. ', '.<break time="500ms"/> ')
    text = text.replace(', ', ',<break time="200ms"/> ')

    return text
```

### 3. Batch Processing

```python
# Efficient batch processing
def process_batch_efficiently(engine, segments, output_dir):
    """Process segments efficiently with cleanup."""
    results = []

    for i, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f"segment_{i:04d}.mp3")

        try:
            audio_path = engine.generate(segment.text, output_path)
            results.append({
                "index": i,
                "audio_path": audio_path,
                "provider": engine.last_provider_used
            })

        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            results.append({"index": i, "error": str(e)})

        # Optional: cleanup between segments
        if (i + 1) % 10 == 0:
            engine.cleanup()

    return results
```

### 4. Memory Management

```python
# Always cleanup after use
engine = TTSEngine()

try:
    # Generate all audio segments
    for segment in segments:
        engine.generate(segment.text, f"audio/{segment.id}.mp3")

finally:
    # Clean up to free memory
    engine.cleanup()
```

## Future Enhancements

1. **Additional Providers**: Support for Google TTS, Amazon Polly, etc.
2. **Voice Cloning**: Custom voice synthesis
3. **Emotion Control**: Adjust emotional tone
4. **Prosody Control**: Fine-tune pitch, rate, volume
5. **Real-time Streaming**: Live audio generation
6. **Audio Post-Processing**: Normalization, noise reduction
7. **Multi-language Voices**: Extended language support
8. **SSML Support**: Full SSML markup support
