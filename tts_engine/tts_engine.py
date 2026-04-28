"""
TTS Engine Module (M6)

Converts finalized text scripts (from LLM Synthesizer) into high-fidelity MP3 audio files.
Uses Microsoft Edge-TTS for enterprise-grade Neural Voices with SpeechT5 as local fallback.
  

"""

import asyncio
import edge_tts
import os
import logging
from typing import Optional, List

# Import shared exceptions
try:
    from .exceptions import TTSEngineError, TTSConnectionError, TTSTimeoutError
except ImportError:
    from exceptions import TTSEngineError, TTSConnectionError, TTSTimeoutError

# Handle both relative and absolute imports for providers
try:
    from .providers import EdgeTTSProvider, SpeechT5Provider, TTSProvider, get_provider
except ImportError:
    from providers import EdgeTTSProvider, SpeechT5Provider, TTSProvider, get_provider

# Set up logging
logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech Engine with provider fallback.

    Primary: Microsoft Edge-TTS (high quality, requires internet)
    Fallback: Microsoft SpeechT5 (good quality, works offline)

    Handles async/sync bridging to work seamlessly with synchronous pipelines.
    """

    # Default voice
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
            voice: Voice ID to use for Edge-TTS (default: en-US-AndrewMultilingualNeural)
            fallback_enabled: Enable fallback to SpeechT5 if Edge-TTS fails (default: True)
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
            logger.info(f"[TTS Engine] Using preferred provider: {preferred_provider}")
        else:
            # Default: Edge-TTS first, then SpeechT5 if fallback enabled
            self._providers.append(EdgeTTSProvider(voice=voice))
            if fallback_enabled:
                self._providers.append(SpeechT5Provider())
                logger.info("[TTS Engine] Initialized with Edge-TTS + SpeechT5 fallback")
            else:
                logger.info("[TTS Engine] Initialized with Edge-TTS only (no fallback)")

        # Track which provider was used for last generation
        self._last_provider_used = None

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
        if output_dir:  # Only create if there's a directory part
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

    @property
    def last_provider_used(self) -> Optional[str]:
        """Get the name of the provider used for the last generation."""
        return self._last_provider_used

    def is_fallback_enabled(self) -> bool:
        """Check if fallback is enabled."""
        return self.fallback_enabled

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

    @staticmethod
    def list_available_voices():
        """
        Development tool to audition available Edge-TTS voices.

        Fetches all voices, filters for English, and prints them.
        """
        # Fetch all voices
        voices = TTSEngine._run_async(edge_tts.list_voices())

        # Filter for English voices
        english_voices = [
            v for v in voices
            if v['Locale'].startswith('en-')
        ]

        print("\n=== Available English Voices ===\n")
        for voice in english_voices:
            print(f"Name: {voice['Name']}")
            print(f"  ID: {voice['ShortName']}")
            print(f"  Gender: {voice['Gender']}")
            print(f"  Locale: {voice['Locale']}")
            print()

    def list_available_providers(self):
        """
        List all available TTS providers.

        Shows which providers are installed and available for use.
        """
        print("\n=== Available TTS Providers ===\n")

        # Check Edge-TTS
        try:
            import edge_tts
            edge_available = True
        except ImportError:
            edge_available = False
        print(f"Edge-TTS: {'[OK] Available' if edge_available else '[X] Not installed'}")

        # Check SpeechT5
        try:
            from transformers import SpeechT5Processor
            import soundfile
            speech_t5_available = True
        except ImportError:
            speech_t5_available = False
        print(f"SpeechT5: {'[OK] Available' if speech_t5_available else '[X] Not installed'}")

        # Show current configuration
        print(f"\nCurrent Configuration:")
        print(f"  Primary: Edge-TTS")
        print(f"  Fallback enabled: {self.fallback_enabled}")
        print(f"  Last provider used: {self._last_provider_used or 'None'}")
        print()


# Example usage
if __name__ == "__main__":
    # Test basic generation (backward compatible - uses Edge-TTS)
    print("=== Test 1: Default (Edge-TTS with fallback) ===")
    engine = TTSEngine()

    test_path = engine.generate(
        "The graph shows velocity increasing over time.",
        "test_output.mp3"
    )

    print(f"Test audio saved to: {test_path}")
    print(f"File exists: {os.path.exists(test_path)}")
    print(f"Provider used: {engine.last_provider_used}")

    # List available providers
    engine.list_available_providers()

    # List available voices
    # TTSEngine.list_available_voices()
