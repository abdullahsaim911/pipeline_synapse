"""
TTS Engine Module (M6)

Provides text-to-speech functionality with automatic fallback:
- Primary: Microsoft Edge-TTS (high quality, requires internet)
- Fallback: Microsoft SpeechT5 (good quality, works offline)
"""

# Import exceptions from the new exceptions module
from .exceptions import TTSEngineError, TTSConnectionError, TTSTimeoutError
# Import main engine
from .tts_engine import TTSEngine
# Import providers (for advanced usage)
from .providers import (
    TTSProvider,
    EdgeTTSProvider,
    SpeechT5Provider,
    get_provider
)

__all__ = [
    # Main engine
    "TTSEngine",
    # Exceptions
    "TTSEngineError",
    "TTSConnectionError",
    "TTSTimeoutError",
    # Providers (for advanced usage)
    "TTSProvider",
    "EdgeTTSProvider",
    "SpeechT5Provider",
    "get_provider"
]
