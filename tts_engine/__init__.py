"""Pocket TTS primary, Kokoro fallback, sharing the backend API runtime."""
from .exceptions import TTSEngineError, TTSConnectionError, TTSPermanentError, TTSTimeoutError
from .tts_engine import TTSEngine, get_shared_tts_engine, shutdown_shared_tts_engine
from .providers import PocketTTSProvider, KokoroProvider, get_provider

__all__ = ['TTSEngine', 'get_shared_tts_engine', 'shutdown_shared_tts_engine',
           'TTSEngineError', 'TTSConnectionError', 'TTSPermanentError', 'TTSTimeoutError',
           'PocketTTSProvider', 'KokoroProvider', 'get_provider']
