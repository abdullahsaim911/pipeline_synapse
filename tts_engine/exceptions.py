"""
TTS Engine Exceptions

Shared exception classes for TTS providers and engine.
"""


class TTSEngineError(Exception):
    """Base exception for TTS Engine errors."""
    pass


class TTSConnectionError(TTSEngineError):
    """Network connection error."""
    pass


class TTSTimeoutError(TTSEngineError):
    """Timeout during TTS generation."""
    pass
