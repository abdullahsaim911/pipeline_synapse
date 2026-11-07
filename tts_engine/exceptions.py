"""
TTS Engine Exceptions

Shared exception classes for TTS providers and engine.
"""


class TTSEngineError(Exception):
    """Base sanitized TTS error with a stable classification."""

    code = "tts_failed"
    retryable = False

    def __init__(self, message: str = "Text-to-speech generation failed", *, code=None):
        super().__init__(message)
        if code is not None:
            self.code = code


class TTSConnectionError(TTSEngineError):
    """Network connection error."""
    code = "tts_connection_failed"
    retryable = True


class TTSTimeoutError(TTSEngineError):
    """Timeout during TTS generation."""
    code = "tts_timeout"
    retryable = True


class TTSPermanentError(TTSEngineError):
    """Retrying the same provider request cannot correct this failure."""

    code = "tts_permanent_failure"
    retryable = False
