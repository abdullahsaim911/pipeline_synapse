"""
TTS Providers Module

Abstract base class and concrete implementations for different TTS backends.
Enables fallback chain: Edge-TTS → SpeechT5.

"""

import asyncio
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional
import logging
import nest_asyncio
nest_asyncio.apply()

# Import shared exceptions
try:
    from .exceptions import (
        TTSEngineError, TTSConnectionError, TTSPermanentError, TTSTimeoutError,
    )
except ImportError:
    from exceptions import (
        TTSEngineError, TTSConnectionError, TTSPermanentError, TTSTimeoutError,
    )

logger = logging.getLogger(__name__)

# Import Edge-TTS
try:
    import edge_tts
    from edge_tts.exceptions import (
        NoAudioReceived,
        UnexpectedResponse,
        UnknownResponse,
        WebSocketError,
    )
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge_tts not available")

# Import SpeechT5 (transformers - should be available)
try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import torch
    import soundfile as sf
    import numpy as np
    import requests
    SPEECH_T5_AVAILABLE = True
except ImportError:
    SPEECH_T5_AVAILABLE = False
    logger.warning("SpeechT5 dependencies not available")


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    def generate(self, text: str, output_path: str) -> str:
        """Generate audio from text.

        Args:
            text: Text to convert to speech
            output_path: Path where audio file should be saved

        Returns:
            Path to generated audio file

        Raises:
            TTSEngineError: If generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and ready.

        Returns:
            True if provider can be used
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass


def _edge_error(exc: Exception) -> TTSEngineError:
    """Classify Edge failures without returning provider payloads or secrets."""
    if isinstance(exc, TTSEngineError):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return TTSTimeoutError("Edge TTS request timed out")
    if EDGE_TTS_AVAILABLE and isinstance(
        exc, (NoAudioReceived, UnexpectedResponse, UnknownResponse, WebSocketError)
    ):
        return TTSConnectionError(
            "Edge TTS returned no usable audio",
            code="tts_service_unavailable",
        )
    if isinstance(exc, (PermissionError, FileNotFoundError, IsADirectoryError)):
        return TTSPermanentError(
            "Edge TTS output configuration is invalid",
            code="tts_invalid_output",
        )
    if isinstance(exc, (ConnectionError, OSError)):
        return TTSConnectionError("Edge TTS connection was interrupted")
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return TTSConnectionError(
            "Edge TTS service is temporarily unavailable",
            code="tts_service_unavailable",
        )
    if status in {400, 401, 403, 404, 405, 422}:
        return TTSPermanentError(
            "Edge TTS rejected the request",
            code="tts_request_rejected",
        )
    if isinstance(exc, (TypeError, ValueError)):
        return TTSPermanentError(
            "Edge TTS configuration or input is invalid",
            code="tts_invalid_request",
        )
    return TTSPermanentError(
        "Edge TTS failed unexpectedly",
        code="tts_unexpected_provider_error",
    )


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge-TTS provider (primary, requires internet)."""

    FALLBACK_VOICE = "en-US-AriaNeural"
    _voice_lock = threading.Lock()
    _available_voices: Optional[set[str]] = None

    def __init__(
        self,
        voice: str = "en-US-AndrewMultilingualNeural",
        *,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.25,
    ):
        """Initialize Edge-TTS provider.

        Args:
            voice: Voice ID to use
        """
        self.voice = voice
        self._validate_voice(voice)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def _validate_voice(self, voice: str) -> None:
        """Validate voice format."""
        if not voice:
            raise ValueError("Voice cannot be empty")
        if not any(char.isupper() for char in voice):
            logger.warning(f"Unusual voice format: {voice}")

    def is_available(self) -> bool:
        """Check if Edge-TTS is available."""
        return EDGE_TTS_AVAILABLE

    @property
    def name(self) -> str:
        return f"EdgeTTS({self.voice})"

    def _run_async(self, coro):
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

    def _run_voice_catalog(self):
        """Resolve the voice catalogue without consuming synthesis test hooks."""
        return self._run_coroutine(edge_tts.list_voices())

    @staticmethod
    def _run_coroutine(coro):
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def _resolved_voice(self) -> str:
        """Use the preferred voice when advertised, otherwise a stable fallback."""
        with self._voice_lock:
            if self._available_voices is None:
                try:
                    voices = self._run_voice_catalog()
                    self.__class__._available_voices = {
                        item.get("ShortName") for item in voices
                        if isinstance(item, dict) and item.get("ShortName")
                    }
                except Exception:
                    # Catalogue lookup is advisory; synthesis remains the real
                    # availability check and SpeechT5 is the durable fallback.
                    return self.voice
            available = self._available_voices or set()
        if self.voice in available:
            return self.voice
        if self.FALLBACK_VOICE in available:
            logger.warning("Configured Edge voice unavailable; using fallback voice")
            return self.FALLBACK_VOICE
        return self.voice

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio using Edge-TTS."""
        if not self.is_available():
            raise RuntimeError("Edge-TTS is not available")

        # Ensure output directory exists (if there is one)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        voice = self._resolved_voice()
        for attempt in range(1, self.max_attempts + 1):
            try:
                communicate = edge_tts.Communicate(text, voice)
                self._run_async(communicate.save(output_path))
                logger.debug("[%s] Generated on attempt %s", self.name, attempt)
                return output_path
            except Exception as exc:
                classified = _edge_error(exc)
                try:
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                except OSError:
                    logger.debug("Could not remove partial Edge TTS output")
                if not classified.retryable or attempt >= self.max_attempts:
                    logger.warning(
                        "[%s] Generation stopped: code=%s retryable=%s attempt=%s",
                        self.name, classified.code, classified.retryable, attempt,
                    )
                    raise classified from exc
                logger.warning(
                    "[%s] Transient failure; retrying: code=%s attempt=%s",
                    self.name, classified.code, attempt,
                )
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * attempt)

        raise TTSEngineError("Edge TTS generation failed")


class SpeechT5Provider(TTSProvider):
    """Microsoft SpeechT5 provider (local fallback, works offline)."""

    _model_lock = threading.Lock()
    _inference_lock = threading.Lock()
    _processor = None
    _model = None
    _vocoder = None
    _speaker_embeddings = None
    _device = None
    MAX_CHUNK_TOKENS = 540
    CHUNK_SILENCE_SECONDS = 0.15

    def __init__(self, device: Optional[str] = None):
        """Initialize SpeechT5 provider.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_models(self):
        """Lazy load SpeechT5 models (thread-safe)."""
        with self._model_lock:
            if self._model is not None:
                return  # Already loaded

            logger.debug(f"[SpeechT5] Loading models on {self.device}...")

            # Load models
            self._processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self._model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Move to device
            self._model.to(self.device)
            self._vocoder.to(self.device)

            # Load the fixed speaker embedding from the one-time setup cache.
            try:
                from huggingface_hub import hf_hub_download
                embedding_path = hf_hub_download(
                    repo_id="Xenova/transformers.js-docs",
                    repo_type="dataset",
                    filename="speaker_embeddings.bin",
                    local_files_only=True,
                )
                with open(embedding_path, "rb") as handle:
                    x_vector = torch.frombuffer(
                        bytearray(handle.read()), dtype=torch.float32
                    )

                # SpeechT5 expects 512-dim embeddings
                if len(x_vector) != 512:
                    logger.warning(f"Speaker embeddings has unexpected shape: {x_vector.shape}")
                    # Pad or truncate to 512
                    if len(x_vector) < 512:
                        x_vector = torch.nn.functional.pad(x_vector, (0, 512 - len(x_vector)))
                    else:
                        x_vector = x_vector[:512]

                self._speaker_embeddings = x_vector.unsqueeze(0).to(self.device)
                logger.debug("[SpeechT5] Speaker embeddings loaded successfully")

            except Exception as e:
                logger.warning(
                    "[SpeechT5] Cached speaker embedding unavailable: error_type=%s",
                    type(e).__name__,
                )
                # Deterministic fallback keeps offline generation reproducible.
                generator = torch.Generator(device="cpu").manual_seed(0)
                self._speaker_embeddings = torch.randn(
                    1, 512, generator=generator
                ).to(self.device)

            logger.debug("[SpeechT5] Models loaded successfully")

    def is_available(self) -> bool:
        """Check if SpeechT5 is available."""
        return SPEECH_T5_AVAILABLE

    @property
    def name(self) -> str:
        return f"SpeechT5({self.device})"

    def unload_models(self) -> None:
        """
        Explicitly unload SpeechT5 models and free GPU memory.

        This should be called after all TTS generation is complete
        to ensure models are removed from GPU/CPU memory.

        Thread-safe - can be called from any thread.
        """
        with self._model_lock:
            if self._model is None:
                return  # Not loaded, nothing to unload

            logger.debug(f"[SpeechT5] Unloading models from {self.device}...")

            try:
                # Delete model references first
                if self._processor is not None:
                    del self._processor
                    self._processor = None

                if self._model is not None:
                    self._model.to("cpu")  # Move to CPU first (safer)
                    del self._model
                    self._model = None

                if self._vocoder is not None:
                    self._vocoder.to("cpu")  # Move to CPU first
                    del self._vocoder
                    self._vocoder = None

                if self._speaker_embeddings is not None:
                    del self._speaker_embeddings
                    self._speaker_embeddings = None

                # Force Python garbage collection
                import gc
                gc.collect()

                # Clear CUDA cache if on GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logger.debug("[SpeechT5] Models unloaded successfully, GPU cache cleared")

            except Exception as e:
                logger.error(
                    "[SpeechT5] Unload failed: error_type=%s", type(e).__name__
                )


    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for SpeechT5.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Limit length (SpeechT5 works best with shorter texts)
        # For longer texts, we'll let it process but may have quality tradeoffs
        if len(text) > 1000:
            logger.warning(f"[SpeechT5] Long text ({len(text)} chars) may have reduced quality")

        return text

    def _token_count(self, text: str) -> int:
        encoded = self._processor.tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        )
        return len(encoded["input_ids"])

    def _split_chunks(self, text: str) -> list[str]:
        """Split all text losslessly enough for SpeechT5's 600-position encoder."""
        sentences = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ] or [text]
        atomic: list[str] = []
        for sentence in sentences:
            if self._token_count(sentence) <= self.MAX_CHUNK_TOKENS:
                atomic.append(sentence)
                continue
            current: list[str] = []
            for word in sentence.split():
                candidate = " ".join([*current, word])
                if current and self._token_count(candidate) > self.MAX_CHUNK_TOKENS:
                    atomic.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                atomic.append(" ".join(current))

        chunks: list[str] = []
        current = ""
        for part in atomic:
            candidate = f"{current} {part}".strip()
            if current and self._token_count(candidate) > self.MAX_CHUNK_TOKENS:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio using SpeechT5."""
        if not self.is_available():
            raise RuntimeError("SpeechT5 is not available")

        # Validate text
        text = self._preprocess_text(text)
        if not text.strip():
            logger.warning("[SpeechT5] Empty text after preprocessing, skipping")
            return output_path

        # Ensure output directory exists (if there is one)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Lazy load models
        try:
            self._load_models()
        except Exception as e:
            logger.error(
                "[SpeechT5] Model loading failed: error_type=%s", type(e).__name__
            )
            raise TTSEngineError(
                "SpeechT5 model loading failed", code="tts_fallback_load_failed"
            ) from e

        try:
            chunks = self._split_chunks(text)
            logger.info("[SpeechT5] Generating %d token-safe chunk(s)", len(chunks))
            audio_parts = []
            silence = np.zeros(
                int(16000 * self.CHUNK_SILENCE_SECONDS), dtype=np.float32
            )
            with self._inference_lock:
                for index, chunk in enumerate(chunks):
                    inputs = self._processor(text=chunk, return_tensors="pt")
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        speech = self._model.generate(
                            **inputs,
                            speaker_embeddings=self._speaker_embeddings,
                        )
                        audio = self._vocoder(speech)
                    if index:
                        audio_parts.append(silence)
                    audio_parts.append(
                        np.atleast_1d(audio.squeeze().cpu().numpy())
                    )
            audio_numpy = np.concatenate(audio_parts)

            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_numpy)) > 0:
                audio_numpy = audio_numpy / np.max(np.abs(audio_numpy)) * 0.9

            sf.write(output_path, audio_numpy, samplerate=16000)

            logger.debug(f"[{self.name}] Generated: {output_path}")
            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("[SpeechT5] CUDA out of memory, trying CPU...")
            if self.device == "cuda":
                self.unload_models()
                self.device = "cpu"
                return self.generate(text, output_path)
            else:
                raise TTSEngineError("SpeechT5 ran out of memory on CPU")
        except Exception as e:
            logger.error(
                "[SpeechT5] Generation failed: error_type=%s", type(e).__name__
            )
            raise TTSEngineError(
                "SpeechT5 generation failed", code="tts_fallback_failed"
            ) from e


def get_provider(name: str, **kwargs) -> TTSProvider:
    """Factory function to create a provider by name.

    Args:
        name: Provider name ('edge' or 'speecht5')
        **kwargs: Provider-specific arguments

    Returns:
        TTSProvider instance
    """
    name_lower = name.lower()

    if name_lower == "edge" or name_lower == "edge-tts":
        return EdgeTTSProvider(**kwargs)
    elif name_lower == "speecht5" or name_lower == "speech-t5":
        return SpeechT5Provider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}. Available: 'edge', 'speecht5'")
