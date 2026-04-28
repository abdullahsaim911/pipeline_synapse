"""
TTS Providers Module

Abstract base class and concrete implementations for different TTS backends.
Enables fallback chain: Edge-TTS → SpeechT5.

"""

import asyncio
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional
import logging
import nest_asyncio
nest_asyncio.apply()

# Import shared exceptions
try:
    from .exceptions import TTSEngineError, TTSConnectionError, TTSTimeoutError
except ImportError:
    from exceptions import TTSEngineError, TTSConnectionError, TTSTimeoutError

logger = logging.getLogger(__name__)

# Import Edge-TTS
try:
    import edge_tts
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


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge-TTS provider (primary, requires internet)."""

    def __init__(self, voice: str = "en-US-AndrewMultilingualNeural"):
        """Initialize Edge-TTS provider.

        Args:
            voice: Voice ID to use
        """
        self.voice = voice
        self._validate_voice(voice)

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

    def generate(self, text: str, output_path: str) -> str:
        """Generate audio using Edge-TTS."""
        if not self.is_available():
            raise RuntimeError("Edge-TTS is not available")

        # Ensure output directory exists (if there is one)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            self._run_async(communicate.save(output_path))
            logger.info(f"[{self.name}] Generated: {output_path}")
            return output_path

        except Exception as first_error:
            logger.warning(f"[{self.name}] First attempt failed: {first_error}, retrying...")
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                self._run_async(communicate.save(output_path))
                logger.info(f"[{self.name}] Generated (retry): {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"[{self.name}] Failed completely: {e}")
                if "Connection" in str(e) or "network" in str(e).lower():
                    raise TTSConnectionError(f"Network error: {e}")
                elif "timeout" in str(e).lower():
                    raise TTSTimeoutError(f"Timeout error: {e}")
                raise TTSEngineError(f"Edge-TTS failed: {e}")


class SpeechT5Provider(TTSProvider):
    """Microsoft SpeechT5 provider (local fallback, works offline)."""

    _model_lock = threading.Lock()
    _processor = None
    _model = None
    _vocoder = None
    _speaker_embeddings = None
    _device = None

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

            logger.info(f"[SpeechT5] Loading models on {self.device}...")

            # Load models
            self._processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self._model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Move to device
            self._model.to(self.device)
            self._vocoder.to(self.device)

            # Load speaker embeddings (use a default speaker)
            # This is a neutral speaker from the SpeechT5 training data
            embeddings_url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin"

            # Download the speaker embeddings file
            try:
                response = requests.get(embeddings_url, timeout=30)
                response.raise_for_status()

                # The file is a raw binary tensor (float32)
                # Load it as a tensor
                x_vector = torch.frombuffer(bytearray(response.content), dtype=torch.float32)

                # SpeechT5 expects 512-dim embeddings
                if len(x_vector) != 512:
                    logger.warning(f"Speaker embeddings has unexpected shape: {x_vector.shape}")
                    # Pad or truncate to 512
                    if len(x_vector) < 512:
                        x_vector = torch.nn.functional.pad(x_vector, (0, 512 - len(x_vector)))
                    else:
                        x_vector = x_vector[:512]

                self._speaker_embeddings = x_vector.unsqueeze(0).to(self.device)
                logger.info("[SpeechT5] Speaker embeddings loaded successfully")

            except Exception as e:
                logger.error(f"[SpeechT5] Failed to load speaker embeddings: {e}")
                # Fallback: use random embeddings
                logger.warning("[SpeechT5] Using random speaker embeddings as fallback")
                self._speaker_embeddings = torch.randn(1, 512).to(self.device)

            logger.info("[SpeechT5] Models loaded successfully")

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

            logger.info(f"[SpeechT5] Unloading models from {self.device}...")

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

                logger.info("[SpeechT5] Models unloaded successfully, GPU cache cleared")

            except Exception as e:
                logger.error(f"[SpeechT5] Error during unload: {e}")


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
            logger.error(f"[SpeechT5] Failed to load models: {e}")
            raise TTSEngineError(f"SpeechT5 model loading failed: {e}")

        try:
            # Prepare inputs
            inputs = self._processor(text=text, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate speech
            with torch.no_grad():
                speech = self._model.generate(
                    **inputs,
                    speaker_embeddings=self._speaker_embeddings
                )

                # Use vocoder to convert to audio
                with torch.no_grad():
                    audio = self._vocoder(speech)

            # Convert to numpy and save
            audio_numpy = audio.squeeze().cpu().numpy()

            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_numpy)) > 0:
                audio_numpy = audio_numpy / np.max(np.abs(audio_numpy)) * 0.9

            sf.write(output_path, audio_numpy, samplerate=16000)

            logger.info(f"[{self.name}] Generated: {output_path}")
            return output_path

        except torch.cuda.OutOfMemoryError:
            logger.error("[SpeechT5] CUDA out of memory, trying CPU...")
            if self.device == "cuda":
                self.device = "cpu"
                self._model_lock.release()
                return self.generate(text, output_path)
            else:
                raise TTSEngineError("SpeechT5 ran out of memory on CPU")
        except Exception as e:
            logger.error(f"[SpeechT5] Generation failed: {e}")
            raise TTSEngineError(f"SpeechT5 generation failed: {e}")


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
