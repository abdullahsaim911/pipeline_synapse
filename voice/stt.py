"""
Speech recognition (module 3): faster-whisper on the CPU, loaded on first use and unloaded when idle.

Input is the 16 kHz mono 16-bit WAV the frontend records, so no audio decoder is needed. Silence is rejected
before the model runs, and segments Whisper itself marks as probably-not-speech are dropped (it otherwise
invents text such as "Thank you." on silence).
"""

import io
import os
import re
import logging
import threading
import time
import wave

import numpy as np

from voice.schemas import TranscribeResponse

SAMPLE_RATE = 16_000
MAX_BYTES = 2_000_000
MAX_SECONDS = 30.0
MIN_SECONDS = 0.2
log = logging.getLogger("uvicorn.error")
SILENCE_RMS = 0.0008
NO_SPEECH_PROB_MAX = 0.6
IDLE_UNLOAD_SECONDS = 300
DEFAULT_MODEL = "base"
# Helps Whisper with lecture vocabulary it otherwise mishears ("eigenvectors" -> "e-gen vectors").
VOCABULARY_PROMPT = "Lecture commands. Terms: eigenvector, eigenvalue, derivative, integral, Fourier transform, bookmarks, night theme, night mode, dark mode, sepia theme, paper theme, study mode, accessibility mode."
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


class AudioError(ValueError):
    """The uploaded audio cannot be used."""


def decode_wav(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise AudioError("The audio is too large.")
    try:
        with wave.open(io.BytesIO(data)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
                raise AudioError("Expected 16 kHz mono 16-bit WAV audio.")
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError("The audio is not a valid WAV file.") from exc
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) / SAMPLE_RATE > MAX_SECONDS:
        raise AudioError("The audio is too long.")
    return samples


class Recognizer:
    def __init__(self, model_size=None, loader=None, clock=time.monotonic):
        self.model_size = model_size or os.environ.get("VOICE_WHISPER_MODEL", DEFAULT_MODEL)
        self._loader = loader or self._load_whisper
        self._clock = clock
        self._model = None
        self._last_used = 0.0
        self._lock = threading.Lock()

    def _load_whisper(self):
        from faster_whisper import WhisperModel
        return WhisperModel(self.model_size, device="cpu", compute_type="int8")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._model is not None and self._clock() - self._last_used > IDLE_UNLOAD_SECONDS:
                self._model = None
                return True
        return False

    def warm(self) -> None:
        with self._lock:
            if self._model is None:
                self._model = self._loader()
            self._last_used = self._clock()

    def transcribe(self, data: bytes, language=None) -> TranscribeResponse:
        if language is not None and not _LANG_RE.match(language):
            raise AudioError("Invalid language code.")
        samples = decode_wav(data)
        duration = len(samples) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        log.info("transcribe: %.2fs audio, rms=%.5f, peak=%.4f", duration, rms, float(np.max(np.abs(samples))) if len(samples) else 0.0)
        if duration < MIN_SECONDS or rms < SILENCE_RMS:
            return TranscribeResponse(text="", duration_seconds=duration, no_speech=True)

        # Laptop microphones are often quiet: bring the clip to a normal level (at most 30x) so the model's own
        # speech detection does not discard it.
        peak = float(np.max(np.abs(samples)))
        if 0.0 < peak < 0.5:
            samples = np.clip(samples * min(0.8 / peak, 30.0), -1.0, 1.0).astype(np.float32)

        self.unload_if_idle()
        with self._lock:  # one recognition at a time: the model is not shared across threads
            if self._model is None:
                self._model = self._loader()
            self._last_used = self._clock()
            segments, info = self._model.transcribe(
                samples, language=language, beam_size=1, vad_filter=True,
                condition_on_previous_text=False, initial_prompt=VOCABULARY_PROMPT,
            )
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob <= NO_SPEECH_PROB_MAX).strip()
            self._last_used = self._clock()
        log.info("transcribe: heard %r (language %s)", text, info.language)
        return TranscribeResponse(
            text=text, language=info.language, language_probability=min(1.0, max(0.0, float(info.language_probability))),
            duration_seconds=duration, no_speech=not text,
        )
