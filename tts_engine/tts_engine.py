"""One shared CPU TTS engine, with PCM publication and MP3 encoding."""
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from .exceptions import TTSEngineError
from .providers import PocketTTSProvider, KokoroProvider, SAMPLE_RATE
from .streaming import current_stream

logger = logging.getLogger(__name__)


_speech_priority = ContextVar('speech_priority', default=1)


@contextmanager
def speech_priority(priority):
    """Bind 0=voice, 1=foreground, 2=background, or a callable returning one.

    Callables are re-evaluated for queued admission, allowing live promotion.
    They must be cheap and must not acquire manager locks (e.g. set membership).
    """
    if not callable(priority) and priority not in (0, 1, 2):
        raise ValueError('Speech priority must be 0, 1 or 2')
    token = _speech_priority.set(priority)
    try:
        yield
    finally:
        _speech_priority.reset(token)


class _SpeechPriorityGate:
    """Non-preemptive, shared admission with FIFO order within each priority."""

    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._waiters = []
        self._sequence = 0

    @staticmethod
    def _rank(ticket):
        sequence, priority = ticket
        value = priority() if callable(priority) else priority
        if value not in (0, 1, 2):
            raise ValueError('Speech priority must be 0, 1 or 2')
        return value, sequence

    def acquire(self, blocking=True, timeout=-1, *, check=None):
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout < 0 and timeout != -1:
            raise ValueError('timeout value must be positive')
        deadline = None if timeout == -1 else time.monotonic() + timeout
        with self._condition:
            ticket = (self._sequence, _speech_priority.get())
            self._sequence += 1
            self._waiters.append(ticket)
        try:
            while True:
                # Ownership checks may perform I/O; do not hold the condition.
                if check is not None:
                    check()
                with self._condition:
                    if not self._active and ticket == min(self._waiters, key=self._rank):
                        self._waiters.remove(ticket)
                        self._active = True
                        return True
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if not blocking or (remaining is not None and remaining <= 0):
                        return False
                    # Cancellation and live priority changes have no notification.
                    dynamic = any(callable(waiter[1]) for waiter in self._waiters)
                    interval = .05 if check is not None or dynamic else None
                    if remaining is not None:
                        interval = remaining if interval is None else min(interval, remaining)
                    self._condition.wait(interval)
        finally:
            with self._condition:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()

    def release(self):
        with self._condition:
            if not self._active:
                raise RuntimeError('release unlocked lock')
            self._active = False
            self._condition.notify_all()

    def locked(self):
        with self._condition:
            return self._active

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()

    @contextmanager
    def admission(self, check):
        self.acquire(check=check)
        try:
            yield
        finally:
            self.release()


class TTSEngine:
    DEFAULT_VOICE = 'alba'
    _inference_lock = _SpeechPriorityGate()

    def __init__(self, voice=DEFAULT_VOICE, fallback_enabled=True, preferred_provider=None):
        from .providers import get_provider
        self.voice = voice
        self.fallback_enabled = fallback_enabled
        self._providers = ([get_provider(preferred_provider)] if preferred_provider else
                           [PocketTTSProvider()] + ([KokoroProvider()] if fallback_enabled else []))
        self._last_provider_used = None
        self._stop = threading.Event()
        self._active_stop = None

    def warm(self, stop_event=None, admission_check=None):
        """Load the primary speech model through the shared inference gate."""
        def check():
            if self._stop.is_set() or (stop_event is not None and stop_event.is_set()):
                raise TTSEngineError('Speech warm-up was cancelled', code='tts_cancelled')
            if admission_check is not None and not self.primary_loaded:
                admission_check()

        with speech_priority(2), self._inference_lock.admission(check):
            check()
            self._providers[0].warm()
            check()

    @property
    def primary_loaded(self):
        return self._providers[0].loaded

    def generate(self, text, output_path):
        import numpy as np
        import imageio_ffmpeg
        if not text or not text.strip():
            raise TTSEngineError('Speech text is empty', code='tts_empty_text')
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = current_stream.get()
        def check():
            if stream:
                stream.check()
            if self._stop.is_set():
                raise TTSEngineError('TTS is shutting down', code='tts_cancelled')

        with self._inference_lock.admission(check):
            check()
            providers = self._providers
            if stream and stream.prefer_fallback and len(providers) > 1:
                providers = providers[1:]
            last = None
            for provider in providers:
                stop = threading.Event()
                self._active_stop = stop
                iterator = None
                published = False
                encoder = None
                fd, temporary = tempfile.mkstemp(suffix='.mp3', dir=output.parent)
                os.close(fd)
                try:
                    encoder = subprocess.Popen([
                        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-y', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1',
                        '-i', 'pipe:0', '-c:a', 'libmp3lame', '-b:a', '96k', temporary,
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    samples = 0
                    iterator = provider.chunks(text, stop)
                    for chunk in iterator:
                        check()
                        if self._stop.is_set():
                            raise TTSEngineError('TTS is shutting down', code='tts_cancelled')
                        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if not chunk.size:
                            continue
                        if not np.isfinite(chunk).all():
                            raise TTSEngineError('Invalid speech samples', code='tts_invalid_audio')
                        pcm = (np.clip(chunk, -1, 1) * 32767).astype('<i2').tobytes()
                        for offset in range(0, len(pcm), 12000):
                            check()
                            part = pcm[offset:offset + 12000]
                            encoder.stdin.write(part)
                            if stream:
                                stream.audio(text, provider.name, part)
                                published = True
                        samples += chunk.size
                    if not samples:
                        raise TTSEngineError('No speech was generated', code='tts_empty_audio')
                    encoder.stdin.close()
                    if encoder.wait(timeout=30) != 0:
                        raise TTSEngineError('Audio encoding failed', code='tts_encoding_failed')
                    check()
                    if Path(temporary).stat().st_size < 4:
                        raise TTSEngineError('Empty encoded audio', code='tts_invalid_audio')
                    os.replace(temporary, output)
                    self._last_provider_used = provider.name
                    return str(output)
                except Exception as exc:
                    if stream:
                        check()
                    last = exc
                    logger.warning('TTS provider failed: provider=%s error_type=%s', provider.name, type(exc).__name__)
                    if published:
                        if stream:
                            stream.prefer_fallback = True
                        raise TTSEngineError('Speech was interrupted; retry using fallback',
                                             code='tts_stream_interrupted') from exc
                finally:
                    stop.set()
                    if iterator is not None and hasattr(iterator, 'close'):
                        try:
                            iterator.close()
                        except Exception as exc:
                            logger.warning('Speech iterator cleanup failed: error_type=%s', type(exc).__name__)
                    self._active_stop = None
                    if encoder and encoder.poll() is None:
                        encoder.kill()
                        encoder.wait()
                    if encoder and encoder.stdin and not encoder.stdin.closed:
                        try:
                            encoder.stdin.close()
                        except OSError:
                            pass  # A killed Windows encoder has already closed its pipe.
                    Path(temporary).unlink(missing_ok=True)
            raise TTSEngineError('Speech providers unavailable',
                                 code=getattr(last, 'code', 'tts_unavailable')) from last

    @property
    def last_provider_used(self):
        return self._last_provider_used

    def is_fallback_enabled(self):
        return self.fallback_enabled

    def cleanup(self):
        self._stop.set()
        if self._active_stop is not None:
            self._active_stop.set()
        if self._inference_lock.acquire(timeout=1):
            try:
                for provider in self._providers:
                    provider.unload_models()
            finally:
                self._inference_lock.release()


_shared_engine = None
_shared_engine_lock = threading.Lock()


def get_shared_tts_engine():
    global _shared_engine
    with _shared_engine_lock:
        if _shared_engine is None:
            _shared_engine = TTSEngine()
        return _shared_engine


def shutdown_shared_tts_engine():
    global _shared_engine
    with _shared_engine_lock:
        engine, _shared_engine = _shared_engine, None
    if engine:
        engine.cleanup()
