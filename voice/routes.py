"""HTTP endpoints for voice control: /voice/transcribe, /voice/interpret, /voice/speak, /voice/warmup."""

import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.security import enforce_allowed_origin
from voice.piper_speaker import PiperSpeaker
from voice.process_runtime import ProcessInterpreter, ProcessRecognizer, VoiceProcessRuntime
from voice.schemas import InterpretRequest, InterpretResponse, TranscribeResponse
from voice.stt import MAX_BYTES, AudioError

router = APIRouter(prefix="/voice", tags=["voice"], dependencies=[Depends(enforce_allowed_origin)])

_runtime: Optional[VoiceProcessRuntime] = None
_recognizer: Optional[ProcessRecognizer] = None
_interpreter: Optional[ProcessInterpreter] = None
_tts = None
_instance_lock = threading.Lock()
_voice_turn_gate = threading.BoundedSemaphore(1)
_speech_gate = threading.BoundedSemaphore(1)
SPEECH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "synapse_voice_cache")
SPEECH_CACHE_MAX_FILES = 128
SPEECH_CACHE_MAX_BYTES = 64 * 1024 * 1024
SPEECH_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
# Older Edge-TTS .mp3 replies no longer match and are pruned as stale files.
_CACHE_FILE_RE = re.compile(r"^[0-9a-f]{64}\.wav$")
SPEECH_FILE_EXTENSION = ".wav"
SPEECH_MEDIA_TYPE = "audio/wav"
logger = logging.getLogger(__name__)


@contextmanager
def _work_slot(gate: threading.BoundedSemaphore, detail: str):
    if not gate.acquire(blocking=False):
        raise HTTPException(status_code=429, detail=detail)
    try:
        yield
    finally:
        gate.release()


def _get_runtime_locked() -> VoiceProcessRuntime:
    global _runtime
    if _runtime is None:
        _runtime = VoiceProcessRuntime()
    return _runtime


def start_voice_runtime() -> None:
    """Spawn the native-model worker before the warm-up thread is created."""
    with _instance_lock:
        _get_runtime_locked().start()


def get_recognizer() -> ProcessRecognizer:
    global _recognizer
    with _instance_lock:
        if _recognizer is None:
            _recognizer = ProcessRecognizer(_get_runtime_locked())
    return _recognizer


def get_interpreter() -> ProcessInterpreter:
    global _interpreter
    with _instance_lock:
        if _interpreter is None:
            _interpreter = ProcessInterpreter(_get_runtime_locked())
    return _interpreter


def get_tts():
    """The local Piper voice for spoken replies, created on first use.

    Deliberately independent of the pipeline's explanation TTS (tts_engine).
    """
    global _tts
    with _instance_lock:
        if _tts is None:
            _tts = PiperSpeaker()
    return _tts


def _speech_cache_path(tts, text: str) -> str:
    """Key by voice settings as well as text, so a voice or pace change is never served stale audio."""
    identity = getattr(tts, "cache_identity", "")
    digest = hashlib.sha256(f"{identity}\n{text}".encode("utf-8")).hexdigest()
    return os.path.join(SPEECH_CACHE_DIR, digest + SPEECH_FILE_EXTENSION)


def _prune_speech_cache(now: Optional[float] = None) -> None:
    """Remove partial, stale and oldest replies while keeping the cache bounded."""
    os.makedirs(SPEECH_CACHE_DIR, exist_ok=True)
    current = time.time() if now is None else now
    cached = []
    for entry in os.scandir(SPEECH_CACHE_DIR):
        if not entry.is_file(follow_symlinks=False):
            continue
        if not _CACHE_FILE_RE.fullmatch(entry.name):
            try:
                os.unlink(entry.path)
            except OSError:
                pass
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if current - stat.st_mtime > SPEECH_CACHE_MAX_AGE_SECONDS:
            try:
                os.unlink(entry.path)
            except OSError:
                pass
            continue
        cached.append((stat.st_mtime, stat.st_size, entry.path))

    cached.sort()
    total = sum(size for _, size, _ in cached)
    while cached and (len(cached) > SPEECH_CACHE_MAX_FILES or total > SPEECH_CACHE_MAX_BYTES):
        _, size, path = cached.pop(0)
        try:
            os.unlink(path)
            total -= size
        except OSError:
            pass


def _generate_speech(tts, text: str, path: str) -> None:
    """Generate to a private temporary file, then publish one complete reply atomically."""
    fd, temporary = tempfile.mkstemp(prefix="voice-", suffix=".tmp", dir=SPEECH_CACHE_DIR)
    os.close(fd)
    try:
        tts.generate(text, temporary)
        if not os.path.isfile(temporary) or os.path.getsize(temporary) == 0:
            raise RuntimeError("TTS produced no audio")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=400)


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None),
               recognizer: ProcessRecognizer = Depends(get_recognizer)):
    with _work_slot(_voice_turn_gate, "Another voice request is already running."):
        try:
            data = file.file.read(MAX_BYTES + 1)
            return recognizer.transcribe(data, language)
        except AudioError as exc:
            raise HTTPException(status_code=413 if "large" in str(exc) else 422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Voice transcription is unavailable: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Transcription is not available right now.") from exc


@router.post("/interpret", response_model=InterpretResponse)
def interpret(request: InterpretRequest, interpreter: ProcessInterpreter = Depends(get_interpreter)):
    with _work_slot(_voice_turn_gate, "Another voice request is already running."):
        bad = interpreter.catalog.unknown_actions(request.context.available_actions)
        if bad:
            raise HTTPException(status_code=422, detail=f"Unknown actions: {bad}")
        try:
            return interpreter.interpret(request)
        except Exception as exc:
            logger.warning("Voice interpretation is unavailable: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Voice interpretation is not available right now.") from exc


@router.post("/speak")
def speak(request: SpeakRequest, tts=Depends(get_tts)):
    """Speak a short reply with the local Piper voice. Repeated phrases are served from a disk cache."""
    text = re.sub(r"\s+", " ", request.text).strip()
    with _work_slot(_speech_gate, "Another voice reply is already being prepared."):
        os.makedirs(SPEECH_CACHE_DIR, exist_ok=True)
        path = _speech_cache_path(tts, text)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                _generate_speech(tts, text, path)
            os.utime(path, None)
            _prune_speech_cache()
        except Exception as exc:
            logger.warning("Voice speech generation failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Speech is not available right now.") from exc
    return FileResponse(path, media_type=SPEECH_MEDIA_TYPE)


@router.post("/warmup", status_code=204)
def warmup(recognizer: ProcessRecognizer = Depends(get_recognizer), interpreter: ProcessInterpreter = Depends(get_interpreter)):
    """Load both models now so the first spoken command is not slow."""
    with _work_slot(_voice_turn_gate, "Another voice request is already running."):
        try:
            interpreter.warm()
            recognizer.warm()
        except Exception as exc:
            logger.warning("Voice warm-up failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Voice models are not available right now.") from exc


def warm_up(stop_event: Optional[threading.Event] = None) -> bool:
    """Best-effort owned startup warm-up; never raises or waits behind request work."""
    if not _voice_turn_gate.acquire(blocking=False):
        return False
    try:
        if stop_event is not None and stop_event.is_set():
            return False
        # Load sentence-transformers/PyArrow before faster-whisper/CTranslate2.
        # Reversing this order can terminate Python with 0xC0000005 on Windows.
        get_interpreter().warm()
        if stop_event is not None and stop_event.is_set():
            return False
        get_recognizer().warm()
        _warm_speech()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background voice warm-up failed: %s", type(exc).__name__)
        return False
    finally:
        _voice_turn_gate.release()


def _warm_speech() -> None:
    """Load the reply voice early; a missing voice is reported but never blocks commands."""
    warm = getattr(get_tts(), "warm", None)
    if warm is None:
        return
    try:
        warm()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Voice reply speech is unavailable: %s", exc)


def shutdown_voice_runtime(timeout: float = 1.0) -> bool:
    """Release lazy voice models without waiting indefinitely for active work."""
    global _runtime, _recognizer, _interpreter, _tts
    complete = True

    # Detach and stop the owned process even when a warm-up/request still holds
    # the turn gate. VoiceProcessRuntime.stop() preempts an in-flight call.
    with _instance_lock:
        runtime = _runtime
        _recognizer = _interpreter = _runtime = None
    if runtime is not None and not runtime.stop(timeout):
        complete = False
        logger.warning("Voice model cleanup exceeded shutdown grace")

    if _speech_gate.acquire(timeout=max(0.0, timeout)):
        try:
            with _instance_lock:
                tts, _tts = _tts, None
            cleanup = getattr(tts, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:  # noqa: BLE001
                    complete = False
                    logger.warning("Voice TTS cleanup failed: %s", type(exc).__name__)
            try:
                _prune_speech_cache()
            except Exception as exc:  # noqa: BLE001
                complete = False
                logger.warning("Voice cache cleanup failed: %s", type(exc).__name__)
        finally:
            _speech_gate.release()
    else:
        complete = False
        logger.warning("Voice TTS cleanup exceeded shutdown grace")
    return complete
