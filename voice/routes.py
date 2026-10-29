"""HTTP endpoints for voice control: /voice/transcribe, /voice/interpret, /voice/speak, /voice/warmup."""

import hashlib
import os
import re
import tempfile
import threading
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.security import enforce_allowed_origin
from voice.interpret import Interpreter
from voice.schemas import InterpretRequest, InterpretResponse, TranscribeResponse
from voice.stt import MAX_BYTES, AudioError, Recognizer

router = APIRouter(prefix="/voice", tags=["voice"], dependencies=[Depends(enforce_allowed_origin)])

_recognizer: Optional[Recognizer] = None
_interpreter: Optional[Interpreter] = None
_tts = None
_lock = threading.Lock()
SPEECH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "synapse_voice_cache")


def get_recognizer() -> Recognizer:
    global _recognizer
    with _lock:
        if _recognizer is None:
            _recognizer = Recognizer()
    return _recognizer


def get_interpreter() -> Interpreter:
    global _interpreter
    with _lock:
        if _interpreter is None:
            _interpreter = Interpreter()
    return _interpreter


def get_tts():
    """The existing TTSEngine (Edge-TTS, with the offline fallback), created on first use."""
    global _tts
    with _lock:
        if _tts is None:
            from tts_engine import TTSEngine
            _tts = TTSEngine()
    return _tts


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=400)


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None),
               recognizer: Recognizer = Depends(get_recognizer)):
    data = file.file.read(MAX_BYTES + 1)
    try:
        return recognizer.transcribe(data, language)
    except AudioError as exc:
        raise HTTPException(status_code=413 if "large" in str(exc) else 422, detail=str(exc))


@router.post("/interpret", response_model=InterpretResponse)
def interpret(request: InterpretRequest, interpreter: Interpreter = Depends(get_interpreter)):
    bad = interpreter.catalog.unknown_actions(request.context.available_actions)
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown actions: {bad}")
    return interpreter.interpret(request)


@router.post("/speak")
def speak(request: SpeakRequest, tts=Depends(get_tts)):
    """Speak a short reply with the existing TTS engine. Repeated phrases are served from a disk cache."""
    text = re.sub(r"\s+", " ", request.text).strip()
    os.makedirs(SPEECH_CACHE_DIR, exist_ok=True)
    path = os.path.join(SPEECH_CACHE_DIR, hashlib.sha1(text.encode("utf-8")).hexdigest() + ".mp3")
    if not os.path.exists(path):
        try:
            tts.generate(text, path)
        except Exception:
            raise HTTPException(status_code=503, detail="Speech is not available right now.")
    return FileResponse(path, media_type="audio/mpeg")


@router.post("/warmup", status_code=204)
def warmup(recognizer: Recognizer = Depends(get_recognizer), interpreter: Interpreter = Depends(get_interpreter)):
    """Load both models now so the first spoken command is not slow."""
    recognizer.warm()
    interpreter.warm()


def warm_up() -> None:
    """Load both models; never raises (voice control is optional and must not stop the server starting)."""
    try:
        get_recognizer().warm()
        get_interpreter().warm()
    except Exception:  # noqa: BLE001
        pass
