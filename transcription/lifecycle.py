"""Application-owned lifecycle for the persistent transcription worker."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from transcription.process_runtime import TranscriptionProcessRuntime

logger = logging.getLogger(__name__)

WARMUP_ENABLED_ENV = "SYNAPSE_TRANSCRIPTION_WARMUP_ENABLED"
_FALSE_VALUES = {"0", "false", "no", "off"}

_runtime: Optional[TranscriptionProcessRuntime] = None
_runtime_lock = threading.Lock()


def get_transcription_runtime() -> TranscriptionProcessRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = TranscriptionProcessRuntime()
        return _runtime


def transcribe(input_source: str, video_id: Optional[str] = None) -> dict:
    return get_transcription_runtime().transcribe(input_source, video_id)


def shutdown_transcription_runtime(timeout: float = 5.0) -> bool:
    global _runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
    return True if runtime is None else runtime.stop(timeout)


class TranscriptionWarmup:
    def __init__(self) -> None:
        configured = os.environ.get(WARMUP_ENABLED_ENV, "").strip().lower()
        enabled = (
            configured not in _FALSE_VALUES if configured
            else "PYTEST_CURRENT_TEST" not in os.environ
        )
        self.status = "starting" if enabled else "disabled"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.status == "disabled":
            return
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            self.status = "warming"
            logger.info("Video transcription model warm-up started")
            self._thread = threading.Thread(
                target=self._run, name="synapse-transcription-warmup", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            get_transcription_runtime().warm()
            self.status = "ready" if not self._stop.is_set() else "stopped"
            if self.status == "ready":
                logger.info("Video transcription model is ready")
        except Exception as exc:  # noqa: BLE001
            self.status = "unavailable"
            logger.warning("Transcription warm-up failed: %s", str(exc))

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout: float = 5.0) -> bool:
        self.request_stop()
        thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=max(0.0, timeout))
        worker_stopped = shutdown_transcription_runtime(timeout)
        return (thread is None or not thread.is_alive()) and worker_stopped
