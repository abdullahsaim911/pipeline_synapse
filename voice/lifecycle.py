"""Owned, non-blocking lifecycle for optional voice model warm-up."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from voice.routes import start_voice_runtime, warm_up

logger = logging.getLogger(__name__)

VOICE_WARMUP_ENABLED_ENV = "SYNAPSE_VOICE_WARMUP_ENABLED"
VOICE_WARMUP_SHUTDOWN_SECONDS_ENV = "SYNAPSE_VOICE_WARMUP_SHUTDOWN_SECONDS"
DEFAULT_SHUTDOWN_SECONDS = 1.0
_FALSE_VALUES = {"0", "false", "no", "off"}


def voice_warmup_enabled() -> bool:
    configured = os.environ.get(VOICE_WARMUP_ENABLED_ENV, "").strip().lower()
    if configured:
        return configured not in _FALSE_VALUES
    return "PYTEST_CURRENT_TEST" not in os.environ


def _shutdown_seconds() -> float:
    try:
        value = float(os.environ.get(VOICE_WARMUP_SHUTDOWN_SECONDS_ENV, DEFAULT_SHUTDOWN_SECONDS))
    except (TypeError, ValueError):
        value = DEFAULT_SHUTDOWN_SECONDS
    return value if value >= 0 else DEFAULT_SHUTDOWN_SECONDS


class VoiceWarmup:
    """Retains and bounds the optional warm-up thread used by the app lifespan."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.status = "idle" if voice_warmup_enabled() else "disabled"

    def start(self) -> None:
        if self.status == "disabled":
            return
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            try:
                start_voice_runtime()
            except (OSError, RuntimeError) as exc:
                self.status = "unavailable"
                logger.warning("Voice model worker could not start: %s", type(exc).__name__)
                return
            self.status = "warming"
            logger.info("Voice model warm-up started")
            self._thread = threading.Thread(
                target=self._run,
                name="synapse-voice-warmup",
                daemon=True,
            )
            try:
                self._thread.start()
            except (RuntimeError, OSError) as exc:
                self._thread = None
                self.status = "unavailable"
                logger.warning("Voice warm-up could not start: %s", type(exc).__name__)

    def _run(self) -> None:
        try:
            ready = warm_up(self._stop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Voice warm-up failed: %s", str(exc))
            ready = False
        self.status = "ready" if ready else ("stopped" if self._stop.is_set() else "unavailable")
        if ready:
            logger.info("Voice models are ready")

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout: Optional[float] = None) -> bool:
        self.request_stop()
        thread = self._thread
        if thread is None or thread.ident is None:
            return True
        thread.join(_shutdown_seconds() if timeout is None else max(0.0, timeout))
        if thread.is_alive():
            logger.warning("Voice warm-up exceeded shutdown grace")
            return False
        with self._lock:
            if self._thread is thread:
                self._thread = None
        return True
