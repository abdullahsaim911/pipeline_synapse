"""Shared Ollama lifecycle for the local backend.

One server is reused across explanations. A server Synapse did not start is
never stopped; a server Synapse did start lives until the backend shuts down.
Ollama itself unloads the model from VRAM after its keep-alive window, so an
idle server holds no GPU memory.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Any, Callable, Optional

import requests

from synthesizer import TextSynthesisUnavailableError
from stage_tracing import trace_stage

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0

_shared: Optional["OllamaService"] = None
_shared_lock = threading.Lock()


class OllamaUnavailableError(TextSynthesisUnavailableError):
    """Ollama is unreachable or does not have the configured model."""


class OllamaService:
    def __init__(
        self,
        *,
        start: Callable[[], Any],
        stop: Callable[[Any], None],
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        poll_interval: float = 0.25,
    ) -> None:
        self._start = start
        self._stop = stop
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._owned_process: Any = None
        self._model_confirmed = False
        self._lock = threading.Lock()

    @property
    def owns_process(self) -> bool:
        return self._owned_process is not None

    def _installed_models(self) -> Optional[list[str]]:
        """Return installed model names, or ``None`` if the server is unreachable."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=3)
            if response.status_code != 200:
                return None
            payload = response.json()
        except (requests.exceptions.RequestException, ValueError):
            return None
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return None
        return [
            str(item.get("name") or item.get("model") or "")
            for item in models
            if isinstance(item, dict)
        ]

    def _process_exited(self) -> bool:
        poll = getattr(self._owned_process, "poll", None)
        return poll is not None and poll() is not None

    def _start_and_wait(self) -> list[str]:
        started_at = time.monotonic()
        if self._owned_process is None or self._process_exited():
            trace_stage("ollama_startup", "start")
            self._owned_process = self._start()
        deadline = time.monotonic() + self._startup_timeout
        while True:
            models = self._installed_models()
            if models is not None:
                # Losing a port race to another server is fine: it is not ours.
                if self._process_exited():
                    self._owned_process = None
                trace_stage(
                    "ollama_startup", "ready", started_at=started_at,
                    owns_process=self._owned_process is not None,
                )
                return models
            if self._process_exited():
                self._owned_process = None
                raise OllamaUnavailableError(
                    "The Ollama server exited during startup. Run 'ollama serve' "
                    "in a terminal to see the error."
                )
            if time.monotonic() >= deadline:
                raise OllamaUnavailableError(
                    f"Ollama did not respond at {self._base_url} within "
                    f"{self._startup_timeout:.0f} seconds."
                )
            time.sleep(self._poll_interval)

    def ensure_ready(self) -> None:
        """Reuse a running server or start one, and confirm the model exists."""
        with self._lock:
            models = self._installed_models()
            if models is None:
                self._model_confirmed = False
                models = self._start_and_wait()
            else:
                trace_stage(
                    "ollama_startup", "reused", owns_process=self.owns_process,
                )
            if self._model_confirmed:
                return
            if not any(
                name == self._model or name.startswith(f"{self._model}:")
                for name in models
            ):
                raise OllamaUnavailableError(
                    f"Ollama is running but model '{self._model}' is not installed. "
                    f"Install it with: ollama pull {self._model}"
                )
            self._model_confirmed = True
            trace_stage(
                "ollama_model", "ready", model=self._model,
                owns_process=self.owns_process,
            )

    def shutdown(self) -> None:
        """Stop only a server this service started."""
        with self._lock:
            process, self._owned_process = self._owned_process, None
            self._model_confirmed = False
        if process is not None:
            self._stop(process)


def shared_ollama_service(
    start: Callable[[], Any], stop: Callable[[Any], None]
) -> OllamaService:
    """Return the process-wide service, creating it on first use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = OllamaService(start=start, stop=stop)
            atexit.register(_shared.shutdown)
        return _shared


def shutdown_shared_ollama() -> None:
    with _shared_lock:
        service = _shared
    if service is not None:
        service.shutdown()
