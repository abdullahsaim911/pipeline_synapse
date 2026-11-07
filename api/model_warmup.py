"""Non-blocking readiness tracking for the remote Modal VLM.

Probing ``/health`` wakes a scaled-to-zero Modal container and vLLM only answers
once its weights are loaded, so a probe costs GPU seconds but no inference and
no model output tokens. ``/v1/models`` then proves the credentials and model.

``ready`` describes the moment of the last successful probe. Modal scales the
container to zero after its idle window, so it is not a promise about the
future; ``ensure_ready`` therefore probes again before each real extraction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from modal_inference.config import MODEL_ID
from vlm_interface.vlm_types import PROVIDER_ENV_VAR, PROVIDER_MODAL
from stage_tracing import trace_stage

logger = logging.getLogger(__name__)

WARMUP_ENABLED_ENV_VAR = "SYNAPSE_MODAL_WARMUP_ON_STARTUP"
WARMUP_TIMEOUT_ENV_VAR = "SYNAPSE_MODAL_WARMUP_TIMEOUT_SECONDS"
WARMUP_ATTEMPT_TIMEOUT_ENV_VAR = "SYNAPSE_MODAL_WARMUP_ATTEMPT_SECONDS"
WARMUP_SHUTDOWN_TIMEOUT_ENV_VAR = "SYNAPSE_MODAL_WARMUP_SHUTDOWN_SECONDS"
DEFAULT_WARMUP_TIMEOUT_SECONDS = 600.0
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 60.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 15.0
_FALSE_VALUES = {"0", "false", "no", "off"}
# Statuses that a further attempt cannot change.
_PERMANENT_HTTP_STATUSES = {400, 401, 403, 405}

STATUS_DISABLED = "disabled"
STATUS_WARMING = "warming"
STATUS_READY = "ready"
STATUS_UNAVAILABLE = "unavailable"


class _PermanentProbeError(RuntimeError):
    """Retrying cannot help, for example rejected credentials."""


def modal_warmup_enabled() -> bool:
    """Warm by default for Modal; allow an explicit local-development opt-out."""
    provider = os.environ.get(PROVIDER_ENV_VAR, PROVIDER_MODAL).strip().lower()
    configured = os.environ.get(WARMUP_ENABLED_ENV_VAR, "").strip().lower()
    return provider == PROVIDER_MODAL and configured not in _FALSE_VALUES


def _positive_seconds(variable: str, default: float) -> float:
    raw = os.environ.get(variable, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return value if value > 0 else default


class ModalWarmup:
    """Tracks ``warming``, ``ready`` and ``unavailable`` for the remote model."""

    def __init__(self, lock_path: Optional[Path] = None) -> None:
        self._status = STATUS_WARMING if modal_warmup_enabled() else STATUS_DISABLED
        self._detail: Optional[str] = None
        self._model_verified = False
        self._probe_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock_path = lock_path or Path(tempfile.gettempdir()) / "synapse_modal_warmup.lock"

    @property
    def status(self) -> str:
        return self._status

    @property
    def detail(self) -> Optional[str]:
        return self._detail

    def _set_status(self, status: str, detail: Optional[str] = None) -> None:
        self._status, self._detail = status, detail
        logger.debug("Modal VLM status: %s%s", status, f" ({detail})" if detail else "")

    def start(self) -> None:
        """Begin warming in the background. Never blocks and never raises."""
        if self._status == STATUS_DISABLED:
            return
        with self._state_lock:
            if self._stop.is_set():
                return
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._warm, name="synapse-modal-warmup", daemon=True
            )
            self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Signal stop and join the background probe within a small bound."""
        self.request_stop()
        grace = timeout if timeout is not None else _positive_seconds(
            WARMUP_SHUTDOWN_TIMEOUT_ENV_VAR, DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        )
        thread = self._thread
        if thread is None or thread.ident is None:
            return True
        thread.join(timeout=max(0.0, grace))
        if thread.is_alive():
            logger.warning("Modal warm-up probe exceeded shutdown grace")
            return False
        with self._state_lock:
            if self._thread is thread:
                self._thread = None
        return True

    def ensure_ready(self, timeout: Optional[float] = None) -> bool:
        """Confirm the remote model is up now, waking it if it scaled to zero."""
        if self._status == STATUS_DISABLED:
            return True
        if self._stop.is_set():
            self._set_status(STATUS_UNAVAILABLE, "backend shutting down")
            return False
        with self._probe_lock:
            return self._probe_until(
                time.monotonic() + (timeout or self._total_timeout())
            )

    @staticmethod
    def _total_timeout() -> float:
        return _positive_seconds(WARMUP_TIMEOUT_ENV_VAR, DEFAULT_WARMUP_TIMEOUT_SECONDS)

    def _warm(self) -> None:
        deadline = time.monotonic() + self._total_timeout()
        with self._probe_lock:
            owns_lock = self._acquire_process_lock(deadline)
            try:
                self._probe_until(deadline)
            finally:
                if owns_lock:
                    self._release_process_lock()
        with self._state_lock:
            if self._thread is threading.current_thread():
                self._thread = None

    def _acquire_process_lock(self, deadline: float) -> bool:
        """Let one worker do the slow wake while others wait for its result."""
        while not self._stop.is_set():
            try:
                os.close(os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return True
            except FileExistsError:
                try:
                    age = time.time() - self._lock_path.stat().st_mtime
                    if age > self._total_timeout():
                        self._lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
            except OSError:
                return False  # Coordination is an optimization, not a requirement.
            if time.monotonic() >= deadline or self._stop.wait(2.0):
                return False
        return False

    def _release_process_lock(self) -> None:
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove the Modal warm-up lock", exc_info=True)

    def _probe_until(self, deadline: float) -> bool:
        endpoint = os.environ.get("SYNAPSE_MODAL_URL", "").strip().rstrip("/")
        api_key = os.environ.get("SYNAPSE_MODAL_API_KEY", "").strip()
        if not endpoint.startswith("https://") or not api_key:
            self._set_status(
                STATUS_UNAVAILABLE,
                "SYNAPSE_MODAL_URL (https) and SYNAPSE_MODAL_API_KEY are required",
            )
            return False

        attempt_timeout = _positive_seconds(
            WARMUP_ATTEMPT_TIMEOUT_ENV_VAR, DEFAULT_ATTEMPT_TIMEOUT_SECONDS
        )
        if self._status != STATUS_READY:
            self._set_status(STATUS_WARMING)
        backoff = 2.0
        while True:
            if self._stop.is_set():
                self._set_status(STATUS_UNAVAILABLE, "backend shutting down")
                return False
            attempt_started = time.monotonic()
            try:
                remaining = deadline - time.monotonic()
                trace_stage("modal_warmup", "probe_start")
                self._probe(endpoint, api_key, max(1.0, min(attempt_timeout, remaining)))
                if self._stop.is_set():
                    self._set_status(STATUS_UNAVAILABLE, "backend shutting down")
                    return False
                trace_stage(
                    "modal_warmup", "probe_ready", started_at=attempt_started,
                    model_verified=self._model_verified,
                )
                self._set_status(STATUS_READY)
                return True
            except _PermanentProbeError as exc:
                trace_stage(
                    "modal_warmup", "probe_failed", started_at=attempt_started,
                    retryable=False, error_type=type(exc).__name__,
                )
                self._set_status(STATUS_UNAVAILABLE, str(exc))
                return False
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                trace_stage(
                    "modal_warmup", "probe_failed", started_at=attempt_started,
                    retryable=True, error_type=type(exc).__name__,
                )
                # A client timeout is normal during a cold start: the first
                # request has already started the container.
                reason = f"last attempt failed ({type(exc).__name__})"
                if time.monotonic() + backoff >= deadline:
                    self._set_status(STATUS_UNAVAILABLE, reason)
                    return False
                self._status = STATUS_WARMING
                if self._stop.wait(backoff):
                    self._set_status(STATUS_UNAVAILABLE, "backend shutting down")
                    return False
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    def _probe(self, endpoint: str, api_key: str, timeout: float) -> None:
        self._request(f"{endpoint}/health", api_key, timeout)
        if self._model_verified:
            return
        payload = json.loads(self._request(f"{endpoint}/v1/models", api_key, timeout))
        served = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
        if MODEL_ID not in served:
            raise _PermanentProbeError(f"The endpoint does not serve {MODEL_ID}")
        self._model_verified = True

    @staticmethod
    def _request(url: str, api_key: str, timeout: float) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Synapse-Backend-Warmup/1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in _PERMANENT_HTTP_STATUSES:
                raise _PermanentProbeError(
                    f"Modal rejected the probe with HTTP {exc.code}"
                ) from exc
            raise
