"""Backend-owned startup loading for the shared primary CPU speech model."""

import logging
import os
import threading

import psutil

from api.resource_guard import LocalResourcesUnavailableError, require_processing_headroom
from .tts_engine import get_shared_tts_engine

logger = logging.getLogger(__name__)
MIN_AVAILABLE_RAM_BYTES = 2 * 1024 ** 3


def _require_warmup_headroom():
    require_processing_headroom()
    if psutil.virtual_memory().available < MIN_AVAILABLE_RAM_BYTES:
        raise LocalResourcesUnavailableError('Waiting for RAM before speech warm-up')


class TTSWarmup:
    """Warm Pocket after local workers load, deferring when RAM is tight."""

    def __init__(self, before_warm=lambda: True):
        self._before_warm = before_warm
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.status = 'disabled' if 'PYTEST_CURRENT_TEST' in os.environ else 'idle'

    def start(self):
        with self._lock:
            if self.status == 'disabled' or self._thread is not None or self._stop.is_set():
                return
            self.status = 'warming'
            self._thread = threading.Thread(target=self._run, name='synapse-tts-warmup', daemon=True)
            try:
                self._thread.start()
            except (RuntimeError, OSError) as exc:
                self._thread = None
                self.status = 'unavailable'
                logger.warning('Pocket TTS warm-up could not start: error_type=%s', type(exc).__name__)

    def _run(self):
        logger.info('Pocket TTS warm-up started')
        while not self._stop.is_set():
            try:
                if not self._before_warm():
                    self.status = 'waiting_for_local_models'
                else:
                    engine = get_shared_tts_engine()
                    if not engine.primary_loaded:
                        _require_warmup_headroom()
                    self.status = 'warming'
                    engine.warm(
                        stop_event=self._stop, admission_check=_require_warmup_headroom,
                    )
                    if self._stop.is_set():
                        break
                    self.status = 'ready'
                    logger.info('Pocket TTS is ready')
                    return
            except LocalResourcesUnavailableError:
                if self.status != 'waiting_for_memory':
                    logger.info('Pocket TTS warm-up waiting for at least 2 GiB available RAM')
                self.status = 'waiting_for_memory'
            except Exception as exc:
                self.status = 'stopped' if self._stop.is_set() else 'unavailable'
                logger.warning('Pocket TTS warm-up stopped: error_type=%s', type(exc).__name__)
                return
            if self._stop.wait(1):
                break
        self.status = 'stopped'

    def request_stop(self):
        self._stop.set()

    def stop(self, timeout=1.0):
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))
            return not self._thread.is_alive()
        return True
