"""In-memory, per-video guard so one lecture is never generated twice at the same time."""

import threading
from typing import Set


class VideoLocks:
    """Non-blocking per-key lock. Process-local: it does not coordinate several server processes."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._held: Set[str] = set()

    def try_acquire(self, key: str) -> bool:
        with self._guard:
            if key in self._held:
                return False
            self._held.add(key)
            return True

    def release(self, key: str) -> None:
        with self._guard:
            self._held.discard(key)
