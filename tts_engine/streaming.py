"""Attempt-scoped, disk-backed audio delivery. Durable job state stays in the API."""
import base64
from contextvars import ContextVar
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import uuid

current_stream = ContextVar('tts_stream', default=None)
STREAM_ROOT = Path(__file__).resolve().parents[1] / '.tts-lab' / 'streams'


def event_line(kind, **values):
    return (json.dumps({'type': kind, **values}, separators=(',', ':')) + '\n').encode()


class AudioAttempt:
    def __init__(self, check=lambda: None, prefer_fallback=False, root=STREAM_ROOT):
        root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='audio-', suffix='.ndjson', dir=root)
        import os
        os.close(fd)
        self.path = Path(name)
        self.id = uuid.uuid4().hex
        self.check = check
        self.prefer_fallback = prefer_fallback
        self.condition = threading.Condition()
        self.size = self.sequence = self.samples = 0
        self.started = self.done = False
        self.finished_at = None
        self.created_at = time.monotonic()
        self.readers = 0
        self.retired = False

    @contextmanager
    def bind(self):
        token = current_stream.set(self)
        try:
            yield self
        finally:
            current_stream.reset(token)

    def append(self, line, terminal=False):
        with self.condition:
            if self.done:
                return
            # 64 MiB per attempt bounds disk usage; detach delivery, not inference.
            if self.size + len(line) > 64 * 1024 * 1024:
                line = event_line('error', code='stream_limit', retryable=True)
                terminal = True
            with self.path.open('ab') as handle:
                handle.write(line)
            self.size += len(line)
            self.done = terminal
            if terminal:
                self.finished_at = time.monotonic()
            self.condition.notify_all()

    def audio(self, text, provider, pcm):
        self.check()
        if not self.started:
            self.append(event_line('start', generation_id=self.id, text=text,
                                   sample_rate=24000, channels=1, format='s16le', provider=provider))
            self.started = True
            from stage_tracing import trace_stage
            trace_stage('tts_first_audio', 'published', started_at=self.created_at,
                        provider=provider, generation_id=self.id)
        self.append(event_line('audio', sequence=self.sequence, sample_offset=self.samples,
                               pcm=base64.b64encode(pcm).decode('ascii')))
        self.sequence += 1
        self.samples += len(pcm) // 2

    def text_ready(self, text, **metadata):
        """Publish validated text while this attempt waits for CPU speech."""
        self.check()
        self.append(event_line('text_ready', generation_id=self.id, text=text, **metadata))

    def complete(self, **metadata):
        self.append(event_line('complete', **metadata), terminal=True)
        self.prefer_fallback = False

    def fail(self, code='tts_failed'):
        self.append(event_line('error', code=code, retryable=code not in ('cancelled', 'superseded')),
                    terminal=True)

    def lines(self):
        with self.condition:
            self.readers += 1
        try:
            with self.path.open('rb') as handle:
                while True:
                    if not self.done and getattr(self, 'artifact', None) is None:
                        try:
                            self.check()
                        except Exception:
                            self.fail('cancelled')
                    # Writers publish size only after a whole flushed record;
                    # readers must not observe a partial concurrent file write.
                    with self.condition:
                        line = handle.readline()
                        terminal = self.done and handle.tell() >= self.size
                        if not line and not terminal:
                            self.condition.wait(timeout=1)
                    if line:
                        yield line
                        continue
                    if terminal:
                        return
                    yield event_line('keepalive')
        finally:
            with self.condition:
                self.readers -= 1
                if self.retired and not self.readers:
                    self.path.unlink(missing_ok=True)

    def dispose(self):
        with self.condition:
            if self.readers:
                self.retired = True
                return False
            self.path.unlink(missing_ok=True)
            return True


class AudioStreams:
    """Owned by the existing preparation manager; never starts inference."""
    def __init__(self):
        self.condition = threading.Condition()
        self.attempts = {}
        self.fallback_keys = set()
        self.closed = False

    def begin(self, key, check, *, prefer_fallback=False):
        with self.condition:
            # Keep only recent terminal attempts. Never evict an active subscriber.
            for old_key, old in list(self.attempts.items()):
                if old.done and (time.monotonic() - old.finished_at > 60 or len(self.attempts) >= 8):
                    if old_key != key and old.dispose():
                        del self.attempts[old_key]
            previous = self.attempts.get(key)
            attempt = AudioAttempt(check, prefer_fallback or key in self.fallback_keys or bool(previous and previous.prefer_fallback))
            self.fallback_keys.discard(key)
            if previous:
                previous.fail('superseded')
                # Windows readers retain the file until they leave the old attempt.
                previous.dispose()
            self.attempts[key] = attempt
            self.condition.notify_all()
            return attempt

    def close(self):
        with self.condition:
            self.closed = True
            for attempt in self.attempts.values():
                attempt.fail('cancelled')
                attempt.dispose()
            self.condition.notify_all()
