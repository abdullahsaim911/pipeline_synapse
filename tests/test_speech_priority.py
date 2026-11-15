"""Priority admission and cancellation without loading speech models."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tts_engine.streaming import current_stream
from tts_engine.tts_engine import (
    TTSEngine, _SpeechPriorityGate, _speech_priority, speech_priority,
)


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, 'Worker did not reach expected state'
        time.sleep(.005)


def queued(gate):
    with gate._condition:
        return len(gate._waiters)


def test_active_work_finishes_then_priority_and_fifo():
    gate = _SpeechPriorityGate()
    order, errors, workers = [], [], []
    gate.acquire()

    def work(name, priority):
        try:
            with speech_priority(priority), gate:
                order.append(name)
        except BaseException as exc:
            errors.append(exc)

    try:
        for index, (name, priority) in enumerate([
            ('background-first', 2), ('foreground-first', 1),
            ('voice-first', 0), ('background-second', 2),
            ('voice-second', 0), ('foreground-second', 1),
        ], 1):
            worker = threading.Thread(target=work, args=(name, priority), daemon=True)
            workers.append(worker)
            worker.start()
            wait_until(lambda: queued(gate) == index)
        assert order == []  # Active generation is never preempted.
        order.append('active-finished')
    finally:
        gate.release()
        for worker in workers:
            worker.join(3)
    assert not errors
    assert not any(worker.is_alive() for worker in workers)
    assert order == [
        'active-finished', 'voice-first', 'voice-second',
        'foreground-first', 'foreground-second', 'background-first', 'background-second',
    ]


def test_queued_background_promotes_ahead_of_earlier_background():
    gate = _SpeechPriorityGate()
    priority_keys = set()
    order, errors, workers = [], [], []
    gate.acquire()

    def work(key):
        try:
            with speech_priority(lambda: 1 if key in priority_keys else 2), gate:
                order.append(key)
        except BaseException as exc:
            errors.append(exc)

    try:
        for index, key in enumerate(['background-first', 'promoted', 'background-last'], 1):
            worker = threading.Thread(target=work, args=(key,), daemon=True)
            workers.append(worker)
            worker.start()
            wait_until(lambda: queued(gate) == index)
        # All three arrived as background work; promote only after they queue.
        priority_keys.add('promoted')
        assert not order
    finally:
        gate.release()
        for worker in workers:
            worker.join(3)
    assert not errors
    assert not any(worker.is_alive() for worker in workers)
    assert order == ['promoted', 'background-first', 'background-last']


def test_lock_api_timeout_and_exception_release():
    gate = _SpeechPriorityGate()
    assert gate.acquire(timeout=0)
    assert gate.locked()
    assert not gate.acquire(blocking=False)
    assert not gate.acquire(timeout=.01)
    assert queued(gate) == 0
    gate.release()
    with pytest.raises(RuntimeError, match='body failed'):
        with gate:
            raise RuntimeError('body failed')
    assert not gate.locked()
    with pytest.raises(RuntimeError, match='unlocked'):
        gate.release()
    with pytest.raises(ValueError):
        gate.acquire(False, timeout=1)


def test_priority_context_restores_and_is_thread_local():
    assert _speech_priority.get() == 1
    seen = []
    with speech_priority(2):
        with pytest.raises(RuntimeError), speech_priority(0):
            assert _speech_priority.get() == 0
            raise RuntimeError()
        assert _speech_priority.get() == 2
        worker = threading.Thread(target=lambda: seen.append(_speech_priority.get()))
        worker.start()
        worker.join(3)
    assert seen == [1]
    assert _speech_priority.get() == 1
    with pytest.raises(ValueError):
        with speech_priority(3):
            pass


@pytest.mark.parametrize('cancel_source', ['stream', 'shutdown'])
def test_generation_cancels_while_queued(monkeypatch, tmp_path, cancel_source):
    gate = _SpeechPriorityGate()
    monkeypatch.setattr(TTSEngine, '_inference_lock', gate)
    engine = TTSEngine()
    assert TTSEngine()._inference_lock is engine._inference_lock
    cancelled = threading.Event()
    errors = []

    def check():
        if cancelled.is_set():
            raise RuntimeError('lease lost')

    def generate():
        token = current_stream.set(SimpleNamespace(check=check))
        try:
            engine.generate('Queued speech.', tmp_path / 'speech.mp3')
        except Exception as exc:
            errors.append(exc)
        finally:
            current_stream.reset(token)

    gate.acquire()
    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    try:
        wait_until(lambda: queued(gate) == 1)
        if cancel_source == 'stream':
            cancelled.set()
        else:
            engine._stop.set()
        worker.join(2)
        assert not worker.is_alive()  # Cancellation completes before active work releases.
        assert gate.locked()
        assert queued(gate) == 0
        assert len(errors) == 1
        if cancel_source == 'stream':
            assert str(errors[0]) == 'lease lost'
        else:
            assert errors[0].code == 'tts_cancelled'
        assert not (tmp_path / 'speech.mp3').exists()
    finally:
        gate.release()
        worker.join(3)
    with gate:
        pass


@pytest.mark.parametrize('endpoint', ['/voice/speak', '/voice/speak/stream'])
def test_voice_binds_priority_in_generation_thread(monkeypatch, tmp_path, endpoint):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from voice import routes
    import tts_engine.streaming as streaming

    original_attempt = streaming.AudioAttempt
    monkeypatch.setattr(streaming, 'AudioAttempt', lambda: original_attempt(root=tmp_path))
    monkeypatch.setattr(routes, 'SPEECH_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(routes, '_speech_key', lambda text: 'a' * 64)
    seen = []

    class FakeSpeech:
        def generate(self, text, path):
            seen.append((_speech_priority.get(), threading.current_thread().name))
            Path(path).write_bytes(b'ID3-speech')

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_tts] = FakeSpeech
    with TestClient(app) as client, speech_priority(2):
        assert client.post(endpoint, json={'text': 'Paused.'}).status_code == 200
        assert _speech_priority.get() == 2
    assert len(seen) == 1
    assert seen[0][0] == 0
    if endpoint.endswith('/stream'):
        assert seen[0][1] == 'voice-speech-stream'
