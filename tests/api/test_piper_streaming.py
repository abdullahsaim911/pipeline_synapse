"""Merged voice replies keep Piper, streaming, caching, and shared CPU admission."""
import base64
import json
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tts_engine import streaming
from tts_engine.tts_engine import TTSEngine, _SpeechPriorityGate
from voice import routes
from voice.piper_speaker import PiperSpeaker


@pytest.fixture
def reply(tmp_path, monkeypatch):
    speaker = PiperSpeaker(voice_dir=tmp_path)
    calls = []
    pcm = (np.sin(np.arange(22050) * .1) * 10000).astype('<i2').tobytes()

    def synthesize(text, syn_config):
        calls.append(text)
        yield SimpleNamespace(sample_rate=22050, sample_width=2, sample_channels=1,
                              audio_int16_bytes=pcm)

    voice = SimpleNamespace(synthesize=synthesize)
    monkeypatch.setattr(speaker, '_load', lambda: (voice, None))
    monkeypatch.setattr(routes, 'SPEECH_CACHE_DIR', str(tmp_path / 'cache'))
    original = streaming.AudioAttempt
    monkeypatch.setattr(streaming, 'AudioAttempt', lambda: original(root=tmp_path))
    monkeypatch.setattr(TTSEngine, '_inference_lock', _SpeechPriorityGate())
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_tts] = lambda: speaker
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, speaker=speaker, voice=voice,
                              calls=calls, pcm=pcm, root=tmp_path)


def test_piper_stream_is_browser_compatible_and_reuses_native_wav(reply):
    response = reply.client.post('/voice/speak/stream', json={'text': 'Paused.'})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()
              if json.loads(line)['type'] != 'keepalive']
    assert events[0]['provider'] == 'Piper'
    assert events[0]['sample_rate'] == 24000
    assert events[-1]['type'] == 'complete'
    chunks = [base64.b64decode(event['pcm']) for event in events if event['type'] == 'audio']
    assert all(0 < len(chunk) <= 12000 and len(chunk) % 2 == 0 for chunk in chunks)
    assert sum(map(len, chunks)) == 24000 * 2
    audio = reply.client.post('/voice/speak', json={'text': 'Paused.'})
    assert audio.status_code == 200 and audio.headers['content-type'] == 'audio/wav'
    with wave.open(routes._speech_cache_path(reply.speaker, 'Paused.'), 'rb') as saved:
        assert saved.getframerate() == 22050
        assert saved.readframes(saved.getnframes()) == reply.pcm
    cached = reply.client.post('/voice/speak/stream', json={'text': 'Paused.'})
    assert cached.json() == {'type': 'complete', 'cached': True}
    assert reply.calls == ['Paused.']
    assert not list(reply.root.glob('audio-*.ndjson'))


@pytest.mark.parametrize('after_audio', [False, True])
def test_interrupted_piper_reply_never_publishes_a_partial_cache(reply, after_audio):
    original = reply.voice.synthesize

    def interrupted(text, syn_config):
        if after_audio:
            yield from original(text, syn_config)
        raise RuntimeError('private model failure')

    reply.voice.synthesize = interrupted
    response = reply.client.post('/voice/speak/stream', json={'text': 'Paused.'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'error'
    assert 'private model failure' not in response.text
    assert not list((reply.root / 'cache').iterdir())
    assert not list(reply.root.glob('audio-*.ndjson'))
    assert routes._speech_gate.acquire(blocking=False)
    routes._speech_gate.release()
    assert not TTSEngine._inference_lock.locked()


def test_piper_waits_for_active_explanation_cpu_slot(reply):
    gate = TTSEngine._inference_lock
    results = []
    gate.acquire()
    worker = threading.Thread(target=lambda: results.append(
        reply.client.post('/voice/speak/stream', json={'text': 'Paused.'})))
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with gate._condition:
                queued = bool(gate._waiters)
            if queued:
                break
            time.sleep(.005)
        assert queued
        assert not reply.calls
    finally:
        gate.release()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0].status_code == 200
    assert reply.calls == ['Paused.']


def test_stream_preserves_origin_validation_and_busy_response(reply):
    assert reply.client.post('/voice/speak/stream', json={'text': 'Paused.'},
                             headers={'Origin': 'https://evil.example'}).status_code == 403
    assert routes._speech_gate.acquire(blocking=False)
    try:
        assert reply.client.post('/voice/speak/stream', json={'text': 'Paused.'}).status_code == 429
    finally:
        routes._speech_gate.release()
