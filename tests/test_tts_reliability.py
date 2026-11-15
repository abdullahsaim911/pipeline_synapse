"""CPU speech migration: real encoding, fake inference, cancellation and fallback."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tts_engine import TTSEngine, TTSEngineError
from tts_engine.streaming import AudioAttempt, AudioStreams


class Provider:
    def __init__(self, effects, name='fake'):
        self.effects, self.name, self.calls = effects, name, 0

    def chunks(self, text, stop):
        self.calls += 1
        for effect in self.effects:
            if isinstance(effect, Exception):
                raise effect
            yield effect


def engine_with(*providers):
    engine = TTSEngine()
    engine._providers = list(providers)
    return engine


def test_default_chain_is_local_and_lazy():
    engine = TTSEngine()
    assert [p.name for p in engine._providers] == ['PocketTTS', 'Kokoro']
    assert engine._providers[0].model is None
    assert engine._providers[1].pipeline is None


def test_before_first_audio_falls_back_and_writes_valid_mp3(tmp_path):
    primary = Provider([RuntimeError('private detail')], 'primary')
    fallback = Provider([np.ones(2400) * .1], 'fallback')
    engine = engine_with(primary, fallback)
    output = tmp_path / 'speech.tmp'
    attempt = AudioAttempt(root=tmp_path)
    with attempt.bind():
        engine.generate('Hello.', output)
    assert output.read_bytes().startswith((b'ID3', b'\xff'))
    assert engine.last_provider_used == 'fallback'
    assert primary.calls == fallback.calls == 1
    events = [json.loads(line) for line in attempt.path.read_text().splitlines()]
    assert events[0]['provider'] == 'fallback'
    assert sum(len(__import__('base64').b64decode(e['pcm'])) // 2 for e in events[1:]) == 2400


def test_midstream_failure_never_replays_fallback_or_overwrites_output(tmp_path):
    primary = Provider([np.zeros(2400), RuntimeError('secret')])
    fallback = Provider([np.zeros(2400)])
    engine = engine_with(primary, fallback)
    output = tmp_path / 'audio.mp3'
    output.write_bytes(b'original')
    attempt = AudioAttempt(root=tmp_path)
    with attempt.bind(), pytest.raises(TTSEngineError) as error:
        engine.generate('Hello.', output)
    assert error.value.code == 'tts_stream_interrupted'
    assert 'secret' not in str(error.value)
    assert output.read_bytes() == b'original'
    assert not fallback.calls
    assert attempt.prefer_fallback
    assert list(tmp_path.glob('*.mp3')) == [output]


def test_explicit_retry_uses_kokoro_after_interruption(tmp_path):
    engine = engine_with(Provider([RuntimeError()]), Provider([np.zeros(2400)], 'Kokoro'))
    attempt = AudioAttempt(root=tmp_path, prefer_fallback=True)
    with attempt.bind():
        engine.generate('Retry.', tmp_path / 'retry.mp3')
    assert engine._providers[0].calls == 0
    assert engine.last_provider_used == 'Kokoro'


def test_file_only_generation_can_fallback_after_partial_private_output(tmp_path):
    primary = Provider([np.zeros(2400), RuntimeError('failed')])
    fallback = Provider([np.zeros(2400)], 'Kokoro')
    engine = engine_with(primary, fallback)
    engine.generate('File only.', tmp_path / 'audio.mp3')
    assert fallback.calls == 1
    assert engine.last_provider_used == 'Kokoro'


def test_ownership_loss_cannot_trigger_fallback(tmp_path):
    def check():
        raise RuntimeError('lease lost')
    engine = engine_with(Provider([np.zeros(2400)]), Provider([np.zeros(2400)]))
    with AudioAttempt(check, root=tmp_path).bind(), pytest.raises(RuntimeError, match='lease lost'):
        engine.generate('Cancelled.', tmp_path / 'audio.mp3')
    assert not (tmp_path / 'audio.mp3').exists()
    assert not engine._providers[1].calls


def test_nonfinite_and_empty_audio_fail_without_artifact(tmp_path):
    for samples in (np.array([np.nan]), np.array([])):
        with pytest.raises(TTSEngineError):
            engine_with(Provider([samples])).generate('Test.', tmp_path / 'speech.mp3')
        assert not (tmp_path / 'speech.mp3').exists()


def test_spool_replays_from_beginning_to_late_subscribers(tmp_path):
    attempt = AudioAttempt(root=tmp_path)
    attempt.audio('Hello', 'PocketTTS', b'\0\0' * 120)
    attempt.audio('Hello', 'PocketTTS', b'\0\0' * 120)
    attempt.complete(explanation_id=1)
    for _ in range(2):
        events = [json.loads(line) for line in attempt.lines()]
        assert [e['type'] for e in events] == ['start', 'audio', 'audio', 'complete']
        assert [e['sample_offset'] for e in events if e['type'] == 'audio'] == [0, 120]


def test_retry_retires_old_attempt_and_carries_fallback(monkeypatch, tmp_path):
    import tts_engine.streaming as module
    original = module.AudioAttempt
    monkeypatch.setattr(module, 'AudioAttempt', lambda check, fallback: original(check, fallback, root=tmp_path))
    streams = AudioStreams()
    first = streams.begin(('video', 'point', 'brief'), lambda: None)
    first.prefer_fallback = True
    first.fail('tts_stream_interrupted')
    second = streams.begin(('video', 'point', 'brief'), lambda: None)
    assert second.id != first.id
    assert second.prefer_fallback
    assert not first.path.exists()
    streams.close()


def test_cancellation_publishes_error_not_completion(tmp_path):
    cancelled = threading.Event()
    def check():
        if cancelled.is_set():
            raise RuntimeError('cancelled')
    attempt = AudioAttempt(check, root=tmp_path)
    attempt.audio('Hello', 'PocketTTS', b'\0\0')
    cancelled.set()
    events = [json.loads(line) for line in attempt.lines()]
    assert events[-1]['type'] == 'error'
    assert events[-1]['code'] == 'cancelled'


def test_shared_engine_reused_and_cleaned(monkeypatch):
    from tts_engine import tts_engine as module
    monkeypatch.setattr(module, '_shared_engine', None)
    calls = []
    fake = SimpleNamespace(cleanup=lambda: calls.append('cleaned'))
    monkeypatch.setattr(module, 'TTSEngine', lambda: fake)
    assert module.get_shared_tts_engine() is module.get_shared_tts_engine()
    module.shutdown_shared_tts_engine()
    assert calls == ['cleaned']
