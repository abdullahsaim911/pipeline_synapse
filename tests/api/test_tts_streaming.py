"""Streaming shares durable preparation; no inference calls from the subscriber."""
import json
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base
from api.explanation_preparation import ExplanationPreparationManager
from api.explanation_service import ExplanationArtifact
from api.models import Video, Intervention, Explanation
from api.routers import video
from explanation_cache import generation_manifest, write_manifest
from tts_engine.streaming import current_stream


@pytest.fixture
def pipeline(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "db.sqlite"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    state = SimpleNamespace(calls=0, fail=False, preferred_fallback=[])
    with sessions() as db:
        db.add(Video(id='StreamTest1', title='Stream', source_url='https://youtube.com/watch?v=StreamTest1'))
        db.add(Intervention(id='stream-point', video_id='StreamTest1', timestamp=1, content_type='text', frame_path=''))
        db.commit()

    def generate(**kwargs):
        state.calls += 1
        state.preferred_fallback.append(current_stream.get().prefer_fallback)
        current_stream.get().audio('Speech text.', 'PocketTTS', b'\0\0' * 240)
        if state.fail:
            from tts_engine.exceptions import TTSEngineError
            raise TTSEngineError('private inference error', code='tts_stream_interrupted')
        path = tmp_path / 'audio.mp3'
        path.write_bytes(b'ID3-valid-audio')
        write_manifest(path, generation_manifest('Speech text.', path))
        with sessions() as db:
            row = Explanation(intervention_id='stream-point', output_mode='brief',
                              text_explanation='Speech text.', audio_file_path=str(path))
            db.add(row)
            db.commit()
            identity = row.id
        return ExplanationArtifact(intervention_id='stream-point', output_mode='brief',
                                   text_explanation='Speech text.', audio_file_path=str(path),
                                   text_file_path=None, vlm_snapshot_path=None, explanation_id=identity)

    manager = ExplanationPreparationManager(session_factory=sessions, orchestrator_factory=lambda: object(),
                                             data_dir=str(tmp_path), generator=generate)
    manager.start()
    app = FastAPI()
    app.include_router(video.router)
    app.dependency_overrides[video.get_preparation_manager] = lambda: manager
    state.manager = manager
    with TestClient(app) as client:
        state.client = client
        yield state
    manager.shutdown()
    engine.dispose()


BODY = {'video_id': 'StreamTest1', 'intervention_id': 'stream-point', 'output_mode': 'brief'}


def wait_for_failed(manager):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if manager.get_status('StreamTest1', 'brief')[0].status == 'failed':
            return
        time.sleep(.01)
    raise AssertionError('Preparation did not persist its failed state')


def test_stream_completes_only_after_artifact_and_ready_commit(pipeline):
    response = pipeline.client.post('/video/intervention/explain/stream', json=BODY)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'complete'
    assert pipeline.manager.get_status('StreamTest1', 'brief')[0].status == 'ready'
    # The generation can finish before subscription; that case returns cached metadata.
    if events[0]['type'] == 'start':
        assert events[0]['text'] == 'Speech text.'
        assert any(event['type'] == 'audio' for event in events)
    response = pipeline.client.post('/video/intervention/explain/stream', json=BODY)
    assert json.loads(response.text)['cached'] is True
    assert pipeline.calls == 1


def test_generation_failure_never_emits_success_or_private_error(pipeline):
    pipeline.fail = True
    response = pipeline.client.post('/video/intervention/explain/stream', json=BODY)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'error'
    assert not any(event['type'] == 'complete' for event in events)
    assert 'private inference' not in response.text
    wait_for_failed(pipeline.manager)


def test_origin_and_validation_apply_before_admission(pipeline):
    assert pipeline.client.post('/video/intervention/explain/stream', json=BODY,
                                headers={'Origin': 'https://untrusted.example'}).status_code == 403
    assert pipeline.client.post('/video/intervention/explain/stream', json={**BODY, 'output_mode': 'other'}).status_code == 422
    assert pipeline.calls == 0


def test_retry_keeps_durable_failure_reason_and_uses_fallback(pipeline):
    from api.models import ExplanationPreparation
    pipeline.fail = True
    pipeline.client.post('/video/intervention/explain/stream', json=BODY)
    manager = pipeline.manager
    wait_for_failed(manager)
    with manager._session_factory() as db:
        row = db.query(ExplanationPreparation).one()
        assert row.failure_reason == 'tts_stream_interrupted'
    # Simulate eviction/restart of the ephemeral stream journal. The existing
    # durable failure reason must still choose Kokoro on the explicit retry.
    manager.audio_streams.attempts.pop(('StreamTest1', 'stream-point', 'brief')).dispose()
    pipeline.fail = False
    response = pipeline.client.post('/video/StreamTest1/interventions/stream-point/explanations/retry?output_mode=brief')
    assert response.status_code in (200, 202)
    response = pipeline.client.post('/video/intervention/explain/stream', json=BODY)
    assert json.loads(response.text.splitlines()[-1])['type'] == 'complete'
    assert pipeline.preferred_fallback == [False, True]


def test_short_reply_stream_and_cached_file_do_not_repeat_inference(tmp_path, monkeypatch):
    from voice import routes
    from tts_engine.streaming import AudioAttempt
    import tts_engine.streaming as streaming
    monkeypatch.setattr(streaming, 'AudioAttempt', lambda: AudioAttempt(root=tmp_path))
    monkeypatch.setattr(routes, 'SPEECH_CACHE_DIR', str(tmp_path / 'voice-cache'))
    calls = []

    class FakeSpeech:
        def generate(self, text, path):
            calls.append(text)
            current_stream.get().audio(text, 'PocketTTS', b'\0\0' * 120)
            from pathlib import Path
            Path(path).write_bytes(b'ID3-short-reply')

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_tts] = FakeSpeech
    with TestClient(app) as client:
        response = client.post('/voice/speak/stream', json={'text': 'Paused.'})
        assert [json.loads(line)['type'] for line in response.text.splitlines()
                if json.loads(line)['type'] != 'keepalive'] == ['start', 'audio', 'complete']
        response = client.post('/voice/speak/stream', json={'text': 'Paused.'})
        assert json.loads(response.text) == {'type': 'complete', 'cached': True}
        assert client.post('/voice/speak', json={'text': 'Paused.'}).content == b'ID3-short-reply'
        assert calls == ['Paused.']
    assert not list(tmp_path.glob('audio-*.ndjson'))
