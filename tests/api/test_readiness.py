import asyncio
from types import SimpleNamespace

from fastapi import Response

from api.main import app, readiness_check


def set_components(voice, transcription, modal="warming"):
    app.state.voice_warmup = SimpleNamespace(status=voice)
    app.state.transcription_warmup = SimpleNamespace(status=transcription)
    app.state.model_warmup = SimpleNamespace(status=modal)


def test_demo_readiness_requires_voice_and_video_transcription():
    set_components("warming", "ready")
    response = Response()

    payload = asyncio.run(readiness_check(response))

    assert response.status_code == 503
    assert payload["status"] == "warming"
    assert payload["components"]["modal"] == "warming"


def test_modal_warmup_does_not_block_local_demo_readiness():
    set_components("ready", "ready", modal="warming")
    response = Response()

    payload = asyncio.run(readiness_check(response))

    assert response.status_code == 200
    assert payload["status"] == "ready"
