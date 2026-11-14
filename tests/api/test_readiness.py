import asyncio
from types import SimpleNamespace

from fastapi import Response

from api.main import app, readiness_check


def set_components(voice, transcription, modal="warming", mistral="disabled"):
    app.state.voice_warmup = SimpleNamespace(status=voice)
    app.state.transcription_warmup = SimpleNamespace(status=transcription)
    app.state.model_warmup = SimpleNamespace(status=modal)
    app.state.mistral_warmup = SimpleNamespace(status=mistral)


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


def test_mistral_readiness_is_reported_separately_from_vlm():
    for mistral in ["warming", "ready", "unavailable", "disabled"]:
        set_components("ready", "ready", modal="ready", mistral=mistral)
        response = Response()
        payload = asyncio.run(readiness_check(response))
        assert payload["components"]["mistral"] == mistral
        assert payload["components"]["modal"] == "ready"
        assert response.status_code == 200  # Existing local readiness contract.
