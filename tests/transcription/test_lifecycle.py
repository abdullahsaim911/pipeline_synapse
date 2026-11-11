import logging

from transcription import lifecycle


def test_transcription_warmup_logs_start_and_ready(monkeypatch, caplog):
    class FakeRuntime:
        def warm(self):
            return None

    monkeypatch.setattr(lifecycle, "get_transcription_runtime", FakeRuntime)
    monkeypatch.setenv("SYNAPSE_TRANSCRIPTION_WARMUP_ENABLED", "1")
    owner = lifecycle.TranscriptionWarmup()

    with caplog.at_level(logging.INFO, logger="transcription.lifecycle"):
        owner.start()
        owner._thread.join(timeout=1)

    assert owner.status == "ready"
    assert "Video transcription model warm-up started" in caplog.text
    assert "Video transcription model is ready" in caplog.text
