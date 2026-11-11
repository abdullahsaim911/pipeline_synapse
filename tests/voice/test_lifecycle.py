import threading

from voice import lifecycle


def test_warmup_is_owned_joined_and_logged(monkeypatch, caplog):
    called = threading.Event()

    def fake_warm(stop_event):
        called.set()
        return not stop_event.is_set()

    monkeypatch.setattr(lifecycle, "warm_up", fake_warm)
    monkeypatch.setattr(lifecycle, "start_voice_runtime", lambda: None)
    monkeypatch.setattr(lifecycle, "voice_warmup_enabled", lambda: True)
    owner = lifecycle.VoiceWarmup()
    with caplog.at_level("INFO", logger="voice.lifecycle"):
        owner.start()
    assert called.wait(1)
    assert "Voice model warm-up started" in caplog.text
    assert owner.stop(timeout=1)


def test_warmup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(lifecycle, "voice_warmup_enabled", lambda: False)
    owner = lifecycle.VoiceWarmup()
    owner.start()
    assert owner.status == "disabled"
    assert owner.stop()


def test_worker_start_failure_is_reported_without_starting_thread(monkeypatch):
    monkeypatch.setattr(lifecycle, "voice_warmup_enabled", lambda: True)
    monkeypatch.setattr(
        lifecycle,
        "start_voice_runtime",
        lambda: (_ for _ in ()).throw(OSError("worker unavailable")),
    )
    owner = lifecycle.VoiceWarmup()

    owner.start()

    assert owner.status == "unavailable"
    assert owner.stop()
