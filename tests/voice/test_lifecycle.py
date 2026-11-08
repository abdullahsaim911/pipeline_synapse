import threading

from voice import lifecycle


def test_warmup_is_owned_and_joined(monkeypatch):
    called = threading.Event()

    def fake_warm(stop_event):
        called.set()
        return not stop_event.is_set()

    monkeypatch.setattr(lifecycle, "warm_up", fake_warm)
    monkeypatch.setattr(lifecycle, "voice_warmup_enabled", lambda: True)
    owner = lifecycle.VoiceWarmup()
    owner.start()
    assert called.wait(1)
    assert owner.stop(timeout=1)


def test_warmup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(lifecycle, "voice_warmup_enabled", lambda: False)
    owner = lifecycle.VoiceWarmup()
    owner.start()
    assert owner.status == "disabled"
    assert owner.stop()
