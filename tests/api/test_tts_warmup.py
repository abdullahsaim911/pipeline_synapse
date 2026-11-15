"""Startup speech loading without downloading or executing models."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api.resource_guard import LocalResourcesUnavailableError
from tts_engine import lifecycle
from tts_engine.tts_engine import TTSEngine, _SpeechPriorityGate


@pytest.fixture
def warmup(monkeypatch):
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    engine = SimpleNamespace(primary_loaded=False, warm=Mock())
    monkeypatch.setattr(lifecycle, 'get_shared_tts_engine', lambda: engine)
    monkeypatch.setattr(lifecycle, '_require_warmup_headroom', lambda: None)
    owner = lifecycle.TTSWarmup()
    yield owner, engine
    assert owner.stop(2)


def test_startup_loads_shared_primary_and_reports_ready(warmup):
    owner, engine = warmup
    owner.start()
    owner._thread.join(2)
    assert owner.status == 'ready'
    engine.warm.assert_called_once()
    owner.start()
    engine.warm.assert_called_once()


def test_warmup_defers_on_low_memory_and_shutdown_interrupts_wait(warmup, monkeypatch):
    owner, engine = warmup
    checked = threading.Event()

    def low_memory():
        checked.set()
        raise LocalResourcesUnavailableError('busy')

    monkeypatch.setattr(lifecycle, '_require_warmup_headroom', low_memory)
    owner.start()
    assert checked.wait(2)
    assert owner.stop(2)
    assert owner.status == 'stopped'
    engine.warm.assert_not_called()


def test_deferred_warmup_retries_when_memory_recovers(warmup, monkeypatch):
    owner, engine = warmup
    checks = Mock(side_effect=[LocalResourcesUnavailableError('busy'), None])
    monkeypatch.setattr(lifecycle, '_require_warmup_headroom', checks)
    owner.start()
    owner._thread.join(3)
    assert owner.status == 'ready'
    assert checks.call_count == 2
    engine.warm.assert_called_once()


def test_headroom_guard_respects_floor_and_existing_admission_setting(monkeypatch):
    guard = Mock()
    monkeypatch.setattr(lifecycle, 'require_processing_headroom', guard)
    monkeypatch.setattr(lifecycle.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1024 ** 3))
    with pytest.raises(LocalResourcesUnavailableError):
        lifecycle._require_warmup_headroom()
    guard.assert_called_once()


def test_existing_loaded_primary_can_be_recognized_without_more_memory(warmup, monkeypatch):
    owner, engine = warmup
    engine.primary_loaded = True
    guard = Mock(side_effect=AssertionError('No new allocation required'))
    monkeypatch.setattr(lifecycle, '_require_warmup_headroom', guard)
    owner.start()
    owner._thread.join(2)
    assert owner.status == 'ready'
    guard.assert_not_called()


def test_local_workers_finish_before_pocket_loading(warmup):
    owner, engine = warmup
    checked = threading.Event()
    owner._before_warm = lambda: checked.set() or False
    owner.start()
    assert checked.wait(2)
    assert owner.stop(2)
    engine.warm.assert_not_called()


def test_failed_model_load_reports_unavailable_without_stopping_api(warmup):
    owner, engine = warmup
    engine.warm.side_effect = RuntimeError('model unavailable')
    owner.start()
    owner._thread.join(2)
    assert owner.status == 'unavailable'


def test_warmup_thread_failure_does_not_abort_backend_startup(warmup, monkeypatch):
    owner, engine = warmup
    thread = Mock()
    thread.start.side_effect = RuntimeError('No thread available')
    monkeypatch.setattr(lifecycle.threading, 'Thread', Mock(return_value=thread))
    owner.start()
    assert owner.status == 'unavailable'
    assert owner._thread is None
    engine.warm.assert_not_called()


def test_engine_warms_primary_only_under_shared_gate(monkeypatch):
    engine = TTSEngine()
    gate = _SpeechPriorityGate()
    monkeypatch.setattr(TTSEngine, '_inference_lock', gate)
    primary = SimpleNamespace(loaded=False)
    primary.warm = Mock(side_effect=lambda: setattr(primary, 'loaded', True))
    fallback = SimpleNamespace(warm=Mock())
    engine._providers = [primary, fallback]
    guards = []
    engine.warm(admission_check=lambda: guards.append(gate.locked()))
    assert engine.primary_loaded
    primary.warm.assert_called_once()
    fallback.warm.assert_not_called()
    assert True in guards  # Headroom is rechecked after acquiring the gate.
    assert not gate.locked()


def test_queued_warmup_can_be_cancelled_before_active_speech_finishes(monkeypatch):
    engine = TTSEngine()
    gate = _SpeechPriorityGate()
    monkeypatch.setattr(TTSEngine, '_inference_lock', gate)
    primary = SimpleNamespace(loaded=False, warm=Mock())
    engine._providers = [primary]
    stop = threading.Event()
    errors = []
    waiting = threading.Event()

    def run():
        try:
            engine.warm(stop, admission_check=waiting.set)
        except Exception as exc:
            errors.append(exc)

    with gate:
        worker = threading.Thread(target=run)
        worker.start()
        assert waiting.wait(2)
        stop.set()
        worker.join(2)
        assert not worker.is_alive()
        assert errors[0].code == 'tts_cancelled'
        primary.warm.assert_not_called()
