"""Shared Ollama lifecycle: reuse, ownership and diagnostics."""

from types import SimpleNamespace

import pytest
import requests

from orchestrator import ollama_service
from orchestrator.ollama_service import OllamaService, OllamaUnavailableError
from synthesizer import TextSynthesisUnavailableError


class FakeServer:
    """Answers /api/tags according to whether a server is up."""

    def __init__(self, up=True, models=("mistral:latest",)):
        self.up = up
        self.models = list(models)

    def get(self, url, timeout):
        if not self.up:
            raise requests.exceptions.ConnectionError("refused")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"models": [{"name": name} for name in self.models]},
        )


def service(monkeypatch, server, **kwargs):
    monkeypatch.setattr(ollama_service.requests, "get", server.get)
    calls = SimpleNamespace(started=[], stopped=[])

    def start():
        server.up = True
        process = SimpleNamespace(pid=7, poll=lambda: None)
        calls.started.append(process)
        return process

    defaults = {"start": start, "stop": calls.stopped.append, "poll_interval": 0.001}
    defaults.update(kwargs)
    return OllamaService(**defaults), calls


def test_running_instance_is_reused_and_never_stopped(monkeypatch):
    svc, calls = service(monkeypatch, FakeServer(up=True))
    svc.ensure_ready()
    svc.ensure_ready()
    svc.shutdown()
    assert calls.started == []
    assert calls.stopped == []


def test_owned_server_starts_once_across_interventions_and_stops_at_shutdown(monkeypatch):
    svc, calls = service(monkeypatch, FakeServer(up=False))
    for _ in range(3):
        svc.ensure_ready()
    assert len(calls.started) == 1
    assert svc.owns_process
    svc.shutdown()
    assert calls.stopped == calls.started
    assert not svc.owns_process


def test_server_that_died_between_interventions_is_restarted(monkeypatch):
    server = FakeServer(up=False)
    svc, calls = service(monkeypatch, server)
    svc.ensure_ready()
    server.up = False
    calls.started[0].poll = lambda: 1
    svc.ensure_ready()
    assert len(calls.started) == 2


def test_missing_model_gives_install_diagnostic_and_public_error_code(monkeypatch):
    svc, _ = service(monkeypatch, FakeServer(up=True, models=("llama3:latest",)))
    with pytest.raises(OllamaUnavailableError, match="ollama pull mistral") as caught:
        svc.ensure_ready()
    assert isinstance(caught.value, TextSynthesisUnavailableError)
    assert caught.value.error_code == "text_synthesis_unavailable"


def test_unreachable_server_times_out_with_a_clear_message(monkeypatch):
    server = FakeServer(up=False)
    svc, _ = service(
        monkeypatch,
        server,
        start=lambda: SimpleNamespace(pid=8, poll=lambda: None),
        startup_timeout=0.01,
    )
    with pytest.raises(OllamaUnavailableError, match="did not respond"):
        svc.ensure_ready()


def test_process_exit_during_startup_is_reported_at_once(monkeypatch):
    svc, _ = service(
        monkeypatch,
        FakeServer(up=False),
        start=lambda: SimpleNamespace(pid=9, poll=lambda: 1),
    )
    with pytest.raises(OllamaUnavailableError, match="exited during startup"):
        svc.ensure_ready()


def test_losing_a_port_race_does_not_claim_ownership(monkeypatch):
    server = FakeServer(up=False)

    def start():
        server.up = True
        return SimpleNamespace(pid=10, poll=lambda: 1)

    svc, calls = service(monkeypatch, server, start=start)
    svc.ensure_ready()
    assert not svc.owns_process
    svc.shutdown()
    assert calls.stopped == []


def test_stop_ollama_targets_only_its_own_process_tree(monkeypatch):
    from orchestrator import orchestrator

    commands = []
    monkeypatch.setattr(orchestrator.os, "name", "nt")
    monkeypatch.setattr(
        orchestrator.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    orchestrator.stop_ollama(SimpleNamespace(pid=4321, wait=lambda timeout: 0))
    assert commands == [["taskkill", "/F", "/T", "/PID", "4321"]]


@pytest.mark.parametrize("provider,configured,expected", [
    ("modal", None, "5m"),
    ("local", None, 0),
    ("modal", "0", 0),
    ("modal", "30m", "30m"),
])
def test_keep_alive_policy(monkeypatch, provider, configured, expected):
    from llm_interface.ollama_client import resolve_keep_alive

    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", provider)
    if configured is None:
        monkeypatch.delenv("SYNAPSE_OLLAMA_KEEP_ALIVE", raising=False)
    else:
        monkeypatch.setenv("SYNAPSE_OLLAMA_KEEP_ALIVE", configured)
    assert resolve_keep_alive() == expected
