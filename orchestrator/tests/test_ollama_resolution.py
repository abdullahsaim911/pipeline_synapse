import os
from types import SimpleNamespace

import pytest

from orchestrator import orchestrator


def test_explicit_ollama_path_is_authoritative(monkeypatch, tmp_path):
    executable = tmp_path / "ollama.exe"
    executable.write_bytes(b"")
    monkeypatch.setenv(orchestrator.OLLAMA_PATH_ENV_VAR, str(executable))
    monkeypatch.setattr(orchestrator.shutil, "which", lambda _name: "ignored")

    assert orchestrator.resolve_ollama_executable() == str(executable)


def test_invalid_explicit_ollama_path_fails_clearly(monkeypatch, tmp_path):
    missing = tmp_path / "missing-ollama.exe"
    monkeypatch.setenv(orchestrator.OLLAMA_PATH_ENV_VAR, str(missing))

    with pytest.raises(RuntimeError, match="SYNAPSE_OLLAMA_PATH"):
        orchestrator.resolve_ollama_executable()


def test_path_discovery_is_used_before_platform_fallback(monkeypatch):
    monkeypatch.delenv(orchestrator.OLLAMA_PATH_ENV_VAR, raising=False)
    monkeypatch.setattr(orchestrator.shutil, "which", lambda name: f"found-{name}")

    assert orchestrator.resolve_ollama_executable() == "found-ollama"


@pytest.mark.skipif(os.name != "nt", reason="Windows installation fallback")
def test_standard_windows_user_install_is_discovered(monkeypatch, tmp_path):
    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.delenv(orchestrator.OLLAMA_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(orchestrator.shutil, "which", lambda _name: None)

    assert orchestrator.resolve_ollama_executable() == str(executable)


def test_start_ollama_uses_resolved_executable(monkeypatch):
    calls = []
    fake_process = SimpleNamespace(pid=1234)
    monkeypatch.setattr(
        orchestrator, "resolve_ollama_executable", lambda: "resolved-ollama"
    )
    monkeypatch.setattr(
        orchestrator.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or fake_process,
    )

    assert orchestrator.start_ollama() is fake_process
    assert calls[0][0] == ["resolved-ollama", "serve"]
