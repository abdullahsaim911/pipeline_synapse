import sys

import pytest

from runtime_interpreters import resolve_python


def test_current_interpreter_is_default(monkeypatch):
    monkeypatch.delenv("SYNAPSE_TEST_PYTHON", raising=False)
    assert resolve_python("SYNAPSE_TEST_PYTHON") == sys.executable


def test_missing_configured_interpreter_has_sanitized_error(monkeypatch, tmp_path):
    missing = tmp_path / "private" / "python.exe"
    monkeypatch.setenv("SYNAPSE_TEST_PYTHON", str(missing))

    with pytest.raises(RuntimeError) as caught:
        resolve_python("SYNAPSE_TEST_PYTHON")

    assert "SYNAPSE_TEST_PYTHON" in str(caught.value)
    assert str(missing) not in str(caught.value)
