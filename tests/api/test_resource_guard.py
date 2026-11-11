from types import SimpleNamespace

import pytest

from api import resource_guard


def test_admission_allows_configured_headroom(monkeypatch):
    monkeypatch.setenv("SYNAPSE_MIN_AVAILABLE_RAM_MB", "512")
    monkeypatch.setattr(
        resource_guard.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=513 * 1024 * 1024),
    )
    resource_guard.require_processing_headroom()


def test_admission_failure_is_retryable_without_machine_details(monkeypatch):
    monkeypatch.setenv("SYNAPSE_MIN_AVAILABLE_RAM_MB", "512")
    monkeypatch.setattr(
        resource_guard.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=511 * 1024 * 1024),
    )
    with pytest.raises(resource_guard.LocalResourcesUnavailableError) as caught:
        resource_guard.require_processing_headroom()
    assert "retry" in str(caught.value)
    assert "511" not in str(caught.value)


@pytest.mark.parametrize("configured", ["bad", "-1"])
def test_invalid_memory_threshold_fails_closed(monkeypatch, configured):
    monkeypatch.setenv("SYNAPSE_MIN_AVAILABLE_RAM_MB", configured)
    with pytest.raises(ValueError):
        resource_guard.require_processing_headroom()
