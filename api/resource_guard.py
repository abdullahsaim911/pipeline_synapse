"""Conservative admission guard for local model work."""

from __future__ import annotations

import os

import psutil


class LocalResourcesUnavailableError(RuntimeError):
    """The machine cannot safely admit another heavy local-model job."""


def _minimum_available_bytes() -> int:
    raw = os.environ.get("SYNAPSE_MIN_AVAILABLE_RAM_MB", "512").strip()
    try:
        megabytes = int(raw)
    except ValueError as exc:
        raise ValueError("SYNAPSE_MIN_AVAILABLE_RAM_MB must be an integer") from exc
    if megabytes < 0:
        raise ValueError("SYNAPSE_MIN_AVAILABLE_RAM_MB cannot be negative")
    return megabytes * 1024 * 1024


def require_processing_headroom() -> None:
    """Reject new heavy work before memory pressure can terminate the API."""
    if psutil.virtual_memory().available < _minimum_available_bytes():
        raise LocalResourcesUnavailableError(
            "Local model capacity is temporarily unavailable; retry shortly"
        )
