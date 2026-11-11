"""Resolve optional isolated Python runtimes without leaking environment details."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_python(variable: str) -> str:
    """Return a configured interpreter, or the current interpreter by default."""
    configured = os.environ.get(variable, "").strip()
    if not configured:
        return sys.executable
    candidate = Path(os.path.expandvars(configured)).expanduser()
    if not candidate.is_file():
        raise RuntimeError(f"{variable} does not point to a Python executable")
    return str(candidate.resolve())
