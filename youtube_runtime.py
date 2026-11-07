"""Shared yt-dlp JavaScript runtime configuration for modern YouTube extraction."""

from __future__ import annotations

import shutil


def youtube_runtime_options() -> dict:
    """Enable installed JS runtimes in yt-dlp's Python API."""
    runtimes = {}
    for name in ("deno", "node", "quickjs", "bun"):
        executable = shutil.which(name)
        if executable:
            runtimes[name] = {"path": executable}
    return {"js_runtimes": runtimes} if runtimes else {}
