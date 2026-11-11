"""Shared yt-dlp JavaScript runtime configuration for modern YouTube extraction."""

from __future__ import annotations

import re
import shutil


_YOUTUBE_VIDEO_ID = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([\w-]{11})"
)


def extract_youtube_video_id(youtube_url: str) -> str:
    """Return a validated YouTube ID without importing the media pipeline."""
    match = _YOUTUBE_VIDEO_ID.search(youtube_url)
    if match is None:
        raise ValueError(f"Invalid YouTube URL format: {youtube_url}")
    return match.group(1)


def youtube_runtime_options() -> dict:
    """Enable installed JS runtimes in yt-dlp's Python API."""
    runtimes = {}
    for name in ("deno", "node", "quickjs", "bun"):
        executable = shutil.which(name)
        if executable:
            runtimes[name] = {"path": executable}
    return {"js_runtimes": runtimes} if runtimes else {}
