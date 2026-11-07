"""Resolve an FFmpeg executable without requiring a machine-wide install."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


FFMPEG_PATH_ENV = "SYNAPSE_FFMPEG_PATH"


class FFmpegUnavailableError(RuntimeError):
    """No usable FFmpeg executable could be found."""


def _valid_executable(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    return str(candidate) if candidate.is_file() else None


def resolve_ffmpeg_executable() -> str:
    """Return configured, project-local, PATH, or imageio-bundled FFmpeg."""
    configured = os.environ.get(FFMPEG_PATH_ENV)
    if configured:
        executable = _valid_executable(configured)
        if executable is None:
            raise FFmpegUnavailableError(
                f"{FFMPEG_PATH_ENV} does not point to an FFmpeg executable"
            )
        return executable

    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    packaged = Path(__file__).resolve().parent / "ffmpeg" / "bin" / executable_name
    executable = _valid_executable(packaged)
    if executable is not None:
        return executable

    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return system_executable

    try:
        import imageio_ffmpeg

        executable = _valid_executable(imageio_ffmpeg.get_ffmpeg_exe())
        if executable is not None:
            return executable
    except (ImportError, RuntimeError, OSError):
        pass

    raise FFmpegUnavailableError(
        "FFmpeg is required for video frame extraction. Install project dependencies "
        f"or set {FFMPEG_PATH_ENV} to the FFmpeg executable."
    )
