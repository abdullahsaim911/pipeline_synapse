from __future__ import annotations
import os
import re
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def resolve_stream(source: str, max_height: int = 480) -> str:
    """Return something cv2.VideoCapture can open.

    - existing local file  -> returned unchanged
    - YouTube URL / 11-char video id -> direct stream URL via yt-dlp (nothing downloaded to disk)
    NOTE: if M0 already has a helper that returns a stream URL, call that instead and
    pass the result as `source` (it is then used as-is).
    """
    if os.path.exists(source):
        return source
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", source):
        source = "https://www.youtube.com/watch?v=" + source
    if source.startswith("http"):
        if "youtube.com" not in source and "youtu.be" not in source:
            return source  # assume it is already a direct media URL
        import yt_dlp  # lazy import
        # YouTube often offers only separate video/audio streams, so ask for VIDEO-only
        # (we never need audio) and fall back step by step.
        opts = {"quiet": True, "no_warnings": True,
                "format": f"bv*[height<={max_height}]/b[height<={max_height}]/bv*/b"}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
        if info.get("url"):
            return info["url"]
        for f in info.get("requested_formats") or []:
            if f.get("vcodec") not in (None, "none") and f.get("url"):
                return f["url"]
        raise RuntimeError("yt-dlp returned no playable video URL")
    raise FileNotFoundError(f"Cannot open video source: {source}")


def probe(stream: str) -> dict:
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {stream}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    return {"fps": fps, "frames": n, "width": w, "height": h,
            "duration": (n / fps) if n > 0 else None}


def _resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def iter_frames(stream: str, fps: float, width: int) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (time_seconds, BGR frame downscaled to `width`) at roughly `fps`.
    Frames are only decoded-and-converted when needed (grab() skips the rest)."""
    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {stream}")
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or not np.isfinite(native) or native <= 0:
        native = 30.0
    step = max(1, int(round(native / fps)))
    i = 0
    try:
        while True:
            if not cap.grab():
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield i / native, _resize_to_width(frame, width)
            i += 1
    finally:
        cap.release()


class FrameReader:
    """Random access reader used for the few representative frames we save."""
    def __init__(self, stream: str):
        self.cap = cv2.VideoCapture(stream)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {stream}")

    def read_at(self, t: float, max_width: Optional[int] = None) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return _resize_to_width(frame, max_width) if max_width else frame

    def close(self):
        self.cap.release()
