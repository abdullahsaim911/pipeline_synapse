from __future__ import annotations
import os
import re
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def resolve_stream(source: str, max_height: int = 480) -> str:
    """Return a local video file path for cv2.VideoCapture.

    This module never downloads or streams video. In the pipeline it is given the frame extractor's frames
    (InteractionStream.feed); the file path form exists only for the command line and the tests. A URL or a
    YouTube id is refused on purpose.
    """
    if os.path.exists(source):
        return source
    if re.match(r"^(https?://|[A-Za-z0-9_-]{11}$)", source):
        raise ValueError(
            "synapse_interaction does not download or stream video. Run the pipeline (the frame extractor hands "
            "its frames to InteractionStream), or pass a video file that is already on disk.")
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
