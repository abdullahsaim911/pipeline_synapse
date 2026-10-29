"""Stage 1: cheap ink/motion scan (OpenCV + NumPy only, CPU, no model).

For each sampled frame we build an "ink mask" (pixels that differ from the dominant
background gray level) and compare it to the previous sampled frame:
    new  = ink that appeared        gone = ink that disappeared
 - pen writing / drawing : new >> gone, small localised blob, ink PERSISTS
 - cursor / pointer move : new ~= gone (a small blob translating), no net ink
 - scene cut             : a large fraction of the frame changes at once
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .video_io import iter_frames, _resize_to_width

_K = np.ones((3, 3), np.uint8)


@dataclass
class Step:
    t: float
    valid: bool = False          # False for the first frame (nothing to compare with)
    changed_frac: float = 0.0
    new_frac: float = 0.0
    gone_frac: float = 0.0
    n_blobs: int = 0
    largest_blob_frac: float = 0.0
    cx: float = 0.5              # centroid of NEW pixels (normalised)
    cy: float = 0.5
    bbox: Optional[Tuple[float, float, float, float]] = None  # normalised x0,y0,x1,y1


def _ink_mask(frame: np.ndarray, thr: int) -> np.ndarray:
    """Pixels whose COLOUR differs from the page background (any channel).
    Gray conversion hides dark coloured ink (blue/green pens on black), so we compare channels.
    Background colour = per-channel median of a sparse pixel grid (ink is sparse on a page)."""
    sample = frame[::6, ::6].reshape(-1, 3)
    bg = np.median(sample, axis=0).astype(np.int16)
    return (np.abs(frame.astype(np.int16) - bg).max(axis=2) > thr)


class Scanner:
    """Frame-by-frame version of the scan: call push(t, bgr_frame) for each frame in time order.

    Frames may come from anywhere (the frame extractor's stream, a video file); this class never opens a
    video. Each frame is downscaled to cfg.scan_width before use."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.steps: List[Step] = []
        self.size = None
        self._prev: Optional[np.ndarray] = None

    def push(self, t: float, frame: np.ndarray) -> Step:
        cfg = self.cfg
        frame = _resize_to_width(frame, cfg.scan_width)
        mask = _ink_mask(frame, cfg.ink_thr)
        h, w = mask.shape
        self.size = (w, h)
        prev = self._prev
        self._prev = mask
        if prev is None or prev.shape != mask.shape:
            st = Step(t=t)
            self.steps.append(st)
            return st

        total = float(h * w)
        changed = mask ^ prev
        changed_u8 = changed.astype(np.uint8)
        changed_frac = float(changed_u8.sum()) / total
        st = Step(t=t, valid=True, changed_frac=changed_frac)
        if changed_frac <= cfg.cut_frac and changed_frac > 0:
            dil = cv2.dilate(changed_u8, _K, iterations=2)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
            raw = np.bincount(labels[changed], minlength=n)
            keep = [i for i in range(1, n) if raw[i] >= cfg.min_blob_px]
            if keep:
                keep_mask = np.isin(labels, keep)
                new = mask & ~prev & keep_mask
                gone = prev & ~mask & keep_mask
                st.new_frac = float(new.sum()) / total
                st.gone_frac = float(gone.sum()) / total
                st.n_blobs = len(keep)
                st.largest_blob_frac = float(max(stats[i, cv2.CC_STAT_AREA] for i in keep)) / total
                ys, xs = np.nonzero(new if new.any() else keep_mask)
                st.cx, st.cy = float(xs.mean()) / w, float(ys.mean()) / h
                ys2, xs2 = np.nonzero(keep_mask)
                st.bbox = (float(xs2.min()) / w, float(ys2.min()) / h,
                           float(xs2.max()) / w, float(ys2.max()) / h)
        self.steps.append(st)
        return st


def scan(stream: str, cfg: Config) -> Tuple[List[Step], dict]:
    """Scan a video file / URL (kept for the CLI and tests; the pipeline feeds frames to Scanner instead)."""
    sc = Scanner(cfg)
    for t, frame in iter_frames(stream, cfg.scan_fps, cfg.scan_width):
        sc.push(t, frame)
    return sc.steps, {"scan_size": sc.size}
