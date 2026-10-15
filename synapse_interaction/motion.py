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
from .video_io import iter_frames

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


def scan(stream: str, cfg: Config) -> Tuple[List[Step], dict]:
    steps: List[Step] = []
    prev: Optional[np.ndarray] = None
    size = None
    for t, frame in iter_frames(stream, cfg.scan_fps, cfg.scan_width):
        mask = _ink_mask(frame, cfg.ink_thr)
        h, w = mask.shape
        size = (w, h)
        if prev is None:
            steps.append(Step(t=t))
            prev = mask
            continue

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
        steps.append(st)
        prev = mask
    return steps, {"scan_size": size}
