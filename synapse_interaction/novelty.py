"""Visual-novelty helpers: is this frame's CONTENT already shown by another (selected) frame?

A frame is reduced to a pixel-level "ink mask" (where writing/drawing is, at 480 px wide).
Frame A is redundant if most of its ink is already present in frame B (B shows everything A shows).
Because a page does not scroll, the partial and complete versions of one page share exactly the same strokes,
whereas a different page has different strokes. This is what makes the module pick unique, content-rich
frames instead of one frame per writing line.
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

MASK_W = 480


def ink_mask(img: np.ndarray, thr: int = 24) -> np.ndarray:
    """Bool mask (H x 480): True where a pixel's colour differs from the page background."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (MASK_W, max(1, int(MASK_W * h / w))), interpolation=cv2.INTER_AREA)
    bg = np.median(small[::3, ::3].reshape(-1, 3), axis=0).astype(np.int16)
    return np.abs(small.astype(np.int16) - bg).max(axis=2) > thr


def load_mask(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    img = cv2.imread(path)
    return None if img is None else ink_mask(img)


def contained_fraction(a: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of a's ink pixels that are also inked in ref (ref dilated by 1 px to tolerate small shifts)."""
    if a.shape != ref.shape:
        return 0.0
    n = int(a.sum())
    if n == 0:
        return 1.0
    refd = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return float((a & refd).sum()) / n
