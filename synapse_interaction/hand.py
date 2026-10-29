"""Stage 2 (optional): HAND_POINTING via MediaPipe (CPU).

Frames are PUSHED in (HandScanner.push) by whoever already decodes the video - the frame extractor in the
pipeline - so this module never opens or downloads a video. For each frame (at most hand_fps per second) it
runs MediaPipe Hands and applies the index-finger rule; the hits are merged over time (dwell) and fast
fingertip motion is rejected (waving).

`index_pointing()` and `interactions_from_hits()` are pure functions so they can be unit-tested without
MediaPipe.
"""
from __future__ import annotations
import math
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .schema import Interaction, HAND_POINTING
from .video_io import iter_frames, _resize_to_width

Landmark = Tuple[float, float, float]


def available(cfg: Optional[Config] = None) -> bool:
    try:
        import mediapipe as mp
    except Exception:
        return False
    if hasattr(mp, "solutions"):          # old mediapipe (<=0.10.2x): models are bundled
        return True
    cfg = cfg or Config()
    return os.path.exists(cfg.hand_model_path)


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c) -> float:
    v1 = (a[0] - b[0], a[1] - b[1]); v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def index_pointing(lm: Sequence[Landmark], aspect: float = 1.0
                   ) -> Tuple[bool, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """MediaPipe 21-point hand: index extended, middle/ring/pinky curled.
    Returns (is_pointing, fingertip_xy, direction_unit_vector) in the SAME normalised units as the input.

    MediaPipe landmarks are x/width and y/height, so on a 16:9 frame a horizontal step is ~1.8x shorter than
    an equal vertical one. The lengths and the angle are therefore measured after multiplying x by `aspect`
    (= frame width / height), i.e. in square units. With aspect=1.0 the behaviour is unchanged."""
    if aspect != 1.0:
        sq = [(p[0] * aspect, p[1], p[2]) for p in lm]
        ok, _tip, d = index_pointing(sq, 1.0)
        if not ok:
            return False, None, None
        return True, (lm[8][0], lm[8][1]), d
    wrist = lm[0]
    index_extended = (_d(lm[8], wrist) > 1.15 * _d(lm[6], wrist)) and _angle(lm[5], lm[6], lm[8]) > 150
    curled = all(_d(lm[tip], wrist) < 1.05 * _d(lm[pip], wrist)
                 for tip, pip in ((12, 10), (16, 14), (20, 18)))
    if not (index_extended and curled):
        return False, None, None
    dx, dy = lm[8][0] - lm[5][0], lm[8][1] - lm[5][1]
    n = math.hypot(dx, dy) or 1.0
    return True, (lm[8][0], lm[8][1]), (dx / n, dy / n)


class _Mp:
    """MediaPipe Hands wrapper. Uses the Tasks API (current mediapipe; needs hand_landmarker.task) and falls
    back to the legacy `mp.solutions` API when that is what is installed."""
    def __init__(self, cfg: Config):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                  model_complexity=0, min_detection_confidence=0.5)
        else:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=cfg.hand_model_path),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.5))

    def _hands_landmarks(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.hands.process(rgb)
            return [[(p.x, p.y, p.z) for p in hl.landmark] for hl in (res.multi_hand_landmarks or [])]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.hands.detect(img)
        return [[(p.x, p.y, p.z) for p in hl] for hl in res.hand_landmarks]

    def hand_hits(self, bgr):
        h, w = bgr.shape[:2]
        hands = self._hands_landmarks(bgr)
        hits = []
        for lm in hands:
            ok, tip, d = index_pointing(lm, aspect=w / float(h))
            if ok:
                hits.append((tip, d))
        return bool(hands), hits

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


class HandScanner:
    """Push frames in time order; collects fingertip hits. Never opens a video."""

    def __init__(self, cfg: Config, mp=None):
        self.cfg = cfg
        self.mp = mp                  # injectable for tests; created lazily otherwise
        self.hits = []                # (t, x, y, dx, dy)
        self.aspect = 16 / 9
        self.failed = False
        self._last_t = None

    def start(self, verbose: bool = False) -> bool:
        if self.mp is not None:
            return True
        if not available(self.cfg):
            if verbose:
                print("[hand] mediapipe or the hand model file is missing -> skipping HAND_POINTING")
            self.failed = True
            return False
        try:
            self.mp = _Mp(self.cfg)
        except Exception as e:
            if verbose:
                print(f"[hand] could not start MediaPipe: {e}")
            self.failed = True
            return False
        return True

    def push(self, t: float, frame: np.ndarray) -> None:
        if self.failed or self.mp is None:
            return
        if self._last_t is not None and t - self._last_t < 0.999 / self.cfg.hand_fps:
            return                    # frames may arrive faster than the hand scan needs
        self._last_t = t
        frame = _resize_to_width(frame, self.cfg.hand_width)
        self.aspect = frame.shape[1] / float(frame.shape[0])
        try:
            _, hs = self.mp.hand_hits(frame)
        except Exception:
            return
        if hs:
            (x, y), (dx, dy) = hs[0]
            self.hits.append((t, x, y, dx, dy))

    def close(self) -> None:
        if self.mp is not None and hasattr(self.mp, "close"):
            self.mp.close()

    def interactions(self) -> List[Interaction]:
        return interactions_from_hits(self.hits, self.cfg, self.aspect)


def interactions_from_hits(hits, cfg: Config, aspect: float = 16 / 9) -> List[Interaction]:
    """Merge per-frame fingertip hits (t, x, y, dx, dy) into HAND_POINTING episodes."""
    runs, cur = [], []
    for h in hits:
        if cur and h[0] - cur[-1][0] > cfg.hand_gap_tol:
            runs.append(cur); cur = []
        cur.append(h)
    if cur:
        runs.append(cur)

    out: List[Interaction] = []
    half = 0.5 / cfg.hand_fps
    for run in runs:
        s0, s1 = run[0][0] - half, run[-1][0] + half
        if s1 - s0 < cfg.hand_min_dur:
            continue
        # distances in frame-heights, so a sideways move is not under-counted on a wide frame
        path = sum(math.hypot((run[k + 1][1] - run[k][1]) * aspect, run[k + 1][2] - run[k][2])
                   for k in range(len(run) - 1))
        speed = path / max(1e-6, s1 - s0)
        if speed > cfg.hand_max_speed:
            continue   # waving, not pointing
        xs = [r[1] for r in run]; ys = [r[2] for r in run]
        conf = min(0.9, 0.45 + 0.07 * len(run))
        out.append(Interaction(HAND_POINTING, round(max(0, s0), 2), round(s1, 2), round(conf, 2),
                               bbox=[round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
                               trajectory=[[round(r[0], 2), round(r[1], 4), round(r[2], 4)] for r in run][:cfg.max_trajectory_points],
                               source="mediapipe",
                               meta={"direction": [round(sum(r[3] for r in run) / len(run), 3),
                                                   round(sum(r[4] for r in run) / len(run), 3)],
                                     "speed": round(speed, 3), "hits": len(run)}))
    return out


def detect(stream: str, cfg: Config, verbose: bool = False) -> List[Interaction]:
    """Video-file entry point (CLI / tests). The pipeline pushes frames into a HandScanner instead."""
    sc = HandScanner(cfg)
    if not sc.start(verbose):
        return []
    try:
        for t, fr in iter_frames(stream, cfg.hand_fps, cfg.hand_width):
            sc.push(t, fr)
    finally:
        sc.close()
    return sc.interactions()
