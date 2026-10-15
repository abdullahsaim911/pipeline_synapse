"""Stage 2 (optional): HAND_POINTING via MediaPipe (CPU).

Pipeline: probe a few frames for a person -> if none (screencast / tablet video) stop;
otherwise scan at hand_fps, run MediaPipe Hands, apply the index-finger rule, merge hits
over time (dwell), reject fast fingertip motion (waving).

`index_pointing()` is a pure function on 21 landmarks so it can be unit-tested without MediaPipe.
"""
from __future__ import annotations
import math
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .schema import Interaction, HAND_POINTING
from .video_io import FrameReader, iter_frames, probe

Landmark = Tuple[float, float, float]


def available(cfg: Optional[Config] = None) -> bool:
    try:
        import mediapipe as mp
    except Exception:
        return False
    if hasattr(mp, "solutions"):          # old mediapipe (<=0.10.2x): models are bundled
        return True
    cfg = cfg or Config()
    return os.path.exists(cfg.hand_model_path) and os.path.exists(cfg.pose_model_path)


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c) -> float:
    v1 = (a[0] - b[0], a[1] - b[1]); v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def index_pointing(lm: Sequence[Landmark]) -> Tuple[bool, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """MediaPipe 21-point hand: index extended, middle/ring/pinky curled.
    Returns (is_pointing, fingertip_xy, direction_unit_vector)."""
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
    """MediaPipe wrapper. Uses the Tasks API (current mediapipe; needs the two .task model files)
    and falls back to the legacy `mp.solutions` API when that is what is installed."""
    def __init__(self, cfg: Config):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                  model_complexity=0, min_detection_confidence=0.5)
            self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0,
                                               min_detection_confidence=0.5)
        else:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=cfg.hand_model_path),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.5))
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=cfg.pose_model_path),
                running_mode=vision.RunningMode.IMAGE))

    def _hands_landmarks(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.hands.process(rgb)
            return [[(p.x, p.y, p.z) for p in hl.landmark] for hl in (res.multi_hand_landmarks or [])]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.hands.detect(img)
        return [[(p.x, p.y, p.z) for p in hl] for hl in res.hand_landmarks]

    def hand_hits(self, bgr):
        hands = self._hands_landmarks(bgr)
        hits = []
        for lm in hands:
            ok, tip, d = index_pointing(lm)
            if ok:
                hits.append((tip, d))
        return bool(hands), hits

    def has_person(self, bgr) -> bool:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            return self.pose.process(rgb).pose_landmarks is not None
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return len(self.pose.detect(img).pose_landmarks) > 0

    def close(self):
        try:
            self.hands.close(); self.pose.close()
        except Exception:
            pass


def person_present(stream: str, cfg: Config, mp: _Mp, duration: Optional[float]) -> bool:
    if not duration:
        return True
    reader = FrameReader(stream)
    try:
        for k in range(cfg.hand_probe_frames):
            t = duration * (k + 0.5) / cfg.hand_probe_frames
            fr = reader.read_at(t, cfg.hand_width)
            if fr is None:
                continue
            if mp.has_person(fr) or mp.hand_hits(fr)[0]:
                return True
    finally:
        reader.close()
    return False


def detect(stream: str, cfg: Config, verbose: bool = False) -> List[Interaction]:
    if not available(cfg):
        if verbose:
            print("[hand] mediapipe or model files missing -> skipping HAND_POINTING")
        return []
    info = probe(stream)
    try:
        mp = _Mp(cfg)
    except Exception as e:
        if verbose:
            print(f"[hand] could not start MediaPipe: {e}")
        return []
    try:
        if not person_present(stream, cfg, mp, info["duration"]):
            if verbose:
                print("[hand] no person/hands found in probe frames -> skipping")
            return []
        hits = []   # (t, x, y, dx, dy)
        for t, fr in iter_frames(stream, cfg.hand_fps, cfg.hand_width):
            _, hs = mp.hand_hits(fr)
            if hs:
                (x, y), (dx, dy) = hs[0]
                hits.append((t, x, y, dx, dy))
    finally:
        mp.close()

    # merge hits into runs
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
        path = sum(math.hypot(run[k + 1][1] - run[k][1], run[k + 1][2] - run[k][2]) for k in range(len(run) - 1))
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
