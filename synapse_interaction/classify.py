"""Turn per-step motion features into WRITING_DRAWING / MOUSE_POINTING segments."""
from __future__ import annotations
import math
from typing import List, Optional, Set

import numpy as np

from .config import Config
from .motion import Step
from .schema import Interaction, MOUSE_POINTING, WRITING_DRAWING


def label_step(s: Step, cfg: Config) -> str:
    if not s.valid:
        return "idle"
    if s.changed_frac > cfg.cut_frac:
        return "cut"
    n = s.new_frac + s.gone_frac
    if n < cfg.min_changed_frac:
        return "idle"
    if s.n_blobs > cfg.max_blobs:
        return "busy"
    net_ratio = (s.new_frac - s.gone_frac) / n
    if (net_ratio >= cfg.writing_net_ratio
            and s.largest_blob_frac <= cfg.write_max_blob_frac
            and s.new_frac <= cfg.write_max_new_frac):
        return "write"
    if net_ratio <= -cfg.writing_net_ratio:
        return "erase"
    if s.largest_blob_frac <= cfg.cursor_max_blob_frac:
        return "cursor"
    return "large"          # hand / person-sized motion: left to the hand stage


def _runs(steps: List[Step], labels: List[str], target: Set[str],
          tolerate: Set[str], gap: float) -> List[List[int]]:
    """Group indices of `target` steps into runs, bridging `tolerate` steps up to `gap` seconds."""
    runs, cur, last = [], None, None
    for i, lab in enumerate(labels):
        if lab in target:
            if cur is not None and steps[i].t - steps[last].t > gap:
                runs.append(cur); cur = None
            if cur is None:
                cur = []
            cur.append(i); last = i
        elif lab in tolerate:
            if cur is not None and steps[i].t - steps[last].t > gap:
                runs.append(cur); cur = None
        else:  # blocking label (cut, large, busy, erase, ...)
            if cur is not None:
                runs.append(cur); cur = None
    if cur is not None:
        runs.append(cur)
    return runs


def _traj(steps: List[Step], idx: List[int], max_pts: int):
    pts = [[round(steps[i].t, 2), round(steps[i].cx, 4), round(steps[i].cy, 4)] for i in idx]
    if len(pts) > max_pts:
        sel = np.linspace(0, len(pts) - 1, max_pts).round().astype(int)
        pts = [pts[j] for j in sel]
    return pts


def _bbox(steps: List[Step], idx: List[int]):
    bs = [steps[i].bbox for i in idx if steps[i].bbox]
    if not bs:
        return None
    return [round(min(b[0] for b in bs), 4), round(min(b[1] for b in bs), 4),
            round(max(b[2] for b in bs), 4), round(max(b[3] for b in bs), 4)]


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _merge_into(a: Interaction, b: Interaction, max_pts: int) -> Interaction:
    a.end = max(a.end, b.end)
    a.confidence = round(max(a.confidence, b.confidence), 2)
    if a.bbox and b.bbox:
        a.bbox = [min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                  max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])]
    else:
        a.bbox = a.bbox or b.bbox
    traj = sorted(a.trajectory + b.trajectory, key=lambda p: p[0])
    if len(traj) > max_pts:
        sel = np.linspace(0, len(traj) - 1, max_pts).round().astype(int)
        traj = [traj[j] for j in sel]
    a.trajectory = traj
    for k in ("steps", "net_ink_frac", "path_len"):
        if k in b.meta:
            a.meta[k] = round(a.meta.get(k, 0) + b.meta[k], 5)
    return a


def postprocess(segs: List[Interaction], cfg: Config) -> List[Interaction]:
    """1) a short pointer move sandwiched against writing is part of the writing (pen travelling)
       2) merge same-type segments separated by short pauses -> fewer, cleaner episodes"""
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    # 1) absorb short MOUSE segments adjacent to a WRITING segment
    keep: List[Interaction] = []
    for i, s in enumerate(segs):
        if s.type == MOUSE_POINTING and s.duration <= cfg.absorb_max_dur:
            near_w = any(o.type == WRITING_DRAWING and
                         (abs(o.end - s.start) <= cfg.absorb_gap or abs(s.end - o.start) <= cfg.absorb_gap)
                         for o in segs)
            if near_w:
                continue
        keep.append(s)
    # 2) merge same type within a gap
    out: List[Interaction] = []
    for s in keep:
        gap = cfg.write_merge_gap if s.type == WRITING_DRAWING else cfg.point_merge_gap
        prev = next((o for o in reversed(out) if o.type == s.type), None)
        if prev is not None and s.start - prev.end <= gap:
            _merge_into(prev, s, cfg.max_trajectory_points)
        else:
            out.append(s)
    # absorbed pointing time that sits between two writing segments is covered by merging writing
    return sorted(out, key=lambda s: (s.start, s.end))


def build_segments(steps: List[Step], cfg: Config) -> List[Interaction]:
    if len(steps) < 2:
        return []
    dt = steps[1].t - steps[0].t
    labels = [label_step(s, cfg) for s in steps]
    out: List[Interaction] = []

    # ---------------- writing / drawing ----------------
    writing: List[Interaction] = []
    for run in _runs(steps, labels, {"write"}, {"idle", "cursor"}, cfg.write_gap_tol):
        s0, s1 = steps[run[0]].t - dt, steps[run[-1]].t
        net = sum(steps[i].new_frac - steps[i].gone_frac for i in range(run[0], run[-1] + 1))
        if (s1 - s0) < cfg.min_write_dur or net < cfg.min_write_net_frac:
            continue   # a hand entering/leaving cancels out: net ink ~ 0
        conf = min(0.95, 0.5 + 0.08 * len(run))
        if conf < cfg.min_conf:
            continue
        seg = Interaction(WRITING_DRAWING, round(s0, 2), round(s1, 2), round(conf, 2),
                          bbox=_bbox(steps, run), trajectory=_traj(steps, run, cfg.max_trajectory_points),
                          source="motion", meta={"net_ink_frac": round(net, 5), "steps": len(run)})
        writing.append(seg)
    out += writing

    # ---------------- mouse / pen-cursor pointing ----------------
    for run in _runs(steps, labels, {"cursor"}, {"idle"}, cfg.point_gap_tol):
        s0, s1 = steps[run[0]].t - dt, steps[run[-1]].t
        if (s1 - s0) < cfg.min_point_dur:
            continue
        xs = [steps[i].cx for i in run]; ys = [steps[i].cy for i in run]
        path = sum(math.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k]) for k in range(len(run) - 1))
        if path < cfg.min_point_path:
            continue
        dur = s1 - s0
        if any(_overlap(s0, s1, w.start, w.end) > 0.5 * dur for w in writing):
            continue   # cursor movement that is really part of writing
        conf = min(0.85, 0.40 + 0.08 * len(run) + min(0.2, path))
        if conf < cfg.min_conf:
            continue
        out.append(Interaction(MOUSE_POINTING, round(s0, 2), round(s1, 2), round(conf, 2),
                               bbox=_bbox(steps, run), trajectory=_traj(steps, run, cfg.max_trajectory_points),
                               source="motion", meta={"path_len": round(path, 3), "steps": len(run),
                                                      "note": "movement-based; a motionless hovering cursor is not detected"}))
    return postprocess(out, cfg)
