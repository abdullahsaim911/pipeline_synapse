"""Does the interaction module actually improve intervention-point detection?

Usage (from the project root, after the baseline run created data/<video_id>/ ):
    python -m synapse_interaction.eval_interventions --video-dir data/VIDEO_ID --gt ground_truth/VIDEO_ID.json

Ground-truth file = the moments YOU decided a blind student needs help (seconds or "mm:ss"):
    {"suffer_points": [12.5, "0:41", "1:05.5"]}        (a plain list also works)

It compares M1 alone with M1 + interactions (ablation by interaction type) using one-to-one
matching within a time tolerance, and prints precision / recall / F1.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from typing import Dict, List, Sequence, Tuple, Union

from .schema import load_interactions, save_interactions, WRITING_DRAWING, MOUSE_POINTING, HAND_POINTING


def parse_time(x: Union[int, float, str]) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    parts = str(x).strip().split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def load_ground_truth(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["suffer_points"] if isinstance(data, dict) else data
    return sorted(parse_time(t) for t in items)


def match(pred: Sequence[float], gt: Sequence[float], tol: float) -> Tuple[int, int, int, List[Tuple[float, float]]]:
    """One-to-one matching: closest pairs first, each prediction/gt used at most once."""
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol)
    used_p, used_g, matched = set(), set(), []
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matched.append((pred[i], gt[j]))
    tp = len(matched)
    return tp, len(pred) - tp, len(gt) - tp, matched


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _m1_points(video_dir: str, interactions_path: str = None, types=None) -> List[float]:
    """Fresh Synchronizer each time (fusion edits points in place)."""
    from synchronizer import Synchronizer
    from .fusion import fuse_interactions
    tp_ = os.path.join(video_dir, "transcript.json")
    kp = os.path.join(video_dir, "keyframes", "keyframes.json")
    pts = Synchronizer(transcript_path=tp_, keyframes_path=kp).detect_suffer_points()
    if interactions_path:
        ints = load_interactions(interactions_path)
        if types:
            ints = [i for i in ints if i.type in types]
        tmp = os.path.join(tempfile.mkdtemp(), "i.json")
        save_interactions(tmp, ints)
        pts = fuse_interactions(pts, tmp, tp_, kp)
    return sorted(p.timestamp for p in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--gt", default=None, help="default: eval_data/ground_truth/<video_id>.json inside this package")
    ap.add_argument("--interactions", default=None, help="default: <video-dir>/interactions.json")
    ap.add_argument("--tol", type=float, default=3.0, help="match tolerance in seconds")
    a = ap.parse_args()

    gt_path = a.gt or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data", "ground_truth",
                                   os.path.basename(a.video_dir.rstrip("\\/")) + ".json")
    gt = load_ground_truth(gt_path)
    ipath = a.interactions or os.path.join(a.video_dir, "interactions.json")
    variants: Dict[str, Tuple] = {
        "M1 alone (baseline)": (None, None),
        "+ WRITING_DRAWING": (ipath, {WRITING_DRAWING}),
        "+ MOUSE_POINTING": (ipath, {MOUSE_POINTING}),
        "+ HAND_POINTING": (ipath, {HAND_POINTING}),
        "M1 + all interactions": (ipath, None),
    }
    print(f"ground truth: {len(gt)} points | tolerance: +-{a.tol:.1f}s\n")
    print(f"{'variant':26s} {'points':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    base_pts = None
    for name, (path, types) in variants.items():
        pts = _m1_points(a.video_dir, path, types)
        if base_pts is None:
            base_pts = pts
        tp, fp, fn, _ = match(pts, gt, a.tol)
        p, r, f = prf(tp, fp, fn)
        print(f"{name:26s} {len(pts):6d} {tp:4d} {fp:4d} {fn:4d} {p:6.2f} {r:6.2f} {f:6.2f}")
        last_pts = pts
    added = [t for t in last_pts if min((abs(t - b) for b in base_pts), default=99) > 0.01]
    missed = [g for g in gt if min((abs(g - p) for p in last_pts), default=99) > a.tol]
    print("\npoints added by the module (watch the video at these times to confirm):", [round(t, 1) for t in added])
    print("ground-truth moments still missed:", [round(g, 1) for g in missed])


if __name__ == "__main__":
    main()
