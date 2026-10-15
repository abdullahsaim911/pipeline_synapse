"""Plain-text BEFORE / AFTER comparison.

    python -m synapse_interaction.before_after data\\VIDEO_ID [data\\VIDEO_ID2 ...] [--gt-dir ground_truth] [--tol 3]

BEFORE = intervention points from the original M1 (unchanged).
AFTER  = the same list after the interaction module is applied.
Prints to the screen and saves before_after.txt.
"""
import argparse
import os

from .demo_report import _m1_and_fused, _mmss
from .fusion import load_transcript
from .eval_interventions import load_ground_truth, match, prf


def _report(video_dir, gt_dir, tol, show_text=False):
    vid = os.path.basename(video_dir.rstrip("\\/"))
    base, fused = _m1_and_fused(video_dir)
    tr = load_transcript(os.path.join(video_dir, "transcript.json")) if show_text else []
    base_map = {round(b[0], 2): b for b in base}
    L = [f"VIDEO {vid}", "=" * 60, f"BEFORE (original M1): {len(base)} points"]
    for t, conf, reason in sorted(base):
        L.append(f"  {_mmss(t):>7s}  {str(conf):7s} {reason}")
    new = up = 0
    L += ["", f"AFTER (M1 + interaction module): {len(fused)} points"]
    for p in fused:
        b = base_map.get(round(p.timestamp, 2))
        tag = ""
        if b is None:
            tag, new = "   <-- NEW", new + 1
        elif (p.confidence, p.trigger_reason) != (b[1], b[2]):
            tag, up = f"   <-- UPGRADED (was {b[1]})", up + 1
        L.append(f"  {_mmss(p.timestamp):>7s}  {str(p.confidence):7s} {p.trigger_reason}{tag}")
        if show_text and tag:
            said = " ".join(t["text"] for t in tr if t["end"] > p.timestamp - 3 and t["start"] < p.timestamp + 3)[:160]
            L.append(f"             teacher says: {said or '(silence)'}")
    L += ["", f"CHANGE: {len(base)} -> {len(fused)} points  ({new} new, {up} upgraded)"]
    gp = os.path.join(gt_dir, vid + ".json") if gt_dir else None
    counts = None
    if gp and os.path.exists(gp):
        gt = load_ground_truth(gp)
        rows = []
        for name, pts in (("BEFORE", [b[0] for b in base]), ("AFTER ", [p.timestamp for p in fused])):
            tp, fp, fn, _ = match(pts, gt, tol)
            pr, rc, f1 = prf(tp, fp, fn)
            rows.append((tp, fp, fn))
            L.append(f"  {name}: correct {tp}, wrong {fp}, missed {fn} | precision {pr:.2f} recall {rc:.2f} F1 {f1:.2f}")
        L.append(f"  (answer key: {len(gt)} moments, tolerance +-{tol:.0f}s)")
        counts = rows
    else:
        L.append("  (no answer key -> shows what changed, not whether it is better)")
    return "\n".join(L), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dirs", nargs="+")
    ap.add_argument("--gt-dir", default="ground_truth")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--out", default="before_after.txt")
    a = ap.parse_args()
    blocks, tot_b, tot_a, n = [], [0, 0, 0], [0, 0, 0], 0
    for vd in a.video_dirs:
        text, counts = _report(vd, a.gt_dir, a.tol)
        blocks.append(text)
        if counts:
            n += 1
            for k in range(3):
                tot_b[k] += counts[0][k]; tot_a[k] += counts[1][k]
    out = "\n\n".join(blocks)
    if n > 1:
        pb, rb, fb = prf(*tot_b); pa, ra, fa = prf(*tot_a)
        out += (f"\n\nALL {n} VIDEOS TOGETHER\n{'=' * 60}\n"
                f"  BEFORE: precision {pb:.2f} recall {rb:.2f} F1 {fb:.2f}\n"
                f"  AFTER : precision {pa:.2f} recall {ra:.2f} F1 {fa:.2f}")
    print(out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    main()
