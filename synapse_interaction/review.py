"""Blind review helper: builds the answer key (ground truth) by asking you one question per candidate.

    python -m synapse_interaction.review data\\VIDEO_ID

The candidates are M1's points plus the module's points, mixed together in time order WITHOUT saying
which system produced them (so you can't be biased). For each one you watch the video at that time and
answer y/n. At the end you can add moments that were not listed. Result: synapse_interaction/eval_data/ground_truth/VIDEO_ID.json
"""
import json
import os
import sys

from .fusion import fuse_interactions, load_transcript


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _parse(x: str) -> float:
    sec = 0.0
    for p in x.strip().split(":"):
        sec = sec * 60 + float(p)
    return sec


def main():
    if len(sys.argv) < 2:
        print("usage: python -m synapse_interaction.review data\\VIDEO_ID [--out file.json]")
        return
    video_dir = sys.argv[1].rstrip("\\/")
    vid = os.path.basename(video_dir)
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data", "ground_truth", vid + ".json")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else default_out

    from synchronizer import Synchronizer
    tp = os.path.join(video_dir, "transcript.json")
    kp = os.path.join(video_dir, "keyframes", "keyframes.json")
    ip = os.path.join(video_dir, "interactions.json")
    base = [p.timestamp for p in Synchronizer(transcript_path=tp, keyframes_path=kp).detect_suffer_points()]
    with_mod = [p.timestamp for p in fuse_interactions(
        Synchronizer(transcript_path=tp, keyframes_path=kp).detect_suffer_points(), ip, tp, kp)]

    pool = []
    for t in sorted(base + with_mod):
        if not pool or t - pool[-1] > 2.0:          # merge candidates closer than 2 s
            pool.append(t)

    tr = load_transcript(tp)
    print(f"\n{len(pool)} candidate moments. Open the video in VLC (Ctrl+T = jump to time), listen WITHOUT looking")
    print("at the screen first, then look. Question each time: would a blind student be lost here?\n")
    yes = []
    for i, t in enumerate(pool, 1):
        near = [x["text"] for x in tr if x["end"] > t - 4 and x["start"] < t + 4]
        print(f"[{i}/{len(pool)}] jump to {_fmt(t)}")
        print("    teacher says around then:", (" ".join(near))[:300] or "(silence)")
        while True:
            a = input("    lost without seeing the screen?  y = yes, n = no : ").strip().lower()
            if a in ("y", "n"):
                break
        if a == "y":
            yes.append(round(t, 1))
        print()

    extra = input("Other moments you noticed that were NOT listed? Type times like 1:05, 2:30 (or press Enter): ").strip()
    for part in [p for p in extra.split(",") if p.strip()]:
        try:
            yes.append(round(_parse(part), 1))
        except ValueError:
            print("   skipped (could not read):", part)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"suffer_points": sorted(set(yes))}, f)
    print(f"\nSaved {len(set(yes))} moments to {out}")


if __name__ == "__main__":
    main()
