"""CLI:  python -m synapse_interaction.run <video file> --out <dir> [--no-hand]
(a file already on disk; the pipeline itself feeds frames and never uses this)"""
import argparse
import os

from .config import Config
from .detector import detect_interactions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="a video file on disk (no URLs: this module never downloads or streams video)")
    ap.add_argument("--out", default="interaction_out")
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--dump-steps", action="store_true", help="write steps.csv (per-step features) for debugging")
    a = ap.parse_args()

    cfg = Config(enable_hand=not a.no_hand, save_frames=not a.no_frames)
    os.makedirs(a.out, exist_ok=True)
    if a.dump_steps:
        cfg.steps_csv = os.path.join(a.out, "steps.csv")
    out_json = os.path.join(a.out, "interactions.json")
    res = detect_interactions(a.source, out_json, cfg, verbose=True)
    for s in res:
        print(f"{s.type:16s} {s.start:7.2f} -> {s.end:7.2f}  conf={s.confidence:.2f}  src={s.source}")
    print(f"{len(res)} interactions -> {out_json}")


if __name__ == "__main__":
    main()
