from __future__ import annotations
import os
from typing import List, Optional

import cv2

from .config import Config
from .schema import Interaction, WRITING_DRAWING, save_interactions
from .video_io import FrameReader, probe, resolve_stream
from . import motion, classify


def _frame_time(seg: Interaction) -> float:
    # writing: just after the last stroke (content complete); pointing: middle of the dwell
    return seg.end + 0.3 if seg.type == WRITING_DRAWING else (seg.start + seg.end) / 2.0


def detect_interactions(source: str, out_path: Optional[str] = None, cfg: Optional[Config] = None,
                        verbose: bool = False) -> List[Interaction]:
    """source: local video path, YouTube URL / video id, or a direct stream URL.
    Writes interactions.json to `out_path` (if given) and returns the list."""
    cfg = cfg or Config()
    stream = resolve_stream(source, cfg.scan_height)
    meta = probe(stream)

    steps, _ = motion.scan(stream, cfg)
    if cfg.steps_csv:
        import csv
        with open(cfg.steps_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "label", "changed_frac", "new_frac", "gone_frac", "n_blobs", "largest_blob_frac", "cx", "cy"])
            for s in steps:
                w.writerow([round(s.t, 2), classify.label_step(s, cfg), round(s.changed_frac, 5), round(s.new_frac, 5),
                            round(s.gone_frac, 5), s.n_blobs, round(s.largest_blob_frac, 5), round(s.cx, 3), round(s.cy, 3)])
    interactions = classify.build_segments(steps, cfg)
    if verbose:
        print(f"[motion] {len(steps)} steps -> {len(interactions)} segments")

    if cfg.enable_hand:
        from . import hand
        interactions += hand.detect(stream, cfg, verbose=verbose)

    interactions.sort(key=lambda s: (s.start, s.end))

    if cfg.save_frames and out_path and interactions:
        frames_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "interaction_frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_stream = resolve_stream(source, cfg.frame_height) if stream != source and not os.path.exists(source) else stream
        reader = FrameReader(frame_stream)
        try:
            for k, seg in enumerate(interactions):
                fr = reader.read_at(_frame_time(seg), cfg.frame_max_width)
                if fr is None:
                    continue
                p = os.path.join(frames_dir, f"{k:03d}_{seg.type}_{_frame_time(seg):07.2f}.jpg")
                cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                seg.frame_path = p
        finally:
            reader.close()

    if out_path:
        save_interactions(out_path, interactions, video=str(source), duration=meta.get("duration"))
    return interactions
