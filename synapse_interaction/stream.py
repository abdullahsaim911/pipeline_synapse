"""Frame-fed interaction detection.

The pipeline already decodes the lecture once (the frame extractor). `InteractionStream` receives those very
frames, one at a time, and never opens, downloads or seeks a video itself:

    stream = InteractionStream(cfg, out_path=".../interactions.json")
    extract_pedagogical_keyframes(..., frame_tap=stream.feed)      # frames arrive here as they are decoded
    interactions = stream.finish()                                  # writes interactions.json

Frames must be fed in time order as (seconds, BGR uint8 image). Any rate works; 2 frames per second or more is
recommended (pen strokes and pointers move between samples). Nothing here raises into the caller from `feed`:
a bad frame is counted and skipped so it can never break frame extraction.

Representative frames (the picture saved for each detected event) cannot be re-read from the video, so a
down-scaled JPEG is spooled to disk at most once per `cfg.spool_interval` seconds and the right ones are
picked at the end; the rest are deleted.
"""
from __future__ import annotations
import csv
import glob
import os
import shutil
from typing import List, Optional

import cv2
import numpy as np

from .config import Config
from .schema import Interaction, WRITING_DRAWING, save_interactions
from .video_io import _resize_to_width
from . import classify, motion


def _frame_time(seg: Interaction) -> float:
    # writing: just after the last stroke (content complete); pointing: middle of the dwell
    return seg.end + 0.3 if seg.type == WRITING_DRAWING else (seg.start + seg.end) / 2.0


class InteractionStream:
    def __init__(self, cfg: Optional[Config] = None, out_path: Optional[str] = None,
                 source_label: Optional[str] = None, verbose: bool = False, hand_scanner=None):
        self.cfg = cfg or Config()
        self.out_path = out_path
        self.source_label = source_label
        self.verbose = verbose
        self.frames_fed = 0
        self.frames_failed = 0
        self.last_t = 0.0
        self._motion = motion.Scanner(self.cfg)
        self._hand = None
        if self.cfg.enable_hand:
            from . import hand
            self._hand = hand_scanner or hand.HandScanner(self.cfg)
            if not self._hand.start(verbose):
                self._hand = None
        self._spool_dir: Optional[str] = None
        self._spooled: List[float] = []
        self._last_spool_t = -1e9
        self._finished = False
        if self.cfg.save_frames and out_path:
            # Paths are built from out_path AS GIVEN (no abspath), so they look like the extractor's own keyframe
            # paths (relative to the working directory, e.g. data/<id>/interaction_frames/...).
            frames_dir = os.path.join(os.path.dirname(out_path), "interaction_frames")
            for old in glob.glob(os.path.join(frames_dir, "*.jpg")):     # pictures of an earlier run
                try:
                    os.remove(old)
                except OSError:
                    pass
            self._spool_dir = os.path.join(frames_dir, "_spool")
            shutil.rmtree(self._spool_dir, ignore_errors=True)
            os.makedirs(self._spool_dir, exist_ok=True)

    # ------------------------------------------------------------------ input
    def feed(self, t: float, frame: np.ndarray) -> None:
        """One decoded frame. Never raises."""
        if self._finished or frame is None or getattr(frame, "ndim", 0) != 3:
            return
        try:
            self._motion.push(t, frame)
            if self._hand is not None:
                self._hand.push(t, frame)
            if self._spool_dir and t - self._last_spool_t >= self.cfg.spool_interval:
                small = _resize_to_width(frame, self.cfg.frame_max_width)
                path = os.path.join(self._spool_dir, f"{t:09.2f}.jpg")
                if cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]):
                    self._spooled.append(t)
                    self._last_spool_t = t
            self.frames_fed += 1
            self.last_t = max(self.last_t, t)
        except Exception:
            self.frames_failed += 1

    # ----------------------------------------------------------------- output
    def finish(self) -> List[Interaction]:
        """Build the events, pick their pictures, write interactions.json (if out_path) and return them."""
        if self._finished:
            return []
        self._finished = True
        cfg = self.cfg
        steps = self._motion.steps
        if cfg.steps_csv:
            with open(cfg.steps_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "label", "changed_frac", "new_frac", "gone_frac", "n_blobs", "largest_blob_frac", "cx", "cy"])
                for s in steps:
                    w.writerow([round(s.t, 2), classify.label_step(s, cfg), round(s.changed_frac, 5), round(s.new_frac, 5),
                                round(s.gone_frac, 5), s.n_blobs, round(s.largest_blob_frac, 5), round(s.cx, 3), round(s.cy, 3)])
        interactions = classify.build_segments(steps, cfg)
        if self.verbose:
            print(f"[interaction] {self.frames_fed} frames ({self.frames_failed} failed) -> {len(steps)} steps -> {len(interactions)} motion segments")
        if self._hand is not None:
            self._hand.close()
            hand_events = self._hand.interactions()
            if self.verbose:
                print(f"[interaction] hand pointing: {len(self._hand.hits)} fingertip hits -> {len(hand_events)} events")
            interactions += hand_events
        interactions.sort(key=lambda s: (s.start, s.end))

        if self._spool_dir:
            self._pick_frames(interactions)
            shutil.rmtree(self._spool_dir, ignore_errors=True)
        if self.out_path:
            dt = (steps[1].t - steps[0].t) if len(steps) > 1 else 0.0
            save_interactions(self.out_path, interactions, video=self.source_label, duration=round(self.last_t + dt, 2))
        return interactions

    def _pick_frames(self, interactions: List[Interaction]) -> None:
        if not interactions or not self._spooled:
            return
        frames_dir = os.path.dirname(self._spool_dir)
        times = np.asarray(self._spooled)
        for k, seg in enumerate(interactions):
            ft = _frame_time(seg)
            nearest = float(times[int(np.argmin(np.abs(times - ft)))])
            src = os.path.join(self._spool_dir, f"{nearest:09.2f}.jpg")
            if not os.path.exists(src):
                continue
            dst = os.path.join(frames_dir, f"{k:03d}_{seg.type}_{nearest:07.2f}.jpg")
            shutil.copyfile(src, dst)
            seg.frame_path = dst
