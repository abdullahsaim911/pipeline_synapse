from __future__ import annotations
from typing import List

import cv2

from .schema import Interaction
from .video_io import iter_frames, probe

COLORS = {"WRITING_DRAWING": (0, 200, 0), "MOUSE_POINTING": (0, 165, 255), "HAND_POINTING": (255, 120, 0)}


def render_debug(stream: str, interactions: List[Interaction], out_path: str,
                 fps: float = 10.0, width: int = 640) -> None:
    writer = None
    for t, fr in iter_frames(stream, fps, width):
        if writer is None:
            h, w = fr.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        h, w = fr.shape[:2]
        y = 20
        for s in interactions:
            if s.start <= t <= s.end:
                col = COLORS.get(s.type, (255, 255, 255))
                if s.bbox:
                    cv2.rectangle(fr, (int(s.bbox[0] * w), int(s.bbox[1] * h)),
                                  (int(s.bbox[2] * w), int(s.bbox[3] * h)), col, 2)
                cv2.putText(fr, f"{s.type} {s.confidence:.2f}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                y += 22
        cv2.putText(fr, f"t={t:6.2f}s", (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(fr)
    if writer:
        writer.release()
