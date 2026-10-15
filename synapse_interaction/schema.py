from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

HAND_POINTING = "HAND_POINTING"
MOUSE_POINTING = "MOUSE_POINTING"
WRITING_DRAWING = "WRITING_DRAWING"
ALL_TYPES = (HAND_POINTING, MOUSE_POINTING, WRITING_DRAWING)


@dataclass
class Interaction:
    type: str
    start: float                      # seconds
    end: float                        # seconds
    confidence: float                 # 0..1 (heuristic, see detector docs)
    bbox: Optional[List[float]] = None            # [x0, y0, x1, y1], normalised 0..1
    trajectory: List[List[float]] = field(default_factory=list)  # [[t, x, y], ...] normalised
    source: str = ""                  # "motion" | "mediapipe" | ...
    frame_path: Optional[str] = None  # representative frame saved for this segment
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def save_interactions(path: str, interactions: List[Interaction],
                      video: Optional[str] = None, duration: Optional[float] = None) -> None:
    payload = {"video": video, "duration": duration,
               "interactions": [asdict(i) for i in interactions]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_interactions(path: str) -> List[Interaction]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["interactions"] if isinstance(data, dict) else data
    return [Interaction(**i) for i in items]
