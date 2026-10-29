import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ---- video access -------------------------------------------------
    scan_fps: float = 4.0            # sampling rate of the motion/ink scan
    scan_width: int = 640            # frames are downscaled to this width in memory
    scan_height: int = 480           # max stream height requested for scanning (URLs only)
    frame_height: int = 720          # max stream height for saved representative frames (URLs only)
    frame_max_width: int = 1280      # saved frames are never wider than this

    # ---- ink / motion scan --------------------------------------------
    ink_thr: int = 24                # colour distance from the background above this counts as "ink"
    min_blob_px: int = 6             # ignore changed blobs smaller than this (noise)
    cut_frac: float = 0.20           # >20% of pixels changed => scene cut / slide change
    min_changed_frac: float = 0.0001 # below this a step is "idle"
    max_blobs: int = 8               # more separate changing blobs => "busy" (animation, video)
    cursor_max_blob_frac: float = 0.02   # cursor-like blobs are small
    write_max_blob_frac: float = 0.08    # a single pen stroke blob stays below this
    write_max_new_frac: float = 0.03     # a hand/body entering adds far more than a stroke
    writing_net_ratio: float = 0.30      # (new-gone)/(new+gone) needed for a "write" step

    # ---- segment building ---------------------------------------------
    write_gap_tol: float = 1.2       # pen pauses up to this long stay one segment
    min_write_dur: float = 0.75
    min_write_net_frac: float = 0.0004   # net ink gained over the segment
    point_gap_tol: float = 0.8
    min_point_dur: float = 0.75
    min_point_path: float = 0.03     # cursor must travel >=3% of frame width/height
    min_conf: float = 0.40           # drop segments below this confidence
    # post-processing: turn many fragments into a few "episodes"
    write_merge_gap: float = 3.0     # merge writing segments closer than this (pen pauses)
    point_merge_gap: float = 2.0     # merge pointing segments closer than this
    absorb_max_dur: float = 3.0      # short pointer move between strokes = part of the writing
    absorb_gap: float = 1.5

    # ---- hand pointing (MediaPipe, optional) --------------------------
    enable_hand: bool = True
    hand_fps: float = 2.0
    hand_width: int = 640
    hand_gap_tol: float = 1.0
    hand_min_dur: float = 0.8
    hand_max_speed: float = 0.35     # fingertip speed limit (frame-fractions / s): waving != pointing
    use_arm_fallback: bool = False   # reserved (not implemented yet)
    # MediaPipe Tasks model (python -m synapse_interaction.get_models). Found relative to this package, not to the
    # working directory, so it works wherever the server is started from.
    hand_model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

    # ---- output --------------------------------------------------------
    save_frames: bool = True
    jpeg_quality: int = 88
    spool_interval: float = 1.0      # frame-fed mode: keep a picture at most this often (seconds) for the event frames
    max_trajectory_points: int = 20
    steps_csv: Optional[str] = None  # if set, per-step features + labels are dumped here (debugging)
