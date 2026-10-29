from __future__ import annotations
from typing import List, Optional

from .config import Config
from .schema import Interaction
from .stream import InteractionStream
from .video_io import iter_frames, resolve_stream


def detect_interactions(source: str, out_path: Optional[str] = None, cfg: Optional[Config] = None,
                        verbose: bool = False) -> List[Interaction]:
    """Video-file entry point for the CLI and tests.

    source: local video path, YouTube URL / video id, or a direct stream URL. The PIPELINE does not use this:
    it feeds the frame extractor's own frames to `InteractionStream` so the video is decoded only once and is
    never downloaded or opened a second time. Both paths run exactly the same detection code.
    """
    cfg = cfg or Config()
    stream = resolve_stream(source, cfg.scan_height)
    runner = InteractionStream(cfg, out_path=out_path, source_label=str(source), verbose=verbose)
    for t, frame in iter_frames(stream, cfg.scan_fps, cfg.scan_width):
        runner.feed(t, frame)
    return runner.finish()
