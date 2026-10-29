"""SYNAPSE Visual Interaction module (HAND_POINTING / MOUSE_POINTING / WRITING_DRAWING).

Sidecar design: receives the frame extractor's frames (InteractionStream), writes interactions.json, and (optionally) fuses
that file into M1's intervention list WITHOUT changing any existing module's logic.
"""
from .schema import (Interaction, HAND_POINTING, MOUSE_POINTING, WRITING_DRAWING,
                     save_interactions, load_interactions)
from .config import Config
from .detector import detect_interactions
from .stream import InteractionStream
from .fusion import fuse_interactions

__all__ = ["Interaction", "Config", "detect_interactions", "InteractionStream", "fuse_interactions",
           "save_interactions", "load_interactions",
           "HAND_POINTING", "MOUSE_POINTING", "WRITING_DRAWING"]
