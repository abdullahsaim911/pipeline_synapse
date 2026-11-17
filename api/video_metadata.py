"""Resolve lecture titles from the metadata already produced by extraction."""
import json
from pathlib import Path


def resolve_video_title(video_id, title, video_dir):
    """Keep real titles; repair placeholders without requiring a network request."""
    def usable(value):
        return (isinstance(value, str) and value.strip()
                and value.strip() not in {video_id, f"Video {video_id}", "Untitled video"})

    if usable(title):
        return title.strip()
    try:
        metadata = json.loads((Path(video_dir) / "video_meta.json").read_text(encoding="utf-8"))
        candidate = metadata.get("title") if isinstance(metadata, dict) else None
        if usable(candidate):
            return candidate.strip()
    except (OSError, ValueError):
        pass
    return title if isinstance(title, str) and title.strip() else f"Video {video_id}"
