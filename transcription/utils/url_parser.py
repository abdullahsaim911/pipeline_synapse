"""
URL Parser Utilities

Extracts video IDs from various YouTube URL formats.
"""

import re
from typing import Optional


YOUTUBE_PATTERNS = [
    # Standard format: https://www.youtube.com/watch?v=VIDEO_ID
    r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
    # Short format: youtu.be/VIDEO_ID
    r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
]


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from a URL.

    Args:
        url: YouTube video URL in any format

    Returns:
        11-character video ID, or None if not found

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url:
        return None

    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube URL.

    Args:
        url: URL to check

    Returns:
        True if YouTube URL, False otherwise
    """
    return extract_video_id(url) is not None


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to MM:SS timestamp string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp (e.g., "03:45")
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_with_ms(seconds: float) -> str:
    """
    Format seconds to MM:SS.mmm timestamp string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp with milliseconds (e.g., "03:45.123")
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"
