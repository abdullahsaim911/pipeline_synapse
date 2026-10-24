"""
YouTube search service for Synapse API.

Searches for public, embeddable YouTube videos using YouTube Data API v3,
fetches exact contentDetails (duration) via videos.list, filters out
unsupported videos longer than 600 seconds (10 minutes), and normalizes
the result objects.

Includes a bounded in-memory LRU/TTL cache to reduce repeat requests.
API keys are never logged, leaked in exceptions, or exposed in responses.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import requests

from api.schemas import VideoSearchResult

logger = logging.getLogger(__name__)

# ISO 8601 duration parser (e.g., PT4M13S, PT1H2M3S, PT45S, P1DT2H)
ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

MAX_VIDEO_DURATION_SECONDS = 600  # 10 minutes maximum supported lecture length
DEFAULT_CACHE_MAX_SIZE = 128
DEFAULT_CACHE_TTL_SECONDS = 300.0  # 5 minutes
DEFAULT_TIMEOUT_SECONDS = 10.0


def parse_iso8601_duration(duration_str: str) -> int:
    """Parse an ISO 8601 duration string into integer seconds.

    Returns 0 if duration_str is empty or does not match.
    """
    if not duration_str or not isinstance(duration_str, str):
        return 0
    match = ISO8601_DURATION_RE.match(duration_str.strip())
    if not match:
        return 0
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_duration_seconds(seconds: int) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS string."""
    if seconds < 0:
        return "00:00"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# Exceptions
# ============================================================================


class YouTubeSearchError(Exception):
    """Base exception for YouTube search service."""


class YouTubeSearchNotConfiguredError(YouTubeSearchError):
    """The service is not configured (e.g. missing API key)."""


class YouTubeSearchTimeoutError(YouTubeSearchError):
    """The search provider timed out."""


class YouTubeSearchRateLimitError(YouTubeSearchError):
    """The YouTube API rate limit or quota was exceeded."""


class YouTubeSearchUnavailableError(YouTubeSearchError):
    """The YouTube search provider is unreachable or failed."""


class YouTubeSearchProviderError(YouTubeSearchError):
    """The YouTube API provider rejected the request."""


# ============================================================================
# Cache
# ============================================================================


class SearchCache:
    """Thread-safe bounded in-memory LRU cache with TTL expiration."""

    def __init__(self, max_size: int = DEFAULT_CACHE_MAX_SIZE, ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS):
        self._max_size = max(1, max_size)
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, List[VideoSearchResult]]] = OrderedDict()

    def get(self, query: str) -> Optional[List[VideoSearchResult]]:
        """Return cached search results if present and unexpired."""
        norm_key = query.strip().lower()
        now = time.monotonic()
        with self._lock:
            if norm_key not in self._cache:
                return None
            ts, results = self._cache[norm_key]
            if now - ts > self._ttl_seconds:
                del self._cache[norm_key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(norm_key)
            return [r.model_copy() for r in results]

    def set(self, query: str, results: List[VideoSearchResult]) -> None:
        """Store search results in the cache."""
        norm_key = query.strip().lower()
        now = time.monotonic()
        with self._lock:
            if norm_key in self._cache:
                del self._cache[norm_key]
            elif len(self._cache) >= self._max_size:
                # Evict oldest entry
                self._cache.popitem(last=False)
            self._cache[norm_key] = (now, [r.model_copy() for r in results])

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ============================================================================
# Service
# ============================================================================


class YouTubeSearchService:
    """Official YouTube Data API v3 search service."""

    SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cache: Optional[SearchCache] = None,
        session: Optional[requests.Session] = None,
    ):
        self._explicit_key = api_key
        self._timeout_seconds = timeout_seconds
        self._cache = cache if cache is not None else SearchCache()
        self._session = session or requests.Session()

    def _get_api_key(self) -> str:
        """Retrieve the API key from constructor or environment."""
        key = self._explicit_key or os.environ.get("YOUTUBE_API_KEY")
        if not key or not key.strip():
            raise YouTubeSearchNotConfiguredError("YouTube search is not configured (missing YOUTUBE_API_KEY).")
        return key.strip()

    def _validate_query(self, query: str) -> str:
        """Validate search query length."""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")
        clean = query.strip()
        if len(clean) > 200:
            raise ValueError("Search query exceeds maximum length of 200 characters.")
        return clean

    def search(self, query: str) -> List[VideoSearchResult]:
        """Search for public, embeddable videos with duration <= 600s."""
        clean_query = self._validate_query(query)

        # 1. Check cache
        cached = self._cache.get(clean_query)
        if cached is not None:
            logger.debug("YouTube search cache hit for query: %s", clean_query)
            return cached

        # 2. Get API key
        api_key = self._get_api_key()

        # 3. Call search.list
        video_ids = self._fetch_search_ids(clean_query, api_key)
        if not video_ids:
            self._cache.set(clean_query, [])
            return []

        # 4. Call videos.list to get exact duration, privacy, and embed status
        video_items = self._fetch_video_details(video_ids, api_key)

        # 5. Filter and normalize
        normalized_results = self._normalize_and_filter(video_items, video_ids)

        # 6. Cache and return
        self._cache.set(clean_query, normalized_results)
        return normalized_results

    def _fetch_search_ids(self, query: str, api_key: str) -> List[str]:
        """Fetch candidate video IDs from YouTube search.list endpoint."""
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoEmbeddable": "true",
            "maxResults": 25,
            "key": api_key,
        }

        try:
            resp = self._session.get(
                self.SEARCH_ENDPOINT,
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            logger.warning("YouTube search request timed out")
            raise YouTubeSearchTimeoutError("YouTube search request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            logger.warning("YouTube search connection failed: %s", type(exc).__name__)
            raise YouTubeSearchUnavailableError("YouTube search provider is unavailable.") from exc

        self._handle_http_status(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise YouTubeSearchProviderError("Malformed JSON response from YouTube search API.") from exc

        items = data.get("items", [])
        video_ids: List[str] = []
        for it in items:
            vid = it.get("id", {}).get("videoId")
            if vid and isinstance(vid, str):
                video_ids.append(vid)

        return video_ids

    def _fetch_video_details(self, video_ids: List[str], api_key: str) -> List[Dict[str, Any]]:
        """Fetch details (contentDetails, snippet, status) via videos.list."""
        if not video_ids:
            return []

        params = {
            "part": "contentDetails,snippet,status",
            "id": ",".join(video_ids),
            "key": api_key,
        }

        try:
            resp = self._session.get(
                self.VIDEOS_ENDPOINT,
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            logger.warning("YouTube video details request timed out")
            raise YouTubeSearchTimeoutError("YouTube video details request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            logger.warning("YouTube video details connection failed: %s", type(exc).__name__)
            raise YouTubeSearchUnavailableError("YouTube search provider is unavailable.") from exc

        self._handle_http_status(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise YouTubeSearchProviderError("Malformed JSON response from YouTube videos API.") from exc

        return data.get("items", [])

    def _handle_http_status(self, response: requests.Response) -> None:
        """Handle error HTTP status codes without leaking credentials."""
        status_code = response.status_code
        if 200 <= status_code < 300:
            return

        error_reason = ""
        try:
            error_json = response.json()
            errors = error_json.get("error", {}).get("errors", [])
            if errors and isinstance(errors, list):
                error_reason = errors[0].get("reason", "")
        except Exception:
            pass

        if status_code == 403:
            if error_reason in ("quotaExceeded", "rateLimitExceeded", "userRateLimitExceeded"):
                logger.warning("YouTube API rate limit or quota exceeded: %s", error_reason)
                raise YouTubeSearchRateLimitError("YouTube search quota exceeded or rate limited.")
            logger.warning("YouTube API authorization failure: %s", error_reason or "forbidden")
            raise YouTubeSearchNotConfiguredError("YouTube API key is invalid or unauthorized.")

        if status_code == 400:
            logger.warning("YouTube API bad request: %s", error_reason)
            raise YouTubeSearchProviderError("YouTube search provider rejected request.")

        if status_code in (429,):
            raise YouTubeSearchRateLimitError("YouTube search rate limit exceeded.")

        if status_code >= 500:
            logger.warning("YouTube API server error: status %d", status_code)
            raise YouTubeSearchUnavailableError("YouTube search provider error.")

        raise YouTubeSearchProviderError(f"YouTube search provider error (status {status_code}).")

    def _normalize_and_filter(
        self,
        video_items: List[Dict[str, Any]],
        requested_order: List[str],
    ) -> List[VideoSearchResult]:
        """Filter videos by duration, privacy, and embeddability, and normalize results."""
        item_map: Dict[str, Dict[str, Any]] = {}
        for it in video_items:
            vid = it.get("id")
            if vid:
                item_map[vid] = it

        results: List[VideoSearchResult] = []

        # Preserve relevance ranking from search.list
        for vid in requested_order:
            item = item_map.get(vid)
            if not item:
                continue

            status = item.get("status", {})
            privacy_status = status.get("privacyStatus", "public")
            embeddable = status.get("embeddable", True)

            # Must be public and embeddable
            if privacy_status != "public" or not embeddable:
                continue

            content_details = item.get("contentDetails", {})
            duration_iso = content_details.get("duration", "")
            duration_seconds = parse_iso8601_duration(duration_iso)

            # Exclude videos longer than 600s or invalid duration
            if duration_seconds <= 0 or duration_seconds > MAX_VIDEO_DURATION_SECONDS:
                continue

            snippet = item.get("snippet", {})
            title = snippet.get("title", f"Video {vid}")
            channel_name = snippet.get("channelTitle", "Unknown Channel")
            thumbnails = snippet.get("thumbnails", {})

            # Prefer medium thumbnail, fallback to high, standard, default
            thumbnail_url = (
                thumbnails.get("medium", {}).get("url")
                or thumbnails.get("high", {}).get("url")
                or thumbnails.get("standard", {}).get("url")
                or thumbnails.get("default", {}).get("url")
            )

            canonical_url = f"https://www.youtube.com/watch?v={vid}"

            results.append(
                VideoSearchResult(
                    video_id=vid,
                    title=title,
                    channel_name=channel_name,
                    thumbnail_url=thumbnail_url,
                    duration_seconds=duration_seconds,
                    duration_formatted=format_duration_seconds(duration_seconds),
                    canonical_url=canonical_url,
                )
            )

        return results


# Global singleton service
_youtube_search_service: Optional[YouTubeSearchService] = None


def get_youtube_search_service() -> YouTubeSearchService:
    """Dependency provider for YouTubeSearchService."""
    global _youtube_search_service
    if _youtube_search_service is None:
        _youtube_search_service = YouTubeSearchService()
    return _youtube_search_service

