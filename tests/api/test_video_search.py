"""Unit and integration tests for YouTube lecture search service and GET /video/search endpoint."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.video import router as video_router
from api.schemas import VideoSearchResult
from api.youtube_search import (
    DEFAULT_CACHE_TTL_SECONDS,
    SearchCache,
    YouTubeSearchError,
    YouTubeSearchNotConfiguredError,
    YouTubeSearchProviderError,
    YouTubeSearchRateLimitError,
    YouTubeSearchService,
    YouTubeSearchTimeoutError,
    YouTubeSearchUnavailableError,
    format_duration_seconds,
    get_youtube_search_service,
    parse_iso8601_duration,
)


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

def make_sample_search_response(items):
    return {
        "kind": "youtube#searchListResponse",
        "items": items,
    }


def make_sample_videos_response(items):
    return {
        "kind": "youtube#videoListResponse",
        "items": items,
    }


SAMPLE_SEARCH_ITEMS = [
    {
        "id": {"kind": "youtube#video", "videoId": "vid_short_1"},
        "snippet": {
            "title": "Intro to Calculus Part 1",
            "channelTitle": "3Blue1Brown",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_short_1/mqdefault.jpg"}},
        },
    },
    {
        "id": {"kind": "youtube#video", "videoId": "vid_long_2"},
        "snippet": {
            "title": "Full 3-Hour Calculus Course",
            "channelTitle": "MIT OCW",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_long_2/mqdefault.jpg"}},
        },
    },
    {
        "id": {"kind": "youtube#video", "videoId": "vid_short_3"},
        "snippet": {
            "title": "Derivatives in 5 Minutes",
            "channelTitle": "Khan Academy",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_short_3/mqdefault.jpg"}},
        },
    },
    {
        "id": {"kind": "youtube#video", "videoId": "vid_unembeddable_4"},
        "snippet": {
            "title": "Restricted Lecture",
            "channelTitle": "Restricted Channel",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_unembeddable_4/mqdefault.jpg"}},
        },
    },
]

SAMPLE_VIDEO_ITEMS = [
    {
        "id": "vid_short_1",
        "snippet": {
            "title": "Intro to Calculus Part 1",
            "channelTitle": "3Blue1Brown",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_short_1/mqdefault.jpg"}},
        },
        "contentDetails": {"duration": "PT4M13S"},  # 253s <= 600s: KEEP
        "status": {"privacyStatus": "public", "embeddable": True},
    },
    {
        "id": "vid_long_2",
        "snippet": {
            "title": "Full 3-Hour Calculus Course",
            "channelTitle": "MIT OCW",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_long_2/mqdefault.jpg"}},
        },
        "contentDetails": {"duration": "PT1H15M"},  # 4500s > 600s: EXCLUDE
        "status": {"privacyStatus": "public", "embeddable": True},
    },
    {
        "id": "vid_short_3",
        "snippet": {
            "title": "Derivatives in 5 Minutes",
            "channelTitle": "Khan Academy",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_short_3/mqdefault.jpg"}},
        },
        "contentDetails": {"duration": "PT5M0S"},  # 300s <= 600s: KEEP
        "status": {"privacyStatus": "public", "embeddable": True},
    },
    {
        "id": "vid_unembeddable_4",
        "snippet": {
            "title": "Restricted Lecture",
            "channelTitle": "Restricted Channel",
            "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/vid_unembeddable_4/mqdefault.jpg"}},
        },
        "contentDetails": {"duration": "PT2M"},  # 120s <= 600s, but embeddable=False: EXCLUDE
        "status": {"privacyStatus": "public", "embeddable": False},
    },
]


# ============================================================================
# Duration Parsing Tests
# ============================================================================

def test_parse_iso8601_duration():
    assert parse_iso8601_duration("PT4M13S") == 253
    assert parse_iso8601_duration("PT10M") == 600
    assert parse_iso8601_duration("PT10M1S") == 601
    assert parse_iso8601_duration("PT1H") == 3600
    assert parse_iso8601_duration("PT1H2M3S") == 3723
    assert parse_iso8601_duration("PT45S") == 45
    assert parse_iso8601_duration("P1D") == 86400
    assert parse_iso8601_duration("") == 0
    assert parse_iso8601_duration(None) == 0
    assert parse_iso8601_duration("invalid") == 0


def test_format_duration_seconds():
    assert format_duration_seconds(45) == "00:45"
    assert format_duration_seconds(253) == "04:13"
    assert format_duration_seconds(600) == "10:00"
    assert format_duration_seconds(3665) == "1:01:05"


# ============================================================================
# Service Normalization and Filtering Tests
# ============================================================================

def test_service_normalization_and_filtering():
    """Verify durations >600s and unembeddable videos are excluded, and objects are normalized."""
    session = MagicMock()

    resp_search = MagicMock()
    resp_search.status_code = 200
    resp_search.json.return_value = make_sample_search_response(SAMPLE_SEARCH_ITEMS)

    resp_videos = MagicMock()
    resp_videos.status_code = 200
    resp_videos.json.return_value = make_sample_videos_response(SAMPLE_VIDEO_ITEMS)

    session.get.side_effect = [resp_search, resp_videos]

    service = YouTubeSearchService(api_key="TEST_API_KEY", session=session)
    results = service.search("calculus")

    assert len(results) == 2

    # Check first result
    r0 = results[0]
    assert r0.video_id == "vid_short_1"
    assert r0.title == "Intro to Calculus Part 1"
    assert r0.channel_name == "3Blue1Brown"
    assert r0.thumbnail_url == "https://img.youtube.com/vi/vid_short_1/mqdefault.jpg"
    assert r0.duration_seconds == 253
    assert r0.duration_formatted == "04:13"
    assert r0.canonical_url == "https://www.youtube.com/watch?v=vid_short_1"

    # Check second result
    r1 = results[1]
    assert r1.video_id == "vid_short_3"
    assert r1.title == "Derivatives in 5 Minutes"
    assert r1.channel_name == "Khan Academy"
    assert r1.duration_seconds == 300
    assert r1.duration_formatted == "05:00"
    assert r1.canonical_url == "https://www.youtube.com/watch?v=vid_short_3"


def test_service_missing_api_key_raises_not_configured():
    """When no API key is set in constructor or env, raises YouTubeSearchNotConfiguredError."""
    with patch.dict(os.environ, {}, clear=True):
        service = YouTubeSearchService(api_key=None)
        with pytest.raises(YouTubeSearchNotConfiguredError) as exc_info:
            service.search("biology")
        assert "not configured" in str(exc_info.value).lower()
        # Ensure error message does not contain any secret
        assert "key=" not in str(exc_info.value)


def test_service_rate_limit_handling():
    """Verify 403 quotaExceeded maps to YouTubeSearchRateLimitError."""
    session = MagicMock()
    resp_search = MagicMock()
    resp_search.status_code = 403
    resp_search.json.return_value = {
        "error": {
            "errors": [{"reason": "quotaExceeded"}],
            "message": "The request cannot be completed because you have exceeded your quota.",
        }
    }
    session.get.return_value = resp_search

    service = YouTubeSearchService(api_key="FAKE_KEY", session=session)
    with pytest.raises(YouTubeSearchRateLimitError) as exc_info:
        service.search("physics")
    assert "quota" in str(exc_info.value).lower()
    # Ensure key is never in exception string
    assert "FAKE_KEY" not in str(exc_info.value)


def test_service_timeout_handling():
    """Verify requests Timeout maps to YouTubeSearchTimeoutError."""
    session = MagicMock()
    session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    service = YouTubeSearchService(api_key="FAKE_KEY", session=session)
    with pytest.raises(YouTubeSearchTimeoutError) as exc_info:
        service.search("chemistry")
    assert "timed out" in str(exc_info.value).lower()
    assert "FAKE_KEY" not in str(exc_info.value)


def test_service_network_unavailable_handling():
    """Verify requests ConnectionError maps to YouTubeSearchUnavailableError."""
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("Network is down")

    service = YouTubeSearchService(api_key="FAKE_KEY", session=session)
    with pytest.raises(YouTubeSearchUnavailableError) as exc_info:
        service.search("algebra")
    assert "unavailable" in str(exc_info.value).lower()
    assert "FAKE_KEY" not in str(exc_info.value)


def test_service_empty_results_and_malformed_json():
    """Verify empty search results return [] and malformed JSON raises provider error."""
    session = MagicMock()
    resp_empty = MagicMock()
    resp_empty.status_code = 200
    resp_empty.json.return_value = {"items": []}
    session.get.return_value = resp_empty

    service = YouTubeSearchService(api_key="FAKE_KEY", session=session)
    results = service.search("nonexistent query xyz")
    assert results == []

    # Malformed JSON
    resp_malformed = MagicMock()
    resp_malformed.status_code = 200
    resp_malformed.json.side_effect = ValueError("Invalid JSON")
    session.get.return_value = resp_malformed

    service_malformed = YouTubeSearchService(api_key="FAKE_KEY", session=session, cache=SearchCache())
    with pytest.raises(YouTubeSearchProviderError):
        service_malformed.search("malformed query")


def test_service_query_validation():
    """Verify empty or excessively long queries raise ValueError."""
    service = YouTubeSearchService(api_key="FAKE_KEY")
    with pytest.raises(ValueError, match="empty"):
        service.search("")
    with pytest.raises(ValueError, match="empty"):
        service.search("   ")
    with pytest.raises(ValueError, match="maximum length"):
        service.search("a" * 201)


def test_search_cache_hits_and_bounded_eviction():
    """Verify cache avoids repeated provider requests and respects LRU bounds."""
    cache = SearchCache(max_size=2, ttl_seconds=60.0)
    session = MagicMock()

    resp_search = MagicMock()
    resp_search.status_code = 200
    resp_search.json.return_value = make_sample_search_response([SAMPLE_SEARCH_ITEMS[0]])

    resp_videos = MagicMock()
    resp_videos.status_code = 200
    resp_videos.json.return_value = make_sample_videos_response([SAMPLE_VIDEO_ITEMS[0]])

    session.get.side_effect = [resp_search, resp_videos]

    service = YouTubeSearchService(api_key="FAKE_KEY", session=session, cache=cache)

    # First call: hits provider
    res1 = service.search("calculus")
    assert len(res1) == 1
    assert session.get.call_count == 2

    # Second call (case-insensitive & trimmed): hits cache
    res2 = service.search("  CALCULUS  ")
    assert len(res2) == 1
    assert session.get.call_count == 2  # No new requests!

    # Add 2 more queries to trigger eviction of 'calculus'
    session.get.side_effect = [resp_search, resp_videos, resp_search, resp_videos]
    service.search("linear algebra")
    service.search("geometry")
    assert len(cache) == 2

    # 'calculus' was LRU evicted: should query provider again
    session.get.side_effect = [resp_search, resp_videos]
    service.search("calculus")
    assert session.get.call_count == 8


# ============================================================================
# FastAPI Endpoint Integration Tests: GET /video/search
# ============================================================================

def make_test_app(service: YouTubeSearchService):
    app = FastAPI()
    app.include_router(video_router)
    app.dependency_overrides[get_youtube_search_service] = lambda: service
    return app


def test_endpoint_search_success():
    """GET /video/search returns 200 with normalized list of videos."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    mock_service.search.return_value = [
        VideoSearchResult(
            video_id="vid_short_1",
            title="Intro to Calculus",
            channel_name="3Blue1Brown",
            thumbnail_url="https://img.youtube.com/vi/vid_short_1/mqdefault.jpg",
            duration_seconds=253,
            duration_formatted="04:13",
            canonical_url="https://www.youtube.com/watch?v=vid_short_1",
        )
    ]

    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=calculus")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["video_id"] == "vid_short_1"
    assert data[0]["title"] == "Intro to Calculus"
    assert data[0]["channel_name"] == "3Blue1Brown"
    assert data[0]["duration_seconds"] == 253
    assert data[0]["canonical_url"] == "https://www.youtube.com/watch?v=vid_short_1"
    mock_service.search.assert_called_once_with("calculus")


def test_endpoint_search_empty_query_rejected():
    """Empty or whitespace query returns 400 Bad Request."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=   ")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    mock_service.search.assert_not_called()


def test_endpoint_search_missing_key_503():
    """Unconfigured search service returns 503 Service Unavailable."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    mock_service.search.side_effect = YouTubeSearchNotConfiguredError("Missing key")

    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=calculus")
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"].lower()


def test_endpoint_search_rate_limit_429():
    """Rate limit or quota exhaustion returns 429 Too Many Requests."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    mock_service.search.side_effect = YouTubeSearchRateLimitError("Quota exceeded")

    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=calculus")
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower() or "quota" in response.json()["detail"].lower()


def test_endpoint_search_timeout_504():
    """Network timeout returns 504 Gateway Timeout."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    mock_service.search.side_effect = YouTubeSearchTimeoutError("Timed out")

    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=calculus")
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_endpoint_search_provider_unavailable_503():
    """Provider failure returns 503."""
    mock_service = MagicMock(spec=YouTubeSearchService)
    mock_service.search.side_effect = YouTubeSearchUnavailableError("Connection down")

    app = make_test_app(mock_service)
    client = TestClient(app)

    response = client.get("/video/search?q=calculus")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()

