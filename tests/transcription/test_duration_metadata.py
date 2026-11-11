from unittest import mock

import pytest

from transcription.transcriber.api_fetcher import TranscriptEntry
from transcription.transcriber.transcription_engine import TranscriptionEngine


def _engine(entries, *, api=True):
    engine = object.__new__(TranscriptionEngine)
    engine.progress_callback = lambda _message: None
    engine.prefer_streaming = False
    engine.api_fetcher = mock.Mock(fetch=mock.Mock(return_value=entries))
    engine.whisper_runner = mock.Mock(transcribe=mock.Mock(return_value=entries))
    return engine


def test_api_duration_is_media_timeline_not_fetch_latency():
    entries = [
        TranscriptEntry(0.0, 2.0, "first"),
        TranscriptEntry(427.0, 2.0, "last"),
    ]
    engine = _engine(entries)
    with mock.patch(
        "transcription.transcriber.transcription_engine.time.perf_counter",
        side_effect=[10.0, 11.4],
    ):
        result = engine.transcribe("https://youtu.be/abcdefghijk", "abcdefghijk")
    assert result.duration_seconds == 429.0
    assert result.processing_duration_seconds == pytest.approx(1.4)
    assert result.to_dict()["processing_duration_seconds"] == pytest.approx(1.4)


def test_whisper_duration_is_media_timeline_not_processing_latency():
    entries = [TranscriptEntry(3.0, 12.5, "speech")]
    engine = _engine(entries)
    with mock.patch(
        "transcription.transcriber.transcription_engine.time.perf_counter",
        side_effect=[20.0, 24.0],
    ):
        result = engine.transcribe("audio.mp3", "unknown", skip_api=True)
    assert result.duration_seconds == 15.5
    assert result.processing_duration_seconds == 4.0
