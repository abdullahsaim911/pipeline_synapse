"""Offline REL-011 retry classification and fallback tests."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tts_engine.exceptions import (
    TTSEngineError,
    TTSPermanentError,
    TTSTimeoutError,
)
from tts_engine.providers import EdgeTTSProvider
from tts_engine.tts_engine import TTSEngine


class FakeProvider:
    def __init__(self, name, effects):
        self._name = name
        self.effects = list(effects)
        self.calls = 0

    @property
    def name(self):
        return self._name

    def generate(self, _text, output_path):
        effect = self.effects[min(self.calls, len(self.effects) - 1)]
        self.calls += 1
        if isinstance(effect, Exception):
            raise effect
        Path(output_path).write_bytes(b"ID3-mock")
        return output_path


class TTSReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = str(Path(self.temp.name) / "audio.mp3")

    @mock.patch("tts_engine.providers.edge_tts.Communicate")
    @mock.patch("tts_engine.providers.time.sleep")
    def test_transient_edge_failure_retries_once_then_succeeds(self, sleep, communicate):
        provider = EdgeTTSProvider(retry_backoff_seconds=0.01)
        provider._run_async = mock.Mock(side_effect=[TimeoutError(), None])
        result = provider.generate("hello", self.output)
        self.assertEqual(result, self.output)
        self.assertEqual(communicate.call_count, 2)
        sleep.assert_called_once_with(0.01)

    @mock.patch("tts_engine.providers.edge_tts.Communicate")
    @mock.patch("tts_engine.providers.time.sleep")
    def test_permanent_edge_failure_is_not_retried(self, sleep, communicate):
        provider = EdgeTTSProvider(retry_backoff_seconds=0.01)
        communicate.side_effect = ValueError("invalid voice with secret payload")
        with self.assertRaises(TTSPermanentError) as raised:
            provider.generate("hello", self.output)
        self.assertEqual(raised.exception.code, "tts_invalid_request")
        self.assertNotIn("secret payload", str(raised.exception))
        self.assertEqual(communicate.call_count, 1)
        sleep.assert_not_called()

    @mock.patch("tts_engine.providers.edge_tts.Communicate")
    @mock.patch("tts_engine.providers.time.sleep")
    def test_invalid_output_failure_is_not_retried(self, sleep, communicate):
        provider = EdgeTTSProvider(retry_backoff_seconds=0.01)
        provider._run_async = mock.Mock(
            side_effect=PermissionError("private output path")
        )
        with self.assertRaises(TTSPermanentError) as raised:
            provider.generate("hello", self.output)
        self.assertEqual(raised.exception.code, "tts_invalid_output")
        self.assertNotIn("private output path", str(raised.exception))
        self.assertEqual(communicate.call_count, 1)
        sleep.assert_not_called()

    @mock.patch("tts_engine.providers.edge_tts.Communicate")
    @mock.patch("tts_engine.providers.time.sleep")
    def test_timeout_attempts_and_backoff_are_bounded(self, sleep, communicate):
        provider = EdgeTTSProvider(max_attempts=2, retry_backoff_seconds=0.02)
        provider._run_async = mock.Mock(side_effect=TimeoutError())
        with self.assertRaises(TTSTimeoutError):
            provider.generate("hello", self.output)
        self.assertEqual(communicate.call_count, 2)
        sleep.assert_called_once_with(0.02)

    def test_permanent_primary_failure_uses_existing_fallback_once(self):
        primary = FakeProvider("edge", [TTSPermanentError(
            "rejected", code="tts_request_rejected"
        )])
        fallback = FakeProvider("speecht5", [self.output])
        engine = TTSEngine.__new__(TTSEngine)
        engine._providers = [primary, fallback]
        engine._last_provider_used = None
        result = engine.generate("hello", self.output)
        self.assertEqual(result, self.output)
        self.assertEqual((primary.calls, fallback.calls), (1, 1))
        self.assertEqual(engine.last_provider_used, "speecht5")

    def test_unexpected_programming_error_is_not_called_transient(self):
        primary = FakeProvider("edge", [AttributeError("bug details")])
        fallback = FakeProvider("speecht5", [self.output])
        engine = TTSEngine.__new__(TTSEngine)
        engine._providers = [primary, fallback]
        engine._last_provider_used = None
        with self.assertLogs("tts_engine.tts_engine", level="ERROR") as captured:
            self.assertEqual(engine.generate("hello", self.output), self.output)
        self.assertEqual((primary.calls, fallback.calls), (1, 1))
        self.assertNotIn("bug details", "\n".join(captured.output))

    def test_all_provider_failure_returns_sanitized_classification(self):
        provider = FakeProvider("edge", [TTSPermanentError(
            "credential payload", code="tts_request_rejected"
        )])
        engine = TTSEngine.__new__(TTSEngine)
        engine._providers = [provider]
        engine._last_provider_used = None
        with self.assertRaises(TTSEngineError) as raised:
            engine.generate("hello", self.output)
        self.assertEqual(raised.exception.code, "tts_request_rejected")
        self.assertNotIn("credential payload", str(raised.exception))
        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
