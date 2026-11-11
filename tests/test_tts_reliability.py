"""Offline REL-011 retry classification and fallback tests."""

import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tts_engine.exceptions import (
    TTSEngineError,
    TTSPermanentError,
    TTSTimeoutError,
)
from tts_engine.providers import EdgeTTSProvider, SpeechT5Provider
from tts_engine import tts_engine as tts_module
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
        previous = EdgeTTSProvider._available_voices
        EdgeTTSProvider._available_voices = {
            "en-US-AndrewMultilingualNeural", "en-US-AriaNeural"
        }
        self.addCleanup(
            setattr, EdgeTTSProvider, "_available_voices", previous
        )

    def test_default_speecht5_fallback_uses_cpu(self):
        with mock.patch.dict(os.environ, {"SYNAPSE_TTS_DEVICE": "cpu"}):
            engine = TTSEngine()
        fallback = next(provider for provider in engine._providers if provider.name.startswith("SpeechT5"))
        self.assertEqual(fallback.device, "cpu")

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

    @mock.patch("tts_engine.providers.edge_tts.Communicate")
    def test_unavailable_preferred_voice_uses_stable_edge_fallback(self, communicate):
        EdgeTTSProvider._available_voices = {"en-US-AriaNeural"}
        provider = EdgeTTSProvider(voice="en-US-MissingNeural")
        provider._run_async = mock.Mock(return_value=None)
        provider.generate("hello", self.output)
        communicate.assert_called_once_with("hello", "en-US-AriaNeural")

    def test_long_speecht5_text_is_split_without_dropping_words(self):
        provider = SpeechT5Provider(device="cpu")

        class Tokenizer:
            def __call__(self, text, **_kwargs):
                return {"input_ids": text.split()}

        provider._processor = mock.Mock(tokenizer=Tokenizer())
        text = " ".join(f"word-{index}" for index in range(1300))
        chunks = provider._split_chunks(text)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(provider._token_count(chunk) <= 540 for chunk in chunks))

    def test_speecht5_joins_every_chunk_and_normalizes_once(self):
        provider = SpeechT5Provider(device="cpu")
        provider._load_models = mock.Mock()
        provider._split_chunks = mock.Mock(return_value=["first", "second"])

        class Batch(dict):
            def to(self, _device):
                return self

        provider._processor = mock.Mock(return_value=Batch(input_ids=torch.tensor([[1]])))
        provider._model = mock.Mock(generate=mock.Mock(return_value=torch.ones(1, 2)))
        provider._vocoder = mock.Mock(side_effect=[torch.tensor([[2.0, -2.0]]), torch.tensor([[1.0]])])
        provider._speaker_embeddings = torch.zeros(1, 512)
        written = {}

        def capture_write(path, audio, samplerate):
            written.update(path=path, audio=audio, samplerate=samplerate)

        with mock.patch.object(provider, "is_available", return_value=True), mock.patch(
            "tts_engine.providers.sf.write", side_effect=capture_write
        ):
            provider.generate("long text", self.output)

        expected_silence = int(16000 * provider.CHUNK_SILENCE_SECONDS)
        self.assertEqual(len(written["audio"]), 2 + expected_silence + 1)
        self.assertEqual(written["samplerate"], 16000)
        self.assertAlmostEqual(float(np.max(np.abs(written["audio"]))), 0.9)

    def test_shared_engine_is_reused_and_unloaded_only_at_shutdown(self):
        previous = tts_module._shared_engine
        tts_module._shared_engine = None
        self.addCleanup(setattr, tts_module, "_shared_engine", previous)
        fake = mock.Mock()
        with mock.patch.object(tts_module, "TTSEngine", return_value=fake) as factory:
            first = tts_module.get_shared_tts_engine()
            second = tts_module.get_shared_tts_engine()
            self.assertIs(first, second)
            factory.assert_called_once_with()
            fake.cleanup.assert_not_called()
            tts_module.shutdown_shared_tts_engine()
            fake.cleanup.assert_called_once_with()
            self.assertIsNone(tts_module._shared_engine)

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
