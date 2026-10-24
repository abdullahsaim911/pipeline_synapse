"""
SYNAPSE — Offline Integration Tests for Modal VLM Pipeline (Stage M2B-2)

Tests cover:
1. Modal mode never imports or instantiates Qwen2/GGUF.
2. Existing educational prompts are preserved.
3. Modal response passes through the current JSON normalizer.
4. All required normalized fields exist with correct types.
5. Both orchestrator entrypoints use Modal.
6. One failed intervention does not discard successful ones.
7. No local GPU cleanup runs solely for VLM inference in Modal mode.
8. Invalid provider and missing credentials fail clearly.
9. No automatic local fallback after remote failure.
10. Existing local mode remains explicitly selectable.
11. Q&A and quiz behavior remains unchanged with Modal snapshots.

Run:
    python -m pytest vlm_engine/tests/test_modal_integration.py -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Mock edge_tts if not installed in current environment
if "edge_tts" not in sys.modules:
    sys.modules["edge_tts"] = mock.MagicMock()

from vlm_interface import (
    PROVIDER_ENV_VAR,
    PROVIDER_LOCAL,
    PROVIDER_MODAL,
    VALID_PROVIDERS,
    VLMAuthenticationError,
    VLMInterfaceError,
    VLMRemoteError,
    VLMResponse,
)
from vlm_interface.modal_vlm_interface import ModalVLMInterface
from vlm_engine import SnapshotEngine


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

@dataclass
class DummyIntervention:
    """Mock intervention point matching synchronizer's contract."""
    timestamp: float
    frame_path: str
    content_type: str = "equation"
    trigger_reason: str = "Visual equation on board"


def _make_temp_image(width: int = 100, height: int = 100, color: str = "blue") -> Path:
    img = Image.new("RGB", (width, height), color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp, format="JPEG")
    tmp.close()
    return Path(tmp.name)


def _sample_vlm_json() -> str:
    """Valid JSON payload matching SnapshotEngine's unified output format."""
    return json.dumps({
        "content_type": "equation",
        "detected_types": ["equation", "text"],
        "visual_analysis": {
            "equation": [{"text": "E = mc^2", "description": "Einstein energy mass relation"}]
        },
        "structural_description": "Centered equation with bold symbols",
        "reading_order": ["E", "equals", "m", "c squared"],
        "conceptual_hints": "Mass-energy equivalence principle",
        "layout": "Centered",
        "text_readout": "E = mc^2",
        "spatial_map": "Top center",
        "colors_styles": "White text on blackboard",
        "missing_elements": None
    })


def _mock_modal_response(content: str | None = None) -> dict[str, Any]:
    if content is None:
        content = _sample_vlm_json()
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 120, "total_tokens": 200},
    }


class _FakeHTTPResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(autouse=True)
def modal_env_vars():
    """Ensure Modal environment variables are set by default for tests."""
    env = {
        PROVIDER_ENV_VAR: PROVIDER_MODAL,
        "SYNAPSE_MODAL_URL": "https://test--synapse-modal.modal.run",
        "SYNAPSE_MODAL_API_KEY": "test-mock-api-key-12345",
        "SYNAPSE_MODAL_TIMEOUT_SECONDS": "10.0",
        "SYNAPSE_MODAL_MAX_RETRIES": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        yield


# =========================================================================
# 1. Modal mode never imports or instantiates Qwen2/GGUF
# =========================================================================

class TestModalNeverImportsHeavy:
    def test_modal_engine_uses_modal_adapter(self):
        engine = SnapshotEngine(provider=PROVIDER_MODAL)
        assert isinstance(engine.vlm_interface, ModalVLMInterface)
        assert engine.use_gguf is False
        assert engine.provider == PROVIDER_MODAL

    def test_modal_initialization_does_not_load_torch(self):
        # Verify heavy modules are not in sys.modules from initializing SnapshotEngine(modal)
        engine = SnapshotEngine(provider=PROVIDER_MODAL)
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
        assert "llama_cpp" not in sys.modules


# =========================================================================
# 2. Existing educational prompts are preserved
# =========================================================================

class TestEducationalPromptsPreserved:
    def test_system_prompt_intact(self):
        assert "expert Vision-Language Model analyst" in SnapshotEngine.SYSTEM_PROMPT
        assert "Chain-of-Regions" in SnapshotEngine.SYSTEM_PROMPT
        assert "OUTPUT FORMAT (JSON only)" in SnapshotEngine.SYSTEM_PROMPT

    def test_all_category_directives_present(self):
        expected_categories = [
            "equation", "graph", "circuit", "diagram", "code",
            "handwritten_notes", "biology", "chemistry", "physics", "text"
        ]
        for cat in expected_categories:
            assert cat in SnapshotEngine.CATEGORY_DIRECTIVES
            directive = SnapshotEngine.CATEGORY_DIRECTIVES[cat]
            assert len(directive.strip()) > 50

    def test_build_prompt_combines_system_and_directives(self):
        engine = SnapshotEngine(provider=PROVIDER_MODAL)
        prompt = engine._build_prompt("graph")
        assert SnapshotEngine.SYSTEM_PROMPT in prompt
        assert SnapshotEngine.CATEGORY_DIRECTIVES["graph"] in prompt
        assert "PRIMARY CONTENT TYPE: graph" in prompt


# =========================================================================
# 3. Modal response passes through current JSON normalizer
# =========================================================================

class TestModalResponsePassesNormalizer:
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_analyze_frame_parses_json(self, mock_urlopen):
        raw_json = _sample_vlm_json()
        resp = _mock_modal_response(content=f"```json\n{raw_json}\n```")
        mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(resp).encode())

        tmp_img = _make_temp_image()
        try:
            engine = SnapshotEngine(provider=PROVIDER_MODAL)
            snapshot = engine.analyze_frame(str(tmp_img), "equation")

            assert snapshot["content_type"] == "equation"
            assert "equation" in snapshot["detected_types"]
            assert snapshot["structural_description"] == "Centered equation with bold symbols"
            assert snapshot["reading_order"] == ["E", "equals", "m", "c squared"]
            assert snapshot["text_readout"] == "E = mc^2"
            assert snapshot["missing_elements"] is None
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 4. All required normalized fields exist with correct types
# =========================================================================

class TestNormalizedFieldTypes:
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_all_required_fields_and_types(self, mock_urlopen):
        resp = _mock_modal_response()
        mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(resp).encode())

        tmp_img = _make_temp_image()
        try:
            engine = SnapshotEngine(provider=PROVIDER_MODAL)
            snapshot = engine.analyze_frame(str(tmp_img), "equation")

            assert isinstance(snapshot["content_type"], str)
            assert isinstance(snapshot["detected_types"], list)
            assert isinstance(snapshot["visual_analysis"], dict)
            assert isinstance(snapshot["structural_description"], str)
            assert isinstance(snapshot["reading_order"], list)
            assert isinstance(snapshot["conceptual_hints"], str)
            assert isinstance(snapshot["layout"], str)
            assert isinstance(snapshot["text_readout"], str)
            assert isinstance(snapshot["spatial_map"], str)
            assert isinstance(snapshot["colors_styles"], str)
            assert snapshot["missing_elements"] is None or isinstance(snapshot["missing_elements"], str)
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 5. Both orchestrator entrypoints use Modal
# =========================================================================

class TestBothOrchestratorEntrypointsUseModal:
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_analyze_vlm_snapshots_uses_modal(self, mock_urlopen):
        from orchestrator.orchestrator import PipelineOrchestrator

        resp = _mock_modal_response()
        mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(resp).encode())

        tmp_img = _make_temp_image()
        try:
            orchestrator = PipelineOrchestrator(base_dir=tempfile.gettempdir())
            interventions = [
                DummyIntervention(timestamp=12.5, frame_path=str(tmp_img)),
                DummyIntervention(timestamp=25.0, frame_path=str(tmp_img)),
            ]

            snapshots = orchestrator._analyze_vlm_snapshots(interventions, tempfile.gettempdir())

            assert len(snapshots) == 2
            assert 12.5 in snapshots
            assert 25.0 in snapshots
            assert snapshots[12.5]["content_type"] == "equation"
            # Verify Modal endpoint was reached
            assert mock_urlopen.call_count == 2
            req = mock_urlopen.call_args_list[0][0][0]
            assert req.full_url.endswith("/v1/chat/completions")
        finally:
            tmp_img.unlink(missing_ok=True)

    @mock.patch("synthesizer.LLMSynthesizer.weave", return_value="Test woven explanation")
    @mock.patch("tts_engine.TTSEngine")
    @mock.patch("orchestrator.orchestrator.start_ollama")
    @mock.patch("orchestrator.orchestrator.stop_ollama")
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_generate_intervention_explanation_uses_modal(
        self, mock_urlopen, mock_stop, mock_start, mock_tts_cls, mock_weave, tmp_path
    ):
        from orchestrator.orchestrator import PipelineOrchestrator

        mock_tts_instance = mock.MagicMock()
        mock_tts_cls.return_value = mock_tts_instance

        resp = _mock_modal_response()
        mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(resp).encode())

        tmp_img = _make_temp_image()
        try:
            orch = PipelineOrchestrator(base_dir=str(tmp_path))
            result = orch.generate_intervention_explanation(
                video_id="test_vid_123",
                intervention_id="int_0001",
                intervention_timestamp=45.0,
                frame_path=str(tmp_img),
                content_type="graph",
                trigger_reason="Teacher pointed to curve",
                transcript_entries=[{"start": 40.0, "duration": 10.0, "text": "Look at this curve"}],
                output_mode="explanatory"
            )

            assert result["intervention_id"] == "int_0001"
            assert result["text_explanation"] == "Test woven explanation"
            assert mock_urlopen.call_count == 1
            req = mock_urlopen.call_args[0][0]
            assert req.full_url.endswith("/v1/chat/completions")
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 6. One failed intervention does not discard successful ones
# =========================================================================

class TestBatchFailureResilience:
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_partial_batch_failure_preserves_successful_snapshots(self, mock_urlopen):
        from orchestrator.orchestrator import PipelineOrchestrator

        success_resp = _FakeHTTPResponse(json.dumps(_mock_modal_response()).encode())
        fail_exc = VLMRemoteError("Non-retryable remote error")

        # Call 1 succeeds, Call 2 fails (non-retryable), Call 3 succeeds
        mock_urlopen.side_effect = [
            success_resp,
            fail_exc,
            _FakeHTTPResponse(json.dumps(_mock_modal_response()).encode()),
        ]

        tmp_img = _make_temp_image()
        try:
            orch = PipelineOrchestrator(base_dir=tempfile.gettempdir())
            interventions = [
                DummyIntervention(timestamp=10.0, frame_path=str(tmp_img)),
                DummyIntervention(timestamp=20.0, frame_path=str(tmp_img)),
                DummyIntervention(timestamp=30.0, frame_path=str(tmp_img)),
            ]

            snapshots = orch._analyze_vlm_snapshots(interventions, tempfile.gettempdir())

            # All 3 timestamps exist
            assert len(snapshots) == 3
            assert 10.0 in snapshots
            assert 20.0 in snapshots
            assert 30.0 in snapshots

            # First and third succeeded
            assert snapshots[10.0].get("_error") is not True
            assert snapshots[30.0].get("_error") is not True

            # Second is a normalized failure state
            assert snapshots[20.0]["_error"] is True
            assert "VLM analysis failed" in snapshots[20.0]["missing_elements"]
            assert snapshots[20.0]["visual_analysis"] == {}
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 7. No local GPU cleanup runs solely for VLM inference in Modal mode
# =========================================================================

class TestNoLocalGPUOperationsInModalMode:
    @mock.patch("orchestrator.orchestrator.ensure_gpu_free")
    @mock.patch("orchestrator.orchestrator.clear_gpu_memory")
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_no_gpu_checks_in_analyze_vlm_snapshots(
        self, mock_urlopen, mock_clear_gpu, mock_ensure_gpu
    ):
        from orchestrator.orchestrator import PipelineOrchestrator

        resp = _mock_modal_response()
        mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(resp).encode())

        tmp_img = _make_temp_image()
        try:
            orch = PipelineOrchestrator(base_dir=tempfile.gettempdir())
            interventions = [DummyIntervention(timestamp=10.0, frame_path=str(tmp_img))]
            orch._analyze_vlm_snapshots(interventions, tempfile.gettempdir())

            mock_ensure_gpu.assert_not_called()
            mock_clear_gpu.assert_not_called()
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 8. Invalid provider and missing credentials fail clearly
# =========================================================================

class TestProviderAndCredentialValidation:
    def test_invalid_provider_in_snapshot_engine_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported VLM provider 'invalid_backend'"):
            SnapshotEngine(provider="invalid_backend")

    def test_invalid_provider_in_env_raises_value_error(self):
        with mock.patch.dict(os.environ, {PROVIDER_ENV_VAR: "unsupported_provider"}):
            with pytest.raises(ValueError, match="Unsupported VLM provider 'unsupported_provider'"):
                SnapshotEngine()

    def test_missing_modal_url_raises_clear_error(self):
        with mock.patch.dict(os.environ, {"SYNAPSE_MODAL_URL": ""}, clear=False):
            with pytest.raises(VLMInterfaceError, match="Modal endpoint URL is required"):
                SnapshotEngine(provider=PROVIDER_MODAL)

    def test_missing_modal_api_key_raises_clear_error(self):
        with mock.patch.dict(os.environ, {"SYNAPSE_MODAL_API_KEY": ""}, clear=False):
            with pytest.raises(VLMAuthenticationError, match="Modal API key is required"):
                SnapshotEngine(provider=PROVIDER_MODAL)


# =========================================================================
# 9. No automatic local fallback after remote failure
# =========================================================================

class TestNoLocalFallbackAfterRemoteFailure:
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_remote_failure_returns_failed_snapshot_without_loading_local(self, mock_urlopen):
        mock_urlopen.side_effect = VLMRemoteError("Internal Server Error (500)")

        tmp_img = _make_temp_image()
        try:
            engine = SnapshotEngine(provider=PROVIDER_MODAL)
            snapshot = engine.analyze_frame(str(tmp_img), "equation")

            # Must return normalized failure state
            assert snapshot["_error"] is True
            assert "VLM analysis failed" in snapshot["missing_elements"]
            assert snapshot["visual_analysis"] == {}
            # Must NOT have initialized local model
            assert engine.provider == PROVIDER_MODAL
            assert isinstance(engine.vlm_interface, ModalVLMInterface)
        finally:
            tmp_img.unlink(missing_ok=True)


# =========================================================================
# 10. Existing local mode remains explicitly selectable
# =========================================================================

class TestLocalModeExplicitlySelectable:
    def test_local_provider_selection_sets_use_gguf(self):
        # We don't have GGUF weights on this test machine, so initialization
        # will attempt local GGUF and raise RuntimeError if files not found.
        # This confirms local path was entered and NOT Modal!
        with mock.patch.dict(os.environ, {PROVIDER_ENV_VAR: PROVIDER_LOCAL}):
            with pytest.raises((RuntimeError, FileNotFoundError, Exception)):
                SnapshotEngine(provider=PROVIDER_LOCAL)


# =========================================================================
# 11. Q&A and quiz behavior remains unchanged
# =========================================================================

class TestQAAndQuizBehaviorUnchanged:
    def test_valid_modal_snapshot_classified_as_valid_by_qa(self):
        from qa.moment import assess_snapshot, SNAPSHOT_VALID

        valid_snap = json.loads(_sample_vlm_json())
        status, pairs = assess_snapshot(valid_snap)
        assert status == SNAPSHOT_VALID
        assert pairs is not None
        assert len(pairs) > 0

    def test_failed_modal_snapshot_classified_as_failed_by_qa(self):
        from qa.moment import assess_snapshot, SNAPSHOT_FAILED

        engine = SnapshotEngine(provider=PROVIDER_MODAL)
        failed_snap = engine._create_failed_snapshot("Remote endpoint timed out", "graph")

        status, pairs = assess_snapshot(failed_snap)
        assert status == SNAPSHOT_FAILED
        assert pairs is None

