"""
SYNAPSE — Offline Tests for ModalVLMInterface (M2B-1)

Every test uses mocked HTTP responses.  No real network requests are made,
no Modal endpoint is contacted, no model is downloaded, and no CUDA/GPU
code is initialized.

Run:
    python -m pytest vlm_interface/tests/test_modal_vlm_interface.py -v
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import tempfile
import urllib.error
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from PIL import Image

# Ensure project root is importable when running pytest from repo root
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vlm_interface.vlm_types import (
    PROVIDER_ENV_VAR,
    PROVIDER_LOCAL,
    PROVIDER_MODAL,
    VALID_PROVIDERS,
    VLMAuthenticationError,
    VLMInterfaceError,
    VLMRemoteError,
    VLMResponse,
    VLMTimeoutError,
)
from vlm_interface.modal_vlm_interface import (
    ModalVLMInterface,
    _DEFAULT_MODEL_ID,
    _MAX_IMAGE_DIMENSION,
    _redact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str = "Test response text",
    prompt_tokens: int = 100,
    completion_tokens: int = 42,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible chat completion response dict."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": _DEFAULT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _json_bytes(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


def _make_pil_image(width: int = 64, height: int = 64, color: str = "red") -> Image.Image:
    """Create a simple in-memory PIL image."""
    return Image.new("RGB", (width, height), color=color)


def _make_temp_image(
    suffix: str = ".jpg",
    width: int = 64,
    height: int = 64,
    color: str = "blue",
) -> Path:
    """Write a temporary image file and return its path."""
    img = _make_pil_image(width, height, color)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP"}
    img.save(tmp, format=fmt.get(suffix, "JPEG"))
    tmp.close()
    return Path(tmp.name)


class _FakeHTTPResponse:
    """Minimal mock for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture()
def adapter() -> ModalVLMInterface:
    """Return a ModalVLMInterface with test credentials."""
    return ModalVLMInterface(
        endpoint_url="https://test--modal.modal.run",
        api_key="test-secret-key-12345",
        timeout_seconds=5.0,
        max_retries=2,
        max_tokens=512,
    )


# =========================================================================
# 1.  Construction & Configuration
# =========================================================================


class TestConstruction:
    """Test adapter construction and configuration validation."""

    def test_valid_construction(self, adapter: ModalVLMInterface):
        assert adapter.model_name == _DEFAULT_MODEL_ID
        assert adapter._endpoint_url == "https://test--modal.modal.run"

    def test_missing_url_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(VLMInterfaceError, match="endpoint URL is required"):
                ModalVLMInterface(
                    endpoint_url="",
                    api_key="key",
                )

    def test_http_url_rejected(self):
        with pytest.raises(VLMInterfaceError, match="must use HTTPS"):
            ModalVLMInterface(
                endpoint_url="http://insecure.example.com",
                api_key="key",
            )

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(VLMAuthenticationError, match="API key is required"):
                ModalVLMInterface(
                    endpoint_url="https://test.modal.run",
                    api_key="",
                )

    def test_env_var_fallback(self):
        env = {
            "SYNAPSE_MODAL_URL": "https://env-url.modal.run",
            "SYNAPSE_MODAL_API_KEY": "env-key-999",
            "SYNAPSE_MODAL_TIMEOUT_SECONDS": "30",
            "SYNAPSE_MODAL_MAX_RETRIES": "5",
            "SYNAPSE_MODAL_MAX_TOKENS": "256",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            a = ModalVLMInterface()
            assert a._endpoint_url == "https://env-url.modal.run"
            assert a._timeout == 30.0
            assert a._max_retries == 5
            assert a._max_tokens == 256

    def test_custom_model_id(self):
        a = ModalVLMInterface(
            endpoint_url="https://x.modal.run",
            api_key="k",
            model_id="Custom/MyModel",
        )
        assert a.model_name == "Custom/MyModel"

    def test_trailing_slash_stripped(self):
        a = ModalVLMInterface(
            endpoint_url="https://x.modal.run/",
            api_key="k",
        )
        assert not a._endpoint_url.endswith("/")


# =========================================================================
# 2.  URL, Headers and Model Name in Request
# =========================================================================


class TestRequestFormat:
    """Verify correct HTTPS URL, Bearer auth, and model name in the payload."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_correct_endpoint_url(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        img = _make_pil_image()
        adapter.analyze(img, "Describe this.")

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://test--modal.modal.run/v1/chat/completions"

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_bearer_authorization_header(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(_make_pil_image(), "Hello")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer test-secret-key-12345"

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_model_name_in_payload(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(_make_pil_image(), "Hello")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["model"] == _DEFAULT_MODEL_ID

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_content_type_json(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(_make_pil_image(), "Hello")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"


# =========================================================================
# 3.  Image / Prompt Request Format
# =========================================================================


class TestImagePromptFormat:
    """Verify image and prompt are correctly formatted in the request."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_payload_structure(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(_make_pil_image(), "Describe the frame.")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert "messages" in body
        assert len(body["messages"]) == 1
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2

        text_part = msg["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "Describe the frame."

        image_part = msg["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_stream_is_false(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(_make_pil_image(), "Hello")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["stream"] is False

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_generation_params_override(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(
            _make_pil_image(),
            "Hello",
            generation_params={"max_new_tokens": 256, "temperature": 0.7},
        )

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0.7

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_max_tokens_clamped_to_2048(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        adapter.analyze(
            _make_pil_image(),
            "Hello",
            generation_params={"max_new_tokens": 9999},
        )

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["max_tokens"] == 2048


# =========================================================================
# 4.  Image Input Formats (JPEG, PNG, WebP, BMP)
# =========================================================================


class TestImageInputFormats:
    """Accept JPEG, PNG, WebP, BMP inputs — all converted to JPEG payload."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    @pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
    def test_file_format_accepted(self, mock_urlopen, adapter, suffix):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        tmp = _make_temp_image(suffix=suffix)
        try:
            result = adapter.analyze(str(tmp), "Describe this.")
            assert isinstance(result, VLMResponse)
            # Payload image is always JPEG regardless of input
            body = json.loads(mock_urlopen.call_args[0][0].data)
            assert "data:image/jpeg;base64," in body["messages"][0]["content"][1]["image_url"]["url"]
        finally:
            tmp.unlink(missing_ok=True)

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_pil_image_accepted(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        result = adapter.analyze(_make_pil_image(), "Describe this.")
        assert isinstance(result, VLMResponse)

    def test_unsupported_format_rejected(self, adapter):
        tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
        # Write a tiny valid TIFF header (enough to pass exists check)
        tmp.write(b"\x49\x49\x2A\x00")
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                adapter.analyze(tmp.name, "Hello")
        finally:
            os.unlink(tmp.name)

    def test_missing_file_raises(self, adapter):
        with pytest.raises(FileNotFoundError):
            adapter.analyze("/nonexistent/image.jpg", "Hello")

    def test_invalid_source_type_raises(self, adapter):
        with pytest.raises(TypeError, match="Unsupported image source type"):
            adapter.analyze(12345, "Hello")  # type: ignore[arg-type]


# =========================================================================
# 5.  Image Conversion and Size Validation
# =========================================================================


class TestImageConversion:
    """Verify RGB conversion, thumbnail resizing, and JPEG encoding."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_rgba_converted_to_rgb(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        result = adapter.analyze(rgba, "Hello")
        assert isinstance(result, VLMResponse)

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_large_image_resized(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        big = Image.new("RGB", (4000, 3000), "green")
        adapter.analyze(big, "Hello")

        # Decode the sent JPEG to verify dimensions
        body = json.loads(mock_urlopen.call_args[0][0].data)
        data_uri = body["messages"][0]["content"][1]["image_url"]["url"]
        b64 = data_uri.split(",", 1)[1]
        sent_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert max(sent_img.size) <= _MAX_IMAGE_DIMENSION

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_small_image_not_resized(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        small = Image.new("RGB", (100, 80), "yellow")
        adapter.analyze(small, "Hello")

        body = json.loads(mock_urlopen.call_args[0][0].data)
        data_uri = body["messages"][0]["content"][1]["image_url"]["url"]
        b64 = data_uri.split(",", 1)[1]
        sent_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert sent_img.size == (100, 80)

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_grayscale_converted_to_rgb(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        gray = Image.new("L", (64, 64), 128)
        result = adapter.analyze(gray, "Hello")
        assert isinstance(result, VLMResponse)


# =========================================================================
# 6.  Successful Response Extraction
# =========================================================================


class TestSuccessfulResponse:
    """Verify text extraction and metadata from valid OpenAI responses."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_text_extracted(self, mock_urlopen, adapter):
        resp = _make_openai_response(content="  Extracted text content  ")
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        result = adapter.analyze(_make_pil_image(), "Describe this.")
        assert result.text == "Extracted text content"  # stripped

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_model_name_populated(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        result = adapter.analyze(_make_pil_image(), "Hello")
        assert result.model == _DEFAULT_MODEL_ID

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_usage_metadata(self, mock_urlopen, adapter):
        resp = _make_openai_response(prompt_tokens=50, completion_tokens=20)
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        result = adapter.analyze(_make_pil_image(), "Hello")
        assert result.prompt_eval_count == 50
        assert result.eval_count == 20

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_duration_ms_positive(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        result = adapter.analyze(_make_pil_image(), "Hello")
        assert result.total_duration_ms >= 0

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_missing_usage_defaults_to_zero(self, mock_urlopen, adapter):
        resp = _make_openai_response()
        del resp["usage"]
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        result = adapter.analyze(_make_pil_image(), "Hello")
        assert result.prompt_eval_count == 0
        assert result.eval_count == 0


# =========================================================================
# 7.  Missing / Empty Choices and Message Content
# =========================================================================


class TestMalformedResponses:
    """Verify proper error on missing/empty choices and content."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_empty_choices_raises(self, mock_urlopen, adapter):
        resp = _make_openai_response()
        resp["choices"] = []
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        with pytest.raises(VLMRemoteError, match="choices"):
            adapter.analyze(_make_pil_image(), "Hello")

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_missing_choices_key_raises(self, mock_urlopen, adapter):
        resp = _make_openai_response()
        del resp["choices"]
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        with pytest.raises(VLMRemoteError, match="choices"):
            adapter.analyze(_make_pil_image(), "Hello")

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_missing_message_raises(self, mock_urlopen, adapter):
        resp = _make_openai_response()
        del resp["choices"][0]["message"]
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        with pytest.raises(VLMRemoteError, match="message"):
            adapter.analyze(_make_pil_image(), "Hello")

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_null_content_raises(self, mock_urlopen, adapter):
        resp = _make_openai_response()
        resp["choices"][0]["message"]["content"] = None
        mock_urlopen.return_value = _FakeHTTPResponse(_json_bytes(resp))

        with pytest.raises(VLMRemoteError, match="content"):
            adapter.analyze(_make_pil_image(), "Hello")


# =========================================================================
# 8.  Auth Errors Do Not Retry  (401 / 403)
# =========================================================================


class TestAuthErrors:
    """401/403 must fail immediately without retry."""

    @pytest.mark.parametrize("status", [401, 403])
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_auth_error_no_retry(self, mock_urlopen, adapter, status):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=status,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"bad token"}'),
        )

        with pytest.raises(VLMAuthenticationError):
            adapter.analyze(_make_pil_image(), "Hello")

        # Should have been called exactly once (no retries)
        assert mock_urlopen.call_count == 1

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_400_no_retry(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"bad request"}'),
        )

        with pytest.raises(VLMRemoteError):
            adapter.analyze(_make_pil_image(), "Hello")

        assert mock_urlopen.call_count == 1

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_404_no_retry(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"not found"}'),
        )

        with pytest.raises(VLMRemoteError):
            adapter.analyze(_make_pil_image(), "Hello")

        assert mock_urlopen.call_count == 1


# =========================================================================
# 9.  Transient Errors Retry Within Bounded Limit
# =========================================================================


class TestTransientRetries:
    """429, 500, 502, 503, 504, timeouts should retry up to max_retries."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    @mock.patch("vlm_interface.modal_vlm_interface.time.sleep")  # skip actual delay
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_retryable_status_exhausts_attempts(
        self, mock_urlopen, mock_sleep, adapter, status
    ):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=status,
            msg="Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"transient"}'),
        )

        with pytest.raises(VLMInterfaceError):
            adapter.analyze(_make_pil_image(), "Hello")

        # 1 initial + 2 retries = 3 total
        assert mock_urlopen.call_count == 3

    @mock.patch("vlm_interface.modal_vlm_interface.time.sleep")
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_timeout_retries(self, mock_urlopen, mock_sleep, adapter):
        mock_urlopen.side_effect = urllib.error.URLError("timed out")

        with pytest.raises(VLMInterfaceError):
            adapter.analyze(_make_pil_image(), "Hello")

        assert mock_urlopen.call_count == 3

    @mock.patch("vlm_interface.modal_vlm_interface.time.sleep")
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_retry_succeeds_on_second_attempt(
        self, mock_urlopen, mock_sleep, adapter
    ):
        error = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=503,
            msg="Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"cold start"}'),
        )
        success = _FakeHTTPResponse(_json_bytes(_make_openai_response("Recovered")))

        mock_urlopen.side_effect = [error, success]

        result = adapter.analyze(_make_pil_image(), "Hello")
        assert result.text == "Recovered"
        assert mock_urlopen.call_count == 2

    @mock.patch("vlm_interface.modal_vlm_interface.time.sleep")
    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_backoff_delay_is_exponential(
        self, mock_urlopen, mock_sleep, adapter
    ):
        mock_urlopen.side_effect = urllib.error.URLError("timed out")

        with pytest.raises(VLMInterfaceError):
            adapter.analyze(_make_pil_image(), "Hello")

        # With max_retries=2: delays should be sleep(1.0) and sleep(2.0)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(delays) == 2
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)


# =========================================================================
# 10.  No Token Leakage in Logs / Errors
# =========================================================================


class TestNoTokenLeakage:
    """Credentials must never appear in log output or exception messages."""

    def test_redact_utility(self):
        assert "****" in _redact("supersecrettoken")
        assert "supersecrettoken" not in _redact("supersecrettoken")
        assert _redact("") == "<not set>"
        assert _redact(None) == "<not set>"
        assert _redact("ab") == "****"

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_error_body_redacts_token(self, mock_urlopen, adapter):
        # Simulate an error response that echoes the token back
        token = adapter._api_key
        error_body = f'{{"error": "Invalid token: {token}"}}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.modal.run/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(error_body.encode()),
        )

        with pytest.raises(VLMRemoteError) as exc_info:
            adapter.analyze(_make_pil_image(), "Hello")

        # The token must NOT appear in the exception string
        assert token not in str(exc_info.value)

    def test_init_log_redacts_key(self, caplog):
        with caplog.at_level(logging.INFO, logger="vlm_interface.modal_vlm_interface"):
            ModalVLMInterface(
                endpoint_url="https://x.modal.run",
                api_key="my-ultra-secret-key-9876",
            )
        full_log = caplog.text
        assert "my-ultra-secret-key-9876" not in full_log
        assert "my-u****" in full_log


# =========================================================================
# 11.  No Local Model Initialization / GPU Usage
# =========================================================================


class TestNoLocalGPU:
    """Ensure the adapter never imports or initializes local ML frameworks."""

    def test_no_torch_import(self, adapter):
        """ModalVLMInterface module must not import torch at module level."""
        import vlm_interface.modal_vlm_interface as mod
        source = Path(mod.__file__).read_text()
        # Allow "torch" to appear in comments but not as an import statement
        import_lines = [
            ln for ln in source.splitlines()
            if ln.strip().startswith("import torch")
            or ln.strip().startswith("from torch")
        ]
        assert import_lines == [], f"Found torch imports: {import_lines}"

    def test_no_transformers_import(self, adapter):
        import vlm_interface.modal_vlm_interface as mod
        source = Path(mod.__file__).read_text()
        import_lines = [
            ln for ln in source.splitlines()
            if ln.strip().startswith("import transformers")
            or ln.strip().startswith("from transformers")
        ]
        assert import_lines == [], f"Found transformers imports: {import_lines}"

    def test_no_cuda_reference(self, adapter):
        import vlm_interface.modal_vlm_interface as mod
        source = Path(mod.__file__).read_text()
        # "cuda" should not appear in executable code.
        # Skip comments (#) and docstring/string-literal lines.
        code_lines = [
            ln for ln in source.splitlines()
            if not ln.strip().startswith("#")
            and not ln.strip().startswith(('"""', "'''", "-", '"'))
            and "cuda" in ln.lower()
        ]
        assert code_lines == [], f"Found CUDA references: {code_lines}"


# =========================================================================
# 12.  VLMResponse Compatibility
# =========================================================================


class TestVLMResponseCompat:
    """Verify ModalVLMInterface returns the same VLMResponse type."""

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_returns_vlm_response_dataclass(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        result = adapter.analyze(_make_pil_image(), "Hello")

        assert isinstance(result, VLMResponse)
        assert hasattr(result, "text")
        assert hasattr(result, "model")
        assert hasattr(result, "total_duration_ms")
        assert hasattr(result, "prompt_eval_count")
        assert hasattr(result, "eval_count")

    def test_vlm_response_fields_match_original(self):
        """VLMResponse from vlm_types must have the same 5 fields as the original."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(VLMResponse)}
        expected = {"text", "model", "total_duration_ms", "prompt_eval_count", "eval_count"}
        assert fields == expected

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_response_consumed_like_local(self, mock_urlopen, adapter):
        """Simulate how SnapshotEngine accesses the response."""
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response("Some analysis JSON"))
        )
        result = adapter.analyze(_make_pil_image(), "Analyze frame")

        # SnapshotEngine does: response.text
        text = result.text
        assert isinstance(text, str)
        assert len(text) > 0


# =========================================================================
# 13.  cleanup() is Harmless
# =========================================================================


class TestCleanup:
    """cleanup() must be callable multiple times without error."""

    def test_cleanup_is_noop(self, adapter):
        adapter.cleanup()
        adapter.cleanup()  # idempotent
        # Still usable after cleanup (remote adapter holds no resources)

    @mock.patch("vlm_interface.modal_vlm_interface.urllib.request.urlopen")
    def test_usable_after_cleanup(self, mock_urlopen, adapter):
        adapter.cleanup()
        mock_urlopen.return_value = _FakeHTTPResponse(
            _json_bytes(_make_openai_response())
        )
        result = adapter.analyze(_make_pil_image(), "Hello")
        assert isinstance(result, VLMResponse)


# =========================================================================
# 14.  Lightweight Import — vlm_types does not load heavy deps
# =========================================================================


class TestLightweightImport:
    """Importing vlm_types must not trigger torch/transformers loading."""

    def test_vlm_types_no_heavy_deps(self):
        """vlm_types.py should have no heavy imports in its source."""
        types_path = Path(__file__).resolve().parent.parent / "vlm_types.py"
        source = types_path.read_text()
        for heavy in ("torch", "transformers", "qwen_vl_utils", "llama_cpp"):
            import_lines = [
                ln for ln in source.splitlines()
                if ln.strip().startswith(f"import {heavy}")
                or ln.strip().startswith(f"from {heavy}")
            ]
            assert import_lines == [], f"vlm_types.py imports {heavy}: {import_lines}"



# =========================================================================
# 15.  Provider Selection Constants
# =========================================================================


class TestProviderConstants:
    """Verify the explicit provider-selection contract is correctly defined."""

    def test_provider_env_var_name(self):
        assert PROVIDER_ENV_VAR == "SYNAPSE_VLM_PROVIDER"

    def test_valid_provider_values(self):
        assert PROVIDER_LOCAL == "local"
        assert PROVIDER_MODAL == "modal"

    def test_valid_providers_frozenset(self):
        assert isinstance(VALID_PROVIDERS, frozenset)
        assert VALID_PROVIDERS == {"local", "modal"}

    def test_both_providers_in_valid_set(self):
        assert PROVIDER_LOCAL in VALID_PROVIDERS
        assert PROVIDER_MODAL in VALID_PROVIDERS

    def test_constants_importable_from_package(self):
        """Verify re-export through vlm_interface.__init__."""
        from vlm_interface import (
            PROVIDER_ENV_VAR as pkg_env,
            PROVIDER_LOCAL as pkg_local,
            PROVIDER_MODAL as pkg_modal,
            VALID_PROVIDERS as pkg_valid,
        )
        assert pkg_env == PROVIDER_ENV_VAR
        assert pkg_local == PROVIDER_LOCAL
        assert pkg_modal == PROVIDER_MODAL
        assert pkg_valid == VALID_PROVIDERS

    def test_unknown_provider_not_in_valid_set(self):
        """Guard against accidentally accepting arbitrary strings."""
        assert "gpu" not in VALID_PROVIDERS
        assert "" not in VALID_PROVIDERS
        assert "Modal" not in VALID_PROVIDERS  # case-sensitive
