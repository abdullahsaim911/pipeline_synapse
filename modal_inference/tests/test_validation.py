"""
SYNAPSE — Local Validation Tests for Modal Inference Components
Stage M1A: Verifies schemas, image checks, token boundaries, and auth logic
without requiring remote GPUs, network calls, or model weights.
"""

import base64
import json
import unittest

from modal_inference.config import (
    DEFAULT_GPU,
    GPU_MEMORY_UTILIZATION,
    MAX_CONCURRENT_INPUTS,
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_PROMPT,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MAX_OUTPUT_TOKENS,
    MAX_VIDEOS_PER_PROMPT,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_QUANTIZATION,
    MULTIMODAL_LIMITS_JSON,
    SCALEDOWN_WINDOW_SECONDS,
)
from modal_inference.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
    ImageUrlPart,
    clamp_generation_tokens,
    detect_image_mime,
    validate_and_parse_data_uri,
    validate_auth_header,
    validate_image_payload,
)


class TestImageValidation(unittest.TestCase):
    """Test magic byte detection and image size boundary enforcement."""

    def setUp(self):
        # 1x1 Minimal PNG (valid PNG header and chunks)
        self.valid_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff"
            b"?\x00\x05\xfe\x02\xfe\xa74d\xa2\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        # Minimal JPEG header
        self.valid_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
            b"\xff\xdb\x00C\x00\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11"
            b"\x00\xff\xd9"
        )

    def test_detect_image_mime_png(self):
        mime = detect_image_mime(self.valid_png)
        self.assertEqual(mime, "image/png")

    def test_detect_image_mime_jpeg(self):
        mime = detect_image_mime(self.valid_jpeg)
        self.assertEqual(mime, "image/jpeg")

    def test_detect_image_mime_unsupported(self):
        corrupted = b"NOT AN IMAGE HEADER 12345678"
        self.assertIsNone(detect_image_mime(corrupted))

    def test_validate_image_payload_valid(self):
        is_valid, msg, mime = validate_image_payload(self.valid_png)
        self.assertTrue(is_valid)
        self.assertEqual(mime, "image/png")

    def test_validate_image_payload_empty(self):
        is_valid, msg, mime = validate_image_payload(b"")
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_validate_image_payload_oversized(self):
        # Simulate payload exceeding limit with fake PNG header
        fake_large_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
        is_valid, msg, _ = validate_image_payload(fake_large_png, max_bytes=100)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum allowed size", msg)

    def test_validate_and_parse_data_uri_valid(self):
        b64 = base64.b64encode(self.valid_png).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        is_valid, msg, raw_bytes, mime = validate_and_parse_data_uri(data_uri)
        self.assertTrue(is_valid)
        self.assertEqual(raw_bytes, self.valid_png)
        self.assertEqual(mime, "image/png")

    def test_validate_and_parse_data_uri_invalid_format(self):
        is_valid, msg, _, _ = validate_and_parse_data_uri("not-a-data-uri")
        self.assertFalse(is_valid)

    def test_validate_and_parse_data_uri_corrupt_base64(self):
        is_valid, msg, _, _ = validate_and_parse_data_uri("data:image/png;base64,@@@@CorruptBase64@@@@")
        self.assertFalse(is_valid)


class TestTokenAndAuthValidation(unittest.TestCase):
    """Test token clamping and HTTP authorization logic."""

    def test_clamp_generation_tokens_none(self):
        tokens = clamp_generation_tokens(None, max_allowed=2048, default_tokens=1024)
        self.assertEqual(tokens, 1024)

    def test_clamp_generation_tokens_underflow(self):
        tokens = clamp_generation_tokens(-5, max_allowed=2048)
        self.assertEqual(tokens, 1)

    def test_clamp_generation_tokens_overflow(self):
        tokens = clamp_generation_tokens(50000, max_allowed=MAX_OUTPUT_TOKENS)
        self.assertEqual(tokens, MAX_OUTPUT_TOKENS)

    def test_validate_auth_header_success(self):
        secret = "synapse-test-secret-key-123"
        header = f"Bearer {secret}"
        ok, msg = validate_auth_header(header, secret)
        self.assertTrue(ok)
        self.assertEqual(msg, "Authorized.")

    def test_validate_auth_header_missing(self):
        ok, msg = validate_auth_header(None, "expected-key")
        self.assertFalse(ok)
        self.assertIn("Missing", msg)

    def test_validate_auth_header_wrong_scheme(self):
        ok, msg = validate_auth_header("Basic dXNlcjpwYXNz", "expected-key")
        self.assertFalse(ok)
        self.assertIn("Bearer", msg)

    def test_validate_auth_header_mismatch(self):
        ok, msg = validate_auth_header("Bearer wrong-key", "expected-key")
        self.assertFalse(ok)
        self.assertIn("Invalid", msg)


class TestChatCompletionRequest(unittest.TestCase):
    """Test request serialization and message validation."""

    def setUp(self):
        self.valid_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff"
            b"?\x00\x05\xfe\x02\xfe\xa74d\xa2\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    def test_from_image_and_prompt(self):
        req = ChatCompletionRequest.from_image_and_prompt(
            image_bytes=self.valid_png,
            mime_type="image/png",
            prompt="Analyze this diagram.",
            max_tokens=500,
        )
        is_valid, msg = req.validate()
        self.assertTrue(is_valid, msg)
        self.assertEqual(req.model, MODEL_ID)
        self.assertEqual(req.max_tokens, 500)

        # Verify serialization structure
        d = req.to_dict()
        self.assertEqual(d["model"], MODEL_ID)
        self.assertEqual(len(d["messages"]), 1)
        content = d["messages"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "Analyze this diagram.")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_validate_rejects_empty_messages(self):
        req = ChatCompletionRequest(messages=[])
        is_valid, msg = req.validate()
        self.assertFalse(is_valid)
        self.assertIn("empty", msg)

    def test_validate_rejects_invalid_role(self):
        msg = ChatMessage(role="invalid_role", content="hello")
        req = ChatCompletionRequest(messages=[msg])
        is_valid, err = req.validate()
        self.assertFalse(is_valid)
        self.assertIn("role", err)

    def test_validate_rejects_multiple_images(self):
        b64 = base64.b64encode(self.valid_png).decode("ascii")
        image = ContentPart(
            type="image_url",
            image_url=ImageUrlPart(url=f"data:image/png;base64,{b64}"),
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=[image, image])]
        )
        is_valid, err = req.validate()
        self.assertFalse(is_valid)
        self.assertIn("maximum is 1", err)


class TestL4DeploymentConfiguration(unittest.TestCase):
    """Lock the cost and memory boundaries approved for Stage M1B."""

    def test_l4_single_request_boundaries(self):
        self.assertEqual(DEFAULT_GPU, "L4")
        self.assertEqual(MAX_MODEL_LEN, 4096)
        self.assertEqual(MAX_NUM_SEQS, 1)
        self.assertEqual(MAX_CONCURRENT_INPUTS, 1)
        self.assertEqual(MAX_IMAGES_PER_PROMPT, 1)
        self.assertEqual(MAX_VIDEOS_PER_PROMPT, 0)
        self.assertEqual(SCALEDOWN_WINDOW_SECONDS, 300)
        self.assertLessEqual(GPU_MEMORY_UTILIZATION, 0.85)

    def test_official_fp8_checkpoint_is_explicit(self):
        self.assertEqual(MODEL_ID, "Qwen/Qwen3-VL-8B-Instruct-FP8")
        self.assertEqual(MODEL_DTYPE, "bfloat16")
        self.assertEqual(MODEL_QUANTIZATION, "fp8")

    def test_multimodal_cli_limit_is_valid_json(self):
        limits = json.loads(MULTIMODAL_LIMITS_JSON)
        self.assertEqual(limits, {"image": 1, "video": 0})
        self.assertEqual(limits["image"], MAX_IMAGES_PER_PROMPT)
        self.assertEqual(limits["video"], MAX_VIDEOS_PER_PROMPT)


if __name__ == "__main__":
    unittest.main()

