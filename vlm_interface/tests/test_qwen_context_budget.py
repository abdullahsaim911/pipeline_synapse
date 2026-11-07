"""Deterministic processor-contract tests for Qwen3-VL context accounting."""

from __future__ import annotations

import pytest
from PIL import Image

from vlm_interface.qwen_context_budget import QwenContextBudgetEstimator
from vlm_interface.vlm_types import VLMContextBudgetError


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class _ImageProcessor:
    merge_size = 2


class _Processor:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens
        self.tokenizer = _Tokenizer()
        self.image_processor = _ImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1
        assert [part["type"] for part in messages[0]["content"]] == ["image", "text"]
        return {
            "input_ids": [[0] * self.input_tokens],
            "image_grid_thw": [[1, 40, 40]],
        }


def test_exact_processor_budget_reports_all_required_components():
    estimator = QwenContextBudgetEstimator(
        processor=_Processor(3000), max_model_len=8192
    )
    report = estimator.estimate(
        image=Image.new("RGB", (1344, 1000)),
        prompt="one two three four",
        common_prompt="one two",
        category_directives="CT-01 CT-02 CT-06",
        requested_output_tokens=2048,
    )
    assert report.exact is True
    assert report.input_tokens == 3000
    assert report.prompt_text_tokens == 4
    assert report.common_prompt_tokens == 2
    assert report.category_directive_tokens == 3
    assert report.image_tokens == 400
    assert report.chat_template_tokens == 2600
    assert report.special_token_overhead == 2596
    assert report.remaining_context_tokens == 3144
    assert report.fits is True


def test_over_budget_request_fails_explicitly_and_retains_report():
    estimator = QwenContextBudgetEstimator(
        processor=_Processor(7000), max_model_len=8192
    )
    with pytest.raises(VLMContextBudgetError, match="7000.*2048.*8192"):
        estimator.estimate(
            image=Image.new("RGB", (1344, 1344)),
            prompt="dense prompt",
            requested_output_tokens=2048,
        )
    assert estimator.last_report is not None
    assert estimator.last_report.fits is False
    assert estimator.last_report.remaining_context_tokens == -856


def test_unavailable_processor_reports_uncertainty_without_character_estimate():
    def unavailable():
        raise OSError("processor is not cached")

    estimator = QwenContextBudgetEstimator(processor_loader=unavailable)
    report = estimator.estimate(
        image=Image.new("RGB", (1344, 800)),
        prompt="prompt text",
        requested_output_tokens=512,
    )
    assert report.exact is False
    assert report.processor_available is False
    assert report.input_tokens is None
    assert report.image_tokens is None
    assert "processor unavailable" in report.uncertainty.lower()
