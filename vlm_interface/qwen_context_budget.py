"""Qwen3-VL processor-based context-budget accounting.

Heavy dependencies are imported lazily so the remote adapter remains cheap to
import. Exact checks use the official Hugging Face processor and its expanded
multimodal ``input_ids``. If the processor is not locally available, callers
receive an explicit uncertainty report rather than a character-count estimate.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Callable

from PIL import Image

from modal_inference.config import MAX_MODEL_LEN, MODEL_ID
from vlm_interface.vlm_types import VLMContextBudgetError


@dataclass(frozen=True)
class ContextBudgetReport:
    model_id: str
    max_model_len: int
    requested_output_tokens: int
    exact: bool
    processor_available: bool
    input_tokens: int | None = None
    prompt_text_tokens: int | None = None
    chat_template_tokens: int | None = None
    common_prompt_tokens: int | None = None
    category_directive_tokens: int | None = None
    image_tokens: int | None = None
    special_token_overhead: int | None = None
    remaining_context_tokens: int | None = None
    fits: bool | None = None
    image_size: tuple[int, int] | None = None
    image_token_method: str | None = None
    uncertainty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sequence_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    raise TypeError("Processor output did not expose countable input_ids.")


def _grid_rows(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (int, float)):
        value = [value]
    return [[int(part) for part in row] for row in (value or [])]


class QwenContextBudgetEstimator:
    """Lazy exact accounting for one Qwen3-VL image/chat request."""

    _PROCESSOR_CACHE: dict[tuple[str, bool], Any] = {}
    _LOAD_ERROR_CACHE: dict[tuple[str, bool], str] = {}

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        max_model_len: int = MAX_MODEL_LEN,
        processor: Any | None = None,
        processor_loader: Callable[[], Any] | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_model_len = max_model_len
        self._processor = processor
        self._processor_loader = processor_loader
        if local_files_only is None:
            raw = os.environ.get("SYNAPSE_QWEN_PROCESSOR_LOCAL_FILES_ONLY", "1")
            local_files_only = raw.strip().lower() not in {"0", "false", "no"}
        self.local_files_only = local_files_only
        self._load_error: str | None = None
        self.last_report: ContextBudgetReport | None = None

    def _load_processor(self) -> Any | None:
        if self._processor is not None:
            return self._processor
        if self._load_error is not None:
            return None
        cache_key = (self.model_id, self.local_files_only)
        if self._processor_loader is None and cache_key in self._PROCESSOR_CACHE:
            self._processor = self._PROCESSOR_CACHE[cache_key]
            return self._processor
        if self._processor_loader is None and cache_key in self._LOAD_ERROR_CACHE:
            self._load_error = self._LOAD_ERROR_CACHE[cache_key]
            return None
        try:
            if self._processor_loader is not None:
                self._processor = self._processor_loader()
            else:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                )
                self._PROCESSOR_CACHE[cache_key] = self._processor
            return self._processor
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            if self._processor_loader is None:
                self._LOAD_ERROR_CACHE[cache_key] = self._load_error
            return None

    @staticmethod
    def _text_tokens(tokenizer: Any, text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    def estimate(
        self,
        *,
        image: Image.Image,
        prompt: str,
        requested_output_tokens: int,
        common_prompt: str = "",
        category_directives: str = "",
    ) -> ContextBudgetReport:
        processor = self._load_processor()
        if processor is None:
            report = ContextBudgetReport(
                model_id=self.model_id,
                max_model_len=self.max_model_len,
                requested_output_tokens=requested_output_tokens,
                exact=False,
                processor_available=False,
                image_size=image.size,
                fits=None,
                uncertainty=(
                    "Official Qwen3-VL processor unavailable; exact text/image token "
                    "consumption was not calculated. Request remains bounded to one "
                    "1344-pixel image and the configured output ceiling. "
                    f"Loader detail: {self._load_error or 'unknown'}"
                ),
            )
            self.last_report = report
            return report

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            # Python lists are sufficient for counting and avoid initializing
            # a tensor/CUDA runtime in the lightweight remote client.
            return_tensors=None,
        )
        input_tokens = _sequence_length(processed["input_ids"])
        tokenizer = processor.tokenizer
        prompt_tokens = self._text_tokens(tokenizer, prompt)
        common_tokens = self._text_tokens(tokenizer, common_prompt)
        category_tokens = self._text_tokens(tokenizer, category_directives)

        grid = _grid_rows(processed.get("image_grid_thw"))
        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        image_tokens = sum(
            (row[0] * row[1] * row[2]) // (merge_size * merge_size)
            for row in grid if len(row) == 3
        )
        image_method = "image_grid_thw / merge_size^2"
        if not grid:
            image_tokens = 0
            image_method = "processor returned no image_grid_thw; image count uncertain"

        chat_template_tokens = max(0, input_tokens - image_tokens)
        special_overhead = max(0, input_tokens - prompt_tokens - image_tokens)
        remaining = self.max_model_len - input_tokens - requested_output_tokens
        fits = remaining >= 0
        report = ContextBudgetReport(
            model_id=self.model_id,
            max_model_len=self.max_model_len,
            requested_output_tokens=requested_output_tokens,
            exact=True,
            processor_available=True,
            input_tokens=input_tokens,
            prompt_text_tokens=prompt_tokens,
            chat_template_tokens=chat_template_tokens,
            common_prompt_tokens=common_tokens,
            category_directive_tokens=category_tokens,
            image_tokens=image_tokens,
            special_token_overhead=special_overhead,
            remaining_context_tokens=remaining,
            fits=fits,
            image_size=image.size,
            image_token_method=image_method,
            uncertainty=(None if grid else image_method),
        )
        self.last_report = report
        if not fits:
            raise VLMContextBudgetError(
                "Qwen3-VL context budget exceeded: "
                f"input_tokens={input_tokens} + requested_output_tokens="
                f"{requested_output_tokens} > max_model_len={self.max_model_len}."
            )
        return report
