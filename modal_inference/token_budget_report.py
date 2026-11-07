"""Offline Qwen3-VL processor token-feasibility report.

This performs no inference and requires no GPU. By default it uses only a
locally cached official processor. Pass ``--allow-download`` to let Hugging
Face download processor/tokenizer assets for the configured public model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from modal_inference.config import (  # noqa: E402
    DEFAULT_STRUCTURED_MAX_TOKENS,
    MAX_IMAGE_DIMENSION,
    MAX_MODEL_LEN,
    MODEL_ID,
)
from vlm_engine.snapshot_engine import SnapshotEngine  # noqa: E402
from vlm_interface.qwen_context_budget import QwenContextBudgetEstimator  # noqa: E402


CASES = [
    ("generic", "ambiguous", []),
    ("ct03_scientific", "biology", ["CT-03"]),
    ("mixed_ct01_ct02", "graph", ["CT-01", "CT-02"]),
    ("mixed_ct01_ct02_ct06", "graph", ["CT-01", "CT-02", "CT-06"]),
    ("dense_all_categories", "ambiguous", [f"CT-{number:02d}" for number in range(1, 13)]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    estimator = QwenContextBudgetEstimator(
        model_id=MODEL_ID,
        max_model_len=MAX_MODEL_LEN,
        local_files_only=not args.allow_download,
    )
    image = Image.new("RGB", (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), "white")
    engine = object.__new__(SnapshotEngine)
    results = []
    for name, primary_type, categories in CASES:
        prompt = engine._build_prompt(
            primary_type, canonical_category_ids=categories
        )
        parts = engine._last_prompt_budget_parts
        report = estimator.estimate(
            image=image,
            prompt=prompt,
            requested_output_tokens=DEFAULT_STRUCTURED_MAX_TOKENS,
            common_prompt=parts["common_prompt"],
            category_directives=parts["category_directives"],
        )
        results.append({"case": name, **report.to_dict()})

    print(json.dumps({
        "model": MODEL_ID,
        "inference_performed": False,
        "image_size": [MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION],
        "results": results,
    }, ensure_ascii=False, indent=2))
    return 0 if all(item.get("fits") is True for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
