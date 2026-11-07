"""Opt-in Task 5 prompt comparison using the existing Ollama client.

Dry-run is the default.  ``--live`` performs three calls per case with the same
model and client decoding defaults, and emits responses plus a blank human-review
rubric.  It deliberately does not convert lexical overlap into a quality score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesizer.knowledge_base import KnowledgeBase  # noqa: E402
from synthesizer.llm_synthesizer import LLMSynthesizer  # noqa: E402
from synthesizer.math_linearizer import MathLinearizer  # noqa: E402
from synchronizer import reconcile_visual_evidence  # noqa: E402


FIXTURE_PATH = Path(__file__).with_name("task5_evaluation_cases.json")


def build_snapshot(case: dict) -> dict:
    categories = case["categories"]
    elements = []
    for raw in case.get("elements", []):
        element = {
            "evidence_status": raw.get("evidence_status", "supported"),
            "attributes": {},
            "child_ids": [],
            **copy.deepcopy(raw),
        }
        elements.append(element)
    relations = []
    for raw in case.get("relations", []):
        relations.append({"label": None, "source_refs": [], **copy.deepcopy(raw)})
    evidence = {
        "schema_version": "1.1.0",
        "frame_id": case["id"],
        "extraction_status": "complete",
        "evidence_status": case.get("evidence_status", "supported"),
        "category_ids": categories,
        "regions": [{
            "id": "region-001", "category_ids": categories,
            "elements": elements, "evidence_status": case.get("evidence_status", "supported"),
        }],
        "relations": relations,
        "uncertainties": copy.deepcopy(case.get("uncertainties", [])),
        "structural_description": {
            "overview": "Independent Task 5 evaluation fixture.",
            "orientation": None,
            "component_ids": [item["id"] for item in elements],
            "relation_ids": [item["id"] for item in relations],
            "reading_order": [item["id"] for item in elements],
            "hierarchy_notes": [], "topology_notes": [], "uncertainties": [],
        },
        "exact_content": copy.deepcopy(case.get("exact_content", [])),
        "category_details": copy.deepcopy(case.get("category_details", {})),
        "validation_diagnostics": [],
    }
    snapshot = {"visual_evidence": evidence, "detected_types": [case["content_type"]]}
    snapshot["coverage_plan"] = reconcile_visual_evidence(snapshot, case["transcript"])
    return snapshot


def build_prompts(synth: LLMSynthesizer, case: dict, snapshot: dict) -> dict[str, str]:
    mode = LLMSynthesizer._normalize_output_mode(case["mode"])
    context = {"context_text": "", "previous_concepts": [], "related_principles": [], "topic_timeline": {}}
    mode_name = "explanatory" if mode == "explanatory" else mode
    legacy_builder = getattr(synth, f"_build_{mode_name}_prompt_legacy_baseline")
    current_builder = getattr(synth, f"_build_{mode_name}_prompt")
    return {
        "legacy_template": legacy_builder(case["transcript"], context, copy.deepcopy(snapshot)),
        "candidate_a_direct_grounded": synth._build_task5_prompt(
            case["transcript"], context, copy.deepcopy(snapshot), mode,
            case["content_type"], case["intervention_reason"], prioritize_coverage=False,
        ),
        "candidate_b_priority_grounded": current_builder(
            case["transcript"], context, copy.deepcopy(snapshot),
            case["content_type"], case["intervention_reason"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call the configured local Ollama Mistral model.")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--model", default="mistral")
    args = parser.parse_args()

    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    synth = LLMSynthesizer(
        ollama_url=args.url, model=args.model, timeout=120, max_retries=0,
        transcripts=[], output_mode="explanatory",
    )
    synth.math_linearizer = MathLinearizer()
    synth.knowledge_base = KnowledgeBase()
    results = []
    for case in cases:
        prompts = build_prompts(synth, case, build_snapshot(case))
        record = {
            "case_id": case["id"],
            "expected_essential_facts": case["expected_essential_facts"],
            "expected_relationships": case["expected_relationships"],
            "prohibited_claims": case["prohibited_claims"],
            "strategies": {},
        }
        for strategy, prompt in prompts.items():
            item = {
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "status": "pending_live_mistral",
            }
            if args.live:
                response = synth.client.generate(prompt)
                item.update({
                    "status": "generated_pending_human_review",
                    "response": response.text,
                    "latency_ms": response.total_duration_ms,
                    "prompt_tokens": response.prompt_eval_count,
                    "output_tokens": response.eval_count,
                    "human_review": {
                        "supported_critical_fact_recall": None,
                        "unsupported_claim_count": None,
                        "incorrect_relationship_count": None,
                        "math_code_exactness": None,
                        "teacher_content_redundancy": None,
                        "uncertainty_handling": None,
                        "cross_level_core_fact_consistency": None,
                        "accessibility_usability": None,
                    },
                })
            record["strategies"][strategy] = item
        results.append(record)
    print(json.dumps({"live": args.live, "model": args.model, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
