"""Task 2B mocked end-to-end structured visual evidence tests."""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

from synthesizer.llm_synthesizer import LLMSynthesizer
from visual_evidence import COMPACT_OBSERVATION_SCHEMA, validate_visual_evidence
from vlm_engine.snapshot_engine import SnapshotEngine
from vlm_interface import PROVIDER_MODAL, VLMResponse


def _base_payload(categories, *, status="complete", evidence_status="supported"):
    return {
        "schema": COMPACT_OBSERVATION_SCHEMA,
        "status": status,
        "evidence_status": evidence_status,
        "categories": categories,
        "layout": "One main region centered in the frame.",
        "regions": [],
        "relations": [],
        "structure": {
            "overview": "A single structured educational visual.",
            "orientation": "top to bottom",
            "components": [],
            "relations": [],
            "reading_order": [],
            "hierarchy": [],
            "topology": [],
            "uncertainties": [],
        },
        "exact": [],
        "details": {},
        "uncertainties": [],
        "missing": [],
    }


class FakeVLM:
    def __init__(self, payload, *, completion_tokens=120, finish_reason=None):
        self.payload = payload
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.prompt = None
        self.generation_params = None

    def analyze(self, image_path_or_pil, prompt, generation_params):
        self.prompt = prompt
        self.generation_params = generation_params
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return VLMResponse(
            text=text,
            model="Qwen/Qwen3-VL-8B-Instruct-FP8",
            total_duration_ms=321,
            prompt_eval_count=700,
            eval_count=self.completion_tokens,
            finish_reason=self.finish_reason,
        )

    def cleanup(self):
        return None


def _engine(payload, *, completion_tokens=120, finish_reason=None):
    engine = object.__new__(SnapshotEngine)
    engine.provider = PROVIDER_MODAL
    engine.use_gguf = False
    engine.model_name = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    engine.structured_max_tokens = 768
    engine.vlm_interface = FakeVLM(
        payload,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )
    return engine


def _analyze(
    payload, primary_type, categories, labels=None, *,
    completion_tokens=120, finish_reason=None
):
    engine = _engine(
        payload,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )
    snapshot = engine.analyze_frame(
        "unused.jpg",
        primary_type,
        frame_id="frame-1",
        timestamp_seconds=14.5,
        raw_category_labels=labels or [primary_type],
        canonical_category_ids=categories,
        classification_confidence=0.91,
        confidence_semantics="classifier",
    )
    return engine, snapshot


def _element(ref, kind, *, category, label=None, text=None, status="supported", **extra):
    return {
        "ref": ref,
        "kind": kind,
        "categories": [category],
        "label": label,
        "text": text,
        "description": extra.get("description"),
        "position": extra.get("position"),
        "bbox": None,
        "evidence_status": status,
        "source_ref": None,
        "attributes": extra.get("attributes", {}),
        "parent_ref": extra.get("parent_ref"),
        "child_refs": extra.get("child_refs", []),
    }


def _region(ref, category, elements, description="Visible region"):
    return {
        "ref": ref,
        "categories": [category],
        "position": "center",
        "bbox": None,
        "description": description,
        "evidence_status": "supported",
        "elements": elements,
    }


@pytest.mark.parametrize("category", [f"CT-{number:02d}" for number in range(1, 13)])
def test_each_canonical_category_gets_only_its_relevant_specialized_instruction(category):
    engine = object.__new__(SnapshotEngine)
    prompt = engine._build_prompt("ambiguous", canonical_category_ids=[category])
    assert COMPACT_OBSERVATION_SCHEMA in prompt
    assert SnapshotEngine.CANONICAL_CATEGORY_DIRECTIVES[category] in prompt
    other = "CT-12" if category != "CT-12" else "CT-01"
    assert SnapshotEngine.CANONICAL_CATEGORY_DIRECTIVES[other] not in prompt


@pytest.mark.parametrize("hint", ["diagram", "slide", "unknown"])
def test_ambiguous_hints_get_full_discovery_without_a_forced_category(hint):
    from category_normalization import normalize_category_labels

    prompt = object.__new__(SnapshotEngine)._build_prompt(hint)
    assert not normalize_category_labels(hint).canonical_categories
    assert "CANONICAL CATEGORY HINTS: none" in prompt
    assert all(f"CT-{number:02d}" in prompt for number in range(1, 13))
    assert '"categories":["CT-01"]' not in prompt
    assert "arrowhead, shaft, attachment and visible endpoints" in prompt
    assert "a data chart" in prompt


def test_duplicate_region_quality_gap_is_partial_without_merging_or_reinference():
    import copy

    payload = _base_payload(["CT-03"])
    first = _region("panel-a", "CT-03", [
        _element("source-a", "source", category="CT-03", label="Source")
    ], description="A labeled source and its surrounding field.")
    second = copy.deepcopy(first)
    second["ref"] = "panel-b"
    second["elements"][0]["ref"] = "source-b"
    payload["regions"] = [first, second]
    original = copy.deepcopy(payload)
    engine = _engine(payload, finish_reason="stop")
    calls = []
    analyze = engine.vlm_interface.analyze

    def record_call(*args, **kwargs):
        calls.append(kwargs)
        return analyze(*args, **kwargs)

    engine.vlm_interface.analyze = record_call
    snapshot = engine.analyze_frame("unused.jpg", "diagram")
    evidence = snapshot["visual_evidence"]
    assert len(calls) == 1
    assert evidence["extraction_status"] == "partial"
    assert evidence["evidence_status"] == "partially_supported"
    assert len(evidence["regions"]) == 2
    assert evidence["validation_diagnostics"][-1]["target_id"] == "region-002"
    assert evidence["validation_diagnostics"][-1]["code"].endswith("possible_duplicate_region")
    assert snapshot["_structured_observation"] == original == payload
    assert snapshot["_vlm_metrics"]["possible_truncation"] is False
    assert not snapshot["_error"]
    assert validate_visual_evidence(evidence).structurally_valid


def test_separate_identical_objects_keep_positions_and_inapplicable_structure_empty():
    payload = _base_payload(["CT-11"])
    payload["regions"] = [
        _region("a", "CT-11", [_element("a1", "specimen", category="CT-11", label="Sample")]),
        _region("b", "CT-11", [_element("b1", "specimen", category="CT-11", label="Sample")]),
    ]
    payload["regions"][0]["position"] = "left"
    payload["regions"][1]["position"] = "right"
    _, snapshot = _analyze(payload, "observational_image", ["CT-11"])
    evidence = snapshot["visual_evidence"]
    assert evidence["extraction_status"] == "complete"
    assert len(evidence["regions"]) == 2
    assert evidence["structural_description"]["reading_order"] == []
    assert evidence["structural_description"]["hierarchy_notes"] == []
    assert evidence["structural_description"]["topology_notes"] == []
    assert not evidence["validation_diagnostics"]


def test_ordered_regions_without_structured_navigation_are_diagnosed():
    payload = _base_payload(["CT-06"])
    payload["regions"] = [
        _region("a", "CT-06", [_element("a1", "text", category="CT-06", text="First")]),
        _region("b", "CT-06", [_element("b1", "text", category="CT-06", text="Second")]),
    ]
    for index, region in enumerate(payload["regions"]):
        region.pop("position")
        region["reading_order_index"] = index
    _, snapshot = _analyze(payload, "text", ["CT-06"])
    evidence = snapshot["visual_evidence"]
    codes = {item["code"].rsplit(".", 1)[-1] for item in evidence["validation_diagnostics"]}
    assert codes == {"missing_region_positions", "missing_reading_order_refs"}
    assert evidence["extraction_status"] == "partial"
    assert evidence["structural_description"]["reading_order"] == []
    payload["structure"]["reading_order"] = ["a", "b"]
    payload["regions"][0]["position"] = "top"
    payload["regions"][1]["position"] = "bottom"
    _, complete = _analyze(payload, "text", ["CT-06"])
    assert complete["visual_evidence"]["extraction_status"] == "complete"


@pytest.mark.parametrize(("first", "second", "conflict"), [
    ("left_of", "right_of", True),
    ("above", "below", True),
    ("precedes", "follows", True),
    ("connected_to", "connected_to", False),
])
def test_explicit_spatial_contradictions_are_flagged_without_rewriting_relations(first, second, conflict):
    payload = _base_payload(["CT-04"])
    payload["regions"] = [_region("network", "CT-04", [
        _element("a", "node", category="CT-04", label="A"),
        _element("b", "node", category="CT-04", label="B"),
    ])]
    # Reciprocal connections are legitimate; opposite spatial claims are not.
    payload["relations"] = [
        {"ref": "q1", "source": "a", "target": "b", "type": first, "evidence_status": "supported"},
        {"ref": "q2", "source": "a" if conflict else "b", "target": "b" if conflict else "a", "type": second, "evidence_status": "supported"},
    ]
    _, snapshot = _analyze(payload, "flowchart", ["CT-04"])
    evidence = snapshot["visual_evidence"]
    assert len(evidence["relations"]) == 2
    assert (evidence["extraction_status"] == "partial") is conflict
    assert validate_visual_evidence(evidence).structurally_valid


CATEGORY_FIXTURES = [
    (
        "graph", "CT-01", "chart", "axis", "Temperature (°C)",
        {"axes": [{"label": "Temperature", "unit": "°C"}], "series": [{"label": "A"}], "visible_values": ["20"]},
    ),
    (
        "equation", "CT-02", "mathematics", "equation", "x² − 3x ≤ 4",
        {"exact_expressions": ["x² − 3x ≤ 4"], "symbols": ["²", "−", "≤"]},
    ),
    (
        "biology", "CT-03", "scientific_structure", "anatomical_part", "Cerebellum",
        {"components": ["Cerebellum"], "containment": ["part of brain"]},
    ),
    (
        "flowchart", "CT-04", "flow_network", "decision_node", "Valid?",
        {"nodes": ["Valid?"], "branch_conditions": ["Yes", "No"]},
    ),
    (
        "table", "CT-05", "table", "table_cell", "95",
        {"headers": ["Student", "Score"], "row_count": 2, "column_count": 2, "cells": [{"row": 1, "column": 2, "text": "95", "row_header": "Ada", "column_header": "Score"}]},
    ),
    (
        "text", "CT-06", "structured_text", "warning", "Do not divide by zero.",
        {"headings": ["Warning"], "annotations": ["Do not divide by zero."]},
    ),
    (
        "geometry", "CT-07", "geometry", "angle", "∠ABC = 90°",
        {"points": ["A", "B", "C"], "angles": ["∠ABC = 90°"], "constraints": ["right angle at B"]},
    ),
    (
        "code", "CT-08", "programming", "code_block", "if score >= 90:\n    print(\"A\")",
        {"language": "python", "exact_code": "if score >= 90:\n    print(\"A\")", "displayed_output": "A", "indentation_significant": True},
    ),
    (
        "software_interface", "CT-09", "interface", "toggle", "Wi-Fi: Off",
        {"controls": ["Wi-Fi toggle"], "visible_states": ["Off"]},
    ),
    (
        "map", "CT-10", "map", "map_label", "River Indus",
        {"locations": ["River Indus"], "directions": ["north arrow points up"]},
    ),
    (
        "observational_image", "CT-11", "observational_image", "specimen_region", "rough brown surface",
        {"visible_features": ["rough brown surface"], "comparisons": ["left region is darker than right"]},
    ),
    (
        "physical_illustration", "CT-12", "physical_scene", "force_arrow", "F = 10 N",
        {"objects": ["block"], "apparatus": ["horizontal support"], "observable_states": ["force arrow points right"]},
    ),
]


@pytest.mark.parametrize(
    ("primary", "category", "details_key", "kind", "exact_text", "details"),
    CATEGORY_FIXTURES,
)
def test_all_twelve_categories_parse_normalize_validate_and_reach_mistral_context(
    primary, category, details_key, kind, exact_text, details
):
    payload = _base_payload([category])
    payload["regions"] = [
        _region("main", category, [
            _element("fact", kind, category=category, text=exact_text)
        ])
    ]
    payload["structure"].update({
        "components": ["fact"],
        "reading_order": ["fact"],
    })
    payload["exact"] = [{
        "text": exact_text,
        "status": "supported",
        "notation": "code" if category == "CT-08" else "math" if category in {"CT-02", "CT-07"} else "plain",
        "source_ref": "fact",
    }]
    payload["details"] = {details_key: details}

    _, snapshot = _analyze(payload, primary, [category])
    evidence = snapshot["visual_evidence"]
    assert validate_visual_evidence(evidence).structurally_valid
    assert evidence["category_ids"] == [category]
    assert evidence["exact_content"][0]["source_ref"] == "element-001"
    assert evidence["structural_description"]["component_ids"] == ["element-001"]
    packaged = json.loads(LLMSynthesizer._format_canonical_evidence(
        object.__new__(LLMSynthesizer), snapshot
    ))
    assert packaged["exact_content"][0]["text"] == exact_text


def test_structured_output_budget_is_configurable_and_conservatively_bounded(monkeypatch):
    monkeypatch.setenv("SYNAPSE_VLM_STRUCTURED_MAX_TOKENS", "1024")
    engine = SnapshotEngine(
        provider=PROVIDER_MODAL,
        endpoint_url="https://example.test",
        api_key="not-a-real-key",
    )
    assert engine.structured_max_tokens == 1024
    assert engine.vlm_interface._max_tokens == 1024


def test_structured_output_budget_defaults_to_8192(monkeypatch):
    monkeypatch.delenv("SYNAPSE_VLM_STRUCTURED_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SYNAPSE_MODAL_MAX_TOKENS", raising=False)
    engine = SnapshotEngine(
        provider=PROVIDER_MODAL,
        endpoint_url="https://example.test",
        api_key="not-a-real-key",
    )
    assert engine.structured_max_tokens == 8192
    assert engine.vlm_interface._max_tokens == 8192


@pytest.mark.parametrize("configured", [0, 8193])
def test_invalid_structured_output_budget_is_rejected_not_clamped(monkeypatch, configured):
    monkeypatch.setenv("SYNAPSE_VLM_STRUCTURED_MAX_TOKENS", str(configured))
    with pytest.raises(ValueError, match="between 1 and 8192"):
        SnapshotEngine(
            provider=PROVIDER_MODAL,
            endpoint_url="https://example.test",
            api_key="not-a-real-key",
        )


def test_snapshot_passes_prompt_budget_components_and_target_temperature():
    payload = _base_payload(["CT-01", "CT-02", "CT-06"])
    engine, _ = _analyze(
        payload, "graph", ["CT-01", "CT-02", "CT-06"]
    )
    params = engine.vlm_interface.generation_params
    assert params["max_new_tokens"] == 768
    assert params["temperature"] == 0.1
    assert "compact structured observation" in params["_context_budget_parts"]["common_prompt"]
    directives = params["_context_budget_parts"]["category_directives"]
    assert all(category in directives for category in ("CT-01", "CT-02", "CT-06"))


def test_context_budget_diagnostics_are_preserved_in_snapshot_metrics():
    payload = _base_payload(["CT-03"])
    engine = _engine(payload)
    engine.vlm_interface.last_context_budget = {
        "exact": True,
        "input_tokens": 3109,
        "requested_output_tokens": 768,
        "remaining_context_tokens": 4315,
        "fits": True,
    }
    snapshot = engine.analyze_frame(
        "unused.jpg", "biology", canonical_category_ids=["CT-03"]
    )
    assert snapshot["_vlm_metrics"]["context_budget"]["input_tokens"] == 3109
    assert snapshot["visual_evidence"]["provenance"]["context_budget"]["fits"] is True


def test_labeled_brain_diagram_preserves_anatomy_hierarchy_and_leader_lines():
    payload = _base_payload(["CT-03"])
    payload["regions"] = [_region("brain", "CT-03", [
        _element("brain-shape", "anatomical_structure", category="CT-03", label="Brain", child_refs=["cerebrum", "cerebellum"]),
        _element("cerebrum", "anatomical_part", category="CT-03", label="Cerebrum", position="upper and largest", parent_ref="brain-shape"),
        _element("cerebellum", "anatomical_part", category="CT-03", label="Cerebellum", position="lower rear", parent_ref="brain-shape"),
        _element("label-line", "leader_line", category="CT-03", label="Cerebellum callout"),
    ], "A labeled side-view brain diagram.")]
    payload["relations"] = [
        {
            "ref": "leader", "source": "label-line", "target": "cerebellum",
            "type": "points_to", "label": "Cerebellum", "directed": True,
            "cross_category": False, "evidence_status": "supported", "source_refs": [],
        },
        {
            "ref": "part", "source": "cerebellum", "target": "brain-shape",
            "type": "custom:part_of", "label": None, "directed": False,
            "cross_category": False, "evidence_status": "supported", "source_refs": [],
        },
        {
            "ref": "depth", "source": "cerebellum", "target": "cerebrum",
            "type": "custom:behind", "label": None, "directed": False,
            "cross_category": False, "evidence_status": "supported", "source_refs": [],
        },
    ]
    payload["structure"].update({
        "overview": "A side-view brain with the cerebrum above the cerebellum and a callout pointing to the cerebellum.",
        "components": ["brain-shape", "cerebrum", "cerebellum", "label-line"],
        "relations": ["leader", "part", "depth"],
        "reading_order": ["cerebrum", "cerebellum"],
        "hierarchy": ["Cerebrum and cerebellum are shown as parts of the brain."],
    })
    payload["details"] = {"scientific_structure": {
        "components": ["Brain", "Cerebrum", "Cerebellum"],
        "connections": ["Leader line labeled Cerebellum points to the lower-rear part"],
    }}

    engine, snapshot = _analyze(payload, "biology", ["CT-03"])
    evidence = snapshot["visual_evidence"]

    assert validate_visual_evidence(evidence).structurally_valid
    assert evidence["relations"][0]["relation"] == "points_to"
    assert evidence["relations"][0]["label"] == "Cerebellum"
    assert {relation["relation"] for relation in evidence["relations"]} == {
        "points_to", "custom:part_of", "custom:behind"
    }
    assert evidence["regions"][0]["elements"][2]["position"] == "lower rear"
    assert "CT-03" in engine.vlm_interface.prompt
    assert engine.vlm_interface.generation_params["max_new_tokens"] == 768


@pytest.mark.parametrize("relation_type", ["on", "surrounds"])
def test_unknown_relation_is_preserved_as_custom_before_strict_validation(
    relation_type,
):
    payload = _base_payload(["CT-03"])
    payload["regions"] = [_region("diagram", "CT-03", [
        _element("source", "shape", category="CT-03"),
        _element("target", "shape", category="CT-03"),
    ])]
    payload["relations"] = [{
        "ref": "relation", "source": "source", "target": "target",
        "type": relation_type, "label": None, "directed": False,
        "cross_category": False, "evidence_status": "supported",
        "source_refs": [],
    }]
    _, snapshot = _analyze(payload, "diagram", ["CT-03"])
    assert snapshot["_error"] is False
    assert snapshot["visual_evidence"]["relations"][0]["relation"] == (
        f"custom:{relation_type}"
    )
    assert snapshot["_vlm_metrics"]["relation_normalization_count"] == 1


def test_invalid_observation_is_retained_without_a_second_interpretation():
    invalid = _base_payload(["CT-04"])
    invalid["regions"] = [_region("flow", "CT-04", [
        _element("start", "node", category="CT-04")
    ])]
    invalid["relations"] = [{
        "ref": "bad", "source": "start", "target": "missing",
        "type": "leads_to", "label": None, "directed": True,
        "cross_category": False, "evidence_status": "supported",
        "source_refs": [],
    }]
    corrected = _base_payload(["CT-04"])
    corrected["regions"] = invalid["regions"]

    class CorrectingVLM:
        def __init__(self):
            self.calls = []
            self.payloads = [invalid, corrected]

        def analyze(self, image_path_or_pil, prompt, generation_params):
            self.calls.append(prompt)
            payload = self.payloads.pop(0)
            return VLMResponse(
                text=json.dumps(payload), model="mock", total_duration_ms=1,
                prompt_eval_count=10, eval_count=20, finish_reason="stop",
            )

    engine = _engine(invalid)
    engine.vlm_interface = CorrectingVLM()
    snapshot = engine.analyze_frame(
        "unused.jpg", "flowchart", canonical_category_ids=["CT-04"]
    )
    assert snapshot["_error"] is True
    assert len(engine.vlm_interface.calls) == 1
    assert snapshot["_structured_observation"] == invalid
    assert snapshot["_failure"]["code"] == "vlm_contract_invalid"


def test_graph_with_axes_values_and_equation_preserves_cross_category_evidence():
    payload = _base_payload(["CT-01", "CT-02"])
    payload["regions"] = [
        _region("plot", "CT-01", [
            _element("x-axis", "axis", category="CT-01", label="Time (s)"),
            _element("y-axis", "axis", category="CT-01", label="Distance (m)"),
            _element("curve", "plotted_curve", category="CT-01", label="d(t)"),
        ]),
        _region("formula", "CT-02", [
            _element("equation", "equation", category="CT-02", text="d(t) = t^2")
        ]),
    ]
    payload["relations"] = [{
        "ref": "correspondence", "source": "equation", "target": "curve",
        "type": "corresponds_to", "label": None, "directed": False,
        "cross_category": True, "evidence_status": "supported", "source_refs": [],
    }]
    payload["structure"].update({
        "overview": "A distance-time graph on the left and its equation on the right.",
        "components": ["x-axis", "y-axis", "curve", "equation"],
        "relations": ["correspondence"],
        "reading_order": ["x-axis", "y-axis", "curve", "equation"],
    })
    payload["exact"] = [
        {"text": "Time (s)", "status": "supported", "notation": "plain", "source_ref": "x-axis"},
        {"text": "d(t) = t^2", "status": "supported", "notation": "math", "source_ref": "equation"},
    ]
    payload["details"] = {
        "chart": {"axes": [{"label": "Time", "unit": "s"}, {"label": "Distance", "unit": "m"}], "visible_values": ["0", "1", "4"]},
        "mathematics": {"exact_expressions": ["d(t) = t^2"]},
    }

    _, snapshot = _analyze(payload, "graph", ["CT-01", "CT-02"])
    evidence = snapshot["visual_evidence"]
    assert evidence["category_ids"] == ["CT-01", "CT-02"]
    assert evidence["relations"][0]["cross_category"] is True
    assert evidence["exact_content"][1]["text"] == "d(t) = t^2"
    assert evidence["exact_content"][1]["source_ref"] == "element-004"


def test_table_chart_correspondence_preserves_cell_header_and_cross_region_mapping():
    payload = _base_payload(["CT-05", "CT-01"])
    payload["regions"] = [
        _region("table", "CT-05", [
            _element("score-header", "column_header", category="CT-05", text="Score"),
            _element("ada-score", "table_cell", category="CT-05", text="95", attributes={"row_index": 1, "column_index": 2}),
        ]),
        _region("chart", "CT-01", [
            _element("ada-bar", "bar", category="CT-01", label="Ada", attributes={"value": 95})
        ]),
    ]
    payload["relations"] = [
        {"ref": "header", "source": "ada-score", "target": "score-header", "type": "depends_on", "label": "column header", "directed": True, "cross_category": False, "evidence_status": "supported", "source_refs": []},
        {"ref": "bar-match", "source": "ada-score", "target": "ada-bar", "type": "corresponds_to", "label": "Ada score", "directed": False, "cross_category": True, "evidence_status": "supported", "source_refs": []},
    ]
    payload["structure"].update({
        "overview": "A score table on the left corresponds to a bar chart on the right.",
        "components": ["score-header", "ada-score", "ada-bar"],
        "relations": ["header", "bar-match"],
        "reading_order": ["score-header", "ada-score", "ada-bar"],
    })
    payload["details"] = {
        "table": {"headers": ["Score"], "cells": [{"row": 1, "column": 2, "text": "95", "row_header": "Ada", "column_header": "Score"}]},
        "chart": {"series": [{"label": "Ada", "visible_value": "95"}], "visible_values": ["95"]},
    }
    _, snapshot = _analyze(payload, "table", ["CT-05", "CT-01"])
    evidence = snapshot["visual_evidence"]
    assert evidence["relations"][1]["cross_category"] is True
    assert evidence["relations"][1]["relation"] == "corresponds_to"
    assert evidence["category_details"]["table"]["cells"][0]["column_header"] == "Score"


def test_circuit_junction_crossing_and_open_switch_are_not_conflated():
    payload = _base_payload(["CT-03"])
    payload["regions"] = [_region("circuit", "CT-03", [
        _element("battery", "battery", category="CT-03", label="9 V"),
        _element("junction", "wire_junction", category="CT-03", description="filled junction dot"),
        _element("crossing", "wire_crossing", category="CT-03", description="crossing with no junction dot"),
        _element("switch", "switch", category="CT-03", attributes={"state": "open"}),
    ])]
    payload["relations"] = [
        {"ref": "wire-1", "source": "battery", "target": "junction", "type": "connected_to", "label": "wire", "directed": False, "cross_category": False, "evidence_status": "supported", "source_refs": []},
        {"ref": "wire-2", "source": "junction", "target": "switch", "type": "connected_to", "label": "wire", "directed": False, "cross_category": False, "evidence_status": "supported", "source_refs": []},
    ]
    payload["structure"].update({"components": ["battery", "junction", "crossing", "switch"], "relations": ["wire-1", "wire-2"], "topology": ["The crossing is not a junction."], "reading_order": ["battery", "junction", "crossing", "switch"]})
    payload["details"] = {"scientific_structure": {"components": ["9 V battery", "junction", "wire crossing", "open switch"], "connections": ["battery-junction", "junction-switch"]}}
    _, snapshot = _analyze(payload, "circuit", ["CT-03"])
    evidence = snapshot["visual_evidence"]
    assert evidence["regions"][0]["elements"][3]["attributes"]["state"] == "open"
    assert all(relation["target_id"] != "element-003" for relation in evidence["relations"])
    assert not any(relation["directed"] for relation in evidence["relations"])


def test_unresolved_flowchart_arrow_stays_uncertain_without_invented_edge():
    payload = _base_payload(["CT-04"], status="partial", evidence_status="ambiguous_or_conflicting")
    payload["regions"] = [_region("flow", "CT-04", [
        _element("decision", "decision_node", category="CT-04", label="Ready?"),
        _element("next", "process_node", category="CT-04", label="Continue"),
    ])]
    payload["structure"].update({"components": ["decision", "next"], "reading_order": ["decision", "next"], "uncertainties": ["A faint arrow has no readable target attachment."]})
    payload["uncertainties"] = [{"target": "decision", "reason": "Faint arrow endpoint is unresolved.", "impact": "No branch edge is asserted."}]
    payload["missing"] = ["Branch label and arrow target are unreadable."]
    _, snapshot = _analyze(payload, "flowchart", ["CT-04"])
    assert snapshot["visual_evidence"]["relations"] == []
    assert snapshot["visual_evidence"]["uncertainties"][0]["target_id"] == "element-001"


@pytest.mark.parametrize(
    ("primary_type", "category", "elements", "relations", "details_key", "expected"),
    [
        (
            "circuit", "CT-03",
            [_element("battery", "battery", category="CT-03", label="9 V"), _element("r1", "resistor", category="CT-03", label="R1")],
            [{"ref": "wire", "source": "battery", "target": "r1", "type": "connected_to", "label": None, "directed": False, "cross_category": False, "evidence_status": "supported", "source_refs": []}],
            "scientific_structure", "connected_to",
        ),
        (
            "flowchart", "CT-04",
            [_element("decision", "decision_node", category="CT-04", label="Valid?"), _element("yes", "terminal", category="CT-04", label="Accept"), _element("no", "terminal", category="CT-04", label="Reject")],
            [
                {"ref": "yes-edge", "source": "decision", "target": "yes", "type": "branches_to", "label": "Yes", "directed": True, "cross_category": False, "evidence_status": "supported", "source_refs": []},
                {"ref": "no-edge", "source": "decision", "target": "no", "type": "branches_to", "label": "No", "directed": True, "cross_category": False, "evidence_status": "supported", "source_refs": []},
            ],
            "flow_network", "Yes",
        ),
    ],
)
def test_circuit_and_flowchart_topology(primary_type, category, elements, relations, details_key, expected):
    payload = _base_payload([category])
    payload["regions"] = [_region("main", category, elements)]
    payload["relations"] = relations
    payload["structure"].update({
        "components": [item["ref"] for item in elements],
        "relations": [item["ref"] for item in relations],
        "reading_order": [item["ref"] for item in elements],
    })
    payload["details"] = {details_key: {}}
    _, snapshot = _analyze(payload, primary_type, [category])
    serialized_relations = json.dumps(snapshot["visual_evidence"]["relations"])
    assert expected in serialized_relations


def test_table_math_and_programming_exactness_use_one_outer_contract():
    cases = [
        ("table", "CT-05", "table", {"headers": ["Name", "Score"], "row_count": 2, "column_count": 2, "cells": [{"row": 1, "column": 2, "text": "95"}]}, "95"),
        ("equation", "CT-02", "mathematics", {"exact_expressions": ["(x + 1)^2"]}, "(x + 1)^2"),
        ("code", "CT-08", "programming", {"language": "python", "exact_code": "if x >= 2:\n    print(x)", "indentation_significant": True}, ">="),
    ]
    for primary, category, detail_key, details, exact in cases:
        payload = _base_payload([category])
        payload["details"] = {detail_key: details}
        _, snapshot = _analyze(payload, primary, [category])
        assert snapshot["_structured_observation"]["schema"] == COMPACT_OBSERVATION_SCHEMA
        assert exact in json.dumps(snapshot["visual_evidence"]["category_details"])


def test_three_category_frame_preserves_every_category_without_forcing_hint():
    payload = _base_payload(["CT-01", "CT-02", "CT-06"])
    _, snapshot = _analyze(payload, "slide", [], labels=["slide"])
    assert snapshot["visual_evidence"]["category_ids"] == ["CT-01", "CT-02", "CT-06"]
    prompt = _engine(payload)._build_prompt("slide", canonical_category_ids=[])
    assert "No unambiguous canonical hint" in prompt
    assert "do not force" in prompt.lower()


def test_three_category_scene_preserves_elements_exact_content_and_cross_category_links():
    payload = _base_payload(["CT-01", "CT-02", "CT-06"])
    payload["regions"] = [
        _region("plot", "CT-01", [_element("curve", "curve", category="CT-01", label="y")]),
        _region("math", "CT-02", [_element("formula", "equation", category="CT-02", text="y = x²")]),
        _region("note", "CT-06", [_element("caption", "caption", category="CT-06", text="Quadratic growth")]),
    ]
    payload["relations"] = [
        {"ref": "formula-curve", "source": "formula", "target": "curve", "type": "corresponds_to", "label": "same y", "directed": False, "cross_category": True, "evidence_status": "supported", "source_refs": []},
        {"ref": "caption-curve", "source": "caption", "target": "curve", "type": "labels", "label": "Quadratic growth", "directed": True, "cross_category": True, "evidence_status": "supported", "source_refs": []},
    ]
    payload["structure"].update({"overview": "A curve, formula and caption form one three-part visual.", "components": ["curve", "formula", "caption"], "relations": ["formula-curve", "caption-curve"], "reading_order": ["curve", "formula", "caption"]})
    payload["exact"] = [{"text": "y = x²", "status": "supported", "notation": "math", "source_ref": "formula"}]
    payload["details"] = {"chart": {"series": [{"label": "y"}]}, "mathematics": {"exact_expressions": ["y = x²"]}, "structured_text": {"annotations": ["Quadratic growth"]}}
    _, snapshot = _analyze(payload, "slide", [])
    evidence = snapshot["visual_evidence"]
    assert evidence["category_ids"] == ["CT-01", "CT-02", "CT-06"]
    assert len(evidence["regions"]) == 3
    assert len(evidence["relations"]) == 2
    assert all(relation["cross_category"] for relation in evidence["relations"])
    assert evidence["exact_content"][0]["source_ref"] == "element-002"


def test_unreadable_label_and_ambiguous_connection_remain_partial():
    payload = _base_payload(["CT-03"], status="partial", evidence_status="ambiguous_or_conflicting")
    payload["regions"] = [_region("diagram", "CT-03", [
        _element("a", "component", category="CT-03", label="[ILLEGIBLE]", status="unavailable_or_unreadable"),
        _element("b", "component", category="CT-03", label="B"),
    ])]
    payload["structure"]["components"] = ["a", "b"]
    payload["structure"]["uncertainties"] = ["A faint line may connect A and B."]
    payload["uncertainties"] = [{"target": "scene", "reason": "The possible connector is faint and discontinuous.", "impact": "No connection is asserted."}]
    payload["missing"] = ["The first component label is unreadable."]
    _, snapshot = _analyze(payload, "diagram", [])
    evidence = snapshot["visual_evidence"]
    assert evidence["extraction_status"] == "partial"
    assert evidence["relations"] == []
    assert evidence["uncertainties"][0]["reason"].startswith("The possible connector")
    assert snapshot["missing_elements"] is None


@pytest.mark.parametrize("raw", ["not json", '{"schema":"synapse_visual_observation_v1"'])
def test_non_json_and_truncated_json_are_preserved_as_failed_evidence(raw):
    _, snapshot = _analyze(raw, "graph", ["CT-01"])
    assert snapshot["_error"] is True
    assert snapshot["missing_elements"] == "Parse error"
    assert snapshot["visual_evidence"]["extraction_status"] == "failed"
    assert snapshot["visual_evidence"]["legacy_snapshot"]["_raw_response"] == raw
    assert snapshot["visual_evidence"]["legacy_snapshot"]["structural_description"] == ""


def test_invalid_structured_reference_fails_without_losing_model_output():
    payload = _base_payload(["CT-04"])
    payload["regions"] = [_region("process", "CT-04", [_element("start", "node", category="CT-04")])]
    payload["relations"] = [{
        "ref": "bad", "source": "start", "target": "missing", "type": "leads_to",
        "label": None, "directed": True, "cross_category": False,
        "evidence_status": "supported", "source_refs": [],
    }]
    _, snapshot = _analyze(payload, "flowchart", ["CT-04"])
    assert snapshot["_error"] is True
    assert "unknown observation ref 'missing'" in snapshot["missing_elements"]
    assert snapshot["_structured_observation"] == payload
    assert snapshot["visual_evidence"]["extraction_status"] == "failed"


def test_dangling_exact_content_reference_fails_without_losing_model_output():
    payload = _base_payload(["CT-02"])
    payload["regions"] = [_region("math", "CT-02", [
        _element("equation", "equation", category="CT-02", text="a = b")
    ])]
    payload["structure"].update({"components": ["equation"], "reading_order": ["equation"]})
    payload["exact"] = [{"text": "a = b", "status": "supported", "notation": "math", "source_ref": "missing"}]
    _, snapshot = _analyze(payload, "equation", ["CT-02"])
    assert snapshot["_error"] is True
    assert "exact source_ref references unknown observation ref 'missing'" in snapshot["missing_elements"]
    assert snapshot["_structured_observation"] == payload


def test_output_limit_marks_possible_truncation_as_partial():
    payload = _base_payload(["CT-01"])
    _, snapshot = _analyze(payload, "graph", ["CT-01"], completion_tokens=768)
    evidence = snapshot["visual_evidence"]
    assert snapshot["_vlm_metrics"]["possible_truncation"] is True
    assert snapshot["_vlm_metrics"]["finish_reason"] is None
    assert snapshot["_vlm_metrics"]["truncation_detection"] == "token_count_fallback"
    assert evidence["extraction_status"] == "partial"
    assert any(item["code"] == "extraction.possible_truncation" for item in evidence["validation_diagnostics"])


def test_missing_finish_reason_below_limit_records_non_truncated_fallback():
    payload = _base_payload(["CT-01"])
    _, snapshot = _analyze(payload, "graph", ["CT-01"], completion_tokens=120)
    assert snapshot["_vlm_metrics"]["finish_reason"] is None
    assert snapshot["_vlm_metrics"]["truncation_detection"] == "token_count_fallback"
    assert snapshot["_vlm_metrics"]["possible_truncation"] is False


def test_stop_finish_reason_overrides_token_count_heuristic():
    payload = _base_payload(["CT-01"])
    payload["regions"] = [_region("plot", "CT-01", [
        _element("axis", "axis", category="CT-01", label="x")
    ])]
    payload["structure"].update({"components": ["axis"], "reading_order": ["axis"]})
    _, snapshot = _analyze(
        payload, "graph", ["CT-01"], completion_tokens=768, finish_reason="stop"
    )
    assert snapshot["_vlm_metrics"]["possible_truncation"] is False
    assert snapshot["_vlm_metrics"]["truncation_detection"] == "finish_reason"
    assert snapshot["visual_evidence"]["extraction_status"] == "complete"


def test_length_finish_reason_marks_short_completion_partial():
    payload = _base_payload(["CT-01"])
    _, snapshot = _analyze(
        payload, "graph", ["CT-01"], completion_tokens=120, finish_reason="length"
    )
    assert snapshot["_vlm_metrics"]["possible_truncation"] is True
    assert snapshot["_vlm_metrics"]["finish_reason"] == "length"
    assert snapshot["visual_evidence"]["extraction_status"] == "partial"


def test_unknown_finish_reason_is_not_silently_treated_as_complete():
    payload = _base_payload(["CT-01"])
    _, snapshot = _analyze(
        payload, "graph", ["CT-01"], finish_reason="content_filter"
    )
    assert snapshot["_vlm_metrics"]["possible_truncation"] is True
    assert snapshot["_vlm_metrics"]["truncation_detection"] == "unknown_finish_reason"
    assert snapshot["visual_evidence"]["extraction_status"] == "partial"


def test_legacy_qwen_snapshot_keeps_old_fields_and_adds_canonical_adapter():
    legacy = {
        "content_type": "equation", "detected_types": ["equation"],
        "visual_analysis": {"equation": [{"text": "E = mc^2"}]},
        "structural_description": "Centered equation", "reading_order": ["E", "equals", "m c squared"],
        "conceptual_hints": "Mass-energy equivalence", "layout": "centered",
        "text_readout": "E = mc^2", "spatial_map": "center", "colors_styles": "white",
        "missing_elements": None,
    }
    _, snapshot = _analyze(legacy, "equation", ["CT-02"])
    for key, value in legacy.items():
        assert snapshot[key] == value
    assert snapshot["visual_evidence"]["legacy_snapshot"] == legacy


def test_structural_facts_survive_into_all_mistral_mode_prompts():
    payload = _base_payload(["CT-04"])
    payload["regions"] = [_region("flow", "CT-04", [
        _element("decision", "decision_node", category="CT-04", label="Ready?"),
        _element("next", "process_node", category="CT-04", label="Continue"),
    ])]
    payload["relations"] = [{
        "ref": "yes", "source": "decision", "target": "next", "type": "branches_to",
        "label": "Yes", "directed": True, "cross_category": False,
        "evidence_status": "supported", "source_refs": [],
    }]
    payload["structure"].update({"components": ["decision", "next"], "relations": ["yes"], "reading_order": ["decision", "next"]})
    _, snapshot = _analyze(payload, "flowchart", ["CT-04"])
    synth = object.__new__(LLMSynthesizer)
    context = {"previous_concepts": [], "context_text": "", "related_principles": [], "topic_timeline": {}}
    prompts = [
        synth._build_brief_prompt("The teacher asks if it is ready.", context, snapshot),
        synth._build_explanatory_prompt("The teacher asks if it is ready.", context, snapshot),
        synth._build_detailed_prompt("The teacher asks if it is ready.", context, snapshot),
    ]
    for prompt in prompts:
        assert "CANONICAL VISUAL EVIDENCE" in prompt
        assert "Ready?" in prompt
        assert "branches_to" in prompt
        assert '"label":"Yes"' in prompt
        assert "Never upgrade partial" in prompt


@dataclass
class DummyIntervention:
    timestamp: float = 22.0
    frame_path: str = "unused.jpg"
    content_type: str = "biology"
    classification_confidence: float = 0.9
    confidence_semantics: str = "classifier"
    raw_category_labels: list[str] | None = None
    canonical_categories: list[str] | None = None


def test_batch_orchestrator_attaches_intervention_metadata(monkeypatch):
    from orchestrator.orchestrator import PipelineOrchestrator

    payload = _base_payload(["CT-03"])
    real_engine = _engine(payload)

    class FakeSnapshotEngine:
        model_name = "Qwen/Qwen3-VL-8B-Instruct-FP8"

        def __init__(self, **kwargs):
            pass

        def analyze_frame(self, *args, **kwargs):
            return real_engine.analyze_frame(*args, **kwargs)

        def cleanup(self):
            pass

    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", "modal")
    monkeypatch.setattr("orchestrator.orchestrator.SnapshotEngine", FakeSnapshotEngine)
    intervention = DummyIntervention(
        raw_category_labels=["biology"], canonical_categories=["CT-03"]
    )
    snapshots = PipelineOrchestrator._analyze_vlm_snapshots(
        object.__new__(PipelineOrchestrator), [intervention], "."
    )
    evidence = snapshots[22.0]["visual_evidence"]
    assert evidence["timestamp_seconds"] == 22.0
    assert evidence["raw_category_labels"] == ["biology"]
    assert evidence["category_ids"] == ["CT-03"]


def test_on_demand_flow_supplies_canonical_evidence_and_preserves_mode(tmp_path, monkeypatch):
    from orchestrator.orchestrator import PipelineOrchestrator

    payload = _base_payload(["CT-03"])
    payload["regions"] = [_region("brain", "CT-03", [
        _element("cerebrum", "anatomical_part", category="CT-03", label="Cerebrum")
    ])]
    payload["structure"]["components"] = ["cerebrum"]
    real_engine = _engine(payload)

    class FakeSnapshotEngine:
        model_name = "Qwen/Qwen3-VL-8B-Instruct-FP8"

        def __init__(self, **kwargs):
            pass

        def analyze_frame(self, *args, **kwargs):
            return real_engine.analyze_frame(*args, **kwargs)

        def cleanup(self):
            pass

    captured = {}

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            captured["init_mode"] = kwargs["output_mode"]

        def weave(self, **kwargs):
            captured.update(kwargs)
            return "The cerebrum is the labeled central structure."

        def cleanup(self):
            pass

    class FakeTTS:
        def generate(self, text, path):
            Path(path).write_bytes(b"ID3-task2b")

        def cleanup(self):
            pass

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", "modal")
    monkeypatch.setattr("orchestrator.orchestrator.SnapshotEngine", FakeSnapshotEngine)
    monkeypatch.setattr(
        "orchestrator.orchestrator.get_ollama_service",
        lambda: types.SimpleNamespace(ensure_ready=lambda: None),
    )
    monkeypatch.setattr("orchestrator.orchestrator.log_memory_state", lambda *args: None)
    monkeypatch.setattr("orchestrator.orchestrator.time.sleep", lambda seconds: None)
    monkeypatch.setattr("synthesizer.LLMSynthesizer", FakeSynthesizer)
    fake_tts_module = types.ModuleType("tts_engine")
    fake_tts_module.TTSEngine = FakeTTS
    fake_tts_module.get_shared_tts_engine = lambda: FakeTTS()
    monkeypatch.setitem(sys.modules, "tts_engine", fake_tts_module)

    orchestrator = PipelineOrchestrator(base_dir=str(tmp_path))
    result = orchestrator.generate_intervention_explanation(
        video_id="video-1",
        intervention_id="intervention-1",
        intervention_timestamp=33.0,
        frame_path=str(frame),
        content_type="biology",
        trigger_reason="Deictic",
        transcript_entries=[{"start": 30.0, "duration": 5.0, "text": "This is the cerebrum."}],
        output_mode="detailed",
        raw_category_labels=["biology"],
        canonical_category_ids=["CT-03"],
    )
    assert captured["init_mode"] == "detailed"
    assert captured["output_mode"] == "detailed"
    assert captured["require_visual_explanation"] is True
    assert captured["vlm_snapshot"]["visual_evidence"]["frame_id"] == "intervention-1"
    assert captured["vlm_snapshot"]["coverage_plan"]["schema"] == "synapse_transcript_visual_coverage_v1"
    assert captured["vlm_snapshot"]["coverage_plan"]["fact_count"] >= 1
    saved = json.loads(Path(result["vlm_snapshot_path"]).read_text(encoding="utf-8"))
    assert saved["visual_evidence"]["regions"][0]["elements"][0]["label"] == "Cerebrum"
    assert saved["coverage_plan"] == captured["vlm_snapshot"]["coverage_plan"]

