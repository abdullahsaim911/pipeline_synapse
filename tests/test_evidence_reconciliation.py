"""Task 4 deterministic transcript/visual reconciliation tests and fixtures."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import pytest

from synchronizer import Synchronizer, reconcile_visual_evidence
from synthesizer.llm_synthesizer import LLMSynthesizer


def _element(
    element_id,
    label,
    category,
    *,
    kind="component",
    position=None,
    exact_text=None,
    attributes=None,
    status="supported",
    parent_id=None,
):
    return {
        "id": element_id,
        "kind": kind,
        "label": label,
        "exact_text": exact_text,
        "position": position,
        "attributes": attributes or {},
        "category_ids": [category],
        "evidence_status": status,
        "parent_id": parent_id,
        "child_ids": [],
    }


def _evidence(
    category,
    *,
    elements=None,
    relations=None,
    exact=None,
    details=None,
    status="complete",
    evidence_status="supported",
    uncertainties=None,
    structure=None,
):
    elements = elements or []
    return {
        "schema_version": "1.1.0",
        "frame_id": "frame-eval",
        "timestamp_seconds": 12.0,
        "extraction_status": status,
        "evidence_status": evidence_status,
        "category_ids": [category] if isinstance(category, str) else category,
        "regions": [{
            "id": "region-001",
            "category_ids": [category] if isinstance(category, str) else category,
            "elements": elements,
            "evidence_status": evidence_status,
        }] if elements else [],
        "relations": relations or [],
        "uncertainties": uncertainties or [],
        "structural_description": {
            "overview": "Evaluation fixture",
            "orientation": None,
            "component_ids": [item["id"] for item in elements],
            "relation_ids": [item["id"] for item in relations or []],
            "reading_order": [],
            "hierarchy_notes": [],
            "topology_notes": [],
            "uncertainties": [],
            **(structure or {}),
        },
        "exact_content": exact or [],
        "category_details": details or {},
        "validation_diagnostics": [],
    }


def _plan(evidence, transcript):
    return reconcile_visual_evidence({"visual_evidence": evidence}, transcript)


def _find_fact(plan, text, *, fact_type=None):
    matches = [
        item for item in plan["facts"]
        if text.casefold() in item["description"].casefold()
        and (fact_type is None or item["fact_type"] == fact_type)
    ]
    assert matches, f"No fact containing {text!r}: {[f['description'] for f in plan['facts']]}"
    return matches[0]


CATEGORY_DETAILS = [
    ("CT-01", {"chart": {"axes": [{"label": "Time", "unit": "s"}]}}),
    ("CT-02", {"mathematics": {"exact_expressions": ["x <= 5"]}}),
    ("CT-03", {"scientific_structure": {"connections": ["battery connected to resistor"]}}),
    ("CT-04", {"flow_network": {"branch_conditions": ["Yes"]}}),
    ("CT-05", {"table": {"cells": [{"row": 1, "column": 2, "text": "95"}]}}),
    ("CT-06", {"structured_text": {"annotations": ["Do not divide by zero"]}}),
    ("CT-07", {"geometry": {"constraints": ["right angle at B"]}}),
    ("CT-08", {"programming": {"language": "python", "exact_code": "if x >= 2:\n    print(x)", "indentation_significant": True}}),
    ("CT-09", {"interface": {"visible_states": ["Wi-Fi toggle off"]}}),
    ("CT-10", {"map": {"directions": ["north arrow points up"]}}),
    ("CT-11", {"observational_image": {"visible_features": ["rough brown surface"]}}),
    ("CT-12", {"physical_scene": {"observable_states": ["force arrow points right"]}}),
]


@pytest.mark.parametrize(("category", "details"), CATEGORY_DETAILS)
def test_all_twelve_category_details_become_source_grounded_atomic_facts(category, details):
    plan = _plan(_evidence(category, details=details), "")
    facts = [item for item in plan["facts"] if item["fact_type"] == "category_detail"]
    assert facts
    assert all(item["source_refs"] == ["scene"] for item in facts)
    assert all(category in item["category_ids"] for item in facts)
    assert all(item["transcript_coverage"] == "unspoken" for item in facts)
    assert all(item["missing_information"] == item["description"] for item in facts)


def test_brain_name_does_not_imply_location_or_hierarchy():
    evidence = _evidence("CT-03", elements=[
        _element("brain", "Brain", "CT-03", kind="anatomical_structure"),
        _element("cerebellum", "Cerebellum", "CT-03", kind="anatomical_part", position="lower rear", parent_id="brain"),
    ])
    plan = _plan(evidence, "The cerebellum is shown here.")
    assert _find_fact(plan, "Visible anatomical part: Cerebellum")["transcript_coverage"] == "spoken_explicit"
    assert _find_fact(plan, "Cerebellum is lower rear")["transcript_coverage"] == "spoken_partial"
    assert _find_fact(plan, "Cerebellum is part of Brain")["transcript_coverage"] == "unspoken"


def test_graph_trend_does_not_cover_axes_or_units():
    evidence = _evidence("CT-01", elements=[
        _element("x-axis", "Time (s)", "CT-01", kind="axis", attributes={"unit": "s"}),
        _element("y-axis", "Distance (m)", "CT-01", kind="axis", attributes={"unit": "m"}),
    ], details={"chart": {"observed_trends": ["curve rises left to right"]}})
    plan = _plan(evidence, "The curve rises left to right.")
    assert _find_fact(plan, "curve rises left to right")["transcript_coverage"] == "spoken_explicit"
    assert _find_fact(plan, "Time (s) unit is s")["transcript_coverage"] == "unspoken"
    assert _find_fact(plan, "Distance (m) unit is m")["transcript_coverage"] == "unspoken"


@pytest.mark.parametrize(
    ("visual", "transcript"),
    [
        ("x <= 5", "The expression is x < 5."),
        ("if x >= 2:\n    print(x)", "The code says if x > 2, print x."),
        ("Do not divide by zero.", "Divide by zero."),
    ],
)
def test_exact_math_code_and_negation_are_not_covered_by_similar_text(visual, transcript):
    category = "CT-08" if visual.startswith("if") else "CT-02" if visual.startswith("x") else "CT-06"
    notation = "code" if category == "CT-08" else "math" if category == "CT-02" else "plain"
    evidence = _evidence(category, exact=[{
        "text": visual, "status": "supported", "notation": notation, "source_ref": "scene"
    }])
    fact = _find_fact(_plan(evidence, transcript), visual, fact_type="exact_content")
    assert fact["transcript_coverage"] == "unspoken"


def test_exact_unicode_mathematical_notation_is_preserved_and_matched():
    evidence = _evidence("CT-02", exact=[{
        "text": "x ≤ 5 ≠ y", "status": "supported", "notation": "math", "source_ref": "scene"
    }])
    fact = _find_fact(_plan(evidence, "The expression is x ≤ 5 ≠ y."), "x ≤ 5 ≠ y", fact_type="exact_content")
    assert fact["description"] == "math: x ≤ 5 ≠ y"
    assert fact["transcript_coverage"] == "spoken_explicit"


def _flow_evidence():
    return _evidence("CT-04", elements=[
        _element("decision", "Valid", "CT-04", kind="decision_node"),
        _element("accept", "Accept", "CT-04", kind="terminal_node"),
    ], relations=[{
        "id": "relation-001", "source_id": "decision", "target_id": "accept",
        "relation": "branches_to", "label": "Yes", "directed": True,
        "cross_category": False, "evidence_status": "supported", "source_refs": [],
    }])


def test_directed_flowchart_branch_requires_supported_direction_and_label():
    correct = _find_fact(_plan(_flow_evidence(), "Valid branches to Accept on Yes."), "branches to", fact_type="relation")
    reversed_fact = _find_fact(_plan(_flow_evidence(), "Accept branches to Valid on Yes."), "branches to", fact_type="relation")
    assert correct["transcript_coverage"] == "spoken_explicit"
    assert reversed_fact["transcript_coverage"] == "uncertain_match"


def test_circuit_crossing_language_does_not_cover_connected_junction():
    evidence = _evidence("CT-03", elements=[
        _element("battery", "Battery", "CT-03", kind="battery"),
        _element("junction", "Junction J1", "CT-03", kind="wire_junction"),
    ], relations=[{
        "id": "relation-001", "source_id": "battery", "target_id": "junction",
        "relation": "connected_to", "label": None, "directed": False,
        "cross_category": False, "evidence_status": "supported", "source_refs": [],
    }])
    fact = _find_fact(_plan(evidence, "Two wires cross near Junction J1."), "connected to", fact_type="relation")
    assert fact["transcript_coverage"] == "unspoken"


def test_table_cell_requires_row_column_association():
    evidence = _evidence("CT-05", details={"table": {"cells": [{
        "row_header": "Ada", "column_header": "Score", "text": "95"
    }]}})
    exact = _find_fact(_plan(evidence, "row header Ada; column header Score; text 95"), "row header Ada", fact_type="category_detail")
    partial = _find_fact(_plan(evidence, "The value is 95."), "row header Ada", fact_type="category_detail")
    assert exact["transcript_coverage"] == "spoken_explicit"
    assert partial["transcript_coverage"] == "unspoken"


def test_mixed_category_cross_relation_is_preserved_and_matched():
    evidence = _evidence(["CT-01", "CT-02"], elements=[
        _element("equation", "Equation y equals x squared", "CT-02", kind="equation"),
        _element("curve", "Curve y", "CT-01", kind="curve"),
    ], relations=[{
        "id": "relation-001", "source_id": "equation", "target_id": "curve",
        "relation": "corresponds_to", "label": "same y", "directed": False,
        "cross_category": True, "evidence_status": "supported", "source_refs": [],
    }])
    fact = _find_fact(
        _plan(evidence, "Equation y equals x squared corresponds to Curve y."),
        "corresponds to", fact_type="relation",
    )
    assert fact["transcript_coverage"] == "spoken_explicit"
    assert fact["category_ids"] == ["CT-02", "CT-01"]


def test_deictic_empty_partial_and_failed_evidence_use_safe_labels():
    supported = _evidence("CT-03", elements=[_element("brain", "Cerebrum", "CT-03")])
    assert _find_fact(_plan(supported, "Look here at this."), "Cerebrum")["transcript_coverage"] == "unspoken"

    partial = copy.deepcopy(supported)
    partial["extraction_status"] = "partial"
    fact = _find_fact(_plan(partial, "The cerebrum is visible."), "Cerebrum")
    assert fact["transcript_coverage"] == "uncertain_match"

    failed = copy.deepcopy(supported)
    failed["extraction_status"] = "failed"
    failed["evidence_status"] = "unavailable_or_unreadable"
    failed["regions"][0]["evidence_status"] = "unavailable_or_unreadable"
    failed["regions"][0]["elements"][0]["evidence_status"] = "unavailable_or_unreadable"
    fact = _find_fact(_plan(failed, "The cerebrum is visible."), "Cerebrum")
    assert fact["transcript_coverage"] == "not_usable"


def test_plan_is_bounded_and_reports_omissions():
    elements = [_element(f"e{i}", f"Label {i}", "CT-06") for i in range(10)]
    plan = reconcile_visual_evidence(
        {"visual_evidence": _evidence("CT-06", elements=elements)}, "", max_facts=3
    )
    assert plan["fact_count"] == 3
    assert plan["truncated"] is True
    assert plan["omitted_fact_count"] == 7
    assert plan["total_fact_count"] == 10


def test_all_mistral_modes_receive_same_compact_coverage_plan():
    snapshot = {"visual_evidence": _evidence("CT-03", elements=[
        _element("brain", "Cerebellum", "CT-03", position="lower rear")
    ])}
    snapshot["coverage_plan"] = reconcile_visual_evidence(snapshot, "The cerebellum is shown.")
    snapshot.update({"detected_types": ["biology"], "visual_analysis": {}, "structural_description": "", "reading_order": [], "conceptual_hints": ""})
    synth = object.__new__(LLMSynthesizer)
    context = {"previous_concepts": [], "context_text": "", "related_principles": [], "topic_timeline": {}}
    prompts = [
        synth._build_brief_prompt("The cerebellum is shown.", context, snapshot),
        synth._build_explanatory_prompt("The cerebellum is shown.", context, snapshot),
        synth._build_detailed_prompt("The cerebellum is shown.", context, snapshot),
    ]
    for prompt in prompts:
        assert "TRANSCRIPT-VISUAL COVERAGE PLAN" in prompt
        assert "spoken_partial" in prompt
        assert "Prioritize critical unspoken facts" in prompt


def test_synchronizer_scores_are_independent_of_reconciliation():
    synchronizer = object.__new__(Synchronizer)
    synchronizer.transcripts = []
    frame = {"timestamp_seconds": 10.0, "complexity_score": 0.31}
    before = synchronizer._calculate_score(frame)
    _plan(_evidence("CT-01"), "irrelevant")
    after = synchronizer._calculate_score(frame)
    assert before == after == (50, "High Complexity", None)


def test_batch_synthesis_receives_coverage_plan(tmp_path):
    from orchestrator.orchestrator import PipelineOrchestrator
    from synchronizer import TranscriptEntry

    snapshot = {
        "content_type": "biology",
        "visual_evidence": _evidence("CT-03", elements=[
            _element("brain", "Cerebellum", "CT-03", position="lower rear")
        ]),
    }
    captured = {}

    class FakeSynthesizer:
        def weave(self, **kwargs):
            captured.update(kwargs)
            return "description"

    class FakeTTS:
        def generate(self, text, path):
            return None

        def cleanup(self):
            return None

    orchestrator = object.__new__(PipelineOrchestrator)
    results = orchestrator._synthesize_interventions_only(
        FakeSynthesizer(),
        FakeTTS(),
        [TranscriptEntry(start=10.0, end=15.0, text="The cerebellum is shown.")],
        {12.0: snapshot},
        str(tmp_path),
        "brief",
    )
    assert len(results) == 1
    plan = captured["vlm_snapshot"]["coverage_plan"]
    assert plan["schema"] == "synapse_transcript_visual_coverage_v1"
    assert _find_fact(plan, "lower rear")["transcript_coverage"] == "spoken_partial"


@dataclass
class EvaluationCase:
    name: str
    evidence: dict
    transcript: str
    contains: str
    fact_type: str
    expected: str


def evaluation_cases():
    brain = _evidence("CT-03", elements=[
        _element("cerebellum", "Cerebellum", "CT-03", position="lower rear")
    ])
    graph = _evidence("CT-01", details={"chart": {"observed_trends": ["curve rises left to right"]}})
    math = _evidence("CT-02", exact=[{"text": "x <= 5", "status": "supported", "notation": "math", "source_ref": "scene"}])
    code = _evidence("CT-08", exact=[{"text": "if x >= 2", "status": "supported", "notation": "code", "source_ref": "scene"}])
    negation = _evidence("CT-06", exact=[{"text": "Do not divide by zero", "status": "supported", "notation": "plain", "source_ref": "scene"}])
    table = _evidence("CT-05", details={"table": {"cells": [{"row_header": "Ada", "column_header": "Score", "text": "95"}]}})
    mixed = _evidence(["CT-01", "CT-02"], elements=[
        _element("equation", "Equation y equals x squared", "CT-02"),
        _element("curve", "Curve y", "CT-01"),
    ], relations=[{"id": "r", "source_id": "equation", "target_id": "curve", "relation": "corresponds_to", "label": None, "directed": False, "cross_category": True, "evidence_status": "supported", "source_refs": []}])
    partial = copy.deepcopy(brain)
    partial["extraction_status"] = "partial"
    return [
        EvaluationCase("brain-name", brain, "The cerebellum is shown.", "Visible component: Cerebellum", "component", "spoken_explicit"),
        EvaluationCase("brain-location", brain, "The cerebellum is shown.", "lower rear", "spatial", "spoken_partial"),
        EvaluationCase("brain-location-explicit", brain, "The cerebellum is at the lower rear.", "lower rear", "spatial", "spoken_explicit"),
        EvaluationCase("graph-trend", graph, "The curve rises left to right.", "curve rises", "category_detail", "spoken_explicit"),
        EvaluationCase("math-operator", math, "The expression is x < 5.", "x <= 5", "exact_content", "unspoken"),
        EvaluationCase("code-operator", code, "The code says if x > 2.", "if x >= 2", "exact_content", "unspoken"),
        EvaluationCase("flow-correct", _flow_evidence(), "Valid branches to Accept on Yes.", "branches to", "relation", "spoken_explicit"),
        EvaluationCase("flow-reversed", _flow_evidence(), "Accept branches to Valid on Yes.", "branches to", "relation", "uncertain_match"),
        EvaluationCase("negation", negation, "Divide by zero.", "Do not divide", "exact_content", "unspoken"),
        EvaluationCase("table-association", table, "row header Ada; column header Score; text 95", "row header Ada", "category_detail", "spoken_explicit"),
        EvaluationCase("mixed-relation", mixed, "Equation y equals x squared corresponds to Curve y.", "corresponds to", "relation", "spoken_explicit"),
        EvaluationCase("deictic", brain, "Look here at this.", "Visible component: Cerebellum", "component", "unspoken"),
        EvaluationCase("empty", brain, "", "Visible component: Cerebellum", "component", "unspoken"),
        EvaluationCase("partial-evidence", partial, "The cerebellum is shown.", "Visible component: Cerebellum", "component", "uncertain_match"),
    ]


def _predictions_and_naive_baseline():
    predictions, baseline, expected = [], [], []
    start = time.perf_counter()
    for case in evaluation_cases():
        fact = _find_fact(_plan(case.evidence, case.transcript), case.contains, fact_type=case.fact_type)
        predictions.append(fact["transcript_coverage"])
        expected.append(case.expected)
        subject_tokens = [token for token in case.contains.casefold().split() if len(token) > 2]
        baseline.append(
            "spoken_explicit"
            if any(token.strip(":;,.?") in case.transcript.casefold() for token in subject_tokens)
            else "unspoken"
        )
    runtime_ms = (time.perf_counter() - start) * 1000
    return predictions, baseline, expected, runtime_ms


def _explicit_metrics(predictions, expected):
    true_positive = sum(p == e == "spoken_explicit" for p, e in zip(predictions, expected))
    false_positive = sum(p == "spoken_explicit" and e != "spoken_explicit" for p, e in zip(predictions, expected))
    false_negative = sum(p != "spoken_explicit" and e == "spoken_explicit" for p, e in zip(predictions, expected))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"precision": precision, "recall": recall, "false_matches": false_positive, "false_omissions": false_negative}


def test_held_out_synthetic_evaluation_metrics():
    predictions, baseline, expected, runtime_ms = _predictions_and_naive_baseline()
    selected = _explicit_metrics(predictions, expected)
    naive = _explicit_metrics(baseline, expected)
    assert predictions == expected
    assert selected == {"precision": 1.0, "recall": 1.0, "false_matches": 0, "false_omissions": 0}
    assert naive["false_matches"] > selected["false_matches"]
    assert runtime_ms < 100.0

