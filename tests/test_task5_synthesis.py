"""Task 5 grounded, category-aware Mistral prompt regressions."""

from __future__ import annotations

import copy

import pytest

from synthesizer.llm_synthesizer import LLMSynthesizer
from synthesizer.math_linearizer import MathLinearizer
from synchronizer import reconcile_visual_evidence
from tests.test_evidence_reconciliation import _element, _evidence


CATEGORY_EXPECTATIONS = [
    ("CT-01", "Charts: identify chart kind"),
    ("CT-02", "Mathematics: preserve exact operators"),
    ("CT-03", "Science and technical structures"),
    ("CT-04", "Flows, processes, and networks"),
    ("CT-05", "Tables: preserve row and column headers"),
    ("CT-06", "Instructional text: preserve exact headings"),
    ("CT-07", "Geometry: name points, shapes"),
    ("CT-08", "Programming: preserve exact tokens"),
    ("CT-09", "User interfaces: name controls"),
    ("CT-10", "Maps: state supported orientation"),
    ("CT-11", "Observational images: describe visible attributes"),
    ("CT-12", "Physical scenes: describe apparatus"),
]


def _synth():
    synth = object.__new__(LLMSynthesizer)
    synth.math_linearizer = MathLinearizer()
    synth.context_manager = None
    synth.transcripts = []
    synth.output_mode = "explanatory"
    return synth


def _context():
    return {
        "context_text": "Earlier, the teacher introduced the example.",
        "previous_concepts": [],
        "related_principles": [],
        "topic_timeline": {},
    }


def _snapshot(evidence, transcript=""):
    snapshot = {
        "visual_evidence": evidence,
        "detected_types": [],
        "visual_analysis": {},
        "structural_description": "",
        "reading_order": [],
        "conceptual_hints": "",
    }
    snapshot["coverage_plan"] = reconcile_visual_evidence(snapshot, transcript)
    return snapshot


@pytest.mark.parametrize(("category", "expected_instruction"), CATEGORY_EXPECTATIONS)
def test_all_twelve_categories_select_grounded_narration_instruction(category, expected_instruction):
    snapshot = _snapshot(_evidence(category, elements=[_element("fact", "Fact", category)]))
    prompt = _synth()._build_explanatory_prompt("", _context(), snapshot)
    assert f"- {category}: {expected_instruction}" in prompt


def test_mixed_three_category_frame_preserves_each_directive_and_cross_category_rule():
    evidence = _evidence(["CT-01", "CT-02", "CT-06"], elements=[
        _element("curve", "Curve y", "CT-01"),
        _element("equation", "y = x^2", "CT-02", exact_text="y = x^2"),
        _element("caption", "Warning", "CT-06", exact_text="Do not extrapolate."),
    ], relations=[{
        "id": "relation-001", "source_id": "equation", "target_id": "curve",
        "relation": "corresponds_to", "label": None, "directed": False,
        "cross_category": True, "evidence_status": "supported", "source_refs": [],
    }])
    prompt = _synth()._build_brief_prompt("The curve rises.", _context(), _snapshot(evidence, "The curve rises."))
    assert all(f"- {category}:" in prompt for category in ("CT-01", "CT-02", "CT-06"))
    assert "Mixed frame: preserve each category" in prompt
    assert "y = x^2" in prompt and "corresponds_to" in prompt and "Curve y" in prompt


def test_science_and_text_mixed_frame_keeps_leader_line_hierarchy_and_warning():
    evidence = _evidence(["CT-03", "CT-06"], elements=[
        _element("brain", "Brain", "CT-03", kind="anatomical_structure"),
        _element("cerebellum", "Cerebellum", "CT-03", kind="anatomical_part", position="lower rear", parent_id="brain"),
        _element("label", "Cerebellum label", "CT-06", kind="label"),
    ], relations=[{
        "id": "leader", "source_id": "label", "target_id": "cerebellum",
        "relation": "points_to", "label": "Cerebellum", "directed": True,
        "cross_category": True, "evidence_status": "supported", "source_refs": [],
    }], details={"structured_text": {"annotations": ["Do not confuse this with the cerebrum."]}})
    prompt = _synth()._build_detailed_prompt("", _context(), _snapshot(evidence))
    for expected in ("lower rear", "part of Brain", "points_to", "Do not confuse", "CT-03", "CT-06"):
        assert expected in prompt


def test_chart_labels_units_and_supported_trend_are_available_without_value_inference():
    evidence = _evidence("CT-01", elements=[
        _element("x", "Time", "CT-01", kind="axis", attributes={"unit": "s"}),
        _element("y", "Distance", "CT-01", kind="axis", attributes={"unit": "m"}),
    ], details={"chart": {"observed_trends": ["curve rises left to right"]}})
    prompt = _synth()._build_explanatory_prompt("The curve rises.", _context(), _snapshot(evidence, "The curve rises."))
    assert "Time" in prompt and '"unit":"s"' in prompt
    assert "Distance" in prompt and '"unit":"m"' in prompt
    assert "Do not turn approximate geometry into exact values" in prompt


def test_math_fraction_scope_retains_exact_source_and_adds_non_authoritative_spoken_hint():
    expression = "(x+2)/(y-1)"
    evidence = _evidence("CT-02", exact=[{
        "text": expression, "status": "supported", "notation": "math", "source_ref": "scene"
    }])
    prompt = _synth()._build_brief_prompt("", _context(), _snapshot(evidence))
    assert expression in prompt
    assert "exact_source" in prompt and "spoken_hint" in prompt
    assert "the quantity ... over the quantity" in prompt
    assert "bracket and fraction scope" in prompt


def test_code_tokens_and_indentation_remain_exact_and_are_not_math_linearized():
    code = "if score >= 90:\n    print(\"A\")"
    evidence = _evidence("CT-08", exact=[{
        "text": code, "status": "supported", "notation": "code", "source_ref": "scene"
    }], details={"programming": {"exact_code": code, "indentation_significant": True}})
    prompt = _synth()._build_explanatory_prompt("The condition uses >.", _context(), _snapshot(evidence, "The condition uses >."))
    assert "score >= 90" in prompt
    assert "    print" in prompt
    assert "visible source code from displayed output" in prompt


def test_flowchart_table_circuit_and_warning_risks_are_explicit_in_prompt_contract():
    cases = [
        (_evidence("CT-04", details={"flow_network": {"branch_conditions": ["Yes -> Accept", "No -> Retry"]}}), ("directed edges", "branch labels")),
        (_evidence("CT-05", details={"table": {"headers": ["Name", "Score"], "cells": [{"row_header": "Ada", "column_header": "Score", "text": "95"}]}}), ("cell-to-header associations", "Ada")),
        (_evidence("CT-03", details={"scientific_structure": {"connections": ["wire crossing without junction", "R1 connected to R2"]}}), ("junctions from wire crossings", "without junction")),
        (_evidence("CT-06", details={"structured_text": {"annotations": ["Do not divide by zero."]}}), ("warning negation", "Do not divide")),
    ]
    for evidence, expected in cases:
        prompt = _synth()._build_explanatory_prompt("", _context(), _snapshot(evidence))
        assert all(item in prompt for item in expected)


def test_uncertain_partial_and_truncated_sources_are_not_presented_as_exhaustive():
    evidence = _evidence("CT-10", status="partial", evidence_status="partially_supported")
    evidence["validation_diagnostics"] = [{
        "code": "extraction.possible_truncation", "message": "Output may be incomplete"
    }]
    snapshot = _snapshot(evidence)
    snapshot["coverage_plan"].update({"truncated": True, "omitted_fact_count": 9})
    prompt = _synth()._build_detailed_prompt("", _context(), snapshot)
    assert "extraction_status=partial" in prompt
    assert "extraction.possible_truncation" in prompt
    assert "coverage_plan_omitted_facts=9" in prompt
    assert "Do not claim the description is exhaustive" in prompt
    assert "uncertain_match as unresolved" in prompt
    assert "Never assert not_usable facts" in prompt


def test_three_modes_share_core_contract_without_fixed_length_or_paragraph_quota():
    snapshot = _snapshot(_evidence("CT-07", exact=[{
        "text": "∠ABC = 90°", "status": "supported", "notation": "math", "source_ref": "scene"
    }]))
    synth = _synth()
    prompts = {
        "brief": synth._build_brief_prompt("", _context(), snapshot),
        "standard": synth._build_explanatory_prompt("", _context(), snapshot),
        "detailed": synth._build_detailed_prompt("", _context(), snapshot),
    }
    for prompt in prompts.values():
        for component in ("Orientation", "Structure", "Salient visual information", "Educational meaning", "Lecture connection"):
            assert component in prompt
        assert "∠ABC = 90°" in prompt
        assert "not five paragraphs" in prompt.lower()
        assert "word limit" not in prompt.lower()
    assert "BRIEF:" in prompts["brief"]
    assert "STANDARD:" in prompts["standard"]
    assert "DETAILED:" in prompts["detailed"]


@pytest.mark.parametrize(("alias", "expected"), [("short", "brief"), ("standard", "explanatory"), ("explanatory", "explanatory"), ("detailed", "detailed")])
def test_legacy_mode_aliases_normalize_compatibly(alias, expected):
    assert LLMSynthesizer._normalize_output_mode(alias) == expected


def test_generation_routes_alias_and_intervention_context_to_one_prompt_call(monkeypatch):
    synth = _synth()
    captured = {}
    monkeypatch.setattr(synth, "_call_llm", lambda prompt: captured.setdefault("prompt", prompt) or "unused")
    snapshot = _snapshot(_evidence("CT-09", elements=[_element("toggle", "Wi-Fi Off", "CT-09")]))
    result = synth._generate_enhanced_explanation("Look at this.", snapshot, "software_interface", "Deictic", "standard")
    assert result == captured["prompt"]
    assert "STANDARD:" in captured["prompt"]
    assert 'Intervention reason: "Deictic"' in captured["prompt"]


def test_failed_canonical_evidence_uses_truthful_transcript_fallback(monkeypatch):
    synth = _synth()
    called = False

    def fail_if_called(_prompt):
        nonlocal called
        called = True
        return "invented"

    synth._call_llm = fail_if_called
    evidence = _evidence("CT-11", status="failed", evidence_status="unavailable_or_unreadable")
    result = synth._generate_enhanced_explanation(
        "The teacher continues the lesson.", _snapshot(evidence), "observational_image", "High Complexity", "brief"
    )
    assert result == "The teacher continues the lesson."
    assert called is False


def test_strict_on_demand_mode_rejects_failed_visual_evidence():
    from synthesizer import VisualEvidenceUnavailableError

    synth = _synth()
    evidence = _evidence("CT-11", status="failed", evidence_status="unavailable_or_unreadable")
    with pytest.raises(VisualEvidenceUnavailableError):
        synth.weave(
            "The teacher continues the lesson.",
            _snapshot(evidence),
            "observational_image",
            "High Complexity",
            "brief",
            require_visual_explanation=True,
        )


def test_strict_on_demand_mode_accepts_partial_visual_evidence(monkeypatch):
    synth = _synth()
    evidence = _evidence("CT-03", status="partial")
    monkeypatch.setattr(synth, "_call_llm", lambda _prompt: "A cautious visual explanation.")

    result = synth.weave(
        "The teacher continues the lesson.",
        _snapshot(evidence),
        "biology",
        "Deictic",
        "explanatory",
        require_visual_explanation=True,
    )

    assert result == "A cautious visual explanation."


def test_strict_on_demand_mode_rejects_empty_mistral_response(monkeypatch):
    from synthesizer import TextSynthesisUnavailableError

    synth = _synth()
    monkeypatch.setattr(synth, "_call_llm", lambda _prompt: "")
    snapshot = _snapshot(_evidence("CT-03"))
    with pytest.raises(TextSynthesisUnavailableError):
        synth.weave(
            "The teacher continues the lesson.",
            snapshot,
            "biology",
            "Deictic",
            "detailed",
            require_visual_explanation=True,
        )


def test_canonical_mixed_evidence_is_not_dropped_by_legacy_keyword_heuristic():
    synth = _synth()
    snapshot = _snapshot(_evidence(["CT-03", "CT-06"], exact=[{
        "text": "Circuit warning text", "status": "supported", "notation": "plain", "source_ref": "scene"
    }]))
    snapshot["text_readout"] = "Circuit warning text"
    assert synth._detect_cross_domain_hallucination(snapshot, "circuit") is False


def test_prompt_serialization_is_bounded_and_does_not_mutate_source():
    elements = [
        _element(f"item-{index}", f"Label {index} " + "x" * 300, "CT-06", exact_text="warning " + "y" * 300)
        for index in range(100)
    ]
    snapshot = _snapshot(_evidence("CT-06", elements=elements))
    original = copy.deepcopy(snapshot)
    synth = _synth()
    canonical = synth._format_canonical_evidence(snapshot)
    coverage = synth._format_coverage_plan(snapshot)
    synth._linearize_vlm_data(snapshot)
    assert len(canonical) <= synth.MAX_CANONICAL_PROMPT_CHARS
    assert len(coverage) <= synth.MAX_COVERAGE_PROMPT_CHARS
    assert "_omitted_items" in canonical or "payload_truncated" in canonical
    assert snapshot == original


def test_transcript_and_surrounding_context_bounds_are_explicit():
    synth = _synth()
    context = _context()
    context["context_text"] = "context " * 2000
    prompt = synth._build_brief_prompt(
        "teacher " * 2000,
        context,
        _snapshot(_evidence("CT-06", details={"structured_text": {"headings": ["Bounded"]}})),
    )
    assert prompt.count("TRANSCRIPT TRUNCATED:") == 2


def test_priority_plan_orders_critical_missing_before_spoken_supporting_fact():
    plan = {
        "schema": "synapse_transcript_visual_coverage_v1",
        "facts": [
            {"fact_id": "fact-1", "fact_type": "component", "description": "Already said", "importance": "supporting", "transcript_coverage": "spoken_explicit"},
            {"fact_id": "fact-2", "fact_type": "relation", "description": "Missing directed edge", "importance": "critical", "transcript_coverage": "unspoken"},
        ],
    }
    payload = _synth()._format_coverage_plan({"coverage_plan": plan})
    assert payload.index("Missing directed edge") < payload.index("Already said")
