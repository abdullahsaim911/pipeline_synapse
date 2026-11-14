"""Task 5 grounded, category-aware Mistral prompt regressions."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

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
    for expected in ("lower rear", '"parent_id":"brain"', "points_to", "Do not confuse", "CT-03", "CT-06"):
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
    assert "Generation may be incomplete" in prompt
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
    monkeypatch.setattr(synth, "_call_llm", lambda prompt, **kwargs: captured.setdefault("prompt", prompt) or "unused")
    snapshot = _snapshot(_evidence("CT-09", elements=[_element("toggle", "Wi-Fi Off", "CT-09")]))
    result = synth._generate_enhanced_explanation("Look at this.", snapshot, "software_interface", "Deictic", "standard")
    assert result == captured["prompt"]
    assert "STANDARD:" in captured["prompt"]
    assert 'Intervention reason: "Deictic"' in captured["prompt"]


def test_failed_canonical_evidence_uses_truthful_transcript_fallback(monkeypatch):
    synth = _synth()
    called = False

    def fail_if_called(_prompt, **kwargs):
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
    monkeypatch.setattr(synth, "_call_llm", lambda _prompt, **kwargs: "A cautious visual explanation.")

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
    monkeypatch.setattr(synth, "_call_llm", lambda _prompt, **kwargs: "")
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
    assert prompt.count("TRANSCRIPT TRUNCATED:") == 1
    assert "unaligned surrounding context omitted" in prompt


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


def _prompt_json(prompt, heading):
    lines = prompt.splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith(heading))
    return json.loads(lines[index + 1])


def test_long_exact_code_is_whole_when_it_fits_and_never_previewed_when_it_does_not():
    synth = _synth()
    code = "if count >= 2:\n" + "    print(count)\n" * 100
    evidence = _evidence("CT-08", exact=[{
        "text": code, "notation": "code", "status": "supported", "source_ref": "scene",
    }])
    packed = json.loads(synth._format_canonical_evidence(_snapshot(evidence)))
    assert packed["exact_content"][0]["text"] == code
    oversized = code * 20
    evidence["exact_content"][0]["text"] = oversized
    text = synth._format_canonical_evidence(_snapshot(evidence))
    packed = json.loads(text)
    assert len(text) <= synth.MAX_CANONICAL_PROMPT_CHARS
    assert packed["payload_truncated"]
    assert packed["_omitted_items"]["exact_content"] == 1
    assert packed["critical_records_omitted"] >= 1
    assert "preview" not in packed and "exact_content" not in packed


def test_budget_selection_keeps_late_critical_edges_dependencies_and_small_category(monkeypatch):
    synth = _synth()
    monkeypatch.setattr(synth, "MAX_CANONICAL_PROMPT_CHARS", 2400)
    elements = [_element(f"filler-{i}", "Decorative label " + "x" * 150, "CT-03") for i in range(60)]
    elements += [
        _element("source", "Source", "CT-03"), _element("target", "Target", "CT-03"),
        _element("toggle", "Network", "CT-09", attributes={"state": "disabled"}),
    ]
    evidence = _evidence(["CT-03", "CT-09"], elements=elements, relations=[{
        "id": "late-edge", "source_id": "source", "target_id": "target", "relation": "points_to",
        "label": "Direction", "directed": True, "evidence_status": "supported",
    }], structure={"reading_order": [element["id"] for element in elements]})
    snapshot = _snapshot(evidence)
    original = copy.deepcopy(snapshot)
    text = synth._format_canonical_evidence(snapshot)
    packed = json.loads(text)
    ids = {item["id"] for section in ("elements", "regions", "relations") for item in packed.get(section, [])}
    assert len(text) <= 2400
    assert "late-edge" in ids and {"source", "target", "toggle"} <= ids
    assert any(item.get("state") == "disabled" for item in packed["attributes"])
    for relation in packed.get("relations", []):
        assert relation["source_id"] in ids and relation["target_id"] in ids
    for element in packed.get("elements", []):
        assert element["region_id"] in ids
        if element.get("parent_id"):
            assert element["parent_id"] in ids
    assert set(packed.get("structure", {}).get("reading_order", [])) <= ids
    assert packed["payload_truncated"] and packed["_omitted_items"]
    assert snapshot == original
    assert text == synth._format_canonical_evidence(snapshot)


def test_attention_does_not_repeat_source_descriptions_or_merge_distinct_objects():
    sentence = "A component with three visible curved connectors."
    left = _element("left", "Left component", "CT-03", position="left")
    right = _element("right", "Right component", "CT-03", position="right")
    left["description"] = right["description"] = sentence
    snapshot = _snapshot(_evidence("CT-03", elements=[left, right]))
    prompt = _synth()._build_explanatory_prompt("", _context(), snapshot)
    canonical = _prompt_json(prompt, "CANONICAL VISUAL EVIDENCE")
    attention = _prompt_json(prompt, "TRANSCRIPT-VISUAL COVERAGE PLAN")
    # Shared wording stays once; each occurrence is nested in its own region.
    assert json.dumps(canonical).count(sentence) == 1
    shared = next(item for item in canonical["descriptions"] if item["text"] == sentence)
    assert shared["source_refs"] == ["left", "right"]
    assert {item["id"] for region in canonical["regions"] for item in region["elements"]} == {"left", "right"}
    assert "attention_groups" in attention and "facts" not in attention
    assert attention["selected_fact_count"] == snapshot["coverage_plan"]["fact_count"]
    assert all("description" not in item for group in attention["attention_groups"] for item in group["items"])


def test_selected_unlabeled_endpoints_keep_their_visual_description():
    elements = [_element(f"node-{i}", "", "CT-03") for i in range(20)]
    for index, node in enumerate(elements):
        node["description"] = f"Observed component {index} with its specific connector geometry."
    packed = _synth()._select_canonical_payload(_snapshot(_evidence("CT-03", elements=elements)), max_chars=2500)
    descriptions = {item.get("source_ref") for item in packed.get("descriptions", [])}
    assert packed["elements"]
    assert all(node["id"] in descriptions for node in packed["elements"])


def test_prompt_keeps_each_unique_observation_inside_its_own_region():
    left = _element("left", "Left", "CT-03", position="left")
    right = _element("right", "Right", "CT-03", position="right")
    left["description"] = "Red source with inward arrowheads."
    right["description"] = "Blue source with outward arrowheads."
    evidence = _evidence("CT-03", elements=[left])
    second = copy.deepcopy(evidence["regions"][0])
    second.update(id="second-region", position="right", elements=[right])
    evidence["regions"][0]["position"] = "left"
    evidence["regions"].append(second)
    snapshot = _snapshot(evidence)
    original = copy.deepcopy(snapshot)
    prompt = _synth()._build_explanatory_prompt("", _context(), snapshot)
    packed = _prompt_json(prompt, "CANONICAL VISUAL EVIDENCE")
    regions = {region["position"]: region for region in packed["regions"]}
    assert regions["left"]["elements"][0]["observations"][0]["text"] == left["description"]
    assert regions["right"]["elements"][0]["observations"][0]["text"] == right["description"]
    assert prompt.count(left["description"]) == prompt.count(right["description"]) == 1
    assert snapshot == original


def test_conflicting_exact_text_cannot_leak_through_descriptions_details_or_math_hints():
    rejected = "x > ERROR_EXACT_LITERAL"
    node = _element("condition", rejected, "CT-02", exact_text=rejected, attributes={"value": rejected})
    node["description"] = "The source condition is " + rejected
    evidence = _evidence("CT-02", elements=[node], exact=[{
        "text": rejected, "notation": "math", "status": "ambiguous_or_conflicting", "source_ref": "condition",
    }], details={"mathematics": {"exact_expressions": [rejected]}},
        structure={"overview": "A condition shown as " + rejected})
    snapshot = _snapshot(evidence)
    original = copy.deepcopy(snapshot)
    prompt = _synth()._build_detailed_prompt("The teacher discusses conditions.", _context(), snapshot)
    assert rejected not in prompt
    assert "Exact content is unresolved" in prompt
    assert snapshot == original


def test_attention_locates_exact_attributes_and_table_rows_without_repeating_values():
    evidence = _evidence(["CT-01", "CT-05"], elements=[
        _element("axis", "Time", "CT-01", kind="axis", attributes={"unit": "s"}),
    ], details={"table": {"headers": ["Name", "Score"], "cells": [
        {"row_header": "Ada", "column_header": "Score", "text": "95"},
        {"row_header": "Bob", "column_header": "Score", "text": "72"},
    ]}})
    snapshot = _snapshot(evidence)
    prompt = _synth()._build_explanatory_prompt("", _context(), snapshot)
    canonical = _prompt_json(prompt, "CANONICAL VISUAL EVIDENCE")
    attention = _prompt_json(prompt, "TRANSCRIPT-VISUAL COVERAGE PLAN")
    items = [item for group in attention["attention_groups"] for item in group["items"]]
    assert any(item.get("field") == "unit" for item in items)
    assert any(item.get("detail_records") for item in items)
    assert all(index < len(canonical["category_details"]) for item in items for index in item.get("detail_records", []))
    assert "Ada" in prompt and "Bob" in prompt


def test_unaligned_context_is_omitted_and_known_aligned_excerpt_is_not_duplicated():
    evidence = _evidence("CT-09", elements=[_element("toggle", "Network", "CT-09", attributes={"state": "disabled"})])
    snapshot = _snapshot(evidence, "An unrelated general principle.")
    context = _context()
    context["context_text"] = "UNRELATED_BACKGROUND " * 100
    prompt = _synth()._build_explanatory_prompt("An unrelated general principle.", context, snapshot)
    assert "UNRELATED_BACKGROUND" not in prompt
    assert "An unrelated general principle." in prompt
    assert "Do not force a lecture connection" in prompt
    fact = next(item for item in snapshot["coverage_plan"]["facts"] if item["fact_type"] == "attribute")
    fact.update(transcript_coverage="spoken_partial", supporting_transcript_span="The Network control is shown.")
    context["context_text"] = "The Network control is shown. UNRELATED_BACKGROUND"
    prompt = _synth()._build_explanatory_prompt("Look here.", context, snapshot)
    assert "The Network control is shown." in prompt
    assert "UNRELATED_BACKGROUND" not in prompt


@pytest.mark.parametrize("mode", ["brief", "explanatory", "detailed"])
def test_real_generation_call_returns_weak_output_unchanged_once_in_each_mode(mode):
    synth = _synth()
    calls = []
    weak_text = "A generic explanation that overlooks the displayed relationships."

    def generate(prompt, **kwargs):
        calls.append(prompt)
        return SimpleNamespace(text=weak_text, total_duration_ms=1, prompt_eval_count=1, eval_count=1, model="fake", done=True, done_reason="stop")

    synth.client = SimpleNamespace(generate=generate)
    snapshot = _snapshot(_evidence("CT-09", status="partial", elements=[_element("toggle", "Network", "CT-09")]))
    original = copy.deepcopy(snapshot)
    result = synth.weave("Look here.", snapshot, "software_interface", "Deictic", mode, require_visual_explanation=True)
    assert result == weak_text
    assert len(calls) == 1
    assert "Network" in calls[0] and "attention_groups" in calls[0]
    assert snapshot == original


def test_valid_partial_canonical_survives_stale_legacy_missing_flag_without_extra_call(monkeypatch):
    synth = _synth()
    snapshot = _snapshot(_evidence("CT-03", status="partial", elements=[_element("part", "Part", "CT-03")]))
    snapshot["missing_elements"] = "Legacy OCR was incomplete"
    captured = []
    monkeypatch.setattr(synth, "_call_llm", lambda prompt, **kwargs: captured.append(prompt) or "The Part is present; some details are unresolved.")
    result = synth.weave("Teacher speech.", snapshot, "diagram", "Deictic", "brief", require_visual_explanation=True)
    assert result.startswith("The Part") and len(captured) == 1
    snapshot["_error"] = True
    from synthesizer import VisualEvidenceUnavailableError
    with pytest.raises(VisualEvidenceUnavailableError):
        synth.weave("Teacher speech.", snapshot, "diagram", "Deictic", "brief", require_visual_explanation=True)
    assert len(captured) == 1


def test_canonical_categories_are_not_overridden_by_generic_legacy_hints():
    snapshot = _snapshot(_evidence("CT-01", elements=[_element("axis", "Time", "CT-01")]))
    assert _synth()._resolve_category_ids(snapshot, "diagram") == ["CT-01"]
    assert _synth()._resolve_category_ids({}, "diagram") == []


def test_conflicts_are_expressed_as_limits_without_raw_diagnostic_dumps():
    evidence = _evidence("CT-07", elements=[_element("point", "P", "CT-07")])
    evidence["validation_diagnostics"] = [{
        "code": "extraction.quality.conflicting_spatial_order", "target_id": "point",
        "message": "RAW_INTERNAL_DIAGNOSTIC " * 100,
    }]
    prompt = _synth()._build_explanatory_prompt("", _context(), _snapshot(evidence))
    assert "Conflicting source observations" in prompt
    assert "RAW_INTERNAL_DIAGNOSTIC" not in prompt
    canonical = _prompt_json(prompt, "CANONICAL VISUAL EVIDENCE")
    assert canonical["limitations"][0]["target_id"] == "point"
    assert "do not assert" in canonical["limitations"][0]["reason"]


def test_unusable_relationship_volume_does_not_crowd_out_supported_visual_description(monkeypatch):
    synth = _synth()
    monkeypatch.setattr(synth, "MAX_CANONICAL_PROMPT_CHARS", 1800)
    observation = _element("sample", "Sample", "CT-11")
    observation["description"] = "The Sample has a striped surface with a darker outer band."
    evidence = _evidence(["CT-03", "CT-11"], elements=[
        _element("a", "A", "CT-03"), _element("b", "B", "CT-03"), observation,
    ], relations=[{
        "id": f"uncertain-{index}", "source_id": "a", "target_id": "b", "relation": f"custom:unclear_{index}",
        "evidence_status": "ambiguous_or_conflicting", "label": "Unresolved path " + "x" * 80,
    } for index in range(30)])
    packed = json.loads(synth._format_canonical_evidence(_snapshot(evidence)))
    assert any(item.get("text") == observation["description"] for item in packed["descriptions"])
    assert any(item["id"] == "sample" for item in packed["elements"])
    assert packed["payload_truncated"]


def test_context_index_handles_batch_dictionary_transcripts_without_using_the_intro(monkeypatch):
    synth = _synth()
    synth.transcripts = [{"text": "An earlier unrelated introduction."}, {"text": "Look at the Network control."}]
    calls = []
    synth.context_manager = SimpleNamespace(get_context=lambda index, mode: calls.append((index, mode)) or _context())
    monkeypatch.setattr(synth, "_call_llm", lambda prompt, **kwargs: "The Network control is shown.")
    snapshot = _snapshot(_evidence("CT-09", elements=[_element("toggle", "Network", "CT-09")]))
    result = synth.weave("Look at the Network control.", snapshot, "software_interface", "Deictic", "brief")
    assert result == "The Network control is shown."
    assert calls == [(1, "brief")]
