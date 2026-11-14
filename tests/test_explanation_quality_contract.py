"""Boundary regressions for the agreed visual explanation contract."""

import copy
import json
from types import SimpleNamespace

import jsonschema
import pytest

from llm_interface.ollama_client import OllamaClient
from synthesizer.context_manager import ContextManager
from synthesizer import TextSynthesisUnavailableError
from llm_interface.providers import GenerationResult, LLMConfigurationError, LLMRateLimitError
from tests.test_task5_synthesis import _synth, _snapshot, _evidence, _element, _context
from visual_evidence.compact_schema import COMPACT_OBSERVATION_JSON_SCHEMA
from tests.test_structured_vlm_integration import _base_payload, _region


def test_adjacent_caption_fragments_survive_without_an_ocr_match():
    manager = ContextManager([
        {"start": 200, "duration": 2, "text": "Unrelated earlier example."},
        {"start": 403, "duration": 2, "text": "There are other kinds of potential energy."},
        {"start": 405, "end": 406.8, "text": "The principle of conservation of energy"},
        {"start": 406.8, "duration": 2, "text": "applies to all forms of energy:"},
        {"start": 409, "duration": 3, "text": "electric, magnetic, even light waves."},
        {"start": 430, "duration": 2, "text": "Different topic after a caption gap."},
    ])
    aligned = manager.get_aligned_context(406)
    assert aligned["aligned_segments"][1]["end"] == 406.8
    snapshot = _snapshot(_evidence("CT-03", elements=[_element("field", "Field", "CT-03")]))
    snapshot["_lecture_context"] = aligned
    for mode in ["brief", "explanatory", "detailed"]:
        prompt = _synth()._build_task5_prompt("The principle of conservation of energy", _context(), snapshot, mode)
        assert "electric, magnetic, even light waves." in prompt
        assert "Unrelated earlier example" not in prompt
        assert "Different topic" not in prompt
        assert "proximity alone does not establish a connection" in prompt
        assert '\\"start\\"' not in prompt  # No JSON-string-inside-JSON caption passage.


def test_video_evaluation_selection_covers_each_lecture_and_hint_without_claiming_truth():
    from scripts.evaluate_saved_lectures import select_cases
    rows = [
        {"video_id": "first", "intervention_id": "a", "timestamp": 1, "classifier_hint": "equation"},
        {"video_id": "first", "intervention_id": "b", "timestamp": 2, "classifier_hint": "diagram"},
        {"video_id": "second", "intervention_id": "c", "timestamp": 3, "classifier_hint": "code"},
        {"video_id": "second", "intervention_id": "d", "timestamp": 4, "classifier_hint": "non-educational"},
    ]
    selected = select_cases(rows)
    assert {row["video_id"] for row in selected} == {"first", "second"}
    assert {row["classifier_hint"] for row in selected} == {"equation", "diagram", "code"}
    assert len({row["intervention_id"] for row in selected}) == len(selected)
    assert selected == select_cases(list(reversed(rows)))


def test_distant_transcript_does_not_become_aligned_context():
    assert not ContextManager([{"start": 50, "end": 52, "text": "Distant."}]).get_aligned_context(100)["aligned_segments"]


@pytest.mark.parametrize("reason,done", [("length", True), ("stop", False), ("unexpected", True), (None, None)])
def test_incomplete_generation_cannot_be_published(reason, done):
    synth = _synth()
    response = OllamaClient()._parse_response({
        "response": "A nonempty but incomplete explanation", "done": done,
        "done_reason": reason, "eval_count": 1024,
    })
    assert response.done is done and response.done_reason == reason
    synth.client = SimpleNamespace(generate=lambda *args, **kwargs: response)
    with pytest.raises(TextSynthesisUnavailableError, match="did not finish"):
        synth._call_llm("source prompt")


@pytest.mark.parametrize("finish", [None, "max_tokens", "safety", "stop"])
def test_cloud_explanation_requires_explicit_normal_completion(finish):
    synth = _synth()
    calls = []

    def generate(prompt, settings):
        calls.append((prompt, settings))
        return GenerationResult("A complete source-grounded explanation", "gemini", "configured-model", 1,
                                finish_reason=finish)

    synth.text_provider = SimpleNamespace(generate=generate)
    if finish == "stop":
        assert synth._call_llm("Source facts", "brief") == "A complete source-grounded explanation"
        assert calls[0][1].max_output_tokens == synth.MODE_OUTPUT_TOKENS["brief"]
    else:
        with pytest.raises(TextSynthesisUnavailableError, match="did not finish"):
            synth._call_llm("Source facts", "brief")


def test_cloud_rate_limit_never_falls_back_to_local_generation():
    synth = _synth()

    def fail(*args):
        raise LLMRateLimitError("Temporary quota limit", provider="gemini")

    synth.text_provider = SimpleNamespace(generate=fail)
    synth.client = SimpleNamespace(generate=lambda *args, **kwargs: pytest.fail("No silent local fallback"))
    with pytest.raises(TextSynthesisUnavailableError, match="LLMRateLimitError"):
        synth._call_llm("Source facts")


def test_explanation_provider_is_separate_from_ask_provider(monkeypatch):
    from synthesizer.configuration import explanation_profile
    monkeypatch.delenv("SYNAPSE_EXPLANATION_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert explanation_profile() == {"provider": "ollama", "model": "mistral:latest"}
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "unknown")
    with pytest.raises(LLMConfigurationError):
        explanation_profile()


def test_text_provider_selection_cannot_change_the_remote_vlm(monkeypatch):
    from orchestrator.orchestrator import PipelineOrchestrator
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "gemini")
    observed = []
    fake = SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct-FP8")
    monkeypatch.setattr("orchestrator.orchestrator.SnapshotEngine",
                        lambda **kwargs: observed.append(kwargs) or fake)
    engine, model = PipelineOrchestrator._create_intervention_vlm_engine("modal")
    assert engine is fake and model == fake.model_name
    assert observed == [{"provider": "modal"}]


def test_dense_prompt_is_repacked_with_output_reserve_without_mutation(monkeypatch):
    synth = _synth()
    synth.client = OllamaClient()
    monkeypatch.setenv("SYNAPSE_OLLAMA_NUM_CTX", "8192")
    # Deterministic tokenizer stand-in makes this a packing test, not a model test.
    monkeypatch.setattr(synth.client, "prompt_input_tokens", lambda prompt: len(prompt.encode()) // 3 + 100)
    snapshot = _snapshot(_evidence("CT-06", elements=[
        _element(f"text-{i}", f"Heading {i}", "CT-06", exact_text="important text " * 15)
        for i in range(100)
    ]))
    original = copy.deepcopy(snapshot)
    sources = []
    for mode, reserve in [("brief", 768), ("explanatory", 1024), ("detailed", 1536)]:
        prompt = synth._build_task5_prompt("Teacher speech.", _context(), snapshot, mode)
        assert synth.client.prompt_input_tokens(prompt) + reserve <= 8192
        assert "payload_truncated" in prompt
        canonical = json.loads(prompt.split("CANONICAL VISUAL EVIDENCE (whole source records; refs preserve distinct objects)\n")[1].splitlines()[0])
        sources.append(canonical)
        for item in canonical.get("exact_content", []):
            assert item["text"] == "important text " * 15
    assert sources[0] == sources[1] == sources[2]
    assert snapshot == original


def test_structured_decoding_prevents_pixel_boxes_and_invalid_categories():
    payload = _base_payload(["CT-03"])
    payload["structure"].pop("components")
    payload["structure"].pop("relations")
    payload["coordinate_system"] = "qwen_1000"
    payload["regions"] = [_region("r_panel", "CT-03", [])]
    payload["regions"][0]["position"] = "left"
    payload["regions"][0]["bbox"] = None
    jsonschema.validate(payload, COMPACT_OBSERVATION_JSON_SCHEMA)
    payload["regions"][0]["ref"] = "scene"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, COMPACT_OBSERVATION_JSON_SCHEMA)
    payload["regions"][0]["ref"] = "r_panel"
    payload["regions"][0]["bbox"] = [0, 0, 1280, 720]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, COMPACT_OBSERVATION_JSON_SCHEMA)
    payload["regions"][0].pop("bbox")
    payload["categories"] = ["diagram"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, COMPACT_OBSERVATION_JSON_SCHEMA)


def test_declared_qwen_coordinates_are_normalized_without_guessing_legacy_units():
    from visual_evidence import assemble_compact_qwen_observation
    payload = _base_payload(["CT-03"])
    payload["coordinate_system"] = "qwen_1000"
    payload["regions"] = [_region("panel", "CT-03", [])]
    payload["regions"][0]["bbox"] = [100, 200, 600, 800]
    evidence = assemble_compact_qwen_observation(payload)
    assert evidence.regions[0].bbox_normalized.as_list() == [0.1, 0.2, 0.6, 0.8]
    payload.pop("coordinate_system")
    with pytest.raises(ValueError):
        assemble_compact_qwen_observation(payload)  # Never guess whether old boxes are pixels.


@pytest.mark.parametrize("box", [[100, 300, 500, 200], [200, 100, 200, 400]])
def test_invalid_declared_crop_anchor_is_discarded_without_inventing_a_replacement(box):
    from visual_evidence import assemble_compact_qwen_observation
    payload = _base_payload(["CT-03"])
    payload["coordinate_system"] = "qwen_1000"
    payload["regions"] = [_region("panel", "CT-03", [{
        "ref": "part", "kind": "object", "evidence_status": "supported",
        "description": "Visible component", "position": "left", "bbox": box,
    }])]
    original = copy.deepcopy(payload)
    evidence = assemble_compact_qwen_observation(payload)
    node = evidence.regions[0].elements[0]
    assert node.bbox_normalized is None and node.position == "left"
    assert evidence.extraction_status == "partial"
    assert any(item.code == "extraction.invalid_crop_anchor" for item in evidence.validation_diagnostics)
    assert payload == original


@pytest.mark.parametrize("descriptions,expected", [
    (["The displayed value is 12 ms.", "The other value is 136 ms."], "element-001"),
    (["The displayed value is 12 ms.", "Repeated value 12 ms."], None),
    (["The displayed value is 112 ms.", "The other value is 136 ms."], None),
])
def test_missing_exact_source_link_requires_one_explicit_literal_occurrence(descriptions, expected):
    from visual_evidence import assemble_compact_qwen_observation
    payload = _base_payload(["CT-05"])
    payload["regions"] = [_region("panel", "CT-05", [
        {"ref": f"value-{index}", "kind": "text", "description": description, "evidence_status": "supported"}
        for index, description in enumerate(descriptions)
    ])]
    payload["exact"] = [{"text": "12 ms", "status": "supported", "notation": "plain"}]
    original = copy.deepcopy(payload)
    evidence = assemble_compact_qwen_observation(payload)
    assert evidence.exact_content[0].source_ref == expected
    assert payload == original


def test_observed_region_categories_override_wrong_classifier_hint():
    from visual_evidence import assemble_compact_qwen_observation
    payload = _base_payload([])
    payload["regions"] = [_region("science", "CT-03", [])]
    evidence = assemble_compact_qwen_observation(payload, category_ids=["CT-01"], raw_category_labels=["graph"])
    assert evidence.category_ids == ["CT-03"]
    assert evidence.raw_category_labels == ["graph"]


def test_wire_message_order_matches_qwen_budget_processor():
    from vlm_interface.modal_vlm_interface import ModalVLMInterface
    adapter = ModalVLMInterface(endpoint_url="https://test.modal.run", api_key="test-key",
                                require_exact_context_budget=False)
    payload = json.loads(adapter._build_payload(b"jpeg", "Inspect the frame", 768, 0.1))
    assert [item["type"] for item in payload["messages"][0]["content"]] == ["image_url", "text"]


def test_decode_hierarchy_has_one_parent_pointer_without_duplicate_child_inventory():
    from visual_evidence.compact_schema import ELEMENT
    from visual_evidence import assemble_compact_qwen_observation
    assert "parent_ref" in ELEMENT["properties"] and "child_refs" not in ELEMENT["properties"]
    payload = _base_payload(["CT-03"])
    payload["regions"] = [_region("region", "CT-03", [
        {"ref": "outer", "kind": "compartment", "evidence_status": "supported"},
        {"ref": "inner", "kind": "part", "evidence_status": "supported", "parent_ref": "outer"},
    ])]
    outer, inner = assemble_compact_qwen_observation(payload).regions[0].elements
    assert inner.parent_id == outer.id and outer.child_ids == [inner.id]


@pytest.mark.parametrize("expression", ["x<=5", "(x+2)/(y-1)"])
def test_math_guidance_never_injects_an_altered_expression(expression):
    snapshot = _snapshot(_evidence("CT-02", exact=[{
        "text": expression, "status": "supported", "notation": "math", "source_ref": "scene",
    }]))
    hint = json.loads(_synth()._spoken_math_hints(snapshot))[0]
    assert hint["exact_source"] == expression
    assert "exact operators and grouping" in hint["spoken_hint"]
    assert "<equals" not in hint["spoken_hint"]


@pytest.mark.parametrize("self_parent", [False, True])
def test_containment_cycles_are_rejected_without_rejecting_flow_cycles(self_parent):
    from visual_evidence.models import VisualEvidence
    elements = [_element("a", "A", "CT-04", parent_id="a" if self_parent else "b")]
    if not self_parent:
        elements.append(_element("b", "B", "CT-04", parent_id="a"))
    with pytest.raises(ValueError, match="cycle"):
        VisualEvidence.model_validate(_evidence("CT-04", elements=elements))
    elements = [_element("a", "A", "CT-04"), _element("b", "B", "CT-04")]
    relations = [{"id": f"edge-{index}", "source_id": source, "target_id": target,
                  "relation": "connected_to", "directed": True, "evidence_status": "supported"}
                 for index, (source, target) in enumerate([("a", "b"), ("b", "a")])]
    VisualEvidence.model_validate(_evidence("CT-04", elements=elements, relations=relations))


@pytest.mark.parametrize("speech,expected", [
    ("Alpha node is left of Beta node. Actually Alpha node is right of Beta node.", "uncertain_match"),
    ("Alpha node is left of Beta node. Alpha node is not left of Beta node.", "uncertain_match"),
    ("Alpha node is left of Beta node. Beta node is right of Alpha node.", "spoken_explicit"),
])
def test_conflicting_teacher_relations_are_not_suppressed_as_already_explained(speech, expected):
    evidence = _evidence("CT-07", elements=[_element("a", "Alpha node", "CT-07"),
                                              _element("b", "Beta node", "CT-07")], relations=[{
        "id": "relation", "source_id": "a", "target_id": "b", "relation": "left_of",
        "evidence_status": "supported", "directed": False,
    }])
    facts = _snapshot(evidence, speech)["coverage_plan"]["facts"]
    fact = next(item for item in facts if item["fact_type"] == "relation")
    assert fact["transcript_coverage"] == expected




