import json
from unittest.mock import Mock

import pytest

from llm_interface.providers import GenerationResult
from synthesizer.configuration import explanation_profile, configured_explanation_provider
from synthesizer.context_manager import ContextManager
from synthesizer import TextSynthesisUnavailableError
from tests.test_task5_synthesis import _synth, _snapshot, _evidence, _element, _context


def test_remote_profile_is_independent_of_ask_quiz_and_binds_revision(monkeypatch):
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "modal_mistral")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("SYNAPSE_MISTRAL_MODAL_URL", "https://example.modal.run")
    monkeypatch.setenv("SYNAPSE_MODAL_API_KEY", "test-key")
    profile = explanation_profile()
    assert profile["provider"] == "modal_mistral" and len(profile["revision"]) == 40
    assert configured_explanation_provider().model == profile["model"]


def test_reasoning_generation_gets_own_budget_and_settings():
    synth = _synth()
    synth.text_provider = Mock(generation_temperature=0.7, max_output_tokens=6144)
    synth.text_provider.generate.return_value = GenerationResult("Final spoken answer", "modal_mistral", "model", 1, finish_reason="stop")
    assert synth._call_llm("Prompt", "brief") == "Final spoken answer"
    settings = synth.text_provider.generate.call_args.args[1]
    assert settings.max_output_tokens == 6144 and settings.temperature == 0.7


def test_long_teacher_correction_is_kept_with_complete_caption_records():
    records = [{"start": t, "end": t + 4, "text": f"caption {t}"} for t in range(0, 125, 5)]
    records[16]["text"] = "This is a misconception: continued motion does not require a net force."
    records.append({"start": 160, "end": 164, "text": "A separate later lesson."})
    aligned = ContextManager(records).get_aligned_context(50)
    assert any("misconception" in item["text"] for item in aligned["aligned_segments"])
    assert not any("separate later" in item["text"] for item in aligned["aligned_segments"])
    snapshot = _snapshot(_evidence("CT-03", elements=[_element("force", "Force", "CT-03")]))
    snapshot["_lecture_context"] = aligned
    for mode in ["brief", "explanatory", "detailed"]:
        prompt = _synth()._build_task5_prompt("caption 50", _context(), snapshot, mode)
        assert "continued motion does not require a net force" in prompt
        assert "refuted misconception" in prompt or "refute" in prompt


def test_remote_budget_repacking_keeps_same_source_facts_across_modes():
    class Budget:
        max_output_tokens = 6144
        def prompt_input_tokens(self, prompt):
            return len(prompt) // 4
        def context_limit(self):
            return 12288
    synth = _synth()
    synth.text_provider = Budget()
    snapshot = _snapshot(_evidence("CT-03", elements=[_element("force", "Force", "CT-03")]))
    prompts = [synth._build_task5_prompt("Teacher context", _context(), snapshot, mode) for mode in ["brief", "explanatory", "detailed"]]
    sources = [text.split("CANONICAL VISUAL EVIDENCE", 1)[1].split("OPTIONAL SPOKEN-MATH HINTS", 1)[0] for text in prompts]
    assert sources[0] == sources[1] == sources[2]


def test_compact_remote_prompt_preserves_source_and_aligned_teacher_records():
    synth = _synth()
    synth.text_provider = type("Provider", (), {"supports_structured_output": True})()
    snapshot = _snapshot(_evidence("CT-03", elements=[_element("force", "SOURCE_FORCE", "CT-03")]))
    snapshot["_lecture_context"] = {"frame_timestamp": 10, "aligned_segments": [
        {"start": 8, "end": 11, "text": "The displayed claim is a misconception."},
        {"start": 11, "end": 15, "text": "The chair continues moving after the push ends."},
    ]}
    prompt = synth._build_task5_prompt("The displayed claim is a misconception.", _context(), snapshot, "explanatory")
    compact = synth._generation_prompt(prompt)
    assert "SOURCE_FORCE" in compact
    assert "The chair continues moving after the push ends." in compact
    assert "The displayed claim is a misconception." in compact
    assert "STANDARD:" in compact and "INTERNAL RESPONSE CONTRACT" in compact
    assert "TRANSCRIPT-VISUAL COVERAGE PLAN" not in compact
    assert len(compact) < len(prompt) + 1000


def test_integral_hints_preserve_differential_variable_without_rewriting_source():
    evidence = _evidence("CT-02", exact=[{
        "text": "∫ u dv = uv - ∫ v du", "notation": "math", "status": "supported", "source_ref": "scene",
    }])
    hints = json.loads(_synth()._spoken_math_hints(_snapshot(evidence)))
    assert hints[0]["exact_source"] == "∫ u dv = uv - ∫ v du"
    assert hints[0]["differential_readings"]["du"] == "with respect to u, not with respect to du"
    assert hints[0]["differential_readings"]["dv"] == "with respect to v, not with respect to dv"


def test_spoken_differential_normalization_uses_only_terminal_source_differentials():
    from visual_evidence.notation import integral_differentials
    assert integral_differentials("∫ du dx") == ["x"]
    assert integral_differentials("∫ u dv = uv - ∫ v du") == ["v", "u"]
    payload = {"visual_description": "The integral of v with respect to du.", "educational_explanation": "", "lecture_connection": "", "uncertainty": ""}
    prompt = 'OPTIONAL SPOKEN-MATH HINTS\n' + json.dumps([{"exact_source": "∫ v du"}])
    assert _synth()._normalize_spoken_differentials(payload, prompt)["visual_description"] == "The integral of v with respect to u."
    assert _synth()._normalize_spoken_differentials(payload, 'OPTIONAL SPOKEN-MATH HINTS\n[]') == payload


@pytest.mark.parametrize("mode,separator", [("brief", " "), ("explanatory", "\n\n"), ("detailed", "\n\n")])
def test_structured_answer_is_returned_as_existing_plain_script(mode, separator):
    synth = _synth()
    payload = {"visual_description": "Two rulers are connected by a V-shape.",
               "educational_explanation": "The teacher explains parallel processing.",
               "lecture_connection": "", "uncertainty": "The V-shape has no supplied meaning."}
    synth.text_provider = Mock(supports_structured_output=True, generation_temperature=0.1, max_output_tokens=1536)
    synth.text_provider.generate.return_value = GenerationResult(json.dumps(payload), "modal_mistral", "model", 1, finish_reason="stop")
    result = synth._call_llm("Original source prompt", mode)
    assert result == separator.join(value for value in payload.values() if value)
    prompt, settings = synth.text_provider.generate.call_args.args
    assert "INTERNAL RESPONSE CONTRACT" in prompt and "Original source prompt" in prompt
    assert settings.response_schema == synth.SPOKEN_RESPONSE_SCHEMA
    assert '"visual_description"' not in result


@pytest.mark.parametrize("payload", [
    {}, {"visual_description": "Only one field"},
    {"visual_description": "", "educational_explanation": "", "lecture_connection": "", "uncertainty": ""},
    {"visual_description": "\\frac{x}{y}", "educational_explanation": "", "lecture_connection": "", "uncertainty": ""},
])
def test_malformed_structured_explanation_never_reaches_tts_contract(payload):
    synth = _synth()
    synth.text_provider = Mock(supports_structured_output=True, generation_temperature=0.1, max_output_tokens=1536)
    synth.text_provider.generate.return_value = GenerationResult(json.dumps(payload), "modal_mistral", "model", 1, finish_reason="stop")
    with pytest.raises(TextSynthesisUnavailableError):
        synth._call_llm("Original source prompt", "brief")


def test_paired_emphasis_is_normalized_without_stripping_operators_or_source_text():
    payload = {"visual_description": "The title is **What Keeps Objects at Rest?**",
               "educational_explanation": "", "lecture_connection": "", "uncertainty": ""}
    result = _synth()._parse_contract(json.dumps(payload), _synth().SPOKEN_RESPONSE_SCHEMA)
    assert result["visual_description"] == "The title is What Keeps Objects at Rest?"
    payload["visual_description"] = "x * y"
    with pytest.raises(TextSynthesisUnavailableError):
        _synth()._parse_contract(json.dumps(payload), _synth().SPOKEN_RESPONSE_SCHEMA)


@pytest.mark.parametrize("retains_teacher", [True, False])
def test_source_review_retains_only_supported_existing_sentences(retains_teacher):
    payload = {"visual_description": "A grid and triangular marks are visible.",
        "educational_explanation": "The teacher explains parallel processing. The triangles likely represent processing.",
        "lecture_connection": "", "uncertainty": "Their meaning is not established."}
    def result(value):
        return GenerationResult(json.dumps(value), "modal_mistral", "model", 1, finish_reason="stop")
    synth = _synth()
    synth.text_provider = Mock(supports_structured_output=True, generation_temperature=0.1, max_output_tokens=1536)
    synth.text_provider.generate.side_effect = [result(payload), result({
        "sentence_support": {"sentence_1": True, "sentence_2": retains_teacher, "sentence_3": False, "sentence_4": True},
    })]
    if retains_teacher:
        text = synth._call_llm("Original source prompt", "explanatory")
        assert "The teacher explains parallel processing." in text
        assert "likely represent" not in text
        assert "Their meaning is not established." in text
    else:
        with pytest.raises(TextSynthesisUnavailableError, match="unsupported claims"):
            synth._call_llm("Original source prompt", "explanatory")
    calls = synth.text_provider.generate.call_args_list
    assert len(calls) == 2
    assert calls[1].args[1].system_instruction and calls[1].args[1].max_output_tokens == 1024


def test_ownership_loss_before_review_is_not_wrapped_or_allowed_to_generate_more():
    class Cancelled(RuntimeError):
        pass
    class Provider:
        supports_structured_output = True
        generation_temperature = 0.1
        max_output_tokens = 1536
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, settings):
            self.calls += 1
            return GenerationResult(json.dumps({"visual_description": "A triangle.",
                "educational_explanation": "The triangle likely represents processing.", "lecture_connection": "",
                "uncertainty": "Its meaning is unknown."}), "modal_mistral", "model", 1, finish_reason="stop")
    synth = _synth()
    synth.text_provider = Provider()
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise Cancelled("Owner lost")
    synth.ownership_check = check
    snapshot = _snapshot(_evidence("CT-03", elements=[_element("shape", "Triangle", "CT-03")]))
    with pytest.raises(Cancelled, match="Owner lost"):
        synth.weave("Teacher speech", snapshot, "diagram", "test", "brief", require_visual_explanation=True)
    assert synth.text_provider.calls == 1
