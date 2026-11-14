import json
from unittest.mock import Mock

import pytest
import requests

from llm_interface.modal_mistral_provider import ModalMistralProvider
from llm_interface.providers import (
    GenerationSettings, LLMAuthenticationError, LLMConfigurationError,
    LLMEmptyResponseError, LLMMalformedResponseError, LLMRequestError,
    LLMTimeoutError,
)
from modal_inference.mistral_config import MAX_MODEL_LEN, build_serve_command


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


def provider(*responses):
    session = Mock()
    session.post.side_effect = list(responses)
    return ModalMistralProvider("https://example.modal.run", "secret-value", session=session), session


def answer(content="The force is resolved into two components.", finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "PRIVATE REASONING"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 600}}


def test_final_answer_only_and_identical_preflight_messages():
    p, session = provider(response({"count": 100}), response(answer()))
    result = p.generate("Lecture evidence")
    assert result.text == "The force is resolved into two components."
    assert "PRIVATE" not in result.text
    assert result.finish_reason == "stop" and result.usage.prompt_tokens == 100
    token_call, generation = session.post.call_args_list
    assert token_call.kwargs["json"]["messages"] == generation.kwargs["json"]["messages"]
    assert isinstance(generation.kwargs["json"]["messages"][0]["content"], str)
    assert generation.kwargs["json"]["temperature"] == 0.1
    assert generation.kwargs["json"]["max_tokens"] == 4096
    assert "thinking_token_budget" not in generation.kwargs["json"]
    assert "regex" in generation.kwargs["json"]["structured_outputs"]
    assert "_" not in generation.kwargs["json"]["bad_words"]
    assert "\\" not in generation.kwargs["json"]["bad_words"]
    assert generation.kwargs["allow_redirects"] is False
    assert "secret-value" not in repr(p)


@pytest.mark.parametrize("text,finish,error", [
    (None, "stop", LLMEmptyResponseError), ("", "stop", LLMEmptyResponseError),
    ("Partial answer", "length", LLMMalformedResponseError),
    ("[THINK]hidden[/THINK]answer", "stop", LLMMalformedResponseError),
])
def test_incomplete_or_leaked_answers_are_rejected(text, finish, error):
    p, session = provider(response({"count": 100}), response(answer(text, finish)))
    with pytest.raises(error):
        p.generate("Evidence")
    assert session.post.call_count == 2  # No content retry.


def test_budget_overflow_rejected_before_generation():
    p, session = provider(response({"count": MAX_MODEL_LEN}))
    with pytest.raises(LLMRequestError):
        p.generate("Evidence")
    assert session.post.call_count == 1


def test_caller_owned_json_schema_is_forwarded_without_changing_messages():
    schema = {"type": "object", "properties": {"spoken": {"type": "string"}}}
    p, session = provider(response({"count": 100}), response(answer('{"spoken":"A final answer"}')))
    result = p.generate("Caller instructions", GenerationSettings(max_output_tokens=100, response_schema=schema))
    assert json.loads(result.text)["spoken"] == "A final answer"
    assert session.post.call_args.kwargs["json"]["structured_outputs"] == {"json": schema, "disable_any_whitespace": True}


def test_review_system_instruction_is_identical_in_tokenizer_and_generation():
    p, session = provider(response({"count": 20}), response(answer('{"supported":true,"issues":[]}')))
    p.generate("Review source", GenerationSettings(max_output_tokens=100, system_instruction="Judge source support"))
    first, second = session.post.call_args_list
    assert first.kwargs["json"]["messages"] == second.kwargs["json"]["messages"]
    assert first.kwargs["json"]["messages"][0]["content"] == "Judge source support"


def test_speech_pattern_and_length_are_both_enforced_without_mutating_schema():
    schema = {"type": "object", "properties": {"spoken": {"type": "string", "pattern": "^[^*]*$", "maxLength": 600}}}
    p, session = provider(response({"count": 20}), response(answer('{"spoken":"Answer"}')))
    p.generate("Instructions", GenerationSettings(max_output_tokens=100, response_schema=schema))
    wire = session.post.call_args.kwargs["json"]["structured_outputs"]["json"]
    assert "pattern" not in wire["properties"]["spoken"]
    assert wire["properties"]["spoken"]["maxLength"] == 600
    assert "*" in session.post.call_args.kwargs["json"]["bad_words"]
    assert schema["properties"]["spoken"]["maxLength"] == 600


@pytest.mark.parametrize("text,accepted", [('{"spoken":"Complete sentence."}  \n', True), ('{"spoken":"Partial', False), ('{"wrong":"Missing required field"}', False)])
def test_token_limit_can_only_accept_independently_complete_schema_valid_json(text, accepted):
    schema = {"type": "object", "properties": {"spoken": {"type": "string"}}, "required": ["spoken"], "additionalProperties": False}
    p, _ = provider(response({"count": 20}), response(answer(text, "length")))
    if accepted:
        assert p.generate("Instructions", GenerationSettings(max_output_tokens=100, response_schema=schema)).finish_reason == "complete_json_at_limit"
    else:
        with pytest.raises(LLMMalformedResponseError):
            p.generate("Instructions", GenerationSettings(max_output_tokens=100, response_schema=schema))


def test_redirect_and_auth_error_never_forward_credentials_or_upstream_body():
    for status, error in [(302, LLMRequestError), (401, LLMAuthenticationError)]:
        p, session = provider(response({"secret": "sensitive prompt"}, status))
        with pytest.raises(error) as exc:
            p.prompt_input_tokens("private lecture")
        assert "sensitive" not in str(exc.value)
        assert session.post.call_count == 1


def test_transient_retry_is_bounded():
    p, session = provider(response({}, 503), response({"count": 12}))
    assert p.prompt_input_tokens("short") == 12
    assert session.post.call_count == 2


def test_timeout_is_safe_and_not_retried():
    p, session = provider(requests.Timeout("sensitive URL"))
    with pytest.raises(LLMTimeoutError) as exc:
        p.prompt_input_tokens("private")
    assert "sensitive" not in str(exc.value) and session.post.call_count == 1


@pytest.mark.parametrize("url", ["http://example.com", "https://user:password@example.com", "https://example.com/?key=x"])
def test_invalid_endpoint_rejected(url):
    with pytest.raises(LLMConfigurationError):
        ModalMistralProvider(url, "key")


def test_tokenizer_malformed_data_cannot_disable_budget_checks():
    p, _ = provider(response({"count": True}))
    with pytest.raises(LLMMalformedResponseError):
        p.prompt_input_tokens("test")


def test_deployment_has_separate_text_only_fp8_profile():
    cmd = build_serve_command()
    assert "--language-model-only" in cmd
    assert "--reasoning-parser" not in cmd
    assert cmd[cmd.index("--quantization") + 1] == "fp8"
    assert "--api-key" not in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "1"
