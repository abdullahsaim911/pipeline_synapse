"""Focused mocked tests for Synapse's Ollama generation request."""

import os
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest import mock

import pytest
import requests

from llm_interface.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def no_installed_model_in_mocked_http_tests(monkeypatch):
    # Request-contract tests must not load a machine-specific GGUF vocabulary.
    monkeypatch.setattr(OllamaClient, "prompt_input_tokens", lambda self, prompt: len(prompt.encode()) // 3 + 64)


def _ollama_response():
    response = mock.Mock(status_code=200)
    response.json.return_value = {
        "response": "generated explanation",
        "model": "mistral:latest",
        "done": True,
    }
    return response


def test_generate_posts_cpu_and_2048_context_options_by_default():
    client = OllamaClient(model="mistral", max_retries=1)

    with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
        "llm_interface.ollama_client.FUSION_AVAILABLE", False
    ), mock.patch(
        "llm_interface.ollama_client.requests.post", return_value=_ollama_response()
    ) as post:
        result = client.generate("Explain the visual evidence")

    assert result.text == "generated explanation"
    assert post.call_args.args == ("http://localhost:11434/api/generate",)
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "mistral"
    assert payload["options"]["num_gpu"] == 0
    assert payload["options"]["num_ctx"] == 2048
    assert payload["options"]["num_predict"] == 512
    assert post.call_args.kwargs["timeout"] == (5, 120)


def test_generate_uses_validated_ollama_environment_overrides():
    client = OllamaClient(max_retries=1)
    environment = {
        "SYNAPSE_OLLAMA_NUM_GPU": "1",
        "SYNAPSE_OLLAMA_NUM_CTX": "4096",
    }

    with mock.patch.dict(os.environ, environment, clear=True):
        payload = client._build_payload("prompt")

    assert payload["options"]["num_gpu"] == 1
    assert payload["options"]["num_ctx"] == 4096


@pytest.mark.parametrize(
    ("name", "value"),
    [("SYNAPSE_OLLAMA_NUM_GPU", "auto"), ("SYNAPSE_OLLAMA_NUM_CTX", "0")],
)
def test_invalid_ollama_environment_option_fails_before_http(name, value):
    client = OllamaClient(max_retries=1)

    with mock.patch.dict(os.environ, {name: value}, clear=True), mock.patch(
        "llm_interface.ollama_client.requests.post"
    ) as post, pytest.raises(ValueError, match=name):
        client.generate("prompt")

    post.assert_not_called()


@pytest.mark.parametrize(("mode", "budget"), [
    ("brief", 768), ("explanatory", 1024), ("standard", 1024), ("detailed", 1536),
])
def test_synthesis_mode_options_reach_the_actual_http_request(mode, budget, monkeypatch):
    from tests.test_task5_synthesis import _synth, _snapshot, _evidence, _element

    monkeypatch.setenv("SYNAPSE_OLLAMA_NUM_CTX", "8192")
    monkeypatch.setattr("llm_interface.ollama_client.FUSION_AVAILABLE", False)
    synth = _synth()
    synth.client = OllamaClient(max_retries=1)
    snapshot = _snapshot(_evidence("CT-09", elements=[_element("control", "Network", "CT-09")]))
    original = copy.deepcopy(snapshot)
    expected_prompt = synth._build_task5_prompt("Look here.", {}, snapshot, mode, "software_interface", "Deictic")
    with mock.patch("llm_interface.ollama_client.requests.post", return_value=_ollama_response()) as post:
        result = synth.weave("Look here.", snapshot, "software_interface", "Deictic", mode,
                             require_visual_explanation=True)
    assert result == "generated explanation"
    post.assert_called_once()
    assert post.call_args.args == ("http://localhost:11434/api/generate",)
    payload = post.call_args.kwargs["json"]
    assert payload["prompt"] == expected_prompt
    assert payload["model"] == "mistral:latest"
    assert payload["options"]["num_predict"] == budget
    assert payload["options"]["num_ctx"] == 8192
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["repeat_penalty"] == 1.0
    for key in ("top_k", "top_p"):
        assert payload["options"][key] == OllamaClient.DEFAULT_GENERATION_PARAMS[key]
    assert snapshot == original
    assert synth.output_mode == "explanatory"  # Per-request mode never mutates instance defaults.


def test_retries_keep_original_options_even_if_caller_changes_its_dictionary(monkeypatch):
    monkeypatch.setenv("SYNAPSE_OLLAMA_NUM_CTX", "8192")
    monkeypatch.setattr("llm_interface.ollama_client.FUSION_AVAILABLE", False)
    options = {"num_predict": 1536}
    payloads = []

    def post(_url, **kwargs):
        payloads.append(copy.deepcopy(kwargs["json"]))
        if len(payloads) == 1:
            options["num_predict"] = 7
            raise requests.exceptions.Timeout("controlled timeout")
        return _ollama_response()

    with mock.patch("llm_interface.ollama_client.requests.post", side_effect=post), mock.patch(
        "llm_interface.ollama_client.time.sleep"
    ) as sleep:
        assert OllamaClient().generate("prompt", options=options).text == "generated explanation"
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert payloads[1]["options"]["num_predict"] == 1536
    sleep.assert_called_once_with(1)
    assert OllamaClient.DEFAULT_GENERATION_PARAMS["num_predict"] == 512


def test_concurrent_modes_on_one_synthesizer_keep_request_local_options(monkeypatch):
    from tests.test_task5_synthesis import _synth, _snapshot, _evidence, _element

    monkeypatch.setenv("SYNAPSE_OLLAMA_NUM_CTX", "8192")
    monkeypatch.setattr("llm_interface.ollama_client.FUSION_AVAILABLE", False)
    synth = _synth()
    synth.client = OllamaClient(max_retries=1)
    snapshot = _snapshot(_evidence("CT-09", elements=[_element("control", "Network", "CT-09")]))
    barrier = Barrier(3)
    payloads = []

    def post(_url, **kwargs):
        payloads.append(copy.deepcopy(kwargs["json"]))
        barrier.wait(timeout=5)
        return _ollama_response()

    def generate(mode):
        return synth.weave("Look here.", snapshot, "software_interface", "Deictic", mode,
                           require_visual_explanation=True)

    with mock.patch("llm_interface.ollama_client.requests.post", side_effect=post):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(generate, ["brief", "explanatory", "detailed"]))
    assert results == ["generated explanation"] * 3
    assert len(payloads) == 3
    assert {p["options"]["num_predict"] for p in payloads} == {768, 1024, 1536}
    assert all(p["options"]["num_ctx"] == 8192 for p in payloads)
    for payload in payloads:
        expected = 768 if "BRIEF:" in payload["prompt"] else 1536 if "DETAILED:" in payload["prompt"] else 1024
        assert payload["options"]["num_predict"] == expected
    assert synth.output_mode == "explanatory"


def test_legacy_fusion_runner_is_kept_only_for_callers_without_options(monkeypatch):
    monkeypatch.setattr("llm_interface.ollama_client.FUSION_AVAILABLE", True)
    monkeypatch.setenv("SYNAPSE_OLLAMA_NUM_CTX", "8192")
    client = OllamaClient(max_retries=1)
    with mock.patch("llm_interface.ollama_client.base_generate", return_value="legacy") as legacy, mock.patch(
        "llm_interface.ollama_client.requests.post", return_value=_ollama_response()
    ) as post:
        assert client.generate("default prompt").text == "legacy"
        options = {"num_predict": 384}
        assert client.generate("explicit prompt", options=options).text == "generated explanation"
    legacy.assert_called_once_with("default prompt")
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["options"]["num_predict"] == 384
    assert options == {"num_predict": 384}
