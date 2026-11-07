"""Focused mocked tests for Synapse's Ollama generation request."""

import os
from unittest import mock

import pytest

from llm_interface.ollama_client import OllamaClient


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
