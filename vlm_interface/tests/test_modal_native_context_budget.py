import json

from PIL import Image
import pytest

from modal_inference.config import MAX_MODEL_LEN
from vlm_interface.modal_vlm_interface import ModalVLMInterface
from vlm_interface.vlm_types import VLMContextBudgetError


def adapter():
    return ModalVLMInterface(endpoint_url="https://test.modal.run", api_key="test", max_retries=0)


def completion():
    return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 894, "completion_tokens": 2}}


def test_native_budget_counts_the_identical_multimodal_request_without_local_processor(monkeypatch):
    calls = []
    client = adapter()

    def post(url, payload):
        calls.append((url, json.loads(payload)))
        return {"count": 894, "max_model_len": MAX_MODEL_LEN} if url.endswith("/tokenize") else completion()

    monkeypatch.setattr(client, "_http_post", post)
    monkeypatch.setattr("vlm_interface.qwen_context_budget.QwenContextBudgetEstimator._load_processor",
                        lambda self: pytest.fail("A remote request must not load a local processor"))
    response = client.analyze(Image.new("RGB", (1280, 720)), "Inspect the frame")
    assert response.prompt_eval_count == 894
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["tokenize", "completions"]
    assert calls[0][1]["messages"] == calls[1][1]["messages"]
    assert calls[0][1]["add_generation_prompt"] is True
    assert client.last_context_budget["exact"] is True
    assert client.last_context_budget["input_tokens"] == 894


@pytest.mark.parametrize("count", [None, True, -1, "894", 0])
def test_invalid_native_count_never_reaches_generation(monkeypatch, count):
    client = adapter()
    calls = []

    def post(url, payload):
        calls.append(url)
        return {"count": count, "max_model_len": MAX_MODEL_LEN}

    monkeypatch.setattr(client, "_http_post", post)
    with pytest.raises(VLMContextBudgetError):
        client.analyze(Image.new("RGB", (100, 100)), "Inspect")
    assert len(calls) == 1


def test_native_count_reduces_only_the_output_tail_with_a_reserve(monkeypatch):
    client = adapter()
    calls = []

    def post(url, payload):
        calls.append(json.loads(payload))
        return {"count": 5000, "max_model_len": MAX_MODEL_LEN} if url.endswith("/tokenize") else completion()

    monkeypatch.setattr(client, "_http_post", post)
    client.analyze(Image.new("RGB", (100, 100)), "Inspect")
    assert calls[1]["max_tokens"] == MAX_MODEL_LEN - 5000 - 32


def test_incompatible_server_context_is_not_silently_accepted(monkeypatch):
    client = adapter()
    monkeypatch.setattr(client, "_http_post", lambda *args: {"count": 894, "max_model_len": 2048})
    with pytest.raises(VLMContextBudgetError, match="does not match"):
        client.analyze(Image.new("RGB", (100, 100)), "Inspect")
