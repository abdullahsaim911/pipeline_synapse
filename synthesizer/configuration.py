"""Explanation-owned provider selection; Ask/Quiz settings stay independent."""

import os

from llm_interface.providers import LLMConfigurationError, get_text_provider


def explanation_profile():
    provider = os.environ.get("SYNAPSE_EXPLANATION_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return {"provider": provider, "model": "mistral:latest"}
    if provider == "gemini":
        from llm_interface.gemini_provider import DEFAULT_MODEL
        return {"provider": provider, "model": os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL}
    if provider == "modal_mistral":
        from modal_inference.mistral_config import MODEL_ID, MODEL_REVISION
        return {"provider": provider, "model": MODEL_ID, "revision": MODEL_REVISION}
    raise LLMConfigurationError("SYNAPSE_EXPLANATION_PROVIDER must be ollama, modal_mistral or gemini")


def configured_explanation_provider():
    selected = explanation_profile()["provider"]
    if selected == "modal_mistral":
        from llm_interface.modal_mistral_provider import ModalMistralProvider
        return ModalMistralProvider.from_env()
    return get_text_provider("gemini") if selected == "gemini" else None
