"""Exercise text checkpoint recovery through the real orchestrator method."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from explanation_cache import text_cache_is_current
from orchestrator.orchestrator import PipelineOrchestrator
from synchronizer import TranscriptEntry


@pytest.fixture(params=["ollama", "remote"])
def checkpointed_text(tmp_path, monkeypatch, request):
    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", "ollama")
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.base_dir = str(tmp_path)
    provider = None if request.param == "ollama" else object()
    select_provider = Mock(return_value=provider)
    warm = Mock()
    service = Mock(return_value=SimpleNamespace(warm_model=warm))
    synthesizer = Mock()
    synthesizer.explain_transcript.return_value = "Energy is conserved."
    constructor = Mock(return_value=synthesizer)
    monkeypatch.setattr("orchestrator.orchestrator.configured_explanation_provider", select_provider)
    monkeypatch.setattr("orchestrator.orchestrator.get_ollama_service", service)
    monkeypatch.setattr("synthesizer.LLMSynthesizer", constructor)
    ownership_check = Mock()
    checkpoint_commit = Mock(side_effect=lambda action: action())
    arguments = dict(
        video_id="checkpoint-video",
        intervention_id="checkpoint-point",
        intervention_timestamp=2.0,
        content_type="unknown",
        trigger_reason="User request",
        transcript_entries=[TranscriptEntry(start=0, end=5, text="Energy is conserved.")],
        output_mode="brief",
        vlm_snapshot={"_transcript_only": True, "transcript_context": "Energy is conserved."},
        ownership_check=ownership_check,
        checkpoint_commit=checkpoint_commit,
    )
    artifact = orchestrator.synthesize_intervention_text(**arguments)
    assert not Path(artifact["audio_file_path"]).exists()
    assert text_cache_is_current(
        artifact["text_explanation"], artifact["audio_file_path"],
        artifact["vlm_snapshot_path"], input_hash=artifact["input_hash"],
    )
    for mock in (select_provider, warm, service, synthesizer, constructor,
                 ownership_check, checkpoint_commit):
        mock.reset_mock()
    return SimpleNamespace(
        orchestrator=orchestrator, arguments=arguments, artifact=artifact,
        provider=provider, select_provider=select_provider, warm=warm,
        service=service, synthesizer=synthesizer, constructor=constructor,
    )


def test_valid_text_checkpoint_recovers_without_provider_initialization(checkpointed_text):
    case = checkpointed_text
    case.select_provider.side_effect = RuntimeError("Text provider unavailable")
    case.warm.side_effect = RuntimeError("Mistral unavailable")
    case.constructor.side_effect = RuntimeError("Synthesizer unavailable")

    recovered = case.orchestrator.synthesize_intervention_text(**case.arguments)

    assert recovered == case.artifact
    assert recovered["reuse_generation"] is False  # Speech still needs regeneration.
    case.select_provider.assert_not_called()
    case.service.assert_not_called()
    case.warm.assert_not_called()
    case.constructor.assert_not_called()
    case.synthesizer.cleanup.assert_not_called()
    case.arguments["ownership_check"].assert_called()


def test_stale_text_checkpoint_initializes_provider_and_preserves_prompt(checkpointed_text):
    case = checkpointed_text
    arguments = dict(case.arguments, trigger_reason="A different request")

    regenerated = case.orchestrator.synthesize_intervention_text(**arguments)

    assert regenerated["input_hash"] != case.artifact["input_hash"]
    case.select_provider.assert_called_once_with()
    if case.provider is None:
        case.service.assert_called_once_with()
        case.warm.assert_called_once_with()
    else:
        case.service.assert_not_called()
        case.warm.assert_not_called()
    case.constructor.assert_called_once_with(
        ollama_url="http://localhost:11434", model="mistral:latest",
        timeout=120, max_retries=3, transcripts=arguments["transcript_entries"],
        output_mode="brief", ownership_check=arguments["ownership_check"],
        **({"text_provider": case.provider} if case.provider is not None else {}),
    )
    case.synthesizer.explain_transcript.assert_called_once_with("Energy is conserved.", "brief")
    case.synthesizer.weave.assert_not_called()
    case.synthesizer.cleanup.assert_called_once_with()
    case.arguments["checkpoint_commit"].assert_called_once()
    assert text_cache_is_current(
        regenerated["text_explanation"], regenerated["audio_file_path"],
        regenerated["vlm_snapshot_path"], input_hash=regenerated["input_hash"],
    )


def test_stale_text_checkpoint_cleans_up_after_synthesis_failure(checkpointed_text):
    case = checkpointed_text
    case.synthesizer.explain_transcript.side_effect = RuntimeError("Synthesis failed")

    with pytest.raises(RuntimeError, match="Synthesis failed"):
        case.orchestrator.synthesize_intervention_text(
            **dict(case.arguments, trigger_reason="A different request"),
        )

    case.constructor.assert_called_once()
    case.synthesizer.cleanup.assert_called_once_with()
    case.arguments["checkpoint_commit"].assert_not_called()
    assert Path(case.artifact["text_file_path"]).read_text(encoding="utf-8") == "Energy is conserved."
