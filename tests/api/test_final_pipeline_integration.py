"""Offline production-path contracts; only model/readiness/TTS boundaries are fake."""

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_db
from api.explanation_preparation import ExplanationPreparationManager
from api.models import Explanation, ExplanationPreparation
from api.processing_jobs import _build_interventions, persist_processing_result
from api.routers import video as routes
from orchestrator.orchestrator import PipelineOrchestrator
from tests.test_evidence_reconciliation import _element, _evidence
from tests.test_structured_vlm_integration import _engine
from visual_evidence import assess_snapshot, load_usable_snapshot


VIDEO_ID = "Task5Test01"
WEAK_TEXT = "A generic explanation that misses important displayed relationships."
SOURCE = Path("data/Tf4A93763Bw/interventions/a0a2adaa/vlm_snapshot.json")


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    data = tmp_path / "data"
    video_dir = data / VIDEO_ID
    video_dir.mkdir(parents=True)
    frame = video_dir / "frame.jpg"
    frame.write_bytes(b"fake frame: image boundary is mocked")
    entries = [{"start": 400.0, "duration": 10.0, "text": "The principle of conservation of energy"}]
    (video_dir / "transcript.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    result = {"transcript": {"entries": entries}, "intervention_points": [{
        "timestamp": 406.0, "frame_path": str(frame), "content_type": "diagram",
        "trigger_reason": "Deictic", "raw_category_labels": ["diagram"],
    }]}
    records = _build_interventions(VIDEO_ID, result)
    point = records[0].id
    with sessions() as db:
        persist_processing_result(db, VIDEO_ID, f"https://www.youtube.com/watch?v={VIDEO_ID}", result, records)

    raw = json.loads(SOURCE.read_text(encoding="utf-8"))["_structured_observation"]
    real_engine = _engine(copy.deepcopy(raw))
    calls = {"vlm": [], "vlm_inference": [], "llm": [], "tts": []}
    failure = {"stage": None}
    analyze = real_engine.vlm_interface.analyze

    def analyze_once(*args, **kwargs):
        calls["vlm_inference"].append((args, kwargs))
        return analyze(*args, **kwargs)

    monkeypatch.setattr(real_engine.vlm_interface, "analyze", analyze_once)

    class FakeSnapshotEngine:
        model_name = real_engine.model_name

        def analyze_frame(self, *args, **kwargs):
            calls["vlm"].append((args, kwargs))
            snapshot = real_engine.analyze_frame(*args, **kwargs)
            snapshot["missing_elements"] = "Stale legacy OCR warning"
            if failure["stage"] == "vlm":
                snapshot["_error"] = True
            return snapshot

        def cleanup(self):
            pass

    def generate(_self, prompt, **kwargs):
        calls["llm"].append(prompt)
        return SimpleNamespace(text="" if failure["stage"] == "llm" else WEAK_TEXT,
                               total_duration_ms=1, prompt_eval_count=1, eval_count=1, model="fake", done=True, done_reason="stop")

    def tts(text, path):
        calls["tts"].append((text, path))
        Path(path).write_bytes(b"ID3-offline-test")

    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", "modal")
    monkeypatch.setattr(PipelineOrchestrator, "_create_intervention_vlm_engine",
                        staticmethod(lambda provider: (FakeSnapshotEngine(), real_engine.model_name)))
    monkeypatch.setattr("orchestrator.orchestrator.get_ollama_service",
                        lambda: SimpleNamespace(begin_warmup=lambda: None, warm_model=lambda: None))
    monkeypatch.setattr("api.resource_guard.require_processing_headroom", lambda: None)
    monkeypatch.setattr("llm_interface.OllamaClient.generate", generate)
    monkeypatch.setattr("llm_interface.OllamaClient.prompt_input_tokens", lambda self, prompt: len(prompt.encode()) // 3 + 64)
    monkeypatch.setitem(sys.modules, "tts_engine", SimpleNamespace(
        get_shared_tts_engine=lambda: SimpleNamespace(generate=tts)))
    orchestrator = PipelineOrchestrator(base_dir=str(data))
    manager = ExplanationPreparationManager(session_factory=sessions,
        orchestrator_factory=lambda: orchestrator, data_dir=str(data))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_preparation_manager] = lambda: manager
    app.dependency_overrides[routes.get_data_dir] = lambda: str(data)

    def database():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = database
    manager.start()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield SimpleNamespace(manager=manager, client=client, sessions=sessions,
                calls=calls, point=point, data=data, raw=raw, failure=failure,
                orchestrator=orchestrator)
    finally:
        manager.shutdown()
        engine.dispose()


@pytest.mark.parametrize("path", ["automatic", "manual"])
@pytest.mark.parametrize("mode", ["brief", "explanatory", "detailed"])
def test_repaired_evidence_reaches_prompt_persistence_audio_and_cache(pipeline, path, mode):
    p = pipeline
    request = {"video_id": VIDEO_ID, "intervention_id": p.point, "output_mode": mode}
    if path == "automatic":
        assert p.manager.schedule_video(VIDEO_ID, mode) == 1
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = p.manager.get_status(VIDEO_ID, mode)[0]
            if state.status in {"ready", "failed"}:
                break
            time.sleep(0.01)
        assert state.status == "ready", state
    else:
        response = p.client.post("/video/intervention/explain", json=request)
        assert response.status_code == 200, response.text

    response = p.client.get(f"/video/intervention/{p.point}/explanation", params={"output_mode": mode})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["text_explanation"] == WEAK_TEXT
    assert payload["output_mode"] == mode and payload["intervention_id"] == p.point
    assert p.client.get(payload["audio_url"]).content == b"ID3-offline-test"
    assert Path(payload["text_file_path"]).read_text(encoding="utf-8") == WEAK_TEXT
    snapshot_path = Path(payload["vlm_snapshot_path"])
    saved_bytes = snapshot_path.read_bytes()
    saved = json.loads(saved_bytes)
    canonical = saved["visual_evidence"]
    assert canonical["frame_id"] == p.point and canonical["timestamp_seconds"] == 406
    assert saved["_structured_observation"] == p.raw
    assert len(canonical["regions"]) == 4 and sum(len(r["elements"]) for r in canonical["regions"]) == 12
    assert len(canonical["relations"]) == 5 and len(canonical["exact_content"]) == 3
    codes = {item["code"] for item in canonical["validation_diagnostics"]}
    assert "normalization.consolidated" in codes
    assert any(code.startswith("extraction.quality.") for code in codes)
    assert canonical["extraction_status"] == "partial"
    assert saved["coverage_plan"]["schema"] == "synapse_transcript_visual_coverage_v1"
    assert saved["coverage_plan"]["fact_count"] == 23
    prompt = p.calls["llm"][0]
    assert "attention_groups" in prompt and "The principle of conservation of energy" in prompt
    assert "CANONICAL VISUAL EVIDENCE" in prompt
    assert len(p.calls["vlm"]) == len(p.calls["llm"]) == len(p.calls["tts"]) == 1
    assert len(p.calls["vlm_inference"]) == 1
    assert p.calls["tts"][0][0] == WEAK_TEXT
    assert p.calls["vlm"][0][1]["frame_id"] == p.point
    assert p.calls["vlm"][0][1]["raw_category_labels"] == ["diagram"]
    with p.sessions() as db:
        row = db.query(Explanation).one()
        assert row.intervention_id == p.point and row.output_mode == mode
        assert row.text_explanation == WEAK_TEXT and row.vlm_snapshot_path == str(snapshot_path)
    assert p.client.post("/video/intervention/explain", json=request).json()["text_explanation"] == WEAK_TEXT
    assert p.manager.schedule_video(VIDEO_ID, mode) == 0
    assert len(p.calls["vlm"]) == len(p.calls["llm"]) == len(p.calls["tts"]) == 1
    assert snapshot_path.read_bytes() == saved_bytes


@pytest.mark.parametrize("stage", ["vlm", "llm"])
def test_technical_failure_is_not_persisted_as_a_success(pipeline, stage):
    p = pipeline
    p.failure["stage"] = stage
    response = p.client.post("/video/intervention/explain", json={
        "video_id": VIDEO_ID, "intervention_id": p.point, "output_mode": "brief"})
    assert response.status_code == 500
    with p.sessions() as db:
        assert db.query(Explanation).count() == 0
        state = db.query(ExplanationPreparation).one()
        assert state.status == "failed"
        assert state.error_code == ("visual_evidence_unavailable" if stage == "vlm" else "text_synthesis_unavailable")
    assert not p.calls["tts"]
    assert len(p.calls["llm"]) == (0 if stage == "vlm" else 1)


def test_other_mode_reuses_evidence_without_rewriting_saved_source(pipeline):
    p = pipeline
    first = p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    saved = Path(first.vlm_snapshot_path).read_bytes()
    second = p.manager.generate_manual(VIDEO_ID, p.point, "detailed")
    assert first.output_mode == "brief" and second.output_mode == "detailed"
    assert len(p.calls["vlm"]) == 1
    assert len(p.calls["vlm_inference"]) == 1
    assert len(p.calls["llm"]) == len(p.calls["tts"]) == 2
    assert Path(second.vlm_snapshot_path).read_bytes() == saved
    with p.sessions() as db:
        assert {row.output_mode for row in db.query(Explanation).all()} == {"brief", "detailed"}


def test_stale_generation_regenerates_text_and_audio_together(pipeline):
    from explanation_cache import manifest_path
    p = pipeline
    first = p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    manifest_path(first.audio_file_path).unlink()
    second = p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    assert second.text_explanation == WEAK_TEXT
    assert len(p.calls["vlm"]) == 1
    assert len(p.calls["llm"]) == len(p.calls["tts"]) == 2
    assert manifest_path(second.audio_file_path).is_file()


def test_unversioned_evidence_is_reextracted_before_generation(pipeline):
    p = pipeline
    first = p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    path = Path(first.vlm_snapshot_path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot.pop("_quality_version")
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    assert len(p.calls["vlm"]) == 2
    # The fake returns byte-identical current evidence; its existing current
    # explanation/audio remain valid by hash after the required re-extraction.
    assert len(p.calls["llm"]) == len(p.calls["tts"]) == 1


@pytest.mark.parametrize("finish", ["stop", "max_tokens"])
@pytest.mark.parametrize("provider_name", ["gemini", "modal_mistral"])
def test_configured_cloud_explanation_uses_existing_publication_path_without_ollama(pipeline, monkeypatch, finish, provider_name):
    from llm_interface.providers import GenerationResult
    from synthesizer.configuration import explanation_profile
    p = pipeline
    provider_calls = []

    def generate(prompt, settings):
        provider_calls.append((prompt, settings))
        text = "Cloud explanation grounded in the supplied records."
        if provider_name == "modal_mistral":
            assert settings.response_schema is not None
            text = json.dumps({"visual_description": text, "educational_explanation": "", "lecture_connection": "", "uncertainty": ""})
        return GenerationResult(text, provider_name, "test-model", 1,
                                finish_reason=finish)

    monkeypatch.setenv("SYNAPSE_EXPLANATION_PROVIDER", provider_name)
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr("synthesizer.configuration.get_text_provider",
                        lambda name: SimpleNamespace(generate=generate))
    if provider_name == "modal_mistral":
        monkeypatch.setattr("llm_interface.modal_mistral_provider.ModalMistralProvider.from_env",
                            lambda: SimpleNamespace(generate=generate, max_output_tokens=1536, generation_temperature=0.1,
                                                    supports_structured_output=True))
    monkeypatch.setattr("orchestrator.orchestrator.get_ollama_service",
                        lambda: pytest.fail("Cloud generation must not load a local model"))
    request = {"video_id": VIDEO_ID, "intervention_id": p.point, "output_mode": "brief"}
    response = p.client.post("/video/intervention/explain", json=request)
    assert len(provider_calls) == 1 and len(p.calls["vlm"]) == 1
    assert not p.calls["llm"]
    if finish == "stop":
        assert response.status_code == 200, response.text
        assert response.json()["text_explanation"].startswith("Cloud explanation")
        assert len(p.calls["tts"]) == 1
        from explanation_cache import manifest_path
        manifest = json.loads(manifest_path(response.json()["audio_file_path"]).read_text())
        assert manifest["text_generator"] == explanation_profile()
        assert p.client.post("/video/intervention/explain", json=request).status_code == 200
        assert len(provider_calls) == 1
    else:
        assert response.status_code == 500
        assert not p.calls["tts"]
        with p.sessions() as db:
            assert db.query(Explanation).count() == 0


def test_explicit_transcript_end_reaches_the_correct_current_segment(pipeline):
    p = pipeline
    entries = [{"start": 400, "end": 410, "text": "CURRENT_TEACHER"},
               {"start": 411, "end": 415, "text": "NEXT_TEACHER"}]
    (p.data / VIDEO_ID / "transcript.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    p.manager.generate_manual(VIDEO_ID, p.point, "brief")
    assert 'Current teacher transcript: "CURRENT_TEACHER"' in p.calls["llm"][0]


@pytest.mark.parametrize("failure", [None, "error", "failed", "unavailable", "empty", "invalid", "legacy"])
def test_canonical_availability_outranks_only_stale_legacy_missing_flags(tmp_path, failure):
    evidence = _evidence("CT-09", status="partial", elements=[_element("control", "Network", "CT-09")])
    snapshot = {"visual_evidence": evidence, "missing_elements": "Legacy OCR warning"}
    if failure == "error":
        snapshot["_error"] = True
    elif failure == "failed":
        evidence["extraction_status"] = "failed"
    elif failure == "unavailable":
        evidence["evidence_status"] = "unavailable_or_unreadable"
    elif failure == "empty":
        evidence["regions"] = []
        evidence["structural_description"] = {}
    elif failure == "invalid":
        evidence["schema_version"] = "invalid"
    elif failure == "legacy":
        snapshot.pop("visual_evidence")
    original = copy.deepcopy(snapshot)
    assert assess_snapshot(snapshot).usable is (failure is None)
    assert snapshot == original
    if failure != "legacy":
        path = tmp_path / "snapshot.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        before = path.read_bytes()
        assert (load_usable_snapshot(path) is not None) is (failure is None)
        assert path.read_bytes() == before
