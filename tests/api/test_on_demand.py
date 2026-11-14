"""Private intent, shared generation, lifecycle and recovery using temporary storage."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch
import queue
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from api.database import get_db
from api.explanation_preparation import ExplanationPreparationManager, PreparationJob, _PreparationQueue
from api.models import (ExplanationPreparation, ExplanationRequest, Intervention, LectureSessionInterest,
                        ProcessingJob, Transcript, User, Video)
from api.on_demand import OnDemandExplanations, utcnow
from api.processing_jobs import _delete_invalid_cache
from api.routers.explanation_requests import router
from tests.api import test_explanation_preparation as preparation_fixtures
from tests.api.test_explanation_preparation import FakeExplanationOrchestrator, VIDEO_ID, VIDEO_URL
from tests.api.test_explanation_status import wait_until


@pytest.fixture
def system():
    fixture = preparation_fixtures.ExplanationPreparationTests()
    fixture.setUp()
    fixture.seed()
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    with fixture.Session() as db:
        db.add_all([User(id=owner, profile_key="owner", display_name="Owner"),
                    User(id=other, profile_key="other", display_name="Other")])
        db.add(Transcript(video_id=VIDEO_ID, start_time=0, end_time=60, text="Energy is conserved in this system."))
        db.commit()
    fake = FakeExplanationOrchestrator(fixture.data_dir)
    manager = ExplanationPreparationManager(session_factory=fixture.Session, orchestrator_factory=lambda: fake,
        data_dir=str(fixture.data_dir), evidence_preparer=lambda **_: None)
    processing = Mock()
    service = OnDemandExplanations(fixture.Session, manager, processing)
    from pathlib import Path
    with patch("api.personal_storage.PROJECT_ROOT", Path(fixture.temp.name)):
        manager.start()
        try:
            yield SimpleNamespace(fixture=fixture, manager=manager, service=service, fake=fake,
                owner=owner, other=other, processing=processing)
        finally:
            manager.shutdown()
            fixture.doCleanups()


def submit(system, owner=None, operation="one", session=None, timestamp=12):
    owner = owner or system.owner
    session = session or system.service.open(owner, VIDEO_ID)["id"]
    return system.service.request(owner, session, operation, VIDEO_ID, timestamp, "explanatory")


def test_shared_generation_and_private_requests(system):
    first = submit(system)
    second = submit(system, owner=system.other)
    assert first["id"] != second["id"]
    assert first["intervention_id"] == second["intervention_id"]
    assert wait_until(lambda: len(system.fake.calls) == 1)
    assert wait_until(lambda: system.service.get(system.owner, first["id"])["status"] == "ready")
    system.service.maintain()
    assert system.service.get(system.owner, first["id"])["saved_explanation_id"]
    assert len(system.service.list(system.owner, VIDEO_ID)["items"]) == 1
    with pytest.raises(HTTPException) as error:
        system.service.get(system.other, first["id"])
    assert error.value.status_code == 404


def test_idempotency_and_different_payload_conflict(system):
    identity = system.service.open(system.owner, VIDEO_ID)["id"]
    first = submit(system, session=identity)
    again = submit(system, session=identity)
    assert first["id"] == again["id"]
    with pytest.raises(HTTPException) as error:
        submit(system, session=identity, timestamp=13)
    assert error.value.status_code == 409


def test_library_capture_does_not_copy_another_learners_manual_request(system):
    from api.personal_storage import capture_lecture
    from api.models import SavedExplanation
    item = submit(system)
    assert wait_until(lambda: system.service.get(system.owner, item["id"])["status"] == "ready")
    system.service.maintain()
    with system.fixture.Session() as db:
        capture_lecture(db, system.other, VIDEO_ID)
        db.commit()
        assert db.query(SavedExplanation).filter_by(user_id=system.other).count() == 0


def test_last_session_release_and_signout_continuation(system):
    first = system.service.open(system.owner, VIDEO_ID)["id"]
    second = system.service.open(system.other, VIDEO_ID)["id"]
    with system.fixture.Session() as db:
        job = ProcessingJob(id=str(uuid.uuid4()), video_id=VIDEO_ID, source_url=VIDEO_URL,
            requested_output_mode="explanatory", status="running", stage="processing")
        db.add(job)
        db.commit()
    system.service.release(system.owner, first, "leave")
    system.processing.cancel.assert_not_called()
    system.service.release(system.other, second, "signout")
    system.service.maintain()
    system.processing.cancel.assert_not_called()
    with system.fixture.Session() as db:
        assert db.get(LectureSessionInterest, second).state == "continuing"


def test_expiry_cancels_only_after_all_interests_end(system):
    identity = system.service.open(system.owner, VIDEO_ID)["id"]
    with system.fixture.Session() as db:
        job = ProcessingJob(id=str(uuid.uuid4()), video_id=VIDEO_ID, source_url=VIDEO_URL,
            requested_output_mode="explanatory", status="running", stage="processing")
        db.add(job)
        db.get(LectureSessionInterest, identity).expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    system.service.maintain()
    system.processing.cancel.assert_called_once()


def test_saved_progress_survives_inference_cache_replacement(system):
    item = submit(system)
    assert wait_until(lambda: system.service.get(system.owner, item["id"])["status"] == "ready")
    system.service.maintain()
    system.service.progress(system.owner, item["id"], {"heard": True, "satisfaction": "yes"})
    with system.fixture.Session() as db:
        # Exercise SQLite's actual SET NULL relationship, including the shared cache deletes.
        db.connection().exec_driver_sql("PRAGMA foreign_keys=ON")
        _delete_invalid_cache(db, VIDEO_ID)
        db.commit()
        row = db.get(ExplanationRequest, item["id"])
        assert row.intervention_id is None
        assert row.saved_explanation_id
    restored = system.service.get(system.owner, item["id"])
    assert restored["status"] == "ready"
    assert restored["heard"]
    assert restored["satisfaction"] == "yes"


def test_requests_wait_for_source_without_restarting_detection(system):
    video = "Waiting1234"
    with system.fixture.Session() as db:
        db.add(ProcessingJob(id=str(uuid.uuid4()), video_id=video, source_url=f"https://www.youtube.com/watch?v={video}",
            requested_output_mode="brief", status="running", stage="processing"))
        db.commit()
    identity = system.service.open(system.owner, video)["id"]
    item = system.service.request(system.owner, identity, "waiting", video, 12, "brief")
    assert item["status"] == "waiting_for_context"
    system.processing.create_or_join.assert_not_called()
    with system.fixture.Session() as db:
        db.add(Video(id=video, title="Waiting lecture", source_url=f"https://www.youtube.com/watch?v={video}"))
        db.add(Transcript(video_id=video, start_time=0, end_time=30, text="A supported explanation."))
        db.commit()
    system.service.source_ready(video)
    with system.fixture.Session() as db:
        row = db.get(ExplanationRequest, item["id"])
        assert row.intervention_id
        state = db.query(ExplanationPreparation).filter_by(intervention_id=row.intervention_id).one()
        assert state.priority_at


def test_manual_moments_are_excluded_from_background_preparation(system):
    with system.fixture.Session() as db:
        db.add(Intervention(id="manual", video_id=VIDEO_ID, timestamp=8, origin="manual",
            evidence_kind="transcript", frame_path="", transcript_context="Context"))
        db.commit()
    system.manager.schedule_video(VIDEO_ID, "brief")
    with system.fixture.Session() as db:
        assert db.query(ExplanationPreparation).filter_by(intervention_id="manual").first() is None


def test_priority_fifo_and_provider_deferral():
    first = PreparationJob(VIDEO_ID, "earlier-background", "explanatory")
    second = PreparationJob(VIDEO_ID, "first-request", "explanatory")
    priorities = {first.key, second.key}
    order = {second.key: utcnow(), first.key: utcnow() + timedelta(seconds=1)}
    work = _PreparationQueue(priorities, priority_order=order)
    work.put(first)
    work.put(second)
    assert work.get() is second
    deferred = {second.key}
    work = _PreparationQueue({second.key}, wait_for_priority=True, deferred=deferred)
    work.put(first)
    assert work.get(timeout=0.05) is first
    deferred.clear()
    work.put(first)
    with pytest.raises(queue.Empty):
        work.get(timeout=0.02)


def test_ready_user_request_can_probe_before_a_queued_background_probe():
    import threading
    manager = ExplanationPreparationManager.__new__(ExplanationPreparationManager)
    foreground = PreparationJob(VIDEO_ID, "user", "explanatory")
    background = PreparationJob(VIDEO_ID, "background", "explanatory")
    manager._state_lock = threading.Lock()
    manager._priority_keys = {foreground.key}
    manager._provider_probe_key = background.key
    manager._provider_retry_after = 0
    manager._before_inference = lambda: True
    assert manager._check_inference_readiness(foreground)
    assert manager._provider_probe_key is None


def test_owned_history_is_paginated_without_dropping_older_requests(system):
    with system.fixture.Session() as db:
        start = utcnow()
        for index in range(201):
            db.add(ExplanationRequest(user_id=system.owner, operation_id=f"page-{index}",
                intent_hash="a" * 64, video_id=VIDEO_ID, timestamp=index, output_mode="brief",
                created_at=start + timedelta(milliseconds=index)))
        db.commit()
    first = system.service.list(system.owner, VIDEO_ID)
    second = system.service.list(system.owner, VIDEO_ID, first["next_cursor"])
    assert len(first["items"]) == 200
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None
    assert len({item["id"] for item in first["items"] + second["items"]}) == 201


def test_startup_restores_user_priority_before_background_dispatch(system):
    system.manager.shutdown()
    with system.fixture.Session() as db:
        for point_id in ("point-10", "point-20", "point-30"):
            db.add(ExplanationPreparation(video_id=VIDEO_ID, intervention_id=point_id,
                output_mode="explanatory", status="pending", attempt_count=0))
        db.commit()
    manager = ExplanationPreparationManager(session_factory=system.fixture.Session,
        orchestrator_factory=lambda: system.fake, data_dir=str(system.fixture.data_dir),
        evidence_preparer=lambda **_: None)
    try:
        manager.start(before_dispatch=lambda: manager.request(VIDEO_ID, "point-30", "explanatory"))
        assert wait_until(lambda: len(system.fake.calls) == 3)
        assert system.fake.calls[0]["intervention_id"] == "point-30"
    finally:
        manager.shutdown()


def test_request_reconciliation_failure_preserves_background_scheduling():
    import threading
    from api.processing_jobs import ProcessingJobManager
    manager = ProcessingJobManager.__new__(ProcessingJobManager)
    manager.preparation_manager = Mock()
    manager._stop_event = threading.Event()
    manager.context_ready = Mock(side_effect=RuntimeError("Temporary request-store failure"))
    manager._schedule_modes(VIDEO_ID, ["explanatory"])
    manager.preparation_manager.schedule_video.assert_called_once_with(VIDEO_ID, "explanatory")


def test_changed_source_is_not_silently_reused_after_cache_replacement(system):
    identity = system.service.open(system.owner, VIDEO_ID)["id"]
    item = system.service.request(system.owner, identity, "original-source", VIDEO_ID, 30,
        "explanatory", "point-30")
    assert wait_until(lambda: system.service.get(system.owner, item["id"])["status"] == "ready")
    with system.fixture.Session() as db:
        _delete_invalid_cache(db, VIDEO_ID)
        db.add(Video(id=VIDEO_ID, title="New source", source_url=VIDEO_URL))
        db.add(Transcript(video_id=VIDEO_ID, start_time=0, end_time=60, text="A different lecture segment."))
        db.commit()
    system.service.source_ready(VIDEO_ID)
    restored = system.service.get(system.owner, item["id"])
    assert restored["error_code"] == "source_changed"
    assert len(system.fake.calls) == 1


def test_request_routes_use_current_owner_and_validate_inputs(system):
    from api.auth import get_current_user
    app = FastAPI()
    app.include_router(router)
    app.state.on_demand_explanations = system.service
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=system.owner)
    def database():
        with system.fixture.Session() as db:
            yield db
    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        opened = client.post("/me/lecture-sessions", json={"video_id": VIDEO_ID})
        assert opened.status_code == 201
        body = {"video_id": VIDEO_ID, "session_id": opened.json()["id"], "operation_id": "route",
            "timestamp_seconds": 12, "output_mode": "explanatory"}
        response = client.post("/me/explanation-requests", json=body)
        assert response.status_code == 202, response.text
        assert client.get(f"/me/explanation-requests?video_id={VIDEO_ID}").json()["items"][0]["id"] == response.json()["id"]
        assert client.post("/me/explanation-requests", json={**body, "output_mode": "standard"}).status_code == 422
        assert client.post("/me/explanation-requests", json=body,
            headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_transcript_only_uses_existing_synthesis_tts_and_checkpoint_path(system, monkeypatch):
    from pathlib import Path
    from orchestrator.orchestrator import PipelineOrchestrator
    from api.explanation_service import prepare_visual_evidence
    from types import SimpleNamespace
    with system.fixture.Session() as db:
        db.add(Intervention(id="transcript-point", video_id=VIDEO_ID, timestamp=12,
            origin="manual", evidence_kind="transcript", frame_path="",
            transcript_context="Energy is conserved."))
        db.commit()
    evidence = prepare_visual_evidence(session_factory=system.fixture.Session,
        orchestrator_factory=lambda: (_ for _ in ()).throw(AssertionError("No visual provider required")),
        data_dir=str(system.fixture.data_dir), video_id=VIDEO_ID,
        intervention_id="transcript-point", output_mode="explanatory")
    assert evidence.snapshot["_transcript_only"]
    engine = PipelineOrchestrator.__new__(PipelineOrchestrator)
    engine.base_dir = str(system.fixture.data_dir)
    text = "Energy remains constant. Here is an additional example."
    synthesizer = SimpleNamespace(explain_transcript=Mock(return_value=text), cleanup=Mock())
    monkeypatch.setattr("synthesizer.LLMSynthesizer", lambda **_: synthesizer)
    monkeypatch.setattr("orchestrator.orchestrator.get_ollama_service", lambda: SimpleNamespace(warm_model=lambda: None))
    monkeypatch.setattr("tts_engine.get_shared_tts_engine", lambda: SimpleNamespace(
        generate=lambda script, destination: Path(destination).write_bytes(b"ID3-generated")))
    artifact = engine.synthesize_intervention_explanation(video_id=VIDEO_ID,
        intervention_id="transcript-point", intervention_timestamp=12, content_type="unknown",
        trigger_reason="User request", transcript_entries=[], vlm_snapshot=evidence.snapshot)
    assert artifact["text_explanation"] == text
    assert Path(artifact["audio_file_path"]).read_bytes().startswith(b"ID3")
    synthesizer.explain_transcript.assert_called_once_with("Energy is conserved.", "explanatory")
