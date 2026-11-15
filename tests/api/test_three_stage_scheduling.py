"""Exercise the production VLM -> text -> speech pipeline without model providers."""

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base
from api.explanation_preparation import (
    EXPLANATION_QUEUE_CAPACITY,
    ExplanationPreparationManager,
    STAGE_READY,
    STAGE_TEXT_READY,
)
from api.models import Explanation, ExplanationPreparation, Intervention, Video
from explanation_cache import cache_is_current, manifest_path
from tts_engine import streaming


VIDEO_ID = "ThreeStage1"
MODE = "explanatory"


def eventually(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "Pipeline did not reach the expected checkpoint"


class SplitOrchestrator:
    def __init__(self, root):
        self.root = root
        self.calls = {stage: [] for stage in ("vlm", "text", "tts")}
        self.active = Counter()
        self.peak = Counter()
        self.gates = {}
        self.entered = {}
        self.lock = threading.Lock()
        self.text_artifacts = {}
        self.interrupt_speech = set()
        self.fallback_preferences = []

    def started(self, stage, name):
        with self.lock:
            return self.entered.setdefault((stage, name), threading.Event())

    def hold(self, stage, name):
        gate = threading.Event()
        self.gates[stage, name] = gate
        return gate

    def run(self, stage, name, action):
        with self.lock:
            self.calls[stage].append(name)
            self.active[stage] += 1
            self.peak[stage] = max(self.peak[stage], self.active[stage])
            self.entered.setdefault((stage, name), threading.Event()).set()
        try:
            gate = self.gates.get((stage, name))
            if gate is not None:
                assert gate.wait(10), f"Unreleased test gate: {stage}:{name}"
            return action()
        finally:
            with self.lock:
                self.active[stage] -= 1

    def extract_visual_evidence(self, **kwargs):
        name = kwargs["intervention_id"]

        def extract():
            snapshot = {"intervention_id": name, "visible_text": "x = 2"}
            path = self.root / f"{name}.snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            return dict(snapshot=snapshot, snapshot_path=str(path), usable=True,
                        reused=False, failure=None)

        return self.run("vlm", name, extract)

    def synthesize_intervention_text(self, **kwargs):
        name = kwargs["intervention_id"]

        def synthesize():
            snapshot = self.root / f"{name}.snapshot.json"
            assert json.loads(snapshot.read_text()) == kwargs["vlm_snapshot"]
            text = f"The diagram for {name} shows x = 2."
            path = self.root / f"{name}.txt"
            kwargs["checkpoint_commit"](lambda: path.write_text(text, encoding="utf-8"))
            artifact = dict(intervention_id=name, text_explanation=text,
                            text_file_path=str(path), vlm_snapshot_path=str(snapshot),
                            output_mode=kwargs["output_mode"])
            self.text_artifacts[name] = artifact
            return artifact

        return self.run("text", name, synthesize)

    def generate_intervention_audio(self, text_artifact, **kwargs):
        name = text_artifact["intervention_id"]

        def speak():
            stream = streaming.current_stream.get()
            self.fallback_preferences.append((name, stream.prefer_fallback))
            if name in self.interrupt_speech:
                from tts_engine.exceptions import TTSEngineError

                self.interrupt_speech.remove(name)
                stream.audio(text_artifact["text_explanation"], "PocketTTS", b"\0\0" * 24)
                stream.prefer_fallback = True
                raise TTSEngineError("Controlled partial speech failure", code="tts_stream_interrupted")
            assert text_artifact is self.text_artifacts[name]
            assert Path(text_artifact["text_file_path"]).read_text() == text_artifact["text_explanation"]
            path = self.root / f"{name}.mp3"
            # Deliberately return even after cancellation: the real service must fence
            # this late provider result before publishing provenance or SQL.
            path.write_bytes(b"ID3-test-speech")
            return dict(text_artifact, audio_file_path=str(path))

        return self.run("tts", name, speak)

    def synthesize_intervention_explanation(self, **kwargs):
        raise AssertionError("The production split path must not use combined synthesis")


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # Keep attempt journals with the test artifacts, and use remote-style resource
    # locking so local-provider configuration cannot serialize VLM with text.
    monkeypatch.setenv("SYNAPSE_VLM_PROVIDER", "modal")
    monkeypatch.setattr(streaming, "AudioAttempt",
                        partial(streaming.AudioAttempt, root=tmp_path / "streams"))
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.sqlite').as_posix()}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    fake = SplitOrchestrator(tmp_path)
    manager = ExplanationPreparationManager(
        session_factory=sessions, orchestrator_factory=lambda: fake,
        data_dir=str(tmp_path),
    )
    managers = [manager]

    class Pipeline:
        def seed(self, count):
            with sessions() as db:
                db.add(Video(id=VIDEO_ID, title="Scheduling", source_url="https://example.test"))
                db.add_all([Intervention(id=f"ip{i}", video_id=VIDEO_ID,
                                         timestamp=float(i), frame_path=f"frames/{i}.jpg",
                                         content_type="diagram")
                            for i in range(1, count + 1)])
                db.commit()
            self.manager.start()

        def restart(self):
            assert self.manager.shutdown(5.0)
            self.manager = ExplanationPreparationManager(
                session_factory=sessions, orchestrator_factory=lambda: fake,
                data_dir=str(tmp_path),
            )
            managers.append(self.manager)
            self.manager.start()

        def state(self, name):
            with sessions() as db:
                return db.query(ExplanationPreparation).filter_by(intervention_id=name).one()

        def attempt(self, name):
            return self.manager.audio_streams.attempts.get((VIDEO_ID, name, MODE))

        def journal(self, name):
            attempt = self.attempt(name)
            with attempt.condition:
                return [json.loads(line) for line in attempt.path.read_bytes().splitlines()]

        def ready(self, count):
            eventually(lambda: all(self.state(f"ip{i}").status == "ready"
                                   for i in range(1, count + 1)))

    harness = Pipeline()
    harness.manager, harness.fake, harness.sessions, harness.root = manager, fake, sessions, tmp_path
    try:
        yield harness
    finally:
        for gate in fake.gates.values():
            gate.set()
        stopped = all([item.shutdown(5.0) for item in managers])
        engine.dispose()
        assert stopped, "Test left preparation workers running"


def test_held_tts_allows_next_text_and_third_vlm_to_run(pipeline):
    p = pipeline
    speech = p.fake.hold("tts", "ip1")
    evidence2 = p.fake.hold("vlm", "ip2")
    text2 = p.fake.hold("text", "ip2")
    evidence3 = p.fake.hold("vlm", "ip3")
    p.seed(3)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    evidence2.set()
    assert p.fake.started("text", "ip2").wait(5)
    assert p.fake.started("vlm", "ip3").wait(5)
    assert p.fake.active == Counter(vlm=1, text=1, tts=1)
    for gate in (speech, text2, evidence3):
        gate.set()
    p.ready(3)
    assert p.fake.calls == {stage: ["ip1", "ip2", "ip3"] for stage in ("vlm", "text", "tts")}
    assert p.fake.peak == Counter(vlm=1, text=1, tts=1)


def test_text_journal_precedes_available_tts_and_complete_follows_ready(pipeline, monkeypatch):
    p = pipeline
    completed_states = []
    complete = streaming.AudioAttempt.func.complete

    def observe_complete(attempt, **metadata):
        if metadata.get("intervention_id") == "ip2":
            state = p.state("ip2")
            with p.sessions() as db:
                row = db.query(Explanation).filter_by(intervention_id="ip2").one()
                completed_states.append((state.status, state.stage, row.id))
        return complete(attempt, **metadata)

    monkeypatch.setattr(streaming.AudioAttempt.func, "complete", observe_complete)
    speech = p.fake.hold("tts", "ip1")
    p.seed(2)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    eventually(lambda: p.state("ip2").stage == STAGE_TEXT_READY)
    assert p.state("ip2").status == "generating"
    events = p.journal("ip2")
    assert [event["type"] for event in events] == ["text_ready"]
    assert events[0]["text"] == p.fake.text_artifacts["ip2"]["text_explanation"]
    assert not p.fake.started("tts", "ip2").is_set()
    assert not (p.root / "ip2.mp3").exists()
    with p.sessions() as db:
        assert db.query(Explanation).count() == 0
    speech.set()
    p.ready(2)
    eventually(lambda: p.attempt("ip2").done)
    assert p.state("ip2").stage == STAGE_READY
    events = p.journal("ip2")
    assert [event["type"] for event in events] == ["text_ready", "complete"]
    with p.sessions() as db:
        row = db.query(Explanation).filter_by(intervention_id="ip2").one()
        assert events[-1]["explanation_id"] == row.id
        assert events[-1]["text_explanation"] == row.text_explanation
        assert cache_is_current(row.text_explanation, row.audio_file_path, row.vlm_snapshot_path)
        assert completed_states == [("ready", STAGE_READY, row.id)]


def test_concurrent_manual_and_explicit_request_join_inflight_speech(pipeline, monkeypatch):
    p = pipeline
    joining = threading.Event()
    wait_for_existing = p.manager._wait_for_existing

    def observe_join(job):
        joining.set()
        return wait_for_existing(job)

    monkeypatch.setattr(p.manager, "_wait_for_existing", observe_join)
    speech = p.fake.hold("tts", "ip1")
    p.seed(1)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    original = p.state("ip1")
    with ThreadPoolExecutor(max_workers=2) as executor:
        manual = executor.submit(p.manager.generate_manual, VIDEO_ID, "ip1", MODE)
        try:
            assert joining.wait(5)
            request = executor.submit(p.manager.request, VIDEO_ID, "ip1", MODE).result(5)
            eventually(lambda: p.state("ip1").priority_at is not None)
            assert not manual.done()
            assert request.status == "generating"
            assert not request.queued
            joined = p.state("ip1")
            assert (joined.lease_owner, joined.attempt_count) == (original.lease_owner, 1)
        finally:
            speech.set()
        artifact = manual.result(5)
    p.ready(1)
    with p.sessions() as db:
        row = db.query(Explanation).one()
        assert artifact.explanation_id == row.id
        assert db.query(ExplanationPreparation).count() == 1
    assert p.fake.calls == {stage: ["ip1"] for stage in ("vlm", "text", "tts")}


def test_cancelled_inflight_tts_cannot_publish_late_result(pipeline):
    p = pipeline
    speech = p.fake.hold("tts", "ip1")
    p.seed(1)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    assert p.manager.cancel_video(VIDEO_ID) == 1
    speech.set()
    eventually(lambda: p.attempt("ip1").done)
    eventually(lambda: not p.manager.has_inflight_video(VIDEO_ID))
    assert (p.root / "ip1.mp3").is_file(), "Fake must actually return a late audio artifact"
    assert not manifest_path(p.root / "ip1.mp3").exists()
    assert p.state("ip1").status == "cancelled"
    assert p.state("ip1").lease_owner is None
    with p.sessions() as db:
        assert db.query(Explanation).count() == 0
    events = p.journal("ip1")
    assert [event["type"] for event in events] == ["text_ready", "error"]
    assert events[-1]["code"] == "cancelled"


def test_full_speech_queue_backpressures_text_then_evidence(pipeline):
    p = pipeline
    capacity = EXPLANATION_QUEUE_CAPACITY
    speech = p.fake.hold("tts", "ip1")
    count = 2 * capacity + 5
    p.seed(count)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    # One active speech, C queued texts, one text waiting to hand off;
    # C queued evidence items, and one extraction waiting to hand off.
    eventually(lambda: len(p.fake.calls["text"]) == capacity + 2
               and len(p.fake.calls["vlm"]) == 2 * capacity + 3)
    assert p.manager._tts_queue.maxsize == capacity
    assert p.manager._tts_queue.qsize() == capacity
    assert p.manager._explanation_queue.maxsize == capacity
    assert p.manager._explanation_queue.qsize() == capacity
    assert not p.fake.started("text", f"ip{capacity + 3}").wait(0.2)
    assert not p.fake.started("vlm", f"ip{2 * capacity + 4}").is_set()
    speech.set()
    p.ready(count)
    expected = [f"ip{i}" for i in range(1, count + 1)]
    assert p.fake.calls == {stage: expected for stage in ("vlm", "text", "tts")}


def test_foreground_in_speech_does_not_block_background_text(pipeline):
    p = pipeline
    speech = p.fake.hold("tts", "ip1")
    evidence2 = p.fake.hold("vlm", "ip2")
    p.seed(2)
    # Admit ip1 as foreground before background scheduling can dispatch it.
    assert p.manager.request(VIDEO_ID, "ip1", MODE).queued
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    assert p.state("ip1").priority_at is not None
    evidence2.set()
    assert p.fake.started("text", "ip2").wait(5), (
        "Foreground ip1 has finished text; its speech must not reserve the text worker"
    )
    eventually(lambda: p.state("ip2").stage == STAGE_TEXT_READY)
    assert not speech.is_set()
    assert not p.fake.started("tts", "ip2").is_set()
    assert [event["type"] for event in p.journal("ip2")] == ["text_ready"]
    speech.set()
    p.ready(2)


def test_late_subscriber_replays_text_and_ordered_pcm_before_completion(pipeline, monkeypatch):
    p = pipeline
    generate = p.fake.generate_intervention_audio

    def streamed(text, **kwargs):
        attempt = streaming.current_stream.get()
        attempt.audio(text["text_explanation"], "PocketTTS", b"\0\0" * 24)
        attempt.audio(text["text_explanation"], "PocketTTS", b"\0\0" * 12)
        return generate(text, **kwargs)

    monkeypatch.setattr(p.fake, "generate_intervention_audio", streamed)
    p.seed(1)
    p.manager.schedule_video(VIDEO_ID, MODE)
    p.ready(1)
    eventually(lambda: p.attempt("ip1").done)
    events = [json.loads(line) for line in p.attempt("ip1").lines()]
    assert [event["type"] for event in events] == [
        "text_ready", "start", "audio", "audio", "complete",
    ]
    assert events[0]["generation_id"] == events[1]["generation_id"]
    assert events[0]["text"] == events[1]["text"] == events[-1]["text_explanation"]
    assert [(event["sequence"], event["sample_offset"]) for event in events
            if event["type"] == "audio"] == [(0, 0), (1, 24)]


def test_cancelling_queued_speech_terminates_journal_before_dequeue(pipeline):
    p = pipeline
    speech = p.fake.hold("tts", "ip1")
    p.seed(2)
    p.manager.schedule_video(VIDEO_ID, MODE)
    assert p.fake.started("tts", "ip1").wait(5)
    eventually(lambda: p.state("ip2").stage == STAGE_TEXT_READY
               and p.manager._tts_queue.qsize() == 1)
    attempt = p.attempt("ip2")
    assert not attempt.done
    assert p.manager.cancel_video(VIDEO_ID) == 2
    # Inspect disk directly: consuming lines() would itself notice lease loss
    # and hide a manager that never terminates queued journals on cancellation.
    assert attempt.done, "Cancellation must terminate queued speech before returning"
    assert not speech.is_set()
    assert p.fake.active["tts"] == 1
    assert not p.fake.started("tts", "ip2").is_set()
    events = p.journal("ip2")
    assert [event["type"] for event in events] == ["text_ready", "error"]
    assert events[-1]["code"] == "cancelled"
    assert p.state("ip2").status == "cancelled"
    speech.set()
    eventually(lambda: not p.manager.has_inflight_video(VIDEO_ID))
    assert not p.fake.started("tts", "ip2").is_set()
    assert not (p.root / "ip2.mp3").exists()
    assert p.journal("ip2") == events


@pytest.mark.parametrize("recover_stale_lease", [False, True], ids=["pending-retry", "stale-recovery"])
def test_interrupted_fallback_survives_retry_admission_and_restart(
    pipeline, monkeypatch, recover_stale_lease,
):
    p = pipeline
    p.fake.interrupt_speech.add("ip1")
    p.seed(1)
    p.manager.schedule_video(VIDEO_ID, MODE)
    eventually(lambda: p.state("ip1").status == "failed")
    eventually(lambda: not p.manager._scheduled)
    assert p.state("ip1").failure_reason == "tts_stream_interrupted"
    assert p.fake.fallback_preferences == [("ip1", False)]
    assert p.journal("ip1")[-1]["code"] == "tts_stream_interrupted"

    # Stop between durable retry admission and dispatch, as a process exit can.
    # Only dispatch is intercepted: the real retry transaction must preserve
    # everything the next manager needs, without its predecessor's stream map.
    with monkeypatch.context() as admission:
        admission.setattr(p.manager, "_enqueue", lambda job: True)
        result = p.manager.retry(VIDEO_ID, "ip1", MODE)
    assert result.queued
    assert p.state("ip1").status == "pending"
    assert p.state("ip1").attempt_count == 1
    assert not (p.root / "ip1.mp3").exists()
    assert p.manager.shutdown(5.0)

    if recover_stale_lease:
        # Model a retried process dying after claiming work but before speech.
        # Startup must recover its expired lease without losing fallback intent.
        with p.sessions() as db:
            row = db.query(ExplanationPreparation).one()
            row.status = "generating"
            row.stage = STAGE_TEXT_READY
            row.attempt_count = 2
            row.lease_owner = "dead-retry-process"
            row.lease_started_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
            db.commit()

    speech = p.fake.hold("tts", "ip1")
    old_manager = p.manager
    p.restart()
    assert p.manager is not old_manager
    eventually(lambda: p.fake.active["tts"] == 1)
    attempt = p.attempt("ip1")
    assert attempt is not None
    assert not attempt.done
    assert [event["type"] for event in p.journal("ip1")] == ["text_ready"]
    assert p.fake.fallback_preferences == [("ip1", False)]
    assert attempt.prefer_fallback, (
        "A fresh manager must restore interrupted-speech fallback before speech starts"
    )
    assert p.state("ip1").attempt_count == (3 if recover_stale_lease else 2)
    speech.set()
    p.ready(1)
    eventually(lambda: p.attempt("ip1").done)
    assert p.fake.fallback_preferences == [("ip1", False), ("ip1", True)]
    assert p.journal("ip1")[-1]["type"] == "complete"
    with p.sessions() as db:
        row = db.query(Explanation).one()
        assert cache_is_current(row.text_explanation, row.audio_file_path, row.vlm_snapshot_path)
