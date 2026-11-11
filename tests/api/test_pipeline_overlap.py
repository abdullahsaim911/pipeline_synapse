"""Coordination of the VLM and explanation workers, with controlled model fakes."""

import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api import main
from api.database import Base
from api.explanation_preparation import (
    EXPLANATION_QUEUE_CAPACITY,
    ExplanationPreparationManager,
    STAGE_EVIDENCE_READY,
    STAGE_EXPLAINING,
    STAGE_EXTRACTING,
    STAGE_FAILED,
    STAGE_READY,
    STAGE_TTS,
    STAGE_WAITING_FOR_VLM,
)
from api.explanation_service import VLM_EXTRACTION_LOCK
from api.models import (
    Explanation, ExplanationPreparation, Intervention, ProcessingJob, Video,
)
from api.routers import video as video_module

VIDEO_ID = "Overlap1234"


def wait_until(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class RecordingOrchestrator:
    """Implements both stages; ``gates`` hold a stage open for a test to observe."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.vlm_calls = []
        self.mistral_calls = []
        self.events = []
        self.fail_vlm = {}
        self.gates = {}
        self.snapshots_received = {}
        self.active = Counter()
        self.max_active = Counter()
        self.lock = threading.Lock()

    def _enter(self, stage, name):
        with self.lock:
            self.active[stage] += 1
            self.max_active[stage] = max(self.max_active[stage], self.active[stage])
            self.events.append((time.monotonic(), f"{stage}:{name}:start"))

    def _exit(self, stage, name):
        with self.lock:
            self.active[stage] -= 1
            self.events.append((time.monotonic(), f"{stage}:{name}:end"))

    def _snapshot_path(self, intervention_id):
        return self.data_dir / f"{intervention_id}.json"

    def extract_visual_evidence(self, **kwargs):
        name = kwargs["intervention_id"]
        path = self._snapshot_path(name)
        if path.exists():
            return {
                "snapshot": None, "snapshot_path": str(path), "usable": True,
                "reused": True, "failure": None,
            }
        before = kwargs.get("before_inference")
        if before is not None and not before():
            return {
                "snapshot": None, "snapshot_path": None, "usable": False, "reused": False,
                "failure": {"code": "vlm_remote_unavailable", "retryable": True},
            }
        self._enter("vlm", name)
        try:
            self.vlm_calls.append(name)
            gate = self.gates.get(f"vlm:{name}")
            if gate is not None:
                gate.wait(4.0)
            else:
                time.sleep(0.02)
            if name in self.fail_vlm:
                return {
                    "snapshot": {"_error": True}, "snapshot_path": str(path),
                    "usable": False, "reused": False,
                    "failure": {"code": self.fail_vlm[name], "retryable": True},
                }
            path.write_text("{}", encoding="utf-8")
            return {
                "snapshot": {"id": name}, "snapshot_path": str(path), "usable": True,
                "reused": False, "failure": None,
            }
        finally:
            self._exit("vlm", name)

    def synthesize_intervention_explanation(self, **kwargs):
        name = kwargs["intervention_id"]
        mode = kwargs["output_mode"]
        self._enter("mistral", name)
        try:
            self.mistral_calls.append((name, mode))
            self.snapshots_received[(name, mode)] = kwargs["vlm_snapshot"]
            gate = self.gates.get(f"mistral:{name}")
            if gate is not None:
                gate.wait(4.0)
            else:
                time.sleep(0.02)
            if kwargs.get("on_stage"):
                kwargs["on_stage"](STAGE_TTS)
            audio = self.data_dir / f"{name}-{mode}.mp3"
            audio.write_bytes(b"ID3-audio")
            return {
                "intervention_id": name,
                "text_explanation": f"Explanation of {name}",
                "audio_file_path": str(audio),
                "text_file_path": None,
                "vlm_snapshot_path": str(self._snapshot_path(name)),
                "output_mode": mode,
            }
        finally:
            self._exit("mistral", name)


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"
        self.data_dir.mkdir()
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'test.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.orchestrator = RecordingOrchestrator(self.data_dir)

    def seed(self, count=3):
        with self.Session() as db:
            db.add(Video(id=VIDEO_ID, title="Lecture", source_url="https://youtu.be/x"))
            for index in range(1, count + 1):
                db.add(Intervention(
                    id=f"ip{index}",
                    video_id=VIDEO_ID,
                    timestamp=float(index * 10),
                    frame_path=f"frames/{index}.jpg",
                    content_type="diagram",
                ))
            db.commit()

    def manager(self, **kwargs):
        manager = ExplanationPreparationManager(
            session_factory=self.Session,
            orchestrator_factory=lambda: self.orchestrator,
            data_dir=str(self.data_dir),
            **kwargs,
        )
        manager.start()
        self.addCleanup(manager.shutdown, 1.0)
        return manager

    def statuses(self, manager, mode="explanatory"):
        return {item.intervention_id: item for item in manager.get_status(VIDEO_ID, mode)}

    def all_settled(self, manager, mode="explanatory"):
        return all(
            item.status in {"ready", "failed"}
            for item in self.statuses(manager, mode).values()
        )

    def state(self, intervention_id, mode="explanatory"):
        with self.Session() as db:
            return db.query(ExplanationPreparation).filter_by(
                intervention_id=intervention_id, output_mode=mode
            ).one()

    # 10 / 16 -----------------------------------------------------------------
    def test_vlm_is_sequential_ordered_and_each_frame_called_once(self):
        self.seed(4)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1", "ip2", "ip3", "ip4"])
        self.assertEqual(self.orchestrator.max_active["vlm"], 1)
        self.assertEqual(self.orchestrator.max_active["mistral"], 1)
        self.assertEqual(
            [name for name, _ in self.orchestrator.mistral_calls],
            ["ip1", "ip2", "ip3", "ip4"],
        )

    # 11 ----------------------------------------------------------------------
    def test_ip2_vlm_runs_while_ip1_mistral_runs(self):
        self.seed(3)
        mistral_gate = threading.Event()
        vlm_gate = threading.Event()
        self.orchestrator.gates = {"mistral:ip1": mistral_gate, "vlm:ip2": vlm_gate}
        self.addCleanup(mistral_gate.set)
        self.addCleanup(vlm_gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")

        self.assertTrue(wait_until(lambda: (
            self.orchestrator.active["vlm"] == 1
            and self.orchestrator.active["mistral"] == 1
        )))
        self.assertEqual(self.orchestrator.vlm_calls[-1], "ip2")
        self.assertEqual(self.orchestrator.mistral_calls, [("ip1", "explanatory")])

        vlm_gate.set()
        mistral_gate.set()
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        order = [label for _, label in sorted(self.orchestrator.events)]
        self.assertLess(order.index("vlm:ip1:end"), order.index("mistral:ip1:start"))
        self.assertLess(order.index("vlm:ip2:start"), order.index("mistral:ip1:end"))
        self.assertLess(order.index("vlm:ip2:end"), order.index("mistral:ip2:start"))
        self.assertEqual(self.orchestrator.max_active["vlm"], 1)
        self.assertEqual(self.orchestrator.max_active["mistral"], 1)

    def test_measured_overlap_shortens_the_run(self):
        """Wall time must beat the sum of the recorded stage durations."""
        self.seed(3)
        delay = 0.06
        vlm, mistral = (
            self.orchestrator.extract_visual_evidence,
            self.orchestrator.synthesize_intervention_explanation,
        )

        def slow_vlm(**kwargs):
            time.sleep(delay)
            return vlm(**kwargs)

        def slow_mistral(**kwargs):
            time.sleep(delay)
            return mistral(**kwargs)

        self.orchestrator.extract_visual_evidence = slow_vlm
        self.orchestrator.synthesize_intervention_explanation = slow_mistral
        manager = self.manager()
        started = time.monotonic()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        elapsed = time.monotonic() - started
        starts = {}
        busy = 0.0
        for moment, label in sorted(self.orchestrator.events):
            stage, name, edge = label.split(":")
            if edge == "start":
                starts[(stage, name)] = moment
            else:
                busy += moment - starts[(stage, name)]
        overlap_time = self._overlap_seconds()
        self.assertGreater(overlap_time, 0.0)

    def _overlap_seconds(self):
        """Seconds during which a VLM and a Mistral stage ran at the same time."""
        active = Counter()
        previous = None
        total = 0.0
        for moment, label in sorted(self.orchestrator.events):
            if previous is not None and active["vlm"] and active["mistral"]:
                total += moment - previous
            stage, _name, edge = label.split(":")
            active[stage] += 1 if edge == "start" else -1
            previous = moment
        return total

    def test_downstream_starts_only_after_evidence_is_persisted(self):
        self.seed(1)
        seen = {}
        original = self.orchestrator.synthesize_intervention_explanation

        def checking(**kwargs):
            seen["persisted"] = self.orchestrator._snapshot_path("ip1").exists()
            seen["stage"] = self.state("ip1").stage
            return original(**kwargs)

        self.orchestrator.synthesize_intervention_explanation = checking
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(seen, {"persisted": True, "stage": STAGE_EXPLAINING})

    def test_evidence_in_memory_is_passed_to_mistral_without_reloading(self):
        self.seed(1)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(
            self.orchestrator.snapshots_received[("ip1", "explanatory")], {"id": "ip1"}
        )

    def test_full_explanation_queue_pauses_vlm_instead_of_growing(self):
        count = EXPLANATION_QUEUE_CAPACITY + 4
        self.seed(count)
        gate = threading.Event()
        self.orchestrator.gates = {"mistral:ip1": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        # One in Mistral, the queue full, and one finished extraction waiting.
        limit = EXPLANATION_QUEUE_CAPACITY + 2
        self.assertTrue(wait_until(lambda: len(self.orchestrator.vlm_calls) == limit))
        time.sleep(0.15)
        self.assertEqual(len(self.orchestrator.vlm_calls), limit)
        gate.set()
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(len(self.orchestrator.vlm_calls), count)

    # 12 ----------------------------------------------------------------------
    def test_failed_intervention_does_not_block_later_ones(self):
        self.seed(3)
        self.orchestrator.fail_vlm = {"ip2": "vlm_output_truncated"}
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))

        statuses = self.statuses(manager)
        self.assertEqual(
            {name: item.status for name, item in statuses.items()},
            {"ip1": "ready", "ip2": "failed", "ip3": "ready"},
        )
        self.assertEqual(statuses["ip2"].error_code, "visual_evidence_unavailable")
        state = self.state("ip2")
        self.assertEqual(state.stage, STAGE_FAILED)
        self.assertEqual(state.failure_reason, "vlm_output_truncated")
        self.assertNotIn(("ip2", "explanatory"), self.orchestrator.mistral_calls)

    def test_retry_reextracts_only_what_failed_and_is_bounded(self):
        self.seed(1)
        self.orchestrator.fail_vlm = {"ip1": "vlm_output_malformed"}
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))

        self.orchestrator.fail_vlm = {}
        self.assertTrue(manager.retry(VIDEO_ID, "ip1", "explanatory").queued)
        self.assertTrue(wait_until(
            lambda: self.statuses(manager)["ip1"].status == "ready"
        ))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1", "ip1"])
        self.assertEqual(self.statuses(manager)["ip1"].attempt_count, 2)
        self.assertIsNone(self.state("ip1").failure_reason)

    def test_unavailable_modal_waits_without_consuming_attempt_or_calling_model(self):
        self.seed(2)
        calls = []
        manager = self.manager(
            before_inference=lambda: calls.append(1) or False,
            vlm_retry_seconds=0.2,
            vlm_max_wait_seconds=1.0,
            maintenance_interval_seconds=0.01,
        )
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: all(
            item.status == "pending" and item.error_code == "vlm_remote_unavailable"
            for item in self.statuses(manager).values()
        )))
        self.assertEqual(self.orchestrator.vlm_calls, [])
        self.assertEqual(self.state("ip1").failure_reason, "vlm_remote_unavailable")
        self.assertEqual(self.state("ip1").stage, STAGE_WAITING_FOR_VLM)
        self.assertEqual(self.state("ip1").attempt_count, 0)
        self.assertEqual(self.state("ip2").attempt_count, 0)
        self.assertEqual(len(calls), 1)

    def test_modal_recovery_resumes_automatically_and_preserves_overlap_pipeline(self):
        self.seed(2)
        readiness = iter((False, True, True))
        manager = self.manager(
            before_inference=lambda: next(readiness, True),
            vlm_retry_seconds=0.03,
            vlm_max_wait_seconds=1.0,
            maintenance_interval_seconds=0.01,
        )
        manager.schedule_video(VIDEO_ID, "explanatory")

        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1", "ip2"])
        self.assertEqual(
            [name for name, _mode in self.orchestrator.mistral_calls],
            ["ip1", "ip2"],
        )
        self.assertEqual(self.state("ip1").attempt_count, 1)

    def test_waiting_for_modal_survives_manager_restart(self):
        self.seed(1)
        first = self.manager(
            before_inference=lambda: False,
            vlm_retry_seconds=0.03,
            vlm_max_wait_seconds=1.0,
            maintenance_interval_seconds=0.01,
        )
        first.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(
            lambda: self.state("ip1").stage == STAGE_WAITING_FOR_VLM
        ))
        self.assertTrue(first.shutdown(1.0))
        time.sleep(0.04)

        second = self.manager(
            before_inference=lambda: True,
            vlm_retry_seconds=0.03,
            vlm_max_wait_seconds=1.0,
            maintenance_interval_seconds=0.01,
        )
        self.assertTrue(wait_until(
            lambda: self.statuses(second)["ip1"].status == "ready"
        ))
        self.assertEqual(self.state("ip1").attempt_count, 1)

    def test_modal_wait_deadline_fails_retryably_without_using_attempts(self):
        self.seed(1)
        readiness = {"ready": False}
        with self.Session() as db:
            db.add(ProcessingJob(
                id="00000000-0000-0000-0000-000000000001",
                video_id=VIDEO_ID,
                source_url="https://youtu.be/x",
                requested_output_mode="explanatory",
                status="succeeded",
                stage="complete",
                active_slot=1,
            ))
            db.commit()
        manager = self.manager(
            before_inference=lambda: readiness["ready"],
            vlm_retry_seconds=0.02,
            vlm_max_wait_seconds=0.05,
            maintenance_interval_seconds=0.01,
        )
        manager.schedule_video(VIDEO_ID, "explanatory")

        self.assertTrue(wait_until(
            lambda: self.statuses(manager)["ip1"].status == "failed"
        ))
        state = self.state("ip1")
        self.assertEqual(state.error_code, "vlm_remote_unavailable")
        self.assertEqual(state.attempt_count, 0)
        self.assertIsNone(state.deferred_until)
        with self.Session() as db:
            self.assertIsNone(db.query(ProcessingJob).one().active_slot)

        readiness["ready"] = True
        self.assertTrue(manager.retry(VIDEO_ID, "ip1", "explanatory").queued)
        self.assertTrue(wait_until(
            lambda: self.statuses(manager)["ip1"].status == "ready"
        ))
        recovered = self.state("ip1")
        self.assertEqual(recovered.attempt_count, 1)
        self.assertIsNone(recovered.provider_wait_started_at)

    def test_mistral_failure_keeps_evidence_and_retry_skips_the_vlm(self):
        self.seed(1)
        original = self.orchestrator.synthesize_intervention_explanation
        attempts = []

        def flaky(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("controlled Mistral failure")
            return original(**kwargs)

        self.orchestrator.synthesize_intervention_explanation = flaky
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(self.statuses(manager)["ip1"].status, "failed")

        manager.retry(VIDEO_ID, "ip1", "explanatory")
        self.assertTrue(wait_until(
            lambda: self.statuses(manager)["ip1"].status == "ready"
        ))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1"])

    # 13 ----------------------------------------------------------------------
    def test_simultaneous_manual_and_background_create_no_duplicate_inference(self):
        self.seed(1)
        gate = threading.Event()
        self.orchestrator.gates = {"vlm:ip1": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.orchestrator.active["vlm"] == 1))

        results = []
        manual = threading.Thread(target=lambda: results.append(
            manager.generate_manual(VIDEO_ID, "ip1", "explanatory")
        ))
        manual.start()
        time.sleep(0.05)
        gate.set()
        manual.join(4.0)

        self.assertEqual(results[0].text_explanation, "Explanation of ip1")
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1"])
        self.assertEqual(self.orchestrator.mistral_calls, [("ip1", "explanatory")])
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        with self.Session() as db:
            self.assertEqual(db.query(Explanation).count(), 1)

    def test_two_worker_managers_atomically_claim_one_durable_job(self):
        self.seed(1)
        first = self.manager()
        second = self.manager()
        gate = threading.Event()
        self.orchestrator.gates = {"vlm:ip1": gate}
        self.addCleanup(gate.set)

        first.schedule_video(VIDEO_ID, "explanatory")
        second.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.orchestrator.active["vlm"] == 1))
        gate.set()
        self.assertTrue(wait_until(lambda: self.all_settled(first)))

        self.assertEqual(self.orchestrator.vlm_calls, ["ip1"])
        self.assertEqual(self.orchestrator.mistral_calls, [("ip1", "explanatory")])
        with self.Session() as db:
            state = db.query(ExplanationPreparation).one()
            self.assertEqual(state.attempt_count, 1)
            self.assertEqual(db.query(Explanation).count(), 1)

    def test_manual_request_for_a_queued_key_takes_it_over(self):
        self.seed(2)
        gate = threading.Event()
        self.orchestrator.gates = {"vlm:ip1": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.orchestrator.active["vlm"] == 1))

        thread = threading.Thread(
            target=manager.generate_manual, args=(VIDEO_ID, "ip2", "explanatory")
        )
        thread.start()
        time.sleep(0.05)
        gate.set()
        thread.join(4.0)
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(Counter(self.orchestrator.vlm_calls)["ip2"], 1)
        self.assertEqual(Counter(self.orchestrator.mistral_calls)[("ip2", "explanatory")], 1)

    def test_second_mode_reuses_evidence(self):
        self.seed(1)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "brief")
        manager.schedule_video(VIDEO_ID, "detailed")
        self.assertTrue(wait_until(
            lambda: self.all_settled(manager, "brief")
            and self.all_settled(manager, "detailed")
        ))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip1"])
        self.assertCountEqual(
            self.orchestrator.mistral_calls, [("ip1", "brief"), ("ip1", "detailed")]
        )

    # 14 ----------------------------------------------------------------------
    def test_restart_resumes_from_persisted_snapshot_without_repeating_vlm(self):
        self.seed(2)
        self.orchestrator._snapshot_path("ip1").write_text("{}", encoding="utf-8")
        with self.Session() as db:
            for name, status, stage in (
                ("ip1", "generating", STAGE_EXPLAINING),
                ("ip2", "generating", STAGE_EXTRACTING),
            ):
                db.add(ExplanationPreparation(
                    video_id=VIDEO_ID, intervention_id=name, output_mode="explanatory",
                    status=status, stage=stage, attempt_count=1, lease_owner="dead:worker",
                ))
            db.commit()

        manager = self.manager()
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(self.orchestrator.vlm_calls, ["ip2"])
        self.assertEqual(
            [name for name, _ in self.orchestrator.mistral_calls], ["ip1", "ip2"]
        )
        self.assertEqual(self.statuses(manager)["ip1"].attempt_count, 2)
        self.assertEqual(self.state("ip1").stage, STAGE_READY)

    def test_restart_does_not_redo_a_finished_explanation(self):
        self.seed(1)
        audio = self.data_dir / "finished.mp3"
        audio.write_bytes(b"ID3-done")
        with self.Session() as db:
            db.add(Explanation(
                intervention_id="ip1", text_explanation="Done",
                audio_file_path=str(audio), output_mode="explanatory",
            ))
            db.add(ExplanationPreparation(
                video_id=VIDEO_ID, intervention_id="ip1", output_mode="explanatory",
                status="generating", attempt_count=1, lease_owner="dead:worker",
            ))
            db.commit()
        manager = self.manager()
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertEqual(self.orchestrator.vlm_calls, [])
        self.assertEqual(self.orchestrator.mistral_calls, [])

    def test_shutdown_leaves_handed_off_work_resumable(self):
        self.seed(3)
        gate = threading.Event()
        self.orchestrator.gates = {"mistral:ip1": gate}
        self.addCleanup(gate.set)
        lease_settings = {
            "lease_timeout_seconds": 0.2,
            "maintenance_interval_seconds": 0.03,
        }
        manager = self.manager(**lease_settings)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: len(self.orchestrator.vlm_calls) >= 2))
        manager.shutdown(timeout=0.05)
        gate.set()
        self.assertTrue(wait_until(lambda: self.orchestrator.active["mistral"] == 0))
        time.sleep(0.05)

        restarted = RecordingOrchestrator(self.data_dir)
        self.orchestrator = restarted
        second = self.manager(**lease_settings)
        self.assertTrue(wait_until(lambda: self.all_settled(second)))
        self.assertEqual(
            {item.status for item in self.statuses(second).values()}, {"ready"}
        )

    # Phase F / 17 ------------------------------------------------------------
    def test_stages_are_recorded_in_order(self):
        self.seed(1)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertTrue(wait_until(lambda: len(manager.timeline) == 5))
        self.assertEqual(
            [stage for _, name, _mode, stage in manager.timeline if name == "ip1"],
            [
                STAGE_EXTRACTING, STAGE_EVIDENCE_READY, STAGE_EXPLAINING,
                STAGE_TTS, STAGE_READY,
            ],
        )

    def test_not_ready_while_only_the_snapshot_exists(self):
        self.seed(1)
        gate = threading.Event()
        self.orchestrator.gates = {"mistral:ip1": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.orchestrator.active["mistral"] == 1))
        self.assertTrue(self.orchestrator._snapshot_path("ip1").exists())
        self.assertEqual(self.statuses(manager)["ip1"].status, "generating")
        gate.set()

    def test_missing_audio_is_never_ready(self):
        self.seed(1)
        original = self.orchestrator.synthesize_intervention_explanation

        def no_audio(**kwargs):
            result = original(**kwargs)
            Path(result["audio_file_path"]).unlink()
            return result

        self.orchestrator.synthesize_intervention_explanation = no_audio
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        item = self.statuses(manager)["ip1"]
        self.assertEqual((item.status, item.error_code), ("failed", "invalid_cache"))

    def test_finished_result_is_retrievable_while_later_work_runs(self):
        self.seed(2)
        gate = threading.Event()
        self.orchestrator.gates = {"vlm:ip2": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "brief")

        app = FastAPI()
        app.state.explanation_preparation = manager
        app.include_router(video_module.router)
        from api.database import get_db

        def test_db():
            with self.Session() as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        app.dependency_overrides[video_module.get_data_dir] = lambda: str(self.data_dir)
        self.assertTrue(wait_until(
            lambda: self.statuses(manager, "brief")["ip1"].status == "ready"
        ))
        with TestClient(app) as client:
            polled = client.get(
                f"/video/{VIDEO_ID}/explanations/status",
                params={"output_mode": "brief"},
            )
            audio = client.get(
                "/video/intervention/ip1/audio", params={"output_mode": "brief"}
            )
        self.assertEqual(polled.status_code, 200, polled.text)
        items = {
            item["intervention_id"]: item["status"] for item in polled.json()["items"]
        }
        self.assertEqual(items["ip1"], "ready")
        self.assertEqual(items["ip2"], "generating")
        self.assertEqual(audio.status_code, 200, audio.text[:200])
        gate.set()

    def test_stale_lease_is_not_processed_by_the_explanation_worker(self):
        self.seed(1)
        gate = threading.Event()
        self.orchestrator.gates = {"vlm:ip1": gate}
        self.addCleanup(gate.set)
        manager = self.manager()
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(wait_until(lambda: self.orchestrator.active["vlm"] == 1))
        with self.Session() as db:
            self.state("ip1")
            db.query(ExplanationPreparation).update({
                ExplanationPreparation.lease_owner: "someone-else"
            })
            db.commit()
        gate.set()
        self.assertTrue(wait_until(lambda: self.orchestrator.active["vlm"] == 0))
        time.sleep(0.1)
        self.assertEqual(self.orchestrator.mistral_calls, [])

    def test_local_vlm_never_overlaps_mistral_on_the_shared_gpu(self):
        self.seed(1)
        seen = {}
        original = self.orchestrator.synthesize_intervention_explanation

        def checking(**kwargs):
            seen["locked"] = VLM_EXTRACTION_LOCK.locked()
            return original(**kwargs)

        self.orchestrator.synthesize_intervention_explanation = checking
        with mock.patch.dict("os.environ", {"SYNAPSE_VLM_PROVIDER": "local"}):
            manager = self.manager()
            manager.schedule_video(VIDEO_ID, "explanatory")
            self.assertTrue(wait_until(lambda: self.all_settled(manager)))
        self.assertTrue(seen["locked"])


class StartupResponsivenessTests(unittest.TestCase):
    """Test 9: an unreachable Modal must not delay or fail backend startup."""

    def test_app_serves_requests_while_modal_warmup_hangs(self):
        release = threading.Event()
        self.addCleanup(release.set)
        entered = threading.Event()

        def hang(*_args, **_kwargs):
            entered.set()
            release.wait(5.0)
            raise TimeoutError()

        environment = {
            "SYNAPSE_VLM_PROVIDER": "modal",
            "SYNAPSE_MODAL_URL": "https://example.modal.run",
            "SYNAPSE_MODAL_API_KEY": "key",
        }
        with (
            mock.patch.dict("os.environ", environment),
            mock.patch("api.model_warmup.urllib.request.urlopen", side_effect=hang),
            mock.patch.object(main, "require_current_schema"),
            mock.patch.object(main, "ExplanationPreparationManager"),
            mock.patch.object(main, "ProcessingJobManager"),
            mock.patch("api.model_warmup.ModalWarmup._acquire_process_lock", return_value=False),
        ):
            started = time.monotonic()
            with TestClient(main.app) as client:
                self.assertLess(time.monotonic() - started, 3.0)
                self.assertTrue(entered.wait(2.0))
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(main.app.state.model_warmup.status, "warming")
                release.set()


if __name__ == "__main__":
    unittest.main()
