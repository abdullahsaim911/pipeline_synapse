"""Explicit requests use the durable pipeline, priority dispatch and fencing."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.database import get_db
from api.routers import video as video_routes
from api.processing_jobs import serialize_job

from api.explanation_preparation import PreparationJob, _PreparationQueue
from api.models import ExplanationPreparation
from tests.api import test_explanation_preparation as helpers
from tests.api.test_explanation_status import wait_until
VIDEO_ID = helpers.VIDEO_ID


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.fixture = helpers.ExplanationPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.seed()

    def manager(self, generator):
        return self.fixture.manager(generator)

    def client(self, manager):
        app = FastAPI()
        app.include_router(video_routes.router)
        app.state.explanation_preparation = manager
        def database():
            with self.fixture.Session() as db:
                yield db
        app.dependency_overrides[get_db] = database
        return TestClient(app)

    def test_request_route_is_nonblocking_and_validates_the_target(self):
        started, release = threading.Event(), threading.Event()
        def generate(**kwargs):
            started.set()
            release.wait(3)
        manager = self.manager(generate)
        self.addCleanup(release.set)
        with self.client(manager) as client:
            route = f"/video/{VIDEO_ID}/interventions/point-10/explanations/request"
            response = client.post(route + "?output_mode=detailed")
            self.assertIn(response.status_code, {200, 202}, response.text)
            self.assertTrue(started.wait(2))
            self.assertEqual(client.post(route + "?output_mode=standard").status_code, 422)
            self.assertEqual(client.post(route.replace("point-10", "missing")).status_code, 404)
            self.assertEqual(client.post(route, headers={"Origin": "https://untrusted.example"}).status_code, 403)
        release.set()

    def test_cancel_route_uses_durable_processing_owner(self):
        manager = self.manager(lambda **kwargs: None)
        manager.request(VIDEO_ID, "point-10", "brief")
        with self.client(manager) as client:
            owner = Mock()
            client.app.state.processing_jobs = owner
            response = client.post(f"/video/{VIDEO_ID}/explanations/cancel")
            self.assertEqual(response.status_code, 200)
            owner.cancel.assert_called_once()

    def test_reopen_routes_cancelled_jobs_through_existing_resume(self):
        manager = self.manager(lambda **kwargs: None)
        manager.request(VIDEO_ID, "point-10", "brief")
        with self.fixture.Session() as db:
            from api.models import ProcessingJob
            job = db.query(ProcessingJob).one()
            job.status = "cancelled"
            db.commit()
            snapshot = {**serialize_job(job), "status": "queued", "stage": "queued"}
        with self.client(manager) as client:
            owner = Mock()
            owner.resume.return_value = snapshot
            client.app.state.processing_jobs = owner
            response = client.post(f"/video/{VIDEO_ID}/explanations/open")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "queued")
            owner.resume.assert_called_once()

    def test_initial_request_runs_once_and_cached_request_reuses_it(self):
        fake = helpers.FakeExplanationOrchestrator(self.fixture.data_dir)
        manager = self.fixture.manager(None, orchestrator_factory=lambda: fake)
        first = manager.request(VIDEO_ID, "point-10", "detailed")
        self.assertIn(first.status, {"pending", "generating", "ready"})
        self.assertTrue(wait_until(lambda: manager.get_status(VIDEO_ID, "detailed")[0].status == "ready"))
        self.assertEqual(manager.request(VIDEO_ID, "point-10", "detailed").status, "ready")
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(manager.get_status(VIDEO_ID, "brief")[0].requested)

    def test_concurrent_repeated_request_joins_running_generation(self):
        started, release = threading.Event(), threading.Event()
        fake = helpers.FakeExplanationOrchestrator(self.fixture.data_dir)
        synthesize = fake.synthesize_intervention_explanation
        def generate_persisted(**kwargs):
            started.set()
            release.wait(3)
            return synthesize(**kwargs)
        fake.synthesize_intervention_explanation = generate_persisted
        manager = self.fixture.manager(None, orchestrator_factory=lambda: fake)
        self.addCleanup(release.set)
        manager.request(VIDEO_ID, "point-10", "brief")
        self.assertTrue(started.wait(2))
        for _ in range(3):
            self.assertEqual(manager.request(VIDEO_ID, "point-10", "brief").status, "generating")
        release.set()
        self.assertTrue(wait_until(lambda: manager.get_status(VIDEO_ID, "brief")[0].status == "ready"))
        self.assertEqual(len(fake.calls), 1)

    def test_failure_does_not_implicitly_retry(self):
        calls = []
        def fail(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("controlled failure")
        manager = self.manager(fail)
        manager.request(VIDEO_ID, "point-10", "brief")
        self.assertTrue(wait_until(lambda: manager.get_status(VIDEO_ID, "brief")[0].status == "failed"))
        self.assertEqual(manager.request(VIDEO_ID, "point-10", "brief").status, "failed")
        self.assertEqual(len(calls), 1)
        manager.retry(VIDEO_ID, "point-10", "brief")
        self.assertTrue(wait_until(lambda: len(calls) == 2))

    def test_cancellation_fences_requested_result(self):
        started, release = threading.Event(), threading.Event()
        def generate(**kwargs):
            started.set()
            release.wait(3)
        manager = self.manager(generate)
        self.addCleanup(release.set)
        manager.request(VIDEO_ID, "point-10", "brief")
        self.assertTrue(started.wait(2))
        manager.cancel_video(VIDEO_ID)
        release.set()
        self.assertTrue(wait_until(lambda: manager.get_status(VIDEO_ID, "brief")[0].status == "cancelled"))
        with self.fixture.Session() as db:
            self.assertEqual(db.query(ExplanationPreparation).one().status, "cancelled")

    def test_request_is_next_after_running_work_even_with_full_handoff_queue(self):
        # Fill all handoff slots and block evidence on one more background item.
        with self.fixture.Session() as db:
            from api.models import Intervention
            for i in range(4, 9):
                db.add(Intervention(id=f"point-{i * 10}", video_id=VIDEO_ID, timestamp=i * 10,
                                    frame_path="frame.jpg", content_type="diagram"))
            db.commit()
        started, release = threading.Event(), threading.Event()
        calls = []
        fake = helpers.FakeExplanationOrchestrator(self.fixture.data_dir)
        synthesize = fake.synthesize_intervention_explanation
        def generate(**kwargs):
            calls.append((kwargs["intervention_id"], kwargs["output_mode"]))
            if len(calls) == 1:
                started.set()
                release.wait(4)
            return synthesize(**kwargs)
        fake.synthesize_intervention_explanation = generate
        manager = self.fixture.manager(None, orchestrator_factory=lambda: fake)
        self.addCleanup(release.set)
        manager.schedule_video(VIDEO_ID, "explanatory")
        self.assertTrue(started.wait(2))
        self.assertTrue(wait_until(lambda: manager._explanation_queue.full()))
        manager.request(VIDEO_ID, "point-80", "detailed")
        release.set()
        self.assertTrue(wait_until(lambda: len(calls) >= 2))
        self.assertEqual(calls[1], ("point-80", "detailed"))
        self.assertTrue(wait_until(lambda: all(item.status == "ready" for item in manager.get_status(VIDEO_ID, "explanatory")), timeout=5))


class PriorityQueueTests(unittest.TestCase):
    def test_both_stages_promote_existing_entries_without_duplicates(self):
        for handoff in (False, True):
            priorities = set()
            queue = _PreparationQueue(priorities, maxsize=3)
            jobs = [PreparationJob(VIDEO_ID, f"p{i}", "explanatory") for i in range(3)]
            entries = [SimpleNamespace(job=job) for job in jobs] if handoff else jobs
            for entry in entries:
                queue.put(entry)
            priorities.add(jobs[2].key)
            self.assertIs(queue.get(), entries[2])
            queue.task_done()
            self.assertIs(queue.get(), entries[0])
            queue.task_done()
            self.assertIs(queue.get(), entries[1])
            queue.task_done()
            queue.join()
            self.assertTrue(queue.empty())


if __name__ == "__main__":
    unittest.main()
