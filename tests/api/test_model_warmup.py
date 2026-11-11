"""Offline tests for non-blocking Modal readiness tracking."""

import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from api.model_warmup import ModalWarmup
from modal_inference.config import MODEL_ID


def response(body=""):
    context = mock.MagicMock()
    context.__enter__.return_value = SimpleNamespace(
        read=lambda: body.encode("utf-8"), status=200
    )
    return context


def models_response(model=MODEL_ID):
    return response(json.dumps({"data": [{"id": model}]}))


class ModalWarmupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock_path = Path(self.temp.name) / "warmup.lock"

    def _environment(self, **overrides):
        values = {
            "SYNAPSE_VLM_PROVIDER": "modal",
            "SYNAPSE_MODAL_URL": "https://example--synapse.modal.run/",
            "SYNAPSE_MODAL_API_KEY": "secret-key",
            "SYNAPSE_MODAL_WARMUP_TIMEOUT_SECONDS": "5",
            "SYNAPSE_MODAL_WARMUP_ATTEMPT_SECONDS": "1",
        }
        values.update(overrides)
        return mock.patch.dict("os.environ", values, clear=True)

    def warmup(self):
        return ModalWarmup(lock_path=self.lock_path)

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_warmup_succeeds_only_after_health_and_model_are_confirmed(self, urlopen):
        urlopen.side_effect = [response(), models_response()]
        with self._environment():
            warmup = self.warmup()
            self.assertEqual(warmup.status, "warming")
            self.assertTrue(warmup.ensure_ready())
        self.assertEqual(warmup.status, "ready")
        urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertEqual(urls, [
            "https://example--synapse.modal.run/health",
            "https://example--synapse.modal.run/v1/models",
        ])
        self.assertTrue(all(
            call.args[0].get_method() == "GET" for call in urlopen.call_args_list
        ))

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_background_warmup_logs_start_and_ready_state(self, urlopen):
        urlopen.side_effect = [response(), models_response()]
        with self._environment(), self.assertLogs("api.model_warmup", level="INFO") as logs:
            warmup = self.warmup()
            warmup.start()
            warmup._thread.join(timeout=1)

        self.assertEqual(warmup.status, "ready")
        output = "\n".join(logs.output)
        self.assertIn("Modal VLM warm-up started", output)
        self.assertIn("Modal VLM status: ready", output)
        self.assertNotIn("secret-key", output)

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_later_check_probes_again_because_modal_may_have_scaled_to_zero(self, urlopen):
        urlopen.side_effect = [response(), models_response(), response()]
        with self._environment():
            warmup = self.warmup()
            self.assertTrue(warmup.ensure_ready())
            self.assertTrue(warmup.ensure_ready())
        self.assertEqual(urlopen.call_count, 3)

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_timeout_then_success_retries_a_cold_start(self, urlopen):
        urlopen.side_effect = [TimeoutError(), response(), models_response()]
        with self._environment(), mock.patch.object(
            ModalWarmup, "_acquire_process_lock", return_value=False
        ):
            warmup = self.warmup()
            with mock.patch.object(warmup._stop, "wait", return_value=False):
                self.assertTrue(warmup.ensure_ready())
        self.assertEqual(warmup.status, "ready")

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_persistent_timeout_is_bounded_and_reports_unavailable(self, urlopen):
        urlopen.side_effect = TimeoutError()
        with self._environment():
            warmup = self.warmup()
            with mock.patch.object(warmup._stop, "wait", return_value=False):
                started = time.monotonic()
                self.assertFalse(warmup.ensure_ready(timeout=0.05))
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(warmup.status, "unavailable")
        self.assertIn("TimeoutError", warmup.detail)

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_rejected_credentials_stop_immediately_without_leaking_the_key(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "https://x", 401, "Unauthorized", {}, io.BytesIO(b"")
        )
        with self._environment():
            warmup = self.warmup()
            self.assertFalse(warmup.ensure_ready())
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(warmup.status, "unavailable")
        self.assertNotIn("secret-key", warmup.detail)

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_wrong_served_model_is_not_ready(self, urlopen):
        urlopen.side_effect = [response(), models_response("other/model")]
        with self._environment():
            warmup = self.warmup()
            self.assertFalse(warmup.ensure_ready())
        self.assertEqual(warmup.status, "unavailable")

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_url_alone_is_not_readiness_when_the_key_is_missing(self, urlopen):
        with self._environment(SYNAPSE_MODAL_API_KEY=""):
            warmup = self.warmup()
            self.assertFalse(warmup.ensure_ready())
        urlopen.assert_not_called()
        self.assertEqual(warmup.status, "unavailable")

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_opt_out_and_local_provider_do_not_contact_modal(self, urlopen):
        for override in (
            {"SYNAPSE_MODAL_WARMUP_ON_STARTUP": "false"},
            {"SYNAPSE_VLM_PROVIDER": "local"},
        ):
            with self._environment(**override):
                warmup = self.warmup()
                warmup.start()
                self.assertTrue(warmup.ensure_ready())
                self.assertEqual(warmup.status, "disabled")
        urlopen.assert_not_called()

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_start_is_nonblocking_and_only_one_thread_is_created(self, urlopen):
        urlopen.side_effect = lambda *args, **kwargs: time.sleep(0.2) or response()
        with self._environment():
            warmup = self.warmup()
            started = time.monotonic()
            warmup.start()
            thread = warmup._thread
            warmup.start()
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertIs(warmup._thread, thread)
            warmup.stop()

    def test_another_worker_holding_the_lock_is_waited_for_not_duplicated(self):
        self.lock_path.write_text("")
        with self._environment():
            warmup = self.warmup()
            released = []

            def wait_for_peer(_seconds):
                self.lock_path.unlink()
                released.append(True)
                return False

            with mock.patch.object(warmup._stop, "wait", side_effect=wait_for_peer):
                self.assertTrue(warmup._acquire_process_lock(time.monotonic() + 5))
        self.assertEqual(released, [True])

    def test_idle_stop_is_joined_idempotent_and_prevents_new_probe(self):
        with self._environment():
            warmup = self.warmup()
            self.assertTrue(warmup.stop(timeout=0.01))
            self.assertTrue(warmup.stop(timeout=0.01))
            warmup.start()
            self.assertIsNone(warmup._thread)
            self.assertFalse(warmup.ensure_ready(timeout=0.01))

    def test_job_readiness_does_not_wait_for_the_full_startup_probe(self):
        with self._environment(SYNAPSE_VLM_READINESS_ATTEMPT_SECONDS="0.03"):
            warmup = self.warmup()
            warmup._probe_lock.acquire()
            try:
                started = time.monotonic()
                self.assertFalse(warmup.ensure_ready_for_inference())
                self.assertLess(time.monotonic() - started, 0.2)
            finally:
                warmup._probe_lock.release()

    @mock.patch("api.model_warmup.urllib.request.urlopen")
    def test_blocked_probe_stop_is_bounded_and_eventually_cleans_lock(self, urlopen):
        entered = threading.Event()
        release = threading.Event()

        def blocked(*_args, **_kwargs):
            entered.set()
            release.wait(2)
            return response()

        urlopen.side_effect = blocked
        with self._environment():
            warmup = self.warmup()
            warmup.start()
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            self.assertFalse(warmup.stop(timeout=0.03))
            self.assertLess(time.monotonic() - started, 0.2)
            calls = urlopen.call_count
            warmup.start()
            self.assertEqual(urlopen.call_count, calls)
            release.set()
            self.assertTrue(warmup.stop(timeout=1))
        self.assertFalse(self.lock_path.exists())


if __name__ == "__main__":
    unittest.main()
