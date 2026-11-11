import io
import queue

import pytest

from transcription.process_runtime import TranscriptionProcessRuntime
from worker_protocol import read_frame


class FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def runtime_with_response(response):
    runtime = TranscriptionProcessRuntime(call_timeout=0.1)
    runtime._process = FakeProcess()
    runtime._responses = queue.Queue()
    runtime._responses.put((True, response))
    return runtime


def test_transcription_request_uses_one_persistent_framed_worker():
    payload = {"video_id": "abcdefghijk", "entries": [], "error": None}
    runtime = runtime_with_response(("ok", payload))

    assert runtime.transcribe("https://youtu.be/abcdefghijk", "abcdefghijk") == payload
    runtime._process.stdin.seek(0)
    assert read_frame(runtime._process.stdin) == (
        "transcribe",
        {"input": "https://youtu.be/abcdefghijk", "video_id": "abcdefghijk"},
    )


def test_worker_error_does_not_expose_child_details():
    runtime = runtime_with_response(("error", "RuntimeError"))

    with pytest.raises(RuntimeError, match="RuntimeError"):
        runtime.transcribe("https://youtu.be/abcdefghijk", "abcdefghijk")


def test_stop_preempts_busy_worker():
    runtime = runtime_with_response(("ok", None))
    process = runtime._process
    runtime._call_lock.acquire()
    try:
        assert not runtime.stop(timeout=0)
    finally:
        runtime._call_lock.release()

    assert process.terminated
    assert runtime._process is None
