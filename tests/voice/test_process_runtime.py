import io
import queue

import pytest

from voice.process_runtime import VoiceProcessRuntime, _read_frame


class FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

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
        self.killed = True
        self.returncode = -9


def runtime_with_response(response):
    runtime = VoiceProcessRuntime(call_timeout=0.1)
    runtime._process = FakeProcess()
    runtime._responses = queue.Queue()
    runtime._responses.put((True, response))
    return runtime


def test_runtime_sends_one_framed_command_and_returns_response():
    runtime = runtime_with_response(("ok", {"ready": True}))

    assert runtime.call("warm_interpreter") == {"ready": True}
    runtime._process.stdin.seek(0)
    assert _read_frame(runtime._process.stdin) == ("warm_interpreter", None)


def test_runtime_does_not_expose_worker_error_details():
    runtime = runtime_with_response(("error", "RuntimeError", "secret child detail"))

    with pytest.raises(RuntimeError, match="RuntimeError") as exc_info:
        runtime.call("warm_interpreter")

    assert "secret child detail" not in str(exc_info.value)


def test_stop_preempts_an_in_flight_call():
    runtime = runtime_with_response(("ok", None))
    process = runtime._process
    runtime._call_lock.acquire()
    try:
        assert not runtime.stop(timeout=0)
    finally:
        runtime._call_lock.release()

    assert process.terminated
    assert runtime._process is None
