"""Owned process boundary for native Voice Control models on Windows."""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from typing import Any, Optional

from runtime_interpreters import resolve_python
from voice.catalog import ActionCatalog
from voice.schemas import InterpretRequest, InterpretResponse, TranscribeResponse
from voice.stt import AudioError
from worker_protocol import read_frame as _read_frame, write_frame as _write_frame

logger = logging.getLogger(__name__)

DEFAULT_CALL_TIMEOUT_SECONDS = 180.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _as_dict(value: Any) -> dict:
    dump = getattr(value, "model_dump", None)
    return dump() if dump is not None else value.dict()


def _run_worker(receive, send) -> None:
    """Load and use native models on this child process's main thread."""
    from voice.interpret import Interpreter
    from voice.stt import Recognizer

    interpreter: Optional[Interpreter] = None
    recognizer: Optional[Recognizer] = None
    try:
        while True:
            command, payload = receive()
            try:
                if command == "warm_interpreter":
                    interpreter = interpreter or Interpreter()
                    interpreter.warm()
                    result = None
                elif command == "warm_recognizer":
                    recognizer = recognizer or Recognizer()
                    recognizer.warm()
                    result = None
                elif command == "transcribe":
                    recognizer = recognizer or Recognizer()
                    result = _as_dict(recognizer.transcribe(payload["data"], payload.get("language")))
                elif command == "interpret":
                    interpreter = interpreter or Interpreter()
                    result = _as_dict(interpreter.interpret(InterpretRequest(**payload)))
                elif command == "close_interpreter":
                    if interpreter is not None:
                        interpreter.close()
                        interpreter = None
                    result = None
                elif command == "close_recognizer":
                    if recognizer is not None:
                        recognizer.close()
                        recognizer = None
                    result = None
                elif command == "shutdown":
                    for model in (recognizer, interpreter):
                        if model is not None:
                            model.close()
                    send(("ok", None))
                    return
                else:
                    raise ValueError(f"Unknown voice worker command: {command}")
                send(("ok", result))
            except Exception as exc:  # noqa: BLE001
                send(("error", type(exc).__name__, str(exc)))
    except (EOFError, BrokenPipeError):
        return


def _stdio_worker_main() -> None:
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    _run_worker(lambda: _read_frame(sys.stdin.buffer), lambda value: _write_frame(protocol_output, value))


class VoiceProcessRuntime:
    """One restartable, bounded child process shared by STT and interpretation."""

    def __init__(
        self,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        python_executable: Optional[str] = None,
    ) -> None:
        self._call_timeout = call_timeout
        self._python_executable = python_executable
        self._call_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._responses: Optional[queue.Queue] = None
        self._reader: Optional[threading.Thread] = None

    def start(self) -> None:
        """Create the worker from the application main thread before warm-up."""
        with self._call_lock, self._state_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._discard_locked()
        python_executable = self._python_executable or resolve_python("SYNAPSE_VOICE_PYTHON")
        process = subprocess.Popen(
            [python_executable, "-m", "voice.process_runtime", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        responses: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target=self._read_responses,
            args=(process, responses),
            name="synapse-voice-ipc",
            daemon=True,
        )
        reader.start()
        self._process = process
        self._responses = responses
        self._reader = reader

    @staticmethod
    def _read_responses(process: subprocess.Popen, responses: queue.Queue) -> None:
        try:
            while True:
                responses.put((True, _read_frame(process.stdout)))
        except (EOFError, OSError) as exc:
            responses.put((False, exc))

    def _discard_locked(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=0.2)
        if self._process is not None:
            try:
                self._process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self._responses = None
        self._reader = None

    def _stop_failed_worker_locked(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
        self._discard_locked()

    def call(self, command: str, payload: Any = None) -> Any:
        with self._call_lock:
            with self._state_lock:
                self._start_locked()
                responses = self._responses
                process = self._process
            assert responses is not None and process is not None and process.stdin is not None
            try:
                _write_frame(process.stdin, (command, payload))
                valid, response = responses.get(timeout=self._call_timeout)
                if not valid:
                    raise response
            except (BrokenPipeError, EOFError, OSError, queue.Empty) as exc:
                exit_code = process.poll()
                with self._state_lock:
                    self._stop_failed_worker_locked()
                logger.warning("Voice model worker failed during %s (exit=%s)", command, exit_code)
                raise RuntimeError("Voice model worker is unavailable") from exc
            if response[0] == "ok":
                return response[1]
            _, error_type, message = response
            if error_type == "AudioError":
                raise AudioError(message)
            raise RuntimeError(f"Voice model operation failed: {error_type}")

    def stop(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        acquired = self._call_lock.acquire(timeout=max(0.0, timeout))
        if not acquired:
            with self._state_lock:
                self._stop_failed_worker_locked()
            return False
        try:
            with self._state_lock:
                return self._stop_locked(timeout)
        finally:
            self._call_lock.release()

    def _stop_locked(self, timeout: float) -> bool:
        process = self._process
        if process is None:
            return True
        clean = True
        if process.poll() is None and process.stdin is not None and self._responses is not None:
            try:
                _write_frame(process.stdin, ("shutdown", None))
                valid, _ = self._responses.get(timeout=max(0.0, timeout))
                clean = valid
            except (BrokenPipeError, EOFError, OSError, queue.Empty):
                clean = False
        try:
            process.wait(timeout=0 if not clean else max(0.0, timeout))
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            clean = False
            process.terminate()
            try:
                process.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
        self._discard_locked()
        return clean


class ProcessRecognizer:
    def __init__(self, runtime: VoiceProcessRuntime) -> None:
        self._runtime = runtime

    def warm(self) -> None:
        self._runtime.call("warm_recognizer")

    def transcribe(self, data: bytes, language: Optional[str] = None) -> TranscribeResponse:
        return TranscribeResponse(**self._runtime.call("transcribe", {"data": data, "language": language}))

    def close(self) -> None:
        self._runtime.call("close_recognizer")


class ProcessInterpreter:
    def __init__(self, runtime: VoiceProcessRuntime) -> None:
        self._runtime = runtime
        self.catalog = ActionCatalog.load()

    def warm(self) -> None:
        self._runtime.call("warm_interpreter")

    def interpret(self, request: InterpretRequest) -> InterpretResponse:
        return InterpretResponse(**self._runtime.call("interpret", _as_dict(request)))

    def close(self) -> None:
        self._runtime.call("close_interpreter")


if __name__ == "__main__" and "--worker" in sys.argv:
    _stdio_worker_main()
