"""Persistent, restartable process for video transcription models."""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from typing import Any, Optional

from runtime_interpreters import resolve_python
from worker_protocol import read_frame, write_frame

logger = logging.getLogger(__name__)

DEFAULT_CALL_TIMEOUT_SECONDS = 3600.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _run_worker(receive, send) -> None:
    from transcription.transcriber import TranscriptionEngine

    engine: Optional[TranscriptionEngine] = None
    try:
        while True:
            command, payload = receive()
            try:
                if command == "warm":
                    engine = engine or TranscriptionEngine(whisper_model="base")
                    engine.whisper_runner.load_model()
                    result = {"ready": True}
                elif command == "transcribe":
                    engine = engine or TranscriptionEngine(whisper_model="base")
                    result = engine.transcribe(
                        payload["input"], video_id=payload.get("video_id")
                    ).to_dict()
                elif command == "shutdown":
                    send(("ok", None))
                    return
                else:
                    raise ValueError(f"Unknown transcription worker command: {command}")
                send(("ok", result))
            except Exception as exc:  # noqa: BLE001
                send(("error", type(exc).__name__))
    except (EOFError, BrokenPipeError):
        return


def _worker_main() -> None:
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    _run_worker(
        lambda: read_frame(sys.stdin.buffer),
        lambda value: write_frame(protocol_output, value),
    )


class TranscriptionProcessRuntime:
    """Own one warm transcription process and restart it after a crash."""

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

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._call_lock, self._state_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._discard_locked()
        python_executable = self._python_executable or resolve_python(
            "SYNAPSE_PIPELINE_PYTHON"
        )
        process = subprocess.Popen(
            [python_executable, "-m", "transcription.process_runtime", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        responses: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target=self._read_responses,
            args=(process, responses),
            name="synapse-transcription-ipc",
            daemon=True,
        )
        reader.start()
        self._process, self._responses, self._reader = process, responses, reader

    @staticmethod
    def _read_responses(process: subprocess.Popen, responses: queue.Queue) -> None:
        try:
            while True:
                responses.put((True, read_frame(process.stdout)))
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
        self._process = self._responses = self._reader = None

    def _terminate_locked(self) -> None:
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
                process, responses = self._process, self._responses
            assert process is not None and process.stdin is not None and responses is not None
            try:
                write_frame(process.stdin, (command, payload))
                valid, response = responses.get(timeout=self._call_timeout)
                if not valid:
                    raise response
            except (BrokenPipeError, EOFError, OSError, queue.Empty) as exc:
                exit_code = process.poll()
                with self._state_lock:
                    self._terminate_locked()
                logger.warning(
                    "Transcription worker failed during %s (exit=%s)", command, exit_code
                )
                raise RuntimeError("Transcription worker is unavailable") from exc
            if response[0] == "ok":
                return response[1]
            raise RuntimeError(f"Transcription operation failed: {response[1]}")

    def warm(self) -> None:
        self.call("warm")

    def transcribe(self, input_source: str, video_id: Optional[str] = None) -> dict:
        result = self.call("transcribe", {"input": input_source, "video_id": video_id})
        if result.get("error"):
            raise RuntimeError("Video transcription failed")
        return result

    def stop(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        acquired = self._call_lock.acquire(timeout=max(0.0, timeout))
        if not acquired:
            with self._state_lock:
                self._terminate_locked()
            return False
        try:
            with self._state_lock:
                process, responses = self._process, self._responses
                if process is None:
                    return True
                clean = True
                if process.poll() is None and process.stdin is not None and responses is not None:
                    try:
                        write_frame(process.stdin, ("shutdown", None))
                        valid, _ = responses.get(timeout=max(0.0, timeout))
                        clean = valid
                    except (BrokenPipeError, EOFError, OSError, queue.Empty):
                        clean = False
                try:
                    process.wait(timeout=max(0.0, timeout) if clean else 0)
                except subprocess.TimeoutExpired:
                    clean = False
                if process.poll() is None:
                    self._terminate_locked()
                else:
                    self._discard_locked()
                return clean
        finally:
            self._call_lock.release()


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main()
