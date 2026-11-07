import io
import json
import logging
import time

from api.logging_setup import (
    SynapseJsonFormatter,
    SynapseShellFormatter,
    configure_logging,
)
from stage_tracing import TraceContext, logger as pipeline_logger, trace_stage


def _record(*, level=logging.INFO, event=None, exc_info=None):
    record = logging.LogRecord("synapse.pipeline", level, __file__, 1, "raw-json-placeholder", (), exc_info)
    if event is not None:
        record.pipeline_event = event
    return record


def test_shell_stage_event_is_concise_and_human_readable():
    line = SynapseShellFormatter().format(_record(event={
        "stage": "vlm_inference", "stage_event": "completed",
        "intervention_id": "intervention-123", "request_attempt": 2,
        "status": "usable", "elapsed_ms": 1250,
    }))

    assert "[VLM]" in line
    assert "intervention=interve" in line
    assert "attempt=2" in line
    assert "status=usable" in line
    assert "1.25s" in line
    assert not line.lstrip().startswith("{")


def test_video_persistence_event_includes_short_video_and_count():
    line = SynapseShellFormatter().format(_record(event={
        "stage": "intervention_persistence", "stage_event": "committed",
        "video_id": "ABCDEFGHIJK", "intervention_count": 4, "elapsed_ms": 50,
    }))

    assert "[DB]" in line
    assert "video=ABCDEFGHIJK" in line
    assert "interventions=4" in line


def test_shell_failure_is_one_line_without_traceback():
    try:
        raise RuntimeError("model unavailable")
    except RuntimeError:
        import sys
        line = SynapseShellFormatter().format(_record(level=logging.ERROR, exc_info=sys.exc_info()))

    assert line.count("\n") == 0
    assert "Traceback" not in line


def test_json_diagnostic_has_stable_fields_and_traceback():
    try:
        raise RuntimeError("model unavailable")
    except RuntimeError:
        import sys
        line = SynapseJsonFormatter().format(_record(
            level=logging.ERROR,
            event={"event": "pipeline_stage", "stage": "mistral_attempt",
                   "stage_event": "failed", "trace_id": "trace-1",
                   "preparation_job_id": "job-1", "request_attempt": 3},
            exc_info=sys.exc_info(),
        ))

    payload = json.loads(line)
    assert payload["severity"] == "ERROR"
    assert payload["trace_id"] == "trace-1"
    assert payload["request_attempt"] == 3
    assert "RuntimeError: model unavailable" in payload["traceback"]
    assert payload["timestamp"]


def test_json_redacts_credentials():
    line = SynapseJsonFormatter().format(_record(event={
        "stage": "modal_attempt", "stage_event": "failed",
        "api_key": "never-log-me", "detail": "Bearer secret-value",
    }))

    assert "never-log-me" not in line
    assert "secret-value" not in line
    assert line.count("[REDACTED]") == 2


def test_raw_model_content_is_omitted_but_token_counts_remain(caplog):
    with caplog.at_level(logging.INFO, logger="synapse.pipeline"):
        trace_stage(
            "modal_attempt", "response", raw_response="full model output",
            prompt="private prompt", transcript="private transcript",
            prompt_tokens=12, completion_tokens=34,
        )

    event = caplog.records[-1].pipeline_event
    assert event["raw_response"] == "[OMITTED]"
    assert event["prompt"] == "[OMITTED]"
    assert event["transcript"] == "[OMITTED]"
    assert event["prompt_tokens"] == 12
    assert event["completion_tokens"] == 34


def test_configuration_is_idempotent(monkeypatch):
    monkeypatch.setenv("SYNAPSE_LOG_FORMAT", "shell")
    root = logging.getLogger()
    configure_logging()
    configure_logging()

    owned = [handler for handler in root.handlers if getattr(handler, "_synapse_console_handler", False)]
    assert len(owned) == 1


def test_central_handler_ignores_third_party_logs(monkeypatch):
    monkeypatch.setenv("SYNAPSE_LOG_FORMAT", "shell")
    configure_logging()
    root = logging.getLogger()
    owned = next(handler for handler in root.handlers if getattr(handler, "_synapse_console_handler", False))

    assert bool(owned.filter(_record())) is True
    foreign = logging.LogRecord("httpx", logging.INFO, __file__, 1, "request", (), None)
    assert bool(owned.filter(foreign)) is False


def test_trace_event_preserves_correlated_ids_and_attempt(caplog):
    context = TraceContext(
        trace_id="trace-overlap", video_id="video-1", intervention_id="ip-1",
        processing_job_id="process-1", preparation_job_id="prepare-1", attempt=2,
    )
    with caplog.at_level(logging.INFO, logger="synapse.pipeline"):
        trace_stage("modal_attempt", "response", context=context, request_attempt=3,
                    prompt_tokens=10, completion_tokens=20, status="usable")

    event = caplog.records[-1].pipeline_event
    assert event["trace_id"] == "trace-overlap"
    assert event["processing_job_id"] == "process-1"
    assert event["preparation_job_id"] == "prepare-1"
    assert event["request_attempt"] == 3
    assert event["prompt_tokens"] == 10
    assert event["completion_tokens"] == 20


def test_trace_stage_swallows_logging_sink_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("closed stream")

    monkeypatch.setattr(pipeline_logger, "log", fail)
    trace_stage("vlm_inference", "completed", context=TraceContext(trace_id="trace-safe"))


def test_elapsed_time_is_recorded(caplog):
    with caplog.at_level(logging.INFO, logger="synapse.pipeline"):
        trace_stage("tts_creation", "completed", started_at=time.monotonic() - 0.01)

    assert caplog.records[-1].pipeline_event["elapsed_ms"] >= 1
