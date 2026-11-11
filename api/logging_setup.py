"""Central logging configuration for the Synapse backend.

The default formatter is deliberately human-oriented. Set
``SYNAPSE_LOG_FORMAT=json`` for machine-readable diagnostics; that mode
preserves the complete structured stage event and exception traceback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


_HANDLER_MARKER = "_synapse_console_handler"
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|password|secret|token|access[_-]?token|auth[_-]?token|bearer)$",
    re.I,
)
_CONTENT_KEY = re.compile(
    r"^(?:prompt|transcript|raw[_-]?response|response[_-]?body|image|image[_-]?data)$",
    re.I,
)
_BEARER = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_STAGE_TAGS = {
    "backend": "BACKEND", "video_submission": "VIDEO", "video_processing": "VIDEO",
    "video_cancellation": "CANCEL", "video_resume": "RESUME",
    "transcription": "ASR", "frame_extraction": "FRAMES",
    "intervention_detection": "DETECT", "intervention_persistence": "DB",
    "vlm_queue": "VLM", "modal_warmup": "VLM", "modal_request": "VLM",
    "modal_attempt": "VLM", "vlm_inference": "VLM", "vlm_json_parsing": "VLM",
    "snapshot_validation": "VLM", "snapshot_persistence": "VLM",
    "evidence_preparation": "VLM", "explanation_queue": "JOB",
    "preparation_state": "JOB", "explanation_reconciliation": "RECON",
    "ollama_startup": "OLLAMA", "ollama_model": "OLLAMA", "ollama_readiness": "OLLAMA",
    "mistral_generation": "LLM", "mistral_model": "LLM", "mistral_attempt": "LLM",
    "explanation_text_persistence": "LLM", "tts_creation": "TTS",
    "explanation_persistence": "DB", "frontend_ready": "READY",
}
_SYNAPSE_NAMESPACES = (
    "synapse", "api", "orchestrator", "vlm_interface", "vlm_engine",
    "synthesizer", "llm_interface", "tts_engine", "frame_extraction",
    "transcription", "synchronizer", "qa", "quiz", "voice",
)


def _redact(value: Any, key: str = "") -> Any:
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if key and _CONTENT_KEY.search(key):
        return "[OMITTED]"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _BEARER.sub(r"\1[REDACTED]", value)
    return value


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds")


def _tag_for(record: logging.LogRecord, event: Mapping[str, Any] | None) -> str:
    if event:
        return _STAGE_TAGS.get(str(event.get("stage", "")), "PIPELINE")
    return record.name.split(".", 1)[0].upper()[:12] or "SYNAPSE"


class SynapseShellFormatter(logging.Formatter):
    """Compact, single-line output for a developer terminal."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            event = getattr(record, "pipeline_event", None)
            stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            level = record.levelname[:5].ljust(5)
            tag = _tag_for(record, event)
            if isinstance(event, Mapping):
                action = str(event.get("stage_event", "event")).replace("_", " ")
                bits = [action]
                intervention = event.get("intervention_id")
                if intervention:
                    bits.append(f"intervention={str(intervention)[:8]}")
                elif event.get("video_id"):
                    bits.append(f"video={str(event['video_id'])[:11]}")
                attempt = event.get("request_attempt", event.get("attempt"))
                if attempt not in (None, 0, "0"):
                    bits.append(f"attempt={attempt}")
                status = event.get("status")
                if status and str(status).lower() != action.lower():
                    bits.append(f"status={status}")
                for count_key, count_label in (
                    ("intervention_count", "interventions"),
                    ("interaction_count", "interactions"),
                    ("frame_count", "frames"),
                ):
                    if event.get(count_key) is not None:
                        bits.append(f"{count_label}={event[count_key]}")
                duration = event.get("elapsed_ms")
                if isinstance(duration, (int, float)):
                    bits.append(f"{duration / 1000:.2f}s")
                message = " | ".join(bits)
            else:
                message = _BEARER.sub(r"\1[REDACTED]", record.getMessage())
            # Full exception text is intentionally diagnostic-mode only.
            return f"{stamp} {level} [{tag}] {message}"
        except Exception:
            return "00:00:00 ERROR [LOGGING] unable to format log record"


class SynapseJsonFormatter(logging.Formatter):
    """Stable JSON-lines output for diagnostics and ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            event = getattr(record, "pipeline_event", None)
            if isinstance(event, Mapping):
                payload = dict(_redact(event))
            else:
                payload = {"event": "application_log", "logger": record.name,
                           "message": _redact(record.getMessage())}
            payload.setdefault("timestamp", _timestamp(record))
            payload["severity"] = record.levelname
            if record.exc_info:
                payload["traceback"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        except Exception:
            return json.dumps({"event": "logging_error", "severity": "ERROR"})


class _SafeStreamHandler(logging.StreamHandler):
    """A logging sink must never be able to fail a pipeline job."""

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        return


class _SynapseOnlyFilter(logging.Filter):
    """Do not duplicate third-party logs already handled by their host."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "root" or record.name.startswith(_SYNAPSE_NAMESPACES)


def configure_logging() -> None:
    """Install or update the single Synapse console handler, idempotently."""

    root = logging.getLogger()
    raw_level = os.getenv("SYNAPSE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)
    output_format = os.getenv("SYNAPSE_LOG_FORMAT", "shell").strip().lower()
    formatter: logging.Formatter = SynapseJsonFormatter() if output_format == "json" else SynapseShellFormatter()

    handlers = [handler for handler in root.handlers if getattr(handler, _HANDLER_MARKER, False)]
    if handlers:
        handler = handlers[0]
        for duplicate in handlers[1:]:
            root.removeHandler(duplicate)
    else:
        handler = _SafeStreamHandler(sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(formatter)
    if not any(isinstance(item, _SynapseOnlyFilter) for item in handler.filters):
        handler.addFilter(_SynapseOnlyFilter())
    root.setLevel(min(root.level or level, level))
    for name in _SYNAPSE_NAMESPACES:
        logging.getLogger(name).setLevel(level)


def configure_ask_logging() -> None:
    """Backward-compatible entry point retained for existing imports."""

    configure_logging()
