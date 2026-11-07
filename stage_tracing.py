"""Small, dependency-free structured tracing for the explanation pipeline.

The trace context is process-local, while durable job identifiers come from the
database.  Logs intentionally contain identifiers, timings and bounded status
metadata only; prompts, transcripts, images and credentials are never emitted.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Optional


logger = logging.getLogger("synapse.pipeline")
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|password|secret|token|access[_-]?token|auth[_-]?token|bearer)$",
    re.I,
)
_CONTENT_KEY = re.compile(
    r"^(?:prompt|transcript|raw[_-]?response|response[_-]?body|image|image[_-]?data)$",
    re.I,
)
_BEARER = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    video_id: Optional[str] = None
    intervention_id: Optional[str] = None
    output_mode: Optional[str] = None
    processing_job_id: Optional[str] = None
    preparation_job_id: Optional[str] = None
    attempt: int = 0


_context: ContextVar[Optional[TraceContext]] = ContextVar(
    "synapse_pipeline_trace", default=None
)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace() -> Optional[TraceContext]:
    return _context.get()


@contextmanager
def bind_trace(context: TraceContext) -> Iterator[TraceContext]:
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


@contextmanager
def extend_trace(**changes: Any) -> Iterator[Optional[TraceContext]]:
    current = current_trace()
    if current is None:
        yield None
        return
    with bind_trace(replace(current, **changes)) as updated:
        yield updated


def _bounded(value: Any, key: str = "") -> Any:
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if key and _CONTENT_KEY.search(key):
        return "[OMITTED]"
    if isinstance(value, str):
        value = _BEARER.sub(r"\1[REDACTED]", value)
        return value if len(value) <= 240 else value[:240] + "...[truncated]"
    if isinstance(value, Mapping):
        return {str(item_key): _bounded(item, str(item_key)) for item_key, item in list(value.items())[:24]}
    if isinstance(value, (list, tuple)):
        return [_bounded(item) for item in value[:24]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded(str(value))


def trace_stage(
    stage: str,
    event: str,
    *,
    started_at: Optional[float] = None,
    context: Optional[TraceContext] = None,
    level: int = logging.INFO,
    exc_info: Any = None,
    **metadata: Any,
) -> None:
    """Emit one structured stage event without ever affecting pipeline work."""

    try:
        trace = context or current_trace()
        payload: dict[str, Any] = {
            "event": "pipeline_stage",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "stage": _bounded(stage),
            "stage_event": _bounded(event),
            "monotonic_ms": round(time.monotonic() * 1000, 3),
        }
        if trace is not None:
            payload.update({key: _bounded(value, key) for key, value in asdict(trace).items() if value not in (None, "")})
        if started_at is not None:
            payload["elapsed_ms"] = round((time.monotonic() - started_at) * 1000, 3)
        payload.update({key: _bounded(value, key) for key, value in metadata.items() if value is not None})
        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        logger.log(level, serialized, extra={"pipeline_event": payload}, exc_info=exc_info)
    except Exception:
        # Observability is best-effort and must never change job behavior.
        return
