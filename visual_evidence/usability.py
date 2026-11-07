"""Decide whether a persisted snapshot can feed explanation generation.

The existence of ``vlm_snapshot.json`` proves nothing: failed extractions are
persisted deliberately. This is the single gate used to reuse a saved snapshot
and to start downstream work, and it mirrors the rejection rules in
``LLMSynthesizer.weave`` so an accepted snapshot is not rejected moments later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .assembler import adapt_legacy_snapshot
from .validation import validate_visual_evidence


@dataclass(frozen=True)
class SnapshotAssessment:
    usable: bool
    reason: Optional[str] = None


def _has_grounded_content(evidence: Mapping[str, Any]) -> bool:
    structure = evidence.get("structural_description")
    overview = structure.get("overview") if isinstance(structure, Mapping) else None
    return bool(evidence.get("regions") or evidence.get("exact_content") or overview)


def assess_snapshot(snapshot: Any) -> SnapshotAssessment:
    """Return whether *snapshot* carries real, valid, available evidence."""

    if not isinstance(snapshot, Mapping):
        return SnapshotAssessment(False, "snapshot_not_an_object")
    if snapshot.get("_error"):
        failure = snapshot.get("_failure")
        code = failure.get("code") if isinstance(failure, Mapping) else None
        return SnapshotAssessment(False, code or "extraction_failed")

    missing = snapshot.get("missing_elements")
    if missing and missing not in ("null", "None", ""):
        return SnapshotAssessment(False, "evidence_marked_missing")

    evidence = snapshot.get("visual_evidence")
    if not isinstance(evidence, Mapping):
        return SnapshotAssessment(False, "canonical_evidence_absent")
    validation = validate_visual_evidence(evidence)
    if not validation.structurally_valid:
        return SnapshotAssessment(False, "canonical_evidence_invalid")
    if evidence.get("extraction_status") in {"failed", "not_attempted"}:
        return SnapshotAssessment(False, "canonical_evidence_unavailable")
    if evidence.get("evidence_status") == "unavailable_or_unreadable":
        return SnapshotAssessment(False, "canonical_evidence_unavailable")
    if not _has_grounded_content(evidence):
        return SnapshotAssessment(False, "canonical_evidence_empty")
    return SnapshotAssessment(True)


def load_usable_snapshot(path: str | Path) -> Optional[dict[str, Any]]:
    """Load a saved snapshot, returning it only if it is usable.

    Pre-canonical snapshots are adapted in memory; the saved record is left
    exactly as it was written.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(snapshot, dict):
        return None
    if "visual_evidence" not in snapshot and not snapshot.get("_error"):
        snapshot = dict(snapshot)
        snapshot["visual_evidence"] = adapt_legacy_snapshot(snapshot).to_schema_dict()
    return snapshot if assess_snapshot(snapshot).usable else None
