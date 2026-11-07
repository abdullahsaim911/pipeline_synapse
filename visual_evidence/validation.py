"""Non-throwing validation facade for canonical visual evidence."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import ValidationDiagnostic, VisualEvidence


class EvidenceValidationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structurally_valid: bool
    evidence: VisualEvidence | None = None
    diagnostics: list[ValidationDiagnostic]


def validate_visual_evidence(
    value: VisualEvidence | Mapping[str, Any],
) -> EvidenceValidationResult:
    """Validate without discarding actionable structural diagnostics.

    Valid-but-incomplete records remain structurally valid; their extraction
    and evidence status plus in-record diagnostics communicate incompleteness.
    """

    try:
        evidence = (
            value
            if isinstance(value, VisualEvidence)
            else VisualEvidence.model_validate(value)
        )
    except ValidationError as exc:
        diagnostics = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "scene"
            diagnostics.append(
                ValidationDiagnostic(
                    code=f"invalid.{error['type']}",
                    message=f"{location}: {error['msg']}",
                    target_id=location,
                )
            )
        return EvidenceValidationResult(
            structurally_valid=False,
            evidence=None,
            diagnostics=diagnostics,
        )

    return EvidenceValidationResult(
        structurally_valid=True,
        evidence=evidence,
        diagnostics=list(evidence.validation_diagnostics),
    )

