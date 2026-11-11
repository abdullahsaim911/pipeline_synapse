"""Deterministic assembly of backend metadata and frame observations."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from category_normalization import normalize_category_labels

from .models import (
    SCHEMA_VERSION,
    BoundingBox,
    CanonicalCategory,
    CategoryDetails,
    Element,
    ElementAttributes,
    EvidenceStatus,
    ExactContent,
    ExtractionStatus,
    Notation,
    Provenance,
    Region,
    RegionAttributes,
    Relation,
    CORE_RELATION_TYPES,
    StructuralDescription,
    Uncertainty,
    ValidationDiagnostic,
    VisualEvidence,
)


COMPACT_OBSERVATION_SCHEMA = "synapse_visual_observation_v1"


def normalize_compact_relations(payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Canonicalize model-authored relation vocabulary without losing meaning."""
    normalized = deepcopy(dict(payload))
    relations = normalized.get("relations")
    if not isinstance(relations, list):
        return normalized, 0
    changed = 0
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        value = str(relation.get("type") or "").strip().lower()
        if value in CORE_RELATION_TYPES or value.startswith("custom:"):
            continue
        semantic = re.sub(r"[^a-z0-9_:-]+", "_", value).strip("_:")
        relation["type"] = f"custom:{semantic or 'unspecified'}"
        changed += 1
    return normalized, changed


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ElementObservation(ObservationModel):
    """One directly observed element before backend IDs are assigned."""

    ref: str | None = None
    kind: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    category_ids: list[CanonicalCategory] = Field(default_factory=list)
    label: str | None = None
    exact_text: str | None = None
    description: str | None = None
    position: str | None = None
    bbox_normalized: BoundingBox | None = None
    source_ref: str | None = None
    attributes: ElementAttributes = Field(default_factory=ElementAttributes)
    parent_ref: str | None = None
    child_refs: list[str] = Field(default_factory=list)

    @field_validator("bbox_normalized", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return BoundingBox(
                x_min=value[0], y_min=value[1], x_max=value[2], y_max=value[3]
            )
        return value


class RegionObservation(ObservationModel):
    """One observed visual region; ``ref`` is local to the observation."""

    ref: str | None = None
    category_ids: list[CanonicalCategory]
    elements: list[ElementObservation] = Field(default_factory=list)
    position: str | None = None
    bbox_normalized: BoundingBox | None = None
    reading_order_index: int | None = Field(default=None, ge=0)
    attributes: RegionAttributes = Field(default_factory=RegionAttributes)
    evidence_status: EvidenceStatus | None = None

    @field_validator("bbox_normalized", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return BoundingBox(
                x_min=value[0], y_min=value[1], x_max=value[2], y_max=value[3]
            )
        return value


class RelationObservation(ObservationModel):
    """A grounded relation whose endpoints use observation-local refs."""

    ref: str | None = None
    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    label: str | None = None
    directed: bool = False
    cross_category: bool = False
    source_refs: list[str] = Field(default_factory=list)


class UncertaintyObservation(ObservationModel):
    target_ref: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    impact: str | None = None


class StructuralObservation(ObservationModel):
    overview: str | None = None
    orientation: str | None = None
    component_refs: list[str] = Field(default_factory=list)
    relation_refs: list[str] = Field(default_factory=list)
    reading_order_refs: list[str] = Field(default_factory=list)
    hierarchy_notes: list[str] = Field(default_factory=list)
    topology_notes: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class FrameObservation(ObservationModel):
    """Model-independent observations from exactly one frame."""

    global_layout: str | None = None
    regions: list[RegionObservation] = Field(default_factory=list)
    relations: list[RelationObservation] = Field(default_factory=list)
    uncertainties: list[UncertaintyObservation] = Field(default_factory=list)
    structural_description: StructuralObservation = Field(
        default_factory=StructuralObservation
    )
    category_details: CategoryDetails | None = None
    exact_content: list[ExactContent] = Field(default_factory=list)
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)


class EvidenceMetadata(ObservationModel):
    """Backend-owned metadata; no visual facts are synthesized here."""

    frame_id: str | None = None
    timestamp_seconds: float | None = Field(default=None, ge=0)
    category_ids: list[CanonicalCategory] = Field(default_factory=list)
    raw_category_labels: list[str] = Field(default_factory=list)
    extraction_status: ExtractionStatus
    evidence_status: EvidenceStatus
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def validate_status_pair(self) -> "EvidenceMetadata":
        if self.extraction_status in {"failed", "not_attempted"}:
            if self.evidence_status != "unavailable_or_unreadable":
                raise ValueError(
                    "failed/not_attempted extraction requires unavailable evidence"
                )
        return self


def _register_ref(refs: dict[str, str], local_ref: str | None, canonical_id: str) -> None:
    if local_ref is None:
        return
    if local_ref in refs:
        raise ValueError(f"duplicate observation ref '{local_ref}'")
    refs[local_ref] = canonical_id


def _resolve_ref(refs: Mapping[str, str], local_ref: str, field: str) -> str:
    if local_ref == VisualEvidence.SCENE_TARGET:
        return local_ref
    try:
        return refs[local_ref]
    except KeyError as exc:
        raise ValueError(f"{field} references unknown observation ref '{local_ref}'") from exc


def assemble_visual_evidence(
    metadata: EvidenceMetadata | Mapping[str, Any],
    observation: FrameObservation | Mapping[str, Any] | None = None,
    *,
    legacy_snapshot: Mapping[str, Any] | None = None,
) -> VisualEvidence:
    """Assemble one canonical record with deterministic local IDs.

    IDs depend only on observation order.  Facts, coordinates, relations, and
    category assignments are copied from supplied evidence; none are inferred
    merely to fill the schema.
    """

    metadata = (
        metadata
        if isinstance(metadata, EvidenceMetadata)
        else EvidenceMetadata.model_validate(metadata)
    )
    observation = (
        FrameObservation()
        if observation is None
        else observation
        if isinstance(observation, FrameObservation)
        else FrameObservation.model_validate(observation)
    )

    normalization = normalize_category_labels(metadata.raw_category_labels)
    categories = list(metadata.category_ids)
    for category in normalization.canonical_categories:
        if category not in categories:
            categories.append(category)

    raw_labels = list(normalization.raw_labels)
    diagnostics = list(observation.diagnostics)
    for note in normalization.notes:
        diagnostics.append(
            ValidationDiagnostic(
                code="category.unresolved_raw_label",
                message=note,
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )

    refs: dict[str, str] = {}
    regions: list[Region] = []
    pending_hierarchy: list[tuple[Element, str | None, list[str]]] = []
    element_number = 0

    for region_number, source_region in enumerate(observation.regions, start=1):
        region_id = f"region-{region_number:03d}"
        _register_ref(refs, source_region.ref, region_id)
        elements: list[Element] = []
        for source_element in source_region.elements:
            element_number += 1
            element_id = f"element-{element_number:03d}"
            _register_ref(refs, source_element.ref, element_id)
            element = Element(
                id=element_id,
                kind=source_element.kind,
                evidence_status=source_element.evidence_status,
                category_ids=source_element.category_ids,
                label=source_element.label,
                exact_text=source_element.exact_text,
                description=source_element.description,
                position=source_element.position,
                bbox_normalized=source_element.bbox_normalized,
                source_ref=source_element.source_ref,
                attributes=source_element.attributes,
            )
            elements.append(element)
            pending_hierarchy.append(
                (element, source_element.parent_ref, source_element.child_refs)
            )
        regions.append(
            Region(
                id=region_id,
                category_ids=source_region.category_ids,
                elements=elements,
                position=source_region.position,
                bbox_normalized=source_region.bbox_normalized,
                reading_order_index=source_region.reading_order_index,
                attributes=source_region.attributes,
                evidence_status=source_region.evidence_status,
            )
        )

    for element, parent_ref, child_refs in pending_hierarchy:
        if parent_ref is not None:
            element.parent_id = _resolve_ref(refs, parent_ref, "parent_ref")
        element.child_ids = [
            _resolve_ref(refs, child_ref, "child_refs") for child_ref in child_refs
        ]

    relations: list[Relation] = []
    relation_refs: dict[str, str] = {}
    for relation_number, source_relation in enumerate(observation.relations, start=1):
        relation_id = f"relation-{relation_number:03d}"
        _register_ref(relation_refs, source_relation.ref, relation_id)
        relations.append(
            Relation(
                id=relation_id,
                source_id=_resolve_ref(refs, source_relation.source_ref, "source_ref"),
                target_id=_resolve_ref(refs, source_relation.target_ref, "target_ref"),
                relation=source_relation.relation,
                evidence_status=source_relation.evidence_status,
                label=source_relation.label,
                directed=source_relation.directed,
                cross_category=source_relation.cross_category,
                source_refs=source_relation.source_refs,
            )
        )

    structural = observation.structural_description
    structural_description = StructuralDescription(
        overview=structural.overview,
        orientation=structural.orientation,
        component_ids=[
            _resolve_ref(refs, ref, "structural component_refs")
            for ref in structural.component_refs
        ],
        relation_ids=[
            _resolve_ref(relation_refs, ref, "structural relation_refs")
            for ref in structural.relation_refs
        ],
        reading_order=[
            _resolve_ref(refs, ref, "structural reading_order_refs")
            for ref in structural.reading_order_refs
        ],
        hierarchy_notes=structural.hierarchy_notes,
        topology_notes=structural.topology_notes,
        uncertainties=structural.uncertainties,
    )

    uncertainties = [
        Uncertainty(
            target_id=_resolve_ref(refs, item.target_ref, "uncertainty target_ref"),
            reason=item.reason,
            impact=item.impact,
        )
        for item in observation.uncertainties
    ]
    exact_content = [
        item.model_copy(
            update={"source_ref": refs.get(item.source_ref, item.source_ref)}
        )
        if item.source_ref
        else item
        for item in observation.exact_content
    ]

    return VisualEvidence(
        schema_version=SCHEMA_VERSION,
        frame_id=metadata.frame_id,
        timestamp_seconds=metadata.timestamp_seconds,
        extraction_status=metadata.extraction_status,
        evidence_status=metadata.evidence_status,
        category_ids=categories,
        raw_category_labels=raw_labels,
        global_layout=observation.global_layout,
        reading_order=list(structural_description.reading_order),
        regions=regions,
        relations=relations,
        uncertainties=uncertainties,
        provenance=metadata.provenance,
        structural_description=structural_description,
        category_details=observation.category_details,
        exact_content=exact_content,
        validation_diagnostics=diagnostics,
        legacy_snapshot=deepcopy(dict(legacy_snapshot)) if legacy_snapshot is not None else None,
    )


def assemble_compact_qwen_observation(
    payload: Mapping[str, Any],
    *,
    frame_id: str | None = None,
    timestamp_seconds: float | None = None,
    category_ids: list[str] | None = None,
    raw_category_labels: list[str] | None = None,
    provenance: Mapping[str, Any] | Provenance | None = None,
    possible_truncation: bool = False,
) -> VisualEvidence:
    """Map the compact Task 2B Qwen contract into canonical v1.1 evidence.

    Model-local ``ref`` values are used only to resolve relationships.  The
    canonical assembler replaces them with deterministic backend IDs.
    """

    if payload.get("schema") != COMPACT_OBSERVATION_SCHEMA:
        raise ValueError(
            f"structured observation schema must be '{COMPACT_OBSERVATION_SCHEMA}'"
        )

    detected_categories = payload.get("categories", [])
    if not isinstance(detected_categories, list):
        raise ValueError("categories must be an array")
    combined_categories = list(category_ids or [])
    for category in detected_categories:
        if category not in combined_categories:
            combined_categories.append(category)

    diagnostics: list[ValidationDiagnostic] = []
    missing = payload.get("missing", [])
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ValueError("missing must be an array of strings")
    for item in missing:
        diagnostics.append(
            ValidationDiagnostic(
                code="extraction.missing_evidence",
                message=item,
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )
    if possible_truncation:
        diagnostics.append(
            ValidationDiagnostic(
                code="extraction.possible_truncation",
                message=(
                    "Generation metadata indicates that the completion may be "
                    "truncated or otherwise incomplete."
                ),
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )

    reported_extraction = payload.get("status", "partial")
    if reported_extraction not in {"complete", "partial", "failed", "not_attempted"}:
        raise ValueError(f"unsupported extraction status '{reported_extraction}'")
    extraction_status = reported_extraction
    if possible_truncation or missing:
        if extraction_status == "complete":
            extraction_status = "partial"

    evidence_status = payload.get("evidence_status", "partially_supported")
    allowed_evidence_statuses = {
        "supported",
        "partially_supported",
        "ambiguous_or_conflicting",
        "unavailable_or_unreadable",
    }
    if evidence_status not in allowed_evidence_statuses:
        raise ValueError(f"unsupported evidence status '{evidence_status}'")
    if extraction_status in {"failed", "not_attempted"}:
        evidence_status = "unavailable_or_unreadable"
    elif extraction_status == "partial" and evidence_status == "supported":
        evidence_status = "partially_supported"

    payload, _normalization_count = normalize_compact_relations(payload)
    regions: list[RegionObservation] = []
    source_regions = payload.get("regions", [])
    if not isinstance(source_regions, list):
        raise ValueError("regions must be an array")
    for region in source_regions:
        if not isinstance(region, Mapping):
            raise ValueError("each region must be an object")
        elements = []
        source_elements = region.get("elements", [])
        if not isinstance(source_elements, list):
            raise ValueError("region elements must be an array")
        for element in source_elements:
            if not isinstance(element, Mapping):
                raise ValueError("each element must be an object")
            elements.append(
                ElementObservation(
                    ref=element.get("ref"),
                    kind=element.get("kind"),
                    evidence_status=element.get("evidence_status"),
                    category_ids=element.get("categories", []),
                    label=element.get("label"),
                    exact_text=element.get("text"),
                    description=element.get("description"),
                    position=element.get("position"),
                    bbox_normalized=element.get("bbox"),
                    source_ref=element.get("source_ref"),
                    attributes=element.get("attributes", {}),
                    parent_ref=element.get("parent_ref"),
                    child_refs=element.get("child_refs", []),
                )
            )
        regions.append(
            RegionObservation(
                ref=region.get("ref"),
                category_ids=region.get("categories", []),
                elements=elements,
                position=region.get("position"),
                bbox_normalized=region.get("bbox"),
                reading_order_index=region.get("reading_order_index"),
                attributes={
                    "structural_description": region.get("description"),
                    **(region.get("attributes", {}) if isinstance(region.get("attributes", {}), Mapping) else {}),
                },
                evidence_status=region.get("evidence_status"),
            )
        )

    relations: list[RelationObservation] = []
    source_relations = payload.get("relations", [])
    if not isinstance(source_relations, list):
        raise ValueError("relations must be an array")
    for relation in source_relations:
        if not isinstance(relation, Mapping):
            raise ValueError("each relation must be an object")
        relations.append(
            RelationObservation(
                ref=relation.get("ref"),
                source_ref=relation.get("source"),
                target_ref=relation.get("target"),
                relation=relation.get("type"),
                evidence_status=relation.get("evidence_status"),
                label=relation.get("label"),
                directed=relation.get("directed", False),
                cross_category=relation.get("cross_category", False),
                source_refs=relation.get("source_refs", []),
            )
        )

    structure = payload.get("structure", {})
    if not isinstance(structure, Mapping):
        raise ValueError("structure must be an object")
    structural_observation = StructuralObservation(
        overview=structure.get("overview"),
        orientation=structure.get("orientation"),
        component_refs=structure.get("components", []),
        relation_refs=structure.get("relations", []),
        reading_order_refs=structure.get("reading_order", []),
        hierarchy_notes=structure.get("hierarchy", []),
        topology_notes=structure.get("topology", []),
        uncertainties=structure.get("uncertainties", []),
    )

    uncertainties: list[UncertaintyObservation] = []
    source_uncertainties = payload.get("uncertainties", [])
    if not isinstance(source_uncertainties, list):
        raise ValueError("uncertainties must be an array")
    for uncertainty in source_uncertainties:
        if not isinstance(uncertainty, Mapping):
            raise ValueError("each uncertainty must be an object")
        uncertainties.append(
            UncertaintyObservation(
                target_ref=uncertainty.get("target", VisualEvidence.SCENE_TARGET),
                reason=uncertainty.get("reason"),
                impact=uncertainty.get("impact"),
            )
        )

    exact_content = payload.get("exact", [])
    if not isinstance(exact_content, list):
        raise ValueError("exact must be an array")
    declared_refs = {VisualEvidence.SCENE_TARGET}
    for region in source_regions:
        if isinstance(region, Mapping) and region.get("ref"):
            declared_refs.add(region["ref"])
        for element in region.get("elements", []):
            if isinstance(element, Mapping) and element.get("ref"):
                declared_refs.add(element["ref"])
    for item in exact_content:
        if not isinstance(item, Mapping):
            raise ValueError("each exact item must be an object")
        source_ref = item.get("source_ref")
        if source_ref is not None and source_ref not in declared_refs:
            raise ValueError(
                f"exact source_ref references unknown observation ref '{source_ref}'"
            )

    details = payload.get("details")
    if extraction_status == "complete" and (
        not regions or not structural_observation.overview
    ):
        extraction_status = "partial"
        if evidence_status == "supported":
            evidence_status = "partially_supported"
        diagnostics.append(
            ValidationDiagnostic(
                code="extraction.incomplete_structure",
                message=(
                    "A complete observation requires at least one grounded region "
                    "and a structural overview."
                ),
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )
    provenance_value = (
        provenance
        if isinstance(provenance, Provenance)
        else Provenance.model_validate(provenance or {})
    )
    provenance_value.observation_format = COMPACT_OBSERVATION_SCHEMA

    return assemble_visual_evidence(
        EvidenceMetadata(
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            category_ids=combined_categories,
            raw_category_labels=raw_category_labels or [],
            extraction_status=extraction_status,
            evidence_status=evidence_status,
            provenance=provenance_value,
        ),
        FrameObservation(
            global_layout=payload.get("layout"),
            regions=regions,
            relations=relations,
            uncertainties=uncertainties,
            structural_description=structural_observation,
            category_details=details,
            exact_content=exact_content,
            diagnostics=diagnostics,
        ),
        legacy_snapshot=payload,
    )


def adapt_legacy_snapshot(
    snapshot: Any,
    *,
    frame_id: str | None = None,
    timestamp_seconds: float | None = None,
    category_ids: list[str] | None = None,
    raw_category_labels: list[str] | None = None,
    provenance: Mapping[str, Any] | Provenance | None = None,
) -> VisualEvidence:
    """Losslessly wrap a current SnapshotEngine result as canonical evidence.

    Legacy prose and nested ``visual_analysis`` are retained unchanged.  They
    are deliberately not converted into precise regions, coordinates, or
    connections because the old response does not ground those structures.
    """

    diagnostics: list[ValidationDiagnostic] = []
    if not isinstance(snapshot, Mapping):
        diagnostics.append(
            ValidationDiagnostic(
                code="legacy.invalid_snapshot_type",
                message="Legacy snapshot is not an object and could not be structured.",
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )
        return assemble_visual_evidence(
            EvidenceMetadata(
                frame_id=frame_id,
                timestamp_seconds=timestamp_seconds,
                category_ids=category_ids or [],
                raw_category_labels=raw_category_labels or [],
                extraction_status=ExtractionStatus.FAILED,
                evidence_status=EvidenceStatus.UNAVAILABLE_OR_UNREADABLE,
                provenance=Provenance.model_validate(provenance or {}),
            ),
            FrameObservation(diagnostics=diagnostics),
        )

    preserved = deepcopy(dict(snapshot))
    labels = list(raw_category_labels or [])
    content_type = snapshot.get("content_type")
    if isinstance(content_type, str):
        labels.append(content_type)
    detected = snapshot.get("detected_types")
    if isinstance(detected, list):
        labels.extend(item for item in detected if isinstance(item, str))
    labels = list(dict.fromkeys(labels))

    missing = snapshot.get("missing_elements")
    failed = bool(snapshot.get("_error")) or missing == "Parse error"
    if failed:
        extraction_status = ExtractionStatus.FAILED
        evidence_status = EvidenceStatus.UNAVAILABLE_OR_UNREADABLE
        diagnostics.append(
            ValidationDiagnostic(
                code="legacy.extraction_failed",
                message=str(snapshot.get("_error_detail") or missing or "VLM extraction failed."),
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )
    elif missing:
        extraction_status = ExtractionStatus.PARTIAL
        evidence_status = EvidenceStatus.PARTIALLY_SUPPORTED
        diagnostics.append(
            ValidationDiagnostic(
                code="legacy.incomplete",
                message=f"Legacy snapshot reports missing evidence: {missing}",
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )
    else:
        extraction_status = ExtractionStatus.COMPLETE
        evidence_status = EvidenceStatus.SUPPORTED

    legacy_reading_order = snapshot.get("reading_order")
    if isinstance(legacy_reading_order, list) and legacy_reading_order:
        diagnostics.append(
            ValidationDiagnostic(
                code="legacy.unstructured_reading_order",
                message=(
                    "Legacy reading-order labels were preserved in legacy_snapshot but "
                    "not promoted to canonical ID references."
                ),
                target_id=VisualEvidence.SCENE_TARGET,
            )
        )

    exact_content: list[ExactContent] = []
    text_readout = snapshot.get("text_readout")
    if not failed and isinstance(text_readout, str) and text_readout:
        normalized = normalize_category_labels(labels)
        notation = Notation.PLAIN
        if "CT-02" in normalized.canonical_categories:
            notation = Notation.MATH
        elif "CT-08" in normalized.canonical_categories:
            notation = Notation.CODE
        exact_content.append(
            ExactContent(
                text=text_readout,
                status=evidence_status,
                notation=notation,
                source_ref="legacy_snapshot.text_readout",
            )
        )

    structural_description = snapshot.get("structural_description")
    overview = structural_description if isinstance(structural_description, str) else None
    layout = snapshot.get("layout")
    global_layout = layout if isinstance(layout, str) and layout else None

    provenance_value = (
        provenance
        if isinstance(provenance, Provenance)
        else Provenance.model_validate(provenance or {})
    )
    provenance_value.observation_format = provenance_value.observation_format or "legacy_snapshot"

    return assemble_visual_evidence(
        EvidenceMetadata(
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            category_ids=category_ids or [],
            raw_category_labels=labels,
            extraction_status=extraction_status,
            evidence_status=evidence_status,
            provenance=provenance_value,
        ),
        FrameObservation(
            global_layout=global_layout,
            structural_description=StructuralObservation(overview=overview),
            exact_content=exact_content,
            diagnostics=diagnostics,
        ),
        legacy_snapshot=preserved,
    )

