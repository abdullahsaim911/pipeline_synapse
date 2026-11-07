"""Typed canonical single-frame visual evidence models.

These models implement ``visual_evidence_schema_v1_1.json`` plus the
cross-record checks that JSON Schema cannot express.  They describe evidence,
not a required Qwen response format and not learner-facing narration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)


SCHEMA_VERSION = "1.1.0"


class CanonicalCategory(str, Enum):
    CHART = "CT-01"
    MATHEMATICS = "CT-02"
    SCIENTIFIC_STRUCTURE = "CT-03"
    FLOW_NETWORK = "CT-04"
    TABLE = "CT-05"
    STRUCTURED_TEXT = "CT-06"
    GEOMETRY = "CT-07"
    PROGRAMMING = "CT-08"
    INTERFACE = "CT-09"
    MAP = "CT-10"
    OBSERVATIONAL_IMAGE = "CT-11"
    PHYSICAL_SCENE = "CT-12"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    AMBIGUOUS_OR_CONFLICTING = "ambiguous_or_conflicting"
    UNAVAILABLE_OR_UNREADABLE = "unavailable_or_unreadable"


class ExtractionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class Notation(str, Enum):
    PLAIN = "plain"
    MATH = "math"
    CODE = "code"
    CHEMICAL = "chemical"
    OTHER = "other"


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class BoundingBox(BaseModel):
    """Normalized ``[x_min, y_min, x_max, y_max]`` coordinates."""

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.x_min >= self.x_max:
            raise ValueError("bbox x_min must be less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("bbox y_min must be less than y_max")
        return self

    def as_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


def _coerce_bbox(value: Any) -> Any:
    if value is None or isinstance(value, BoundingBox):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return BoundingBox(
            x_min=value[0], y_min=value[1], x_max=value[2], y_max=value[3]
        )
    return value


class ElementAttributes(BaseModel):
    """Observed element properties; extensions preserve category semantics."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    unit: str | None = None
    value: str | float | int | None = None
    symbol: str | None = None
    state: str | None = None
    style: str | None = None
    indentation_level: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)


class RegionAttributes(BaseModel):
    """Schema-compatible home for local region structure and extensions."""

    model_config = ConfigDict(extra="allow")

    structural_description: str | None = None
    hierarchy_notes: list[str] = Field(default_factory=list)
    topology_notes: list[str] = Field(default_factory=list)


class Element(EvidenceModel):
    # The authoritative JSON Schema intentionally permits extra element fields.
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    label: str | None = None
    exact_text: str | None = None
    description: str | None = None
    position: str | None = None
    bbox_normalized: BoundingBox | None = None
    source_ref: str | None = None
    attributes: ElementAttributes = Field(default_factory=ElementAttributes)
    category_ids: list[CanonicalCategory] = Field(default_factory=list)
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)

    _bbox = field_validator("bbox_normalized", mode="before")(_coerce_bbox)

    @field_serializer("bbox_normalized")
    def serialize_bbox(self, value: BoundingBox | None) -> list[float] | None:
        return value.as_list() if value is not None else None

    @field_validator("category_ids", "child_ids")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value


class Region(EvidenceModel):
    id: str = Field(min_length=1)
    category_ids: list[CanonicalCategory]
    elements: list[Element]
    position: str | None = None
    bbox_normalized: BoundingBox | None = None
    reading_order_index: int | None = Field(default=None, ge=0)
    attributes: RegionAttributes = Field(default_factory=RegionAttributes)
    evidence_status: EvidenceStatus | None = None

    _bbox = field_validator("bbox_normalized", mode="before")(_coerce_bbox)

    @field_serializer("bbox_normalized")
    def serialize_bbox(self, value: BoundingBox | None) -> list[float] | None:
        return value.as_list() if value is not None else None

    @field_validator("category_ids")
    @classmethod
    def unique_categories(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("region category_ids must be unique")
        return value


CORE_RELATION_TYPES: frozenset[str] = frozenset(
    {
        "left_of",
        "right_of",
        "above",
        "below",
        "contains",
        "inside",
        "grouped_with",
        "connected_to",
        "points_to",
        "leads_to",
        "branches_to",
        "depends_on",
        "corresponds_to",
        "parent_of",
        "child_of",
        "precedes",
        "follows",
        "intersects",
        "parallel_to",
        "perpendicular_to",
        "bonded_to",
        "labels",
    }
)


class Relation(EvidenceModel):
    id: str | None = Field(default=None, min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    label: str | None = None
    directed: bool = False
    cross_category: bool = False
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("relation")
    @classmethod
    def supported_relation(cls, value: str) -> str:
        value = value.strip()
        if value in CORE_RELATION_TYPES:
            return value
        if value.startswith("custom:") and len(value) > len("custom:"):
            return value
        raise ValueError(
            "unsupported relation type; use a core type or 'custom:<semantic>' "
            "to preserve a category-specific relation"
        )


class Uncertainty(EvidenceModel):
    target_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    impact: str | None = None


class StructuralDescription(EvidenceModel):
    overview: str | None
    orientation: str | None
    component_ids: list[str]
    relation_ids: list[str]
    reading_order: list[str]
    hierarchy_notes: list[str] = Field(default_factory=list)
    topology_notes: list[str] = Field(default_factory=list)
    uncertainties: list[str]

    @field_validator("component_ids", "relation_ids", "reading_order")
    @classmethod
    def unique_references(cls, value: list[str], info: ValidationInfo) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must contain unique references")
        return value


class ExactContent(EvidenceModel):
    text: str
    status: EvidenceStatus
    notation: Notation | None = None
    source_ref: str | None = None


class ValidationDiagnostic(EvidenceModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    target_id: str | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str | None = None
    model: str | None = None
    frame_path: str | None = None
    observation_format: str | None = None
    classification_confidence: float | None = None
    confidence_semantics: str | None = None


class _CategoryDetail(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChartDetails(_CategoryDetail):
    axes: list[dict[str, Any]] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)
    legends: list[dict[str, Any]] = Field(default_factory=list)
    visible_values: list[str] = Field(default_factory=list)
    observed_trends: list[str] = Field(default_factory=list)


class MathematicsDetails(_CategoryDetail):
    exact_expressions: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)


class ScientificStructureDetails(_CategoryDetail):
    components: list[str] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)
    containment: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)


class FlowNetworkDetails(_CategoryDetail):
    nodes: list[str] = Field(default_factory=list)
    branch_conditions: list[str] = Field(default_factory=list)
    loops: list[str] = Field(default_factory=list)


class TableDetails(_CategoryDetail):
    headers: list[str] = Field(default_factory=list)
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    cells: list[dict[str, Any]] = Field(default_factory=list)


class StructuredTextDetails(_CategoryDetail):
    headings: list[str] = Field(default_factory=list)
    lists: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    emphasized_text: list[str] = Field(default_factory=list)


class GeometryDetails(_CategoryDetail):
    points: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    angles: list[str] = Field(default_factory=list)
    shapes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class ProgrammingDetails(_CategoryDetail):
    language: str | None = None
    exact_code: str | None = None
    displayed_output: str | None = None
    indentation_significant: bool | None = None


class InterfaceDetails(_CategoryDetail):
    regions: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    visible_states: list[str] = Field(default_factory=list)
    observed_outcomes: list[str] = Field(default_factory=list)


class MapDetails(_CategoryDetail):
    locations: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    legend_entries: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)


class ObservationalImageDetails(_CategoryDetail):
    visible_features: list[str] = Field(default_factory=list)
    observed_conditions: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)


class PhysicalSceneDetails(_CategoryDetail):
    objects: list[str] = Field(default_factory=list)
    apparatus: list[str] = Field(default_factory=list)
    depicted_actions: list[str] = Field(default_factory=list)
    observable_states: list[str] = Field(default_factory=list)


class CategoryDetails(EvidenceModel):
    chart: ChartDetails | None = None
    mathematics: MathematicsDetails | None = None
    scientific_structure: ScientificStructureDetails | None = None
    flow_network: FlowNetworkDetails | None = None
    table: TableDetails | None = None
    structured_text: StructuredTextDetails | None = None
    geometry: GeometryDetails | None = None
    programming: ProgrammingDetails | None = None
    interface: InterfaceDetails | None = None
    map: MapDetails | None = None
    observational_image: ObservationalImageDetails | None = None
    physical_scene: PhysicalSceneDetails | None = None


class VisualEvidence(EvidenceModel):
    """Canonical, level-independent evidence for exactly one frame."""

    schema_version: Literal["1.1.0"]
    extraction_status: ExtractionStatus
    evidence_status: EvidenceStatus
    category_ids: list[CanonicalCategory]
    regions: list[Region]
    relations: list[Relation]
    uncertainties: list[Uncertainty]
    structural_description: StructuralDescription
    validation_diagnostics: list[ValidationDiagnostic]
    frame_id: str | None = None
    timestamp_seconds: float | None = Field(default=None, ge=0)
    raw_category_labels: list[str] = Field(default_factory=list)
    global_layout: str | None = None
    reading_order: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None
    category_details: CategoryDetails | None = None
    exact_content: list[ExactContent] = Field(default_factory=list)
    legacy_snapshot: dict[str, Any] | None = None

    SCENE_TARGET: ClassVar[str] = "scene"

    @field_validator("category_ids", "raw_category_labels", "reading_order")
    @classmethod
    def unique_scene_values(cls, value: list[str], info: ValidationInfo) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must contain unique values")
        return value

    @model_validator(mode="after")
    def validate_structure(self) -> "VisualEvidence":
        if self.extraction_status in {"failed", "not_attempted"}:
            if self.evidence_status != "unavailable_or_unreadable":
                raise ValueError(
                    "failed or not_attempted extraction must use "
                    "unavailable_or_unreadable evidence_status"
                )

        ids: set[str] = set()
        region_ids: set[str] = set()
        element_ids: set[str] = set()
        relation_ids: set[str] = set()
        element_by_id: dict[str, Element] = {}

        for region in self.regions:
            if region.id in ids:
                raise ValueError(f"duplicate ID '{region.id}'")
            ids.add(region.id)
            region_ids.add(region.id)
            for element in region.elements:
                if element.id in ids:
                    raise ValueError(f"duplicate ID '{element.id}'")
                ids.add(element.id)
                element_ids.add(element.id)
                element_by_id[element.id] = element

        for relation in self.relations:
            if relation.id is not None:
                if relation.id in ids:
                    raise ValueError(f"duplicate ID '{relation.id}'")
                ids.add(relation.id)
                relation_ids.add(relation.id)

        endpoint_ids = region_ids | element_ids
        for relation in self.relations:
            if relation.source_id not in endpoint_ids:
                raise ValueError(
                    f"relation source_id '{relation.source_id}' does not reference a region or element"
                )
            if relation.target_id not in endpoint_ids:
                raise ValueError(
                    f"relation target_id '{relation.target_id}' does not reference a region or element"
                )

        relation_keys: set[tuple[Any, ...]] = set()
        for relation in self.relations:
            key = (
                relation.source_id,
                relation.target_id,
                relation.relation,
                relation.label,
                relation.directed,
            )
            if key in relation_keys:
                raise ValueError(f"duplicate competing relation {key!r}")
            relation_keys.add(key)

        for element in element_by_id.values():
            if element.parent_id is not None and element.parent_id not in endpoint_ids:
                raise ValueError(
                    f"element '{element.id}' parent_id '{element.parent_id}' is invalid"
                )
            for child_id in element.child_ids:
                if child_id not in element_ids:
                    raise ValueError(
                        f"element '{element.id}' child_id '{child_id}' is invalid"
                    )
                if child_id == element.id:
                    raise ValueError(f"element '{element.id}' cannot be its own child")

        for reference in self.reading_order:
            if reference not in endpoint_ids:
                raise ValueError(f"reading_order reference '{reference}' is invalid")
        for reference in self.structural_description.reading_order:
            if reference not in endpoint_ids:
                raise ValueError(
                    f"structural reading_order reference '{reference}' is invalid"
                )
        for reference in self.structural_description.component_ids:
            if reference not in endpoint_ids:
                raise ValueError(f"structural component_id '{reference}' is invalid")
        for reference in self.structural_description.relation_ids:
            if reference not in relation_ids:
                raise ValueError(f"structural relation_id '{reference}' is invalid")

        uncertainty_targets = ids | {self.SCENE_TARGET}
        if self.frame_id:
            uncertainty_targets.add(self.frame_id)
        for uncertainty in self.uncertainties:
            if uncertainty.target_id not in uncertainty_targets:
                raise ValueError(
                    f"uncertainty target_id '{uncertainty.target_id}' is invalid"
                )

        return self

    def to_schema_dict(self) -> dict[str, Any]:
        """Serialize exactly as a JSON-compatible v1.1 evidence record."""

        data = self.model_dump(mode="json", exclude_none=True)
        # These two nullable members are required by the authoritative schema.
        structure = data["structural_description"]
        structure.setdefault("overview", None)
        structure.setdefault("orientation", None)

        def convert_bbox(container: dict[str, Any]) -> None:
            bbox = container.get("bbox_normalized")
            if isinstance(bbox, dict):
                container["bbox_normalized"] = [
                    bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
                ]

        for region in data.get("regions", []):
            convert_bbox(region)
            for element in region.get("elements", []):
                convert_bbox(element)
        return data

