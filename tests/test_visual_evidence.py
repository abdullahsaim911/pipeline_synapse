"""Focused Task 2A tests for canonical single-frame visual evidence."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from visual_evidence import (
    BoundingBox,
    EvidenceMetadata,
    FrameObservation,
    RegionObservation,
    RelationObservation,
    StructuralObservation,
    UncertaintyObservation,
    adapt_legacy_snapshot,
    assemble_visual_evidence,
    validate_visual_evidence,
)
from visual_evidence.assembler import ElementObservation


def _authoritative_schema_path() -> Path:
    """Locate the approved Task 2 schema without copying or weakening it."""

    repository_root = Path(__file__).resolve().parents[1]
    roots = [repository_root / "task2_spec", repository_root / "task2_specs"]
    matches = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("visual_evidence_schema_v1_1.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one authoritative visual_evidence_schema_v1_1.json "
            f"under task2_spec[s], found {len(matches)}"
        )
    return matches[0]


SCHEMA_PATH = _authoritative_schema_path()


def _metadata(categories=None, labels=None, extraction="complete", evidence="supported"):
    return EvidenceMetadata(
        frame_id="frame-42",
        timestamp_seconds=12.5,
        category_ids=categories or [],
        raw_category_labels=labels or [],
        extraction_status=extraction,
        evidence_status=evidence,
        provenance={
            "provider": "modal",
            "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
            "frame_path": "frames/frame-42.jpg",
        },
    )


def _element(ref, kind, **kwargs):
    return ElementObservation(
        ref=ref,
        kind=kind,
        evidence_status=kwargs.pop("evidence_status", "supported"),
        **kwargs,
    )


def _structured_evidence():
    observation = FrameObservation(
        global_layout="A left-to-right decision process.",
        regions=[
            RegionObservation(
                ref="process",
                category_ids=["CT-04"],
                position="center",
                bbox_normalized=[0.1, 0.1, 0.9, 0.9],
                attributes={"structural_description": "Three connected process nodes."},
                elements=[
                    _element("start", "process_node", label="Start"),
                    _element("decision", "decision_node", label="Valid?"),
                    _element("accept", "terminal_node", label="Accept"),
                ],
            )
        ],
        relations=[
            RelationObservation(
                ref="edge-start",
                source_ref="start",
                target_ref="decision",
                relation="leads_to",
                directed=True,
                evidence_status="supported",
            ),
            RelationObservation(
                ref="edge-yes",
                source_ref="decision",
                target_ref="accept",
                relation="branches_to",
                label="Yes",
                directed=True,
                evidence_status="supported",
            ),
        ],
        structural_description=StructuralObservation(
            overview="Start leads to a decision, whose Yes branch reaches Accept.",
            orientation="left to right",
            component_refs=["start", "decision", "accept"],
            relation_refs=["edge-start", "edge-yes"],
            reading_order_refs=["start", "decision", "accept"],
            topology_notes=["The decision has a visibly labeled Yes branch."],
        ),
    )
    return assemble_visual_evidence(_metadata(["CT-04"], ["flowchart"]), observation)


def test_basic_creation_serialization_matches_authoritative_json_schema():
    evidence = assemble_visual_evidence(_metadata(["CT-06"], ["text"]))
    serialized = evidence.to_schema_dict()

    assert serialized["schema_version"] == "1.1.0"
    assert serialized["frame_id"] == "frame-42"
    assert serialized["structural_description"]["overview"] is None
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        validator = jsonschema.Draft202012Validator(json.load(handle))
    validator.validate(serialized)
    validator.validate(_structured_evidence().to_schema_dict())


@pytest.mark.parametrize("category_number", range(1, 13))
def test_all_twelve_categories_are_representable(category_number):
    category = f"CT-{category_number:02d}"
    evidence = assemble_visual_evidence(_metadata([category]))
    assert evidence.category_ids == [category]


def test_mixed_category_and_three_category_scene_preserves_all_categories():
    evidence = assemble_visual_evidence(
        _metadata(["CT-01", "CT-02", "CT-06"], ["graph", "equation", "text"]),
        FrameObservation(
            regions=[
                RegionObservation(ref="graph", category_ids=["CT-01"], elements=[]),
                RegionObservation(ref="math", category_ids=["CT-02"], elements=[]),
                RegionObservation(ref="caption", category_ids=["CT-06"], elements=[]),
            ]
        ),
    )
    assert evidence.category_ids == ["CT-01", "CT-02", "CT-06"]
    assert [region.category_ids for region in evidence.regions] == [
        ["CT-01"], ["CT-02"], ["CT-06"]
    ]


def test_complete_structural_description_and_directed_branch_ids_are_grounded():
    evidence = _structured_evidence()
    structure = evidence.structural_description

    assert structure.component_ids == ["element-001", "element-002", "element-003"]
    assert structure.relation_ids == ["relation-001", "relation-002"]
    assert evidence.relations[1].directed is True
    assert evidence.relations[1].label == "Yes"
    assert evidence.relations[1].source_id == "element-002"


def test_circuit_topology_and_uncertain_crossing_are_representable():
    evidence = assemble_visual_evidence(
        _metadata(["CT-03"], ["circuit"]),
        FrameObservation(
            regions=[RegionObservation(
                ref="circuit",
                category_ids=["CT-03"],
                elements=[
                    _element("battery", "battery", label="9 V"),
                    _element("r1", "resistor", label="R1"),
                    _element("crossing", "wire_crossing", evidence_status="ambiguous_or_conflicting"),
                ],
            )],
            relations=[RelationObservation(
                ref="wire-1",
                source_ref="battery",
                target_ref="r1",
                relation="connected_to",
                evidence_status="supported",
                source_refs=["visual_analysis.circuit.connections[0]"],
            )],
            uncertainties=[UncertaintyObservation(
                target_ref="crossing",
                reason="No visible junction dot at the wire crossing.",
                impact="Connectivity through the crossing is not confirmed.",
            )],
            structural_description=StructuralObservation(
                overview="A battery is visibly wired to resistor R1.",
                component_refs=["battery", "r1", "crossing"],
                relation_refs=["wire-1"],
                uncertainties=["The wire crossing may or may not be a junction."],
            ),
            category_details={
                "scientific_structure": {
                    "components": ["9 V battery", "R1"],
                    "connections": ["battery to R1"],
                }
            },
        ),
    )
    assert evidence.relations[0].relation == "connected_to"
    assert evidence.uncertainties[0].target_id == "element-003"


def test_mathematical_exactness_is_preserved_verbatim():
    expression = r"\sum_{i=1}^{n} x_i^2"
    evidence = assemble_visual_evidence(
        _metadata(["CT-02"], ["equation"]),
        FrameObservation(
            exact_content=[{
                "text": expression,
                "status": "supported",
                "notation": "math",
                "source_ref": "equation-ocr",
            }],
            category_details={"mathematics": {"exact_expressions": [expression]}},
        ),
    )
    assert evidence.exact_content[0].text == expression
    assert evidence.category_details.mathematics.exact_expressions == [expression]


def test_code_syntax_indentation_and_output_are_preserved():
    code = "if x >= 2:\n    print(x)"
    evidence = assemble_visual_evidence(
        _metadata(["CT-08"], ["code"]),
        FrameObservation(category_details={
            "programming": {
                "language": "python",
                "exact_code": code,
                "displayed_output": "2",
                "indentation_significant": True,
            }
        }),
    )
    assert evidence.category_details.programming.exact_code == code
    assert ">=" in evidence.category_details.programming.exact_code


def test_table_cells_preserve_row_column_association():
    evidence = assemble_visual_evidence(
        _metadata(["CT-05"], ["table"]),
        FrameObservation(category_details={
            "table": {
                "headers": ["Year", "Value"],
                "row_count": 2,
                "column_count": 2,
                "cells": [{"row": 1, "column": 2, "text": "42", "unit": "kg"}],
            }
        }),
    )
    assert evidence.category_details.table.cells[0]["column"] == 2


def test_spatial_positions_bounding_boxes_and_reading_order():
    evidence = assemble_visual_evidence(
        _metadata(["CT-06"]),
        FrameObservation(
            regions=[RegionObservation(
                ref="text-region",
                category_ids=["CT-06"],
                position="upper left",
                bbox_normalized=[0.0, 0.0, 0.5, 0.5],
                elements=[
                    _element("heading", "heading", exact_text="Overview"),
                    _element("body", "paragraph", exact_text="Details"),
                ],
            )],
            structural_description=StructuralObservation(
                component_refs=["heading", "body"],
                reading_order_refs=["heading", "body"],
            ),
        ),
    )
    assert evidence.reading_order == ["element-001", "element-002"]
    assert evidence.to_schema_dict()["regions"][0]["bbox_normalized"] == [0.0, 0.0, 0.5, 0.5]


def test_cross_category_relation_is_explicit_not_inferred_from_proximity():
    evidence = assemble_visual_evidence(
        _metadata(["CT-01", "CT-02"]),
        FrameObservation(
            regions=[
                RegionObservation(ref="chart", category_ids=["CT-01"], elements=[
                    _element("curve", "plotted_curve", label="f(x)")
                ]),
                RegionObservation(ref="formula", category_ids=["CT-02"], elements=[
                    _element("equation", "equation", exact_text="f(x)=x^2")
                ]),
            ],
            relations=[RelationObservation(
                source_ref="equation",
                target_ref="curve",
                relation="corresponds_to",
                cross_category=True,
                evidence_status="supported",
            )],
        ),
    )
    assert evidence.relations[0].cross_category is True


def test_duplicate_ids_are_structurally_invalid():
    payload = _structured_evidence().to_schema_dict()
    payload["regions"][0]["elements"][1]["id"] = "element-001"
    result = validate_visual_evidence(payload)
    assert result.structurally_valid is False
    assert "duplicate ID" in result.diagnostics[0].message


def test_invalid_relation_reference_is_structurally_invalid():
    payload = _structured_evidence().to_schema_dict()
    payload["relations"][0]["target_id"] = "missing-element"
    result = validate_visual_evidence(payload)
    assert result.structurally_valid is False
    assert "does not reference" in result.diagnostics[0].message


@pytest.mark.parametrize(
    "bbox",
    ([0.7, 0.1, 0.2, 0.9], [-0.1, 0.1, 0.2, 0.9], [0.1, 0.1, 1.1, 0.9]),
)
def test_invalid_bounding_boxes_are_rejected(bbox):
    with pytest.raises(ValueError, match="bbox|greater than or equal|less than or equal"):
        RegionObservation(ref="bad", category_ids=["CT-11"], bbox_normalized=bbox)


def test_valid_but_partially_supported_evidence_remains_valid():
    evidence = assemble_visual_evidence(
        _metadata(["CT-11"], extraction="partial", evidence="partially_supported"),
        FrameObservation(
            uncertainties=[UncertaintyObservation(
                target_ref="scene",
                reason="The lower-right label is occluded.",
            )]
        ),
    )
    result = validate_visual_evidence(evidence)
    assert result.structurally_valid is True
    assert result.evidence.evidence_status == "partially_supported"


def test_unreadable_label_and_uncertain_connection_remain_uncertain():
    evidence = assemble_visual_evidence(
        _metadata(["CT-04"], extraction="partial", evidence="ambiguous_or_conflicting"),
        FrameObservation(
            regions=[RegionObservation(ref="network", category_ids=["CT-04"], elements=[
                _element(
                    "node",
                    "network_node",
                    label=None,
                    evidence_status="unavailable_or_unreadable",
                )
            ])],
            uncertainties=[UncertaintyObservation(
                target_ref="node",
                reason="Node label is unreadable and its outgoing edge is occluded.",
            )],
        ),
    )
    assert evidence.regions[0].elements[0].label is None
    assert "unreadable" in evidence.uncertainties[0].reason
    assert evidence.relations == []


def test_legacy_snapshot_is_preserved_without_invented_regions_or_connections():
    legacy = {
        "content_type": "diagram",
        "detected_types": ["diagram", "equation"],
        "visual_analysis": {"diagram": {"details": "Two adjacent boxes"}},
        "structural_description": "Two adjacent boxes; connection unclear.",
        "reading_order": ["left box", "right box"],
        "text_readout": "x + 1",
        "layout": "Two columns",
        "missing_elements": None,
        "future_unknown_field": {"preserve": True},
    }
    evidence = adapt_legacy_snapshot(legacy, frame_id="legacy-frame")

    assert evidence.legacy_snapshot == legacy
    assert evidence.regions == []
    assert evidence.relations == []
    assert evidence.category_ids == ["CT-02"]
    assert evidence.raw_category_labels == ["diagram", "equation"]
    assert any(item.code == "category.unresolved_raw_label" for item in evidence.validation_diagnostics)


def test_failed_vlm_response_remains_unavailable_and_preserved():
    legacy = {
        "_error": True,
        "_error_detail": "TimeoutError: endpoint timed out",
        "content_type": "graph",
        "detected_types": ["graph"],
        "visual_analysis": {},
        "missing_elements": "VLM analysis failed",
    }
    evidence = adapt_legacy_snapshot(legacy)
    assert evidence.extraction_status == "failed"
    assert evidence.evidence_status == "unavailable_or_unreadable"
    assert evidence.exact_content == []
    assert evidence.legacy_snapshot == legacy


def test_schema_version_compatibility_rejects_other_versions():
    payload = _structured_evidence().to_schema_dict()
    payload["schema_version"] = "1.0.0"
    result = validate_visual_evidence(payload)
    assert result.structurally_valid is False
    assert "schema_version" in result.diagnostics[0].message


def test_hierarchy_references_are_resolved_and_validated():
    evidence = assemble_visual_evidence(
        _metadata(["CT-03"]),
        FrameObservation(regions=[RegionObservation(
            ref="cell",
            category_ids=["CT-03"],
            elements=[
                _element("cell-body", "cell", child_refs=["nucleus"]),
                _element("nucleus", "organelle", parent_ref="cell-body"),
            ],
        )]),
    )
    parent, child = evidence.regions[0].elements
    assert parent.child_ids == [child.id]
    assert child.parent_id == parent.id


def test_category_specific_custom_relation_semantics_are_preserved():
    evidence = assemble_visual_evidence(
        _metadata(["CT-03"]),
        FrameObservation(
            regions=[RegionObservation(ref="chem", category_ids=["CT-03"], elements=[
                _element("carbon", "atom", label="C"),
                _element("oxygen", "atom", label="O"),
            ])],
            relations=[RelationObservation(
                source_ref="carbon",
                target_ref="oxygen",
                relation="custom:double_bond",
                evidence_status="supported",
            )],
        ),
    )
    assert evidence.relations[0].relation == "custom:double_bond"

