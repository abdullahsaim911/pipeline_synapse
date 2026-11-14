"""JSON Schema used to constrain Qwen's compact observation response."""

from __future__ import annotations

from .assembler import COMPACT_OBSERVATION_SCHEMA
from .models import CORE_RELATION_TYPES, CanonicalCategory


EVIDENCE_STATUSES = [
    "supported",
    "partially_supported",
    "ambiguous_or_conflicting",
    "unavailable_or_unreadable",
]

STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
CATEGORY_ARRAY = {"type": "array", "items": {"enum": [item.value for item in CanonicalCategory]}}
BBOX = {
    "anyOf": [
        {"type": "null"},
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
    ]
}

ELEMENT = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "pattern": "^e_[a-zA-Z0-9_]+$"},
        "kind": {"type": "string"},
        "evidence_status": {"enum": EVIDENCE_STATUSES},
        "categories": CATEGORY_ARRAY,
        "label": NULLABLE_STRING,
        "text": NULLABLE_STRING,
        "description": {"type": "string", "minLength": 1},
        "position": NULLABLE_STRING,
        "bbox": BBOX,
        "source_ref": NULLABLE_STRING,
        "attributes": {"type": "object"},
        "parent_ref": NULLABLE_STRING,
        # One parent pointer expresses containment without a second, redundant
        # child inventory that can diverge or expand indefinitely during decoding.
    },
    "required": ["ref", "kind", "description", "bbox", "evidence_status"],
    "additionalProperties": False,
}

REGION = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "pattern": "^r_[a-zA-Z0-9_]+$"},
        "categories": CATEGORY_ARRAY,
        "evidence_status": {"enum": EVIDENCE_STATUSES},
        "elements": {"type": "array", "items": ELEMENT},
        "position": NULLABLE_STRING,
        "bbox": BBOX,
        "reading_order_index": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "description": NULLABLE_STRING,
    },
    "required": ["ref", "categories", "evidence_status", "elements", "position", "bbox"],
    "additionalProperties": False,
}

RELATION = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "pattern": "^rel_[a-zA-Z0-9_]+$"},
        "source": {"type": "string"},
        "target": {"type": "string"},
        "type": {
            "anyOf": [
                {"enum": sorted(CORE_RELATION_TYPES)},
                {"type": "string", "pattern": "^custom:[a-z0-9_:-]+$"},
            ]
        },
        "evidence_status": {"enum": EVIDENCE_STATUSES},
        "label": NULLABLE_STRING,
        "directed": {"type": "boolean"},
        "cross_category": {"type": "boolean"},
        "source_refs": STRING_ARRAY,
    },
    "required": ["ref", "source", "target", "type", "evidence_status"],
    "additionalProperties": False,
}

COMPACT_OBSERVATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"const": COMPACT_OBSERVATION_SCHEMA},
        "coordinate_system": {"const": "qwen_1000"},
        "status": {"enum": ["complete", "partial", "failed", "not_attempted"]},
        "evidence_status": {"enum": EVIDENCE_STATUSES},
        "categories": CATEGORY_ARRAY,
        "layout": NULLABLE_STRING,
        "regions": {"type": "array", "items": REGION},
        "relations": {"type": "array", "items": RELATION},
        "structure": {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "orientation": NULLABLE_STRING,
                "reading_order": STRING_ARRAY,
                "hierarchy": STRING_ARRAY,
                "topology": STRING_ARRAY,
                "uncertainties": STRING_ARRAY,
            },
            "required": ["overview"],
            "additionalProperties": False,
        },
        "exact": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "status": {"enum": EVIDENCE_STATUSES},
                    "notation": {"enum": ["plain", "math", "code", "chemical", "other"]},
                    "source_ref": NULLABLE_STRING,
                },
                "required": ["text", "status", "notation"],
                "additionalProperties": False,
            },
        },
        "details": {"type": "object"},
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "impact": NULLABLE_STRING,
                },
                "required": ["target", "reason"],
                "additionalProperties": False,
            },
        },
        "missing": STRING_ARRAY,
    },
    "required": [
        "schema",
        "coordinate_system",
        "status",
        "evidence_status",
        "categories",
        "regions",
        "relations",
        "structure",
        "exact",
    ],
    "additionalProperties": False,
}
