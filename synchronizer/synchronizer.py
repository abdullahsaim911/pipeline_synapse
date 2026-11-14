"""
Synchronizer Module (M1)

Detect "suffer points" where blind students lose visual access.
"""

import json
import re
import unicodedata
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from pathlib import Path

from category_normalization import normalize_category_labels


@dataclass
class TranscriptEntry:
    """Transcript segment with timing information."""
    start: float
    end: float
    text: str


@dataclass
class InterventionPoint:
    """Represents a detected suffer point requiring intervention."""
    timestamp: float
    frame_path: str
    content_type: str
    complexity_score: float
    clip_confidence: Optional[float]
    transcript_context: str
    trigger_reason: str
    confidence: str = "medium"
    classification_confidence: Optional[float] = None
    confidence_semantics: str = "unknown"
    raw_category_labels: Optional[List[str]] = None
    canonical_categories: Optional[List[str]] = None
    category_mapping_notes: Optional[List[str]] = None


@dataclass
class AtomicVisualFact:
    """One source-grounded fact derived from canonical visual evidence."""

    fact_id: str
    fact_type: str
    description: str
    source_refs: List[str]
    category_ids: List[str]
    importance: str
    evidence_reliability: str
    evidence_status: str
    uncertainty_reasons: List[str]
    subject_groups: List[List[str]]
    relation_cues: List[str]
    precision_sensitive: bool = False
    directional: bool = False
    # Internal matching metadata; omitted from the serialized coverage contract.
    predicate: str = ""
    exact_text: str = ""
    notation: str = ""
    relation_type: str = ""
    relation_label: str = ""
    ambiguous_subject: bool = False


_COVERAGE_LABELS = {
    "spoken_explicit", "spoken_partial", "unspoken", "uncertain_match", "not_usable"
}
_IMPORTANCE_ORDER = {"critical": 0, "important": 1, "supporting": 2}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "this", "to", "with", "shown",
}
_DEICTIC_WORDS = {
    "look", "here", "there", "this", "that", "see", "notice", "observe", "screen"
}
_RELATION_CUES = {
    "left_of": ["left of", "to the left"],
    "right_of": ["right of", "to the right"],
    "above": ["above", "over", "on top of"],
    "below": ["below", "under", "beneath"],
    "contains": ["contains", "includes", "holds"],
    "inside": ["inside", "within"],
    "connected_to": ["connected", "joined", "wired"],
    "points_to": ["points to", "points toward", "arrow to"],
    "leads_to": ["leads to", "flows to", "goes to", "then"],
    "branches_to": ["branches to", "branch to", "leads to", "go to", "goes to"],
    "corresponds_to": ["corresponds to", "matches", "represents"],
    "parent_of": ["parent of", "contains"],
    "child_of": ["child of", "part of"],
    "labels": ["labels", "label points to", "named"],
    "intersects": ["intersects", "crosses at"],
    "parallel_to": ["parallel"],
    "perpendicular_to": ["perpendicular", "right angle"],
    "bonded_to": ["bonded", "bond"],
    "custom:part_of": ["part of", "belongs to", "within"],
    "custom:behind": ["behind", "at the rear of"],
}


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> List[str]:
    return re.findall(
        r">=|<=|==|!=|[=<>≤≥≠+*/^\-]|[^\W\d_]\w*(?:'\w+)?|\d+(?:\.\d+)?",
        _normalized_text(value),
    )


def _content_tokens(value: Any) -> List[str]:
    return [token for token in _tokens(value) if token not in _STOP_WORDS]


def _subject_tokens(value: Any) -> List[str]:
    # Labels may contain identifiers such as A or Point A: articles cannot be
    # dropped from an entity name as though they were prose stop words.
    return _tokens(value)


def _flatten_detail(prefix: str, value: Any) -> List[str]:
    """Turn category-detail leaves into concise facts without inventing semantics."""

    if value is None or value == [] or value == {} or value == "":
        return []
    if isinstance(value, dict):
        scalar_parts = [
            f"{key.replace('_', ' ')} {item}"
            for key, item in value.items()
            if not isinstance(item, (dict, list)) and item is not None
        ]
        nested = [
            fact
            for key, item in value.items()
            if isinstance(item, (dict, list))
            for fact in _flatten_detail(f"{prefix} {key.replace('_', ' ')}", item)
        ]
        return ([f"{prefix}: {'; '.join(scalar_parts)}"] if scalar_parts else []) + nested
    if isinstance(value, list):
        return [fact for item in value for fact in _flatten_detail(prefix, item)]
    return [f"{prefix}: {value}"]


_DETAIL_CATEGORIES = dict(zip(
    ("chart", "mathematics", "scientific_structure", "flow_network", "table", "structured_text",
     "geometry", "programming", "interface", "map", "observational_image", "physical_scene"),
    (f"CT-{number:02d}" for number in range(1, 13)),
))
_CRITICAL_DETAIL_FIELDS = {
    "chart": ("axes", "units", "scales", "visible values", "observed trends", "legends"),
    "mathematics": ("exact expressions", "transformations", "bounds"),
    "scientific_structure": ("connections", "containment", "directions"),
    "flow_network": ("branch conditions", "loops"),
    "table": ("headers", "cells"),
    "structured_text": ("definitions", "warnings", "steps", "lists"),
    "geometry": ("constraints", "angles", "markings", "dimensions"),
    "programming": ("exact code", "displayed output", "indentation"),
    "interface": ("visible states",), "map": ("directions", "routes", "legend entries"),
    "observational_image": ("comparisons",), "physical_scene": ("observable states", "contacts"),
}


def _detail_importance(detail_name: str, description: str) -> str:
    path = description.split(":", 1)[0]
    if detail_name == "structured_text" and set(_tokens(description)) & _NEGATIONS:
        return "critical"
    return "critical" if any(field in path for field in _CRITICAL_DETAIL_FIELDS.get(detail_name, ())) else "important"


def extract_atomic_visual_facts(
    visual_evidence: Dict[str, Any], *, max_facts: int = 64
) -> tuple[List[AtomicVisualFact], Dict[str, Any]]:
    """Extract bounded atomic facts from canonical evidence without inference."""

    if not isinstance(visual_evidence, dict):
        return [], {"truncated": False, "omitted_fact_count": 0}
    if not isinstance(max_facts, int) or isinstance(max_facts, bool) or max_facts < 0:
        raise ValueError("max_facts must be a non-negative integer")

    root_status = visual_evidence.get("evidence_status", "unavailable_or_unreadable")
    extraction_status = visual_evidence.get("extraction_status", "failed")
    root_categories = list(visual_evidence.get("category_ids") or [])
    unusable_root = extraction_status in {"failed", "not_attempted"} or root_status == "unavailable_or_unreadable"
    uncertainty_by_target: Dict[str, List[str]] = {}
    for item in visual_evidence.get("uncertainties") or []:
        if isinstance(item, dict):
            uncertainty_by_target.setdefault(item.get("target_id", "scene"), []).append(
                str(item.get("reason") or "Unspecified visual uncertainty")
            )
    uncertainty_by_target.setdefault("scene", []).extend(
        (visual_evidence.get("structural_description") or {}).get("uncertainties") or []
    )

    candidates: List[AtomicVisualFact] = []
    names: Dict[str, str] = {}
    categories_by_ref: Dict[str, List[str]] = {}
    status_by_ref: Dict[str, str] = {}
    elements = []
    for region in visual_evidence.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_id = region.get("id")
        categories = list(region.get("category_ids") or root_categories)
        if region_id:
            names[region_id] = str(region.get("position") or "visual region")
            categories_by_ref[region_id] = categories
            status_by_ref[region_id] = region.get("evidence_status") or root_status
        for element in region.get("elements") or []:
            if isinstance(element, dict) and element.get("id"):
                ref = element["id"]
                names[ref] = str(element.get("label") or element.get("exact_text") or element.get("kind") or "visual element")
                categories_by_ref[ref] = list(element.get("category_ids") or categories)
                status_by_ref[ref] = element.get("evidence_status") or region.get("evidence_status") or root_status
                elements.append(element)
    name_counts = {}
    for element in elements:
        name = _normalized_text(names[element["id"]])
        name_counts[name] = name_counts.get(name, 0) + 1
    elements_by_ref = {element["id"]: element for element in elements}
    relations_by_ref = {item["id"]: item for item in visual_evidence.get("relations") or []
                        if isinstance(item, dict) and item.get("id")}
    order_family = {kind: family for family in (("left_of", "right_of"), ("above", "below"), ("precedes", "follows")) for kind in family}
    order_groups = {}
    for ref, relation in relations_by_ref.items():
        family = order_family.get(relation.get("relation"))
        if family:
            key = (frozenset((relation.get("source_id"), relation.get("target_id"))), family)
            order_groups.setdefault(key, []).append(ref)
    diagnosed_groups = set()
    for diagnostic in visual_evidence.get("validation_diagnostics") or []:
        if diagnostic.get("code") != "extraction.quality.conflicting_spatial_order":
            continue
        relation = relations_by_ref.get(diagnostic.get("target_id"))
        if not relation:
            continue
        key = (frozenset((relation.get("source_id"), relation.get("target_id"))), order_family.get(relation.get("relation")))
        if key not in diagnosed_groups:
            for ref in order_groups.get(key, []):
                status_by_ref[ref] = "ambiguous_or_conflicting"
                uncertainty_by_target.setdefault(ref, []).append(diagnostic.get("message") or "Conflicting spatial/order observations.")
            diagnosed_groups.add(key)
    candidate_keys = {}
    proposition_claims = {}
    next_number = 1

    def add_fact(
        fact_type: str,
        description: Any,
        source_refs: List[str],
        categories: List[str],
        importance: str,
        evidence_status: str,
        *,
        subject_groups: Optional[List[List[str]]] = None,
        relation_cues: Optional[List[str]] = None,
        precision_sensitive: bool = False,
        directional: bool = False,
        predicate: str = "",
        exact_text: str = "",
        notation: str = "",
        relation_type: str = "",
        relation_label: str = "",
        equivalence_key: Any = None,
    ) -> None:
        nonlocal next_number
        clean_description = str(description or "").strip()
        if not clean_description:
            return
        reasons = [
            reason
            for ref in source_refs or ["scene"]
            for reason in uncertainty_by_target.get(ref, [])
        ]
        status = evidence_status or root_status
        if unusable_root or status == "unavailable_or_unreadable" or any(status_by_ref.get(ref) == "unavailable_or_unreadable" for ref in source_refs):
            reliability = "unavailable"
        elif status == "ambiguous_or_conflicting" or any(status_by_ref.get(ref) == "ambiguous_or_conflicting" for ref in source_refs):
            reliability = "ambiguous"
        elif status == "partially_supported" or reasons or any(status_by_ref.get(ref) == "partially_supported" for ref in source_refs):
            reliability = "partial"
        else:
            reliability = "supported"
        # Equivalence requires the same proposition and known source association.
        # Status/reliability differences are never silently reconciled.
        key = (equivalence_key or (fact_type, clean_description, tuple(source_refs)), status, reliability)
        if key in candidate_keys:
            existing = candidate_keys[key]
            existing.source_refs = list(dict.fromkeys(existing.source_refs + source_refs))
            existing.category_ids = list(dict.fromkeys(existing.category_ids + (categories or root_categories)))
            existing.uncertainty_reasons = list(dict.fromkeys(existing.uncertainty_reasons + reasons))
            if _IMPORTANCE_ORDER[importance] < _IMPORTANCE_ORDER[existing.importance]:
                existing.importance = importance
            return
        fact = AtomicVisualFact(
            fact_id=f"fact-{next_number:03d}",
            fact_type=fact_type,
            description=clean_description,
            source_refs=list(dict.fromkeys(source_refs)),
            category_ids=list(dict.fromkeys(categories or root_categories)),
            importance=importance,
            evidence_reliability=reliability,
            evidence_status=status,
            uncertainty_reasons=list(dict.fromkeys(reasons)),
            subject_groups=subject_groups or [],
            relation_cues=relation_cues or [],
            precision_sensitive=precision_sensitive,
            directional=directional,
            predicate=predicate, exact_text=exact_text, notation=notation,
            relation_type=relation_type, relation_label=relation_label,
            ambiguous_subject=any(name_counts.get(_normalized_text(names.get(ref, "")), 0) > 1 for ref in source_refs),
        )
        candidates.append(fact)
        candidate_keys[key] = fact
        proposition_claims.setdefault(key[0], []).append(fact)
        next_number += 1

    # Prefer explicit exact-content notation/status over an inferred duplicate
    # from element.exact_text, but only for the same text and canonical source.
    exact_by_source = {}
    for item in visual_evidence.get("exact_content") or []:
        if isinstance(item, dict) and item.get("text"):
            exact_by_source.setdefault((item.get("source_ref"), item["text"]), []).append(item)

    def exact_context(ref, text, categories):
        if "CT-05" not in categories:
            return {"subject_groups": [_content_tokens(text)]}
        source = elements_by_ref.get(ref, {})
        attributes = source.get("attributes") or {}
        context = [str(attributes[key]) for key in ("row_header", "column_header") if attributes.get(key)]
        if not context and source.get("label") not in (None, "", text):
            context = [source["label"]]
        return {"subject_groups": [_subject_tokens(item) for item in context], "relation_type": "table_value"}

    for region in visual_evidence.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_id = region.get("id")
        region_categories = list(region.get("category_ids") or root_categories)
        if region_id:
            names[region_id] = str(region.get("position") or "visual region")
            categories_by_ref[region_id] = region_categories
        for element in region.get("elements") or []:
            if not isinstance(element, dict) or not element.get("id"):
                continue
            element_id = element["id"]
            label = element.get("label") or element.get("exact_text")
            kind = str(element.get("kind") or "visual element").replace("_", " ")
            name = str(label or kind)
            names[element_id] = name
            categories = list(element.get("category_ids") or region_categories)
            categories_by_ref[element_id] = categories
            status = element.get("evidence_status") or region.get("evidence_status") or root_status
            text_only = kind in {"text label", "label", "text", "heading", "paragraph", "equation", "code block", "code line"}
            if not (text_only and element.get("exact_text")) and not (not label and element.get("description")):
                add_fact(
                    "component", f"Visible {kind}: {name}", [element_id], categories,
                    "important", status, subject_groups=[_subject_tokens(name)],
                )
            if element.get("description") and _normalized_text(element["description"]) not in {
                _normalized_text(element.get("label")), _normalized_text(element.get("exact_text"))
            }:
                add_fact(
                    "structure", element["description"], [element_id], categories,
                    "critical" if set(categories) & {"CT-03", "CT-04", "CT-07", "CT-12"} else "important",
                    status, subject_groups=[_subject_tokens(name)] if label else [],
                    predicate=element["description"],
                    equivalence_key=("statement", element_id, element["description"]),
                )
            if element.get("position"):
                add_fact(
                    "spatial", f"{name} is {element['position']}", [element_id], categories,
                    "critical", status, subject_groups=[_subject_tokens(name)],
                    relation_cues=_content_tokens(element["position"]),
                    predicate=element["position"],
                )
            explicit_exact = exact_by_source.get((element_id, element.get("exact_text")), [])
            if element.get("exact_text") and not any(item.get("status") == status for item in explicit_exact):
                notation = "code" if "CT-08" in categories else "math" if "CT-02" in categories else "exact"
                add_fact(
                    "exact_content", f"{notation}: {element['exact_text']}", [element_id],
                    categories, "critical", status,
                    precision_sensitive=True,
                    exact_text=element["exact_text"], notation=notation,
                    equivalence_key=("exact", element_id, notation, element["exact_text"]),
                    **exact_context(element_id, element["exact_text"], categories),
                )
            for key, value in (element.get("attributes") or {}).items():
                if value is None:
                    continue
                if isinstance(value, (dict, list)):
                    for index, description in enumerate(_flatten_detail(f"{name} {key.replace('_', ' ')}", value)):
                        add_fact("attribute", description, [element_id], categories, "important", status,
                                 subject_groups=[_subject_tokens(name)], predicate=description.split(":", 1)[-1],
                                 equivalence_key=("attribute_detail", element_id, key, index, description))
                    continue
                importance = "critical" if key in {
                    "unit", "value", "symbol", "state", "indentation_level",
                    "row_index", "column_index", "direction", "charge", "polarity"
                } else "supporting"
                if key in {"label", "text", "symbol"} and str(value) == element.get("exact_text"):
                    continue
                add_fact(
                    "attribute", f"{name} {key.replace('_', ' ')} is {value}",
                    [element_id], categories, importance, status,
                    subject_groups=[_subject_tokens(name)],
                    relation_cues=[key.replace("_", " ")],
                    precision_sensitive=importance == "critical",
                    predicate=str(value), relation_type=key,
                )
            if element.get("parent_id"):
                parent_id = element["parent_id"]
                add_fact(
                    "hierarchy", f"{name} is part of {names.get(parent_id, parent_id)}",
                    [element_id, parent_id], categories, "critical", status,
                    subject_groups=[_subject_tokens(name), _subject_tokens(names.get(parent_id, parent_id))],
                    relation_cues=["part of", "within"],
                    relation_type="custom:part_of", equivalence_key=("hierarchy", element_id, parent_id),
                )

        attributes = region.get("attributes") or {}
        for field in ("structural_description", "hierarchy_notes", "topology_notes"):
            notes = attributes.get(field) or []
            for note in notes if isinstance(notes, list) else [notes]:
                add_fact("structure", note, [region_id] if region_id else ["scene"], region_categories,
                         "important" if field == "structural_description" else "critical",
                         region.get("evidence_status") or root_status, predicate=note,
                         equivalence_key=("statement", region_id, note))

    for relation in visual_evidence.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        source_id, target_id = relation.get("source_id"), relation.get("target_id")
        if not source_id or not target_id:
            continue
        relation_type = str(relation.get("relation") or "related_to")
        relation_words = relation_type.replace("custom:", "").replace("_", " ")
        label = f" labeled {relation['label']}" if relation.get("label") else ""
        source_name, target_name = names.get(source_id, source_id), names.get(target_id, target_id)
        categories = list(dict.fromkeys(
            categories_by_ref.get(source_id, []) + categories_by_ref.get(target_id, [])
        )) or root_categories
        cues = list(_RELATION_CUES.get(relation_type, [relation_words]))
        if relation.get("label"):
            cues.append(str(relation["label"]))
        add_fact(
            "relation", f"{source_name} {relation_words} {target_name}{label}",
            [source_id, target_id] + ([relation["id"]] if relation.get("id") else []),
            categories, "critical", relation.get("evidence_status") or root_status,
            subject_groups=[_subject_tokens(source_name), _subject_tokens(target_name)],
            relation_cues=cues,
            directional=bool(relation.get("directed")),
            relation_type=relation_type, relation_label=str(relation.get("label") or ""),
            equivalence_key=("hierarchy", source_id, target_id) if relation_type in {"child_of", "custom:part_of"} and not relation.get("label")
            else ("hierarchy", target_id, source_id) if relation_type == "parent_of" and not relation.get("label")
            else ("relation", source_id, target_id, relation_type, relation.get("label"), bool(relation.get("directed"))),
        )

    for index, item in enumerate(visual_evidence.get("exact_content") or []):
        if not isinstance(item, dict) or not item.get("text"):
            continue
        source_ref = item.get("source_ref") or "scene"
        notation = item.get("notation") or "exact"
        add_fact(
            "exact_content", f"{notation}: {item['text']}", [source_ref],
            categories_by_ref.get(source_ref, root_categories), "critical",
            item.get("status") or root_status,
            precision_sensitive=True,
            exact_text=item["text"], notation=notation,
            equivalence_key=("exact", source_ref, notation, item["text"]) if source_ref in names
            else ("unbound_exact", index, notation, item["text"]),
            **exact_context(source_ref, item["text"], categories_by_ref.get(source_ref, root_categories)),
        )

    structure = visual_evidence.get("structural_description") or {}
    for text in dict.fromkeys([visual_evidence.get("global_layout"), structure.get("overview"), structure.get("orientation")]):
        if text:
            add_fact("structure", text, ["scene"], root_categories, "important", root_status,
                     predicate=text, equivalence_key=("statement", "scene", text))
    for key in ("hierarchy_notes", "topology_notes"):
        for note in structure.get(key) or []:
            add_fact(
                "structure", note, list(structure.get("component_ids") or []),
                root_categories, "critical", root_status,
                subject_groups=[_content_tokens(note)],
                predicate=note,
                equivalence_key=("statement", (structure.get("component_ids") or ["scene"])[0], note)
                if len(structure.get("component_ids") or []) <= 1 else None,
            )
    reading_order = list(structure.get("reading_order") or [])
    if len(reading_order) > 1:
        readable_order = [names.get(ref, ref) for ref in reading_order]
        add_fact(
            "reading_order", "Reading order: " + " then ".join(readable_order),
            reading_order, root_categories, "critical", root_status,
            subject_groups=[_subject_tokens(name) for name in readable_order],
            relation_cues=["then", "followed by", "before"],
            relation_type="precedes", directional=True,
        )

    for detail_name, detail_value in (visual_evidence.get("category_details") or {}).items():
        if detail_value is None:
            continue
        category = _DETAIL_CATEGORIES.get(detail_name)
        descriptions = _flatten_detail(detail_name.replace("_", " "), detail_value)
        if detail_name == "programming" and isinstance(detail_value, dict):
            descriptions = [description for key, value in detail_value.items()
                            for description in _flatten_detail(f"programming {key.replace('_', ' ')}", value)]
        for index, description in enumerate(descriptions):
            detail_payload = description.split(":", 1)[-1]
            add_fact(
                "category_detail", description, ["scene"], [category] if category else root_categories,
                _detail_importance(detail_name, description), root_status, subject_groups=[_content_tokens(detail_payload)],
                precision_sensitive=detail_name in {"mathematics", "programming", "table"},
                predicate=detail_payload,
                exact_text=detail_payload if detail_name in {"mathematics", "programming", "table"} else "",
                notation="code" if detail_name == "programming" else "math" if detail_name == "mathematics" else "plain",
                equivalence_key=("detail", detail_name, index, description),
            )

    unbound_counts = {}
    for fact in candidates:
        if fact.source_refs == ["scene"]:
            key = (fact.fact_type, fact.description)
            unbound_counts[key] = unbound_counts.get(key, 0) + 1
    for fact in candidates:
        if fact.source_refs == ["scene"] and unbound_counts[(fact.fact_type, fact.description)] > 1:
            fact.ambiguous_subject = True
    for claims in proposition_claims.values():
        if len({fact.evidence_status for fact in claims}) > 1:
            for fact in claims:
                fact.uncertainty_reasons.append("Source records disagree on support for this proposition.")
                if fact.evidence_reliability == "supported":
                    fact.evidence_reliability = "partial"

    type_order = {"exact_content": 0, "relation": 1, "hierarchy": 1, "structure": 2,
                  "spatial": 2, "reading_order": 2, "attribute": 3, "category_detail": 3, "component": 4}
    unique = sorted(candidates, key=lambda fact: (
        fact.evidence_reliability in {"ambiguous", "unavailable"},
        _IMPORTANCE_ORDER[fact.importance], type_order[fact.fact_type], int(fact.fact_id.split("-")[-1]),
    ))
    omitted = max(0, len(unique) - max_facts)
    # Seed with the best available facts that add category coverage, then fill
    # by educational priority. This is coverage, not a fixed per-category quota.
    covered = set()
    seeds = []
    for fact in unique:
        if fact.fact_type == "structure" and fact.source_refs == ["scene"]:
            continue  # A generic scene overview does not represent each category's facts.
        if fact.evidence_reliability not in {"ambiguous", "unavailable"} and set(fact.category_ids) - covered:
            seeds.append(fact.fact_id)
            covered.update(fact.category_ids)
    chosen = set(seeds[:max_facts])
    for fact in unique:
        if len(chosen) >= max_facts:
            break
        chosen.add(fact.fact_id)
    selected = [fact for fact in unique if fact.fact_id in chosen]
    for index, fact in enumerate(selected, start=1):
        fact.fact_id = f"fact-{index:03d}"
    return selected, {
        "truncated": omitted > 0,
        "omitted_fact_count": omitted,
        "total_fact_count": len(unique),
    }


_WORD_VARIANTS = {
    "pointing": "points", "point": "points", "towards": "toward",
    "branching": "branches", "branch": "branches", "linked": "connected",
    "increases": "rises", "increasing": "rises", "decreases": "falls", "decreasing": "falls",
}
_INVERSE_CUES = {
    "left_of": ["right of"], "right_of": ["left of"], "above": ["below", "under"],
    "below": ["above", "over"], "contains": ["inside", "within", "part of"],
    "inside": ["contains", "includes"], "parent_of": ["part of", "child of"],
    "child_of": ["contains", "parent of"], "custom:part_of": ["contains", "includes"],
    "points_to": ["pointed to by"], "leads_to": ["reached from"],
    "branches_to": ["reached from"], "precedes": ["follows", "after"], "follows": ["precedes", "before"],
}
_SYMMETRIC_RELATIONS = {"connected_to", "intersects", "parallel_to", "perpendicular_to", "bonded_to", "corresponds_to", "grouped_with"}
_NEGATIONS = {"not", "never", "without", "no", "cannot", "can't", "isn't", "aren't", "doesn't", "don't", "wasn't"}


def _match_tokens(value: str) -> List[str]:
    return [_WORD_VARIANTS.get(token, token) for token in _tokens(value)]


def _transcript_clauses(text: str) -> list:
    # Keep semicolon-associated table fields together; do not split decimals.
    return [(clause.strip(), _match_tokens(clause)) for clause in re.split(
        r"(?<!\d)[.!?](?!\d)|\b(?:but|however|whereas)\b", text, flags=re.IGNORECASE,
    ) if clause.strip()]


def _token_span(needle: List[str], tokens: List[str], *, content_only: bool = False, start: int = 0):
    indexed = [(index, token) for index, token in enumerate(tokens)
               if index >= start and (not content_only or token not in _STOP_WORDS)]
    wanted = [_WORD_VARIANTS.get(token, token) for token in needle if not content_only or token not in _STOP_WORDS]
    if not wanted:
        return None
    values = [token for _, token in indexed]
    for index in range(len(values) - len(wanted) + 1):
        if values[index:index + len(wanted)] == wanted:
            return indexed[index][0], indexed[index + len(wanted) - 1][0] + 1
    return None


def _contains_exact(payload: str, transcript: str, notation: str) -> bool:
    """Match an ordered literal; never use a bag of operators/numbers or case-fold code."""
    payload = unicodedata.normalize("NFC", payload).replace("\r\n", "\n")
    transcript = unicodedata.normalize("NFC", transcript).replace("\r\n", "\n")
    if notation == "math":
        payload = payload.replace("−", "-")
        transcript = transcript.replace("−", "-")
        pattern = r"\s*".join(re.escape(char) for char in payload if not char.isspace())
    elif notation == "code":
        pattern = re.escape(payload)
    else:
        payload = re.sub(r"\s+", " ", payload.casefold()).strip().rstrip(".!?")
        transcript = re.sub(r"\s+", " ", transcript.casefold())
        pattern = re.escape(payload)
    if not pattern:
        return False
    # Avoid matching x=5 inside x=50 or a short identifier inside another name.
    prefix = r"(?m)^" if notation == "code" and payload[:1] in {" ", "\t"} else r"(?<!\w)"
    suffix = r"(?!\w|\.\d)"
    if notation in {"math", "code"}:
        suffix = r"(?!\w|\.\w|\s*[+*/^=<>≤≥≠-])"
    match = re.search(prefix + pattern + suffix, transcript)
    if match and notation in {"math", "code"} and re.search(r"[+*/^=<>≤≥≠-]\s*$", transcript[:match.start()]):
        return False
    return bool(match)


def _relation_covered(fact: AtomicVisualFact, tokens: List[str], spans: list, *, allow_negation: bool = False) -> bool:
    if len(spans) < 2 or any(span is None for span in spans):
        return False
    if len({tuple(group) for group in fact.subject_groups}) < len(fact.subject_groups):
        return False  # Same labels cannot establish distinct endpoint roles.
    connectors = {"is", "are", "was", "were", "be", "will", "can", "does", "the", "a", "an",
                  "to", "toward", "with", "on", "when", "if", "node", "point"}
    connectors.update(_match_tokens(fact.relation_label))
    if allow_negation:
        connectors.update(_NEGATIONS)

    def binds(first, cue, second):
        return (first[1] <= cue[0] and cue[1] <= second[0]
                and set(tokens[first[1]:cue[0]] + tokens[cue[1]:second[0]]) <= connectors)

    if fact.fact_type == "reading_order":
        return all(any((cue_span := _token_span(_match_tokens(cue), tokens, start=left[1])) and binds(left, cue_span, right)
                       for cue in fact.relation_cues) for left, right in zip(spans, spans[1:]))
    source, target = spans[:2]
    cues = _RELATION_CUES.get(fact.relation_type, fact.relation_cues)
    for cue in cues:
        span = _token_span(_match_tokens(cue), tokens)
        if span and binds(source, span, target):
            return True
        if span and fact.relation_type in _SYMMETRIC_RELATIONS and not fact.directional:
            # "A and B are connected" and "B is connected to A" are symmetric.
            if binds(target, span, source):
                return True
            first, second = sorted((source, target))
            if second[1] <= span[0] and set(tokens[first[1]:second[0]] + tokens[second[1]:span[0]]) <= {"and", "is", "are", "both"}:
                return True
    for cue in _INVERSE_CUES.get(fact.relation_type, []):
        span = _token_span(_match_tokens(cue), tokens)
        if span and binds(target, span, source):
            return True
    return False


def _match_atomic_fact(fact: AtomicVisualFact, transcript_text: str, *, clauses=None) -> tuple[str, Optional[str], str]:
    if fact.evidence_reliability in {"unavailable", "ambiguous"}:
        return "not_usable", None, ""
    clauses = _transcript_clauses(transcript_text) if clauses is None else clauses
    if fact.precision_sensitive and fact.notation == "code":
        clauses = [(transcript_text, _match_tokens(transcript_text))]
    best, best_span = "unspoken", None
    contradicted = False
    ranks = {"unspoken": 0, "uncertain_match": 1, "spoken_partial": 2, "spoken_explicit": 3}
    for clause, tokens in clauses:
        if not set(tokens) - _DEICTIC_WORDS - _STOP_WORDS:
            continue
        spans = [_token_span(group, tokens) for group in fact.subject_groups if group]
        all_subjects = bool(spans) and all(span is not None for span in spans)
        negative = sum(token in _NEGATIONS for token in tokens if not (token == "no" and fact.relation_label.casefold() == "no"))
        source_negative = sum(token in _NEGATIONS for token in _match_tokens(fact.exact_text or fact.predicate))
        status = "unspoken"
        if fact.fact_type in {"relation", "hierarchy", "reading_order"}:
            if all_subjects:
                core_covered = _relation_covered(fact, tokens, spans)
                # A later opposite spatial/containment statement must not be
                # hidden by an earlier explicit match. Reciprocal network edges
                # can both be true, so only known asymmetric predicates qualify.
                if (negative and _relation_covered(fact, tokens, spans, allow_negation=True)) or (
                    fact.relation_type in _INVERSE_CUES and not core_covered
                    and _relation_covered(fact, tokens, list(reversed(spans)))
                ):
                    contradicted = True
                status = "spoken_explicit" if core_covered else "spoken_partial"
                if core_covered and fact.relation_label:
                    notation = "math" if re.search(r"[\d=<>≤≥≠+*/^²³]", fact.relation_label) else "plain"
                    if not _contains_exact(fact.relation_label, clause, notation):
                        status = "spoken_partial"
                if negative or (fact.relation_cues and any(_token_span(_match_tokens(cue), tokens) for cue in fact.relation_cues) and not core_covered):
                    status = "uncertain_match"
        elif fact.fact_type == "component":
            if all_subjects:
                status = "spoken_explicit" if set(tokens) & {"shown", "visible", "here", "shows", "displays", "see"} else "spoken_partial"
                if negative:
                    status = "uncertain_match"
        elif fact.fact_type in {"spatial", "attribute"}:
            if all_subjects:
                status = "spoken_partial"
                predicate = _token_span(_match_tokens(fact.predicate), tokens, content_only=True)
                connectors = {"is", "are", "was", "were", "has", "have", "at", "in", "the", "a", "an",
                              "to", "of", "with", "measured", "equals", "located", "set"}
                connectors.update(fact.relation_type.split("_"))
                if predicate and any(span[1] <= predicate[0] and set(tokens[span[1]:predicate[0]]) <= connectors for span in spans):
                    # Row/column indices require their roles, not just a number.
                    role = fact.relation_type.split("_")[0]
                    if role not in {"row", "column", "indentation"} or role in tokens:
                        status = "spoken_explicit"
                if fact.relation_type in {"unit", "symbol", "value"} and not _contains_exact(fact.predicate, clause, "math"):
                    status = "spoken_partial"
                if negative != source_negative:
                    status = "uncertain_match"
        elif fact.precision_sensitive:
            payload = fact.exact_text or fact.description.split(":", 1)[-1].strip()
            # Code can contain newlines; inspect the original segment, not its prose clauses.
            text = transcript_text if fact.notation == "code" else clause
            if _contains_exact(payload, text, fact.notation):
                status = "spoken_explicit" if negative == source_negative else "uncertain_match"
                if fact.relation_type == "table_value" and not all_subjects:
                    status = "spoken_partial"
                elif fact.relation_type == "table_value":
                    value_span = _token_span(_match_tokens(payload), tokens)
                    ordered = sorted(spans)
                    links = {"is", "are", "has", "have", "for", "of", "row", "column", "header", "text", "value", "equals"}
                    if (not value_span or len(set(ordered)) != len(ordered)
                            or any(left[1] > right[0] or set(tokens[left[1]:right[0]]) - links
                                   for left, right in zip(ordered, ordered[1:] + [value_span]))):
                        status = "spoken_partial"
        else:
            predicate = _match_tokens(fact.predicate or fact.description)
            if _token_span(predicate, tokens, content_only=True):
                status = "spoken_explicit" if negative == source_negative else "uncertain_match"
            else:
                needed = set(predicate) - _STOP_WORDS
                overlap = len(needed & set(tokens)) / max(1, len(needed))
                if all_subjects:
                    status = "spoken_partial"
                elif overlap >= 0.4:
                    status = "uncertain_match"
        if status == "spoken_explicit" and (fact.evidence_reliability == "partial" or fact.ambiguous_subject):
            status = "uncertain_match"
        if ranks[status] > ranks[best]:
            best, best_span = status, clause
    if contradicted:
        return "uncertain_match", best_span, fact.description
    return best, best_span, "" if best == "spoken_explicit" else fact.description


def reconcile_visual_evidence(
    vlm_snapshot: Dict[str, Any], transcript_text: str, *, max_facts: int = 64
) -> Dict[str, Any]:
    """Create a compact post-VLM/pre-synthesis transcript coverage plan."""

    evidence = vlm_snapshot.get("visual_evidence") if isinstance(vlm_snapshot, dict) else None
    frame_id = evidence.get("frame_id") if isinstance(evidence, dict) else None
    facts, bounds = extract_atomic_visual_facts(evidence or {}, max_facts=max_facts)
    clauses = _transcript_clauses(transcript_text)
    plan_facts = []
    for fact in facts:
        coverage, span, missing = _match_atomic_fact(fact, transcript_text, clauses=clauses)
        assert coverage in _COVERAGE_LABELS
        item = asdict(fact)
        item.pop("subject_groups", None)
        item.pop("relation_cues", None)
        item.pop("precision_sensitive", None)
        item.pop("directional", None)
        for key in ("predicate", "exact_text", "notation", "relation_type", "relation_label", "ambiguous_subject"):
            item.pop(key, None)
        if coverage == "uncertain_match":
            item["uncertainty_reasons"] = list(dict.fromkeys(
                item["uncertainty_reasons"]
                + ["Transcript support is ambiguous or visual evidence is only partial."]
            ))
        elif coverage == "not_usable":
            item["uncertainty_reasons"] = list(dict.fromkeys(
                item["uncertainty_reasons"]
                + ["Underlying visual evidence is unavailable or ambiguous."]
            ))
        item.update({
            "transcript_coverage": coverage,
            "supporting_transcript_span": span,
            "missing_information": missing or None,
        })
        plan_facts.append(item)
    counts = {label: 0 for label in sorted(_COVERAGE_LABELS)}
    for fact in plan_facts:
        counts[fact["transcript_coverage"]] += 1
    return {
        "schema": "synapse_transcript_visual_coverage_v1",
        "frame_id": frame_id,
        "method": "deterministic_conservative_v2",
        "fact_count": len(plan_facts),
        "coverage_counts": counts,
        "facts": plan_facts,
        **bounds,
    }


class Synchronizer:
    """
    Detect suffer points where blind students lose visual access.

    Uses a scoring system with multiple factors:
    - Deictic phrases (+50)
    - Silent drawing (+80)
    - High complexity (+30)
    - Negative deictic (-50)
    """

    # Deictic phrases to detect
    DEICTIC_PHRASES = [
        "look at", "here we see", "this graph", "notice", "observe",
        "check this", "see this", "here's", "look here"
    ]

    # Negative deictic phrases (indicate moving on)
    NEGATIVE_DEICTIC_PHRASES = [
        "next slide", "moving on", "let's continue", "moving forward",
        "now let's", "next we'll", "moving to", "next topic"
    ]

    # Scoring thresholds
    DEICTIC_SCORE = 50
    SILENT_DRAWING_SCORE = 80
    HIGH_COMPLEXITY_SCORE = 50
    HIGH_COMPLEXITY_THRESHOLD = 0.30
    NEGATIVE_DEICTIC_PENALTY = -50
    SUFFER_POINT_THRESHOLD = 50

    # Matching window (seconds) - aligns with context window
    MATCHING_WINDOW = 30.0

    # Redundancy window (seconds)
    REDUNDANCY_WINDOW = 5.0

    def __init__(
        self,
        transcript_path: str,
        keyframes_path: str
    ):
        """
        Initialize Synchronizer with transcript and keyframes data.

        Args:
            transcript_path: Path to transcript.json
            keyframes_path: Path to keyframes.json
        """
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)

        self.transcripts: List[TranscriptEntry] = []
        for entry in transcript_data.get("entries", []):
            self.transcripts.append(TranscriptEntry(
                start=entry.get("start", 0.0),
                end=entry.get("start", 0.0) + entry.get("duration", 0.0),
                text=entry.get("text", "")
            ))

        # Load keyframes
        with open(keyframes_path, 'r') as f:
            keyframes_data = json.load(f)

        # Handle both formats: list directly or dict with "keyframes" key
        if isinstance(keyframes_data, list):
            self.keyframes: List[Dict] = keyframes_data
        else:
            self.keyframes: List[Dict] = keyframes_data.get("keyframes", [])

    def detect_suffer_points(self) -> List[InterventionPoint]:
        """
        Detect suffer points using the scoring system.

        Returns:
            List of InterventionPoint objects
        """
        interventions = []
        last_intervention_time = None

        for frame in self.keyframes:
            # _calculate_score returns (score, trigger_reason, matched_segment) —
            # eliminates a redundant _match_segment call and gives accurate reasons
            score, trigger_reason, matched_segment = self._calculate_score(frame)

            # Check if score meets threshold OR has meaningful visual content
            complexity = frame.get("complexity_score", 0.0)

            # High confidence: score >= 100 (multiple triggers)
            # Medium confidence: score >= 50 (single trigger)
            # Low confidence: score < 50 but complexity > 0.25 (visual-only)
            meets_threshold = score >= self.SUFFER_POINT_THRESHOLD
            visual_only = complexity > 0.25 and not meets_threshold

            if meets_threshold or visual_only:
                # Check redundancy (skip if within 5s of previous)
                current_time = frame.get("timestamp_seconds", 0.0)

                if last_intervention_time is not None:
                    if abs(current_time - last_intervention_time) < self.REDUNDANCY_WINDOW:
                        continue  # Skip redundant intervention

                transcript_context = matched_segment.text if matched_segment else ""

                # Determine confidence level
                if score >= 100:
                    confidence = "high"
                elif meets_threshold:
                    confidence = "medium"
                else:
                    confidence = "low"

                raw_labels = frame.get("raw_category_labels") or [frame.get("content_type", "text")]
                normalization = normalize_category_labels(raw_labels)

                # Older keyframe manifests contain only ``clip_confidence``;
                # those values were complexity scores in the legacy pipeline
                # and must never be presented as classifier confidence.
                classification_confidence = frame.get("classification_confidence")
                confidence_semantics = frame.get("confidence_semantics")
                if classification_confidence is None:
                    confidence_semantics = confidence_semantics or "legacy_complexity"
                else:
                    confidence_semantics = confidence_semantics or "classifier"

                interventions.append(InterventionPoint(
                    timestamp=current_time,
                    frame_path=frame.get("frame_path", ""),
                    content_type=frame.get("content_type", "text"),
                    complexity_score=complexity,
                    clip_confidence=frame.get("clip_confidence"),
                    transcript_context=transcript_context,
                    trigger_reason=trigger_reason,
                    confidence=confidence,
                    classification_confidence=classification_confidence,
                    confidence_semantics=confidence_semantics,
                    raw_category_labels=list(normalization.raw_labels),
                    canonical_categories=list(normalization.canonical_categories),
                    category_mapping_notes=list(normalization.notes),
                ))

                last_intervention_time = current_time

        return interventions

    def _calculate_score(self, frame: Dict) -> tuple:
        """
        Calculate intervention score for a frame.

        Returns (score, trigger_reason, matched_segment) to avoid a redundant
        _match_segment call in detect_suffer_points and to report accurate
        component-based trigger reasons instead of inaccurate score-based guesses.
        """
        score = 0
        triggered = []

        frame_time = frame.get("timestamp_seconds", 0.0)
        matched_segment = self._match_segment(frame_time)

        if matched_segment:
            transcript_text = matched_segment.text.lower()

            for phrase in self.DEICTIC_PHRASES:
                if phrase in transcript_text:
                    score += self.DEICTIC_SCORE
                    triggered.append("Deictic")
                    break

            for phrase in self.NEGATIVE_DEICTIC_PHRASES:
                if phrase in transcript_text:
                    score += self.NEGATIVE_DEICTIC_PENALTY
                    triggered.append("Negative Deictic")
                    break

        if matched_segment and len(matched_segment.text.strip()) < 10:
            complexity = frame.get("complexity_score", 0.0)
            if complexity > 0.3:
                score += self.SILENT_DRAWING_SCORE
                triggered.append("Silent Drawing")

        complexity = frame.get("complexity_score", 0.0)
        if complexity > self.HIGH_COMPLEXITY_THRESHOLD:
            score += self.HIGH_COMPLEXITY_SCORE
            triggered.append("High Complexity")

        trigger_reason = " + ".join(triggered) if triggered else "Unknown"
        return score, trigger_reason, matched_segment

    def _match_segment(self, timestamp: float) -> Optional[TranscriptEntry]:
        """
        Find transcript segment containing timestamp, or nearest within 30s.

        Falls back to nearest segment endpoint when timestamp falls in a gap
        between segments (e.g., pauses between sentences).
        """
        for segment in self.transcripts:
            if segment.start <= timestamp <= segment.end:
                return segment

        # Fallback: nearest segment endpoint within 30s (handles inter-segment gaps)
        if not self.transcripts:
            return None
        nearest = min(
            self.transcripts,
            key=lambda s: min(abs(s.start - timestamp), abs(s.end - timestamp))
        )
        if min(abs(nearest.start - timestamp), abs(nearest.end - timestamp)) <= self.MATCHING_WINDOW:
            return nearest
        return None


