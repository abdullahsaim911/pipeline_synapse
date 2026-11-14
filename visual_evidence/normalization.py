"""Conservative occurrence normalization at the observation assembly boundary.

No image interpretation or fuzzy text matching: equivalent records must share a
location anchor and exact content/structure. Raw observations remain with the caller.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import TYPE_CHECKING, Any

from .models import ValidationDiagnostic

if TYPE_CHECKING:
    from .assembler import FrameObservation


# A literal singular panel location in model prose can anchor a region when
# coordinates were omitted. Do not interpret generic descriptions or labels.
_PANEL_POSITION = re.compile(
    r"^(leftmost|rightmost|topmost|bottommost|middle|central|upper[- ]left|"
    r"upper[- ]right|lower[- ]left|lower[- ]right)\s+"
    r"(panel|diagram|graph|chart|table|image|block|scene)\b", re.IGNORECASE,
)


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _unique(values):
    return list(dict.fromkeys(values))


def normalize_observation(observation: FrameObservation) -> tuple[FrameObservation, dict[str, str], bool]:
    """Return a copied observation, removed-ref aliases, and unresolved-conflict flag.

    Indexing and graph comparisons are bounded by input size. Groups are processed
    together, rather than comparing every region/element pair or retrying merges.
    """
    result = observation.model_copy(deep=True)
    entities = {}
    for region in result.regions:
        for entity in [region, *region.elements]:
            if entity.ref is not None:
                if entity.ref in entities:
                    raise ValueError(f"duplicate observation ref '{entity.ref}'")
                entities[entity.ref] = entity
    relation_refs = [relation.ref for relation in result.relations if relation.ref is not None]
    if len(relation_refs) != len(set(relation_refs)):
        raise ValueError("duplicate observation relation ref")
    if set(relation_refs) & entities.keys():
        raise ValueError("observation refs must not identify both an entity and a relation")

    unresolved = False
    reported = set()

    def report(code, message, target="scene"):
        nonlocal unresolved
        if code not in reported:
            result.diagnostics.append(ValidationDiagnostic(
                code=f"normalization.{code}", message=message, target_id=target,
            ))
            reported.add(code)
        if code != "consolidated":
            unresolved = True

    def region_anchor(region):
        if region.bbox_normalized is not None:
            return ("bbox", tuple(region.bbox_normalized.as_list()))
        position = region.position or region.attributes.structural_description or ""
        match = _PANEL_POSITION.match(position.strip())
        return ("panel_position", match[0].lower()) if match else None

    def element_content(element, local):
        content = element.model_dump(mode="json", exclude={"ref", "category_ids"})
        for field in ("parent_ref", "source_ref"):
            content[field] = local.get(content[field], content[field])
        content["child_refs"] = [local.get(ref, ref) for ref in element.child_refs]
        return content

    groups = []
    region_buckets = defaultdict(list)
    anchor_contents = defaultdict(set)
    for region in result.regions:
        anchor = region_anchor(region)
        if anchor is None or region.ref is None or not region.elements:
            continue
        local = {region.ref: ("region",)}
        local.update({element.ref: ("element", i) for i, element in enumerate(region.elements) if element.ref})
        content = region.model_dump(mode="json", exclude={"ref", "elements", "category_ids", "reading_order_index"})
        content["elements"] = [element_content(element, local) for element in region.elements]
        signature = _key(content)
        nominal_identity = tuple((element.kind, element.label) for element in region.elements)
        anchor_contents[(anchor, nominal_identity)].add(signature)
        region_buckets[(anchor, signature)].append(region)
    for signatures in anchor_contents.values():
        if len(signatures) > 1:
            report("identity_conflict", "Records at the same region anchor differ; retained separate observations.")
    for bucket in region_buckets.values():
        if len(bucket) > 1 and all(element.ref for region in bucket for element in region.elements):
            # Each member maps its region and ordered elements to the same slots.
            groups.append([[region.ref, *[element.ref for element in region.elements]] for region in bucket])

    # Within one owner, only an exact bounding box anchors an element occurrence.
    # Text, type, a repeated cell value, or a coarse 'center' position cannot do so.
    grouped_regions = {member[0] for group in groups for member in group}
    for region in result.regions:
        if region.ref in grouped_regions:
            continue  # Do not overlap region and element merge proposals.
        buckets = defaultdict(list)
        locations = defaultdict(set)
        for element in region.elements:
            if element.ref and element.bbox_normalized is not None:
                content = element_content(element, {})
                signature = _key(content)
                location = (tuple(element.bbox_normalized.as_list()), element.kind, element.label)
                locations[location].add(signature)
                buckets[signature].append([element.ref])
        if any(len(signatures) > 1 for signatures in locations.values()):
            report("identity_conflict", "Records at the same element anchor differ; retained separate observations.")
        groups.extend(bucket for bucket in buckets.values() if len(bucket) > 1)

    proposed = {ref: representative for group in groups for member in group
                for ref, representative in zip(member, group[0])}
    incidence = defaultdict(set)
    blocked = set()
    for relation in result.relations:
        source = proposed.get(relation.source_ref, relation.source_ref)
        target = proposed.get(relation.target_ref, relation.target_ref)
        if source == target and relation.source_ref != relation.target_ref:
            blocked.update((relation.source_ref, relation.target_ref))
        semantics = (relation.relation, relation.label, relation.directed, relation.evidence_status)
        incidence[relation.source_ref].add(("source", relation.target_ref, *semantics))
        incidence[relation.target_ref].add(("target", relation.source_ref, *semantics))
    # Parent/child and source associations from outside a group also distinguish
    # occurrences. Include incoming references as well as outgoing content.
    for region in result.regions:
        for element in region.elements:
            for target, role in [(element.parent_ref, "parent"), (element.source_ref, "source"),
                                 *[(ref, "child") for ref in element.child_refs]]:
                if target in entities:
                    mapped = proposed.get(element.ref, element.ref)
                    incidence[target].add((role, element.ref))
                    if mapped == proposed.get(target, target) and element.ref != target:
                        blocked.update((element.ref, target))

    aliases = {}
    uncertain_refs = {item.target_ref for item in result.uncertainties}
    for group in groups:
        local = {ref: representative for member in group for ref, representative in zip(member, group[0])}

        def neighbors(ref):
            # External refs remain literal: a failed proposal elsewhere cannot
            # accidentally authorize this merge through a tentative alias.
            return {(role, local.get(target, target), *semantics)
                    for role, target, *semantics in incidence[ref]}

        if any(
            ref in uncertain_refs or entities[ref].evidence_status not in (None, "supported")
            for member in group for ref in member
        ) or "scene" in uncertain_refs or result.structural_description.uncertainties:
            report("uncertain_identity", "Duplicate candidates include uncertain observations; retained separate identities.")
            continue
        if any(ref in blocked for member in group for ref in member) or any(
            neighbors(ref) != neighbors(representative)
            for member in group for ref, representative in zip(member, group[0])
        ):
            report("graph_identity_conflict", "Matching content has distinct connectivity or would collapse an edge; retained both occurrences.")
            continue
        for member in group[1:]:
            for ref, representative in zip(member, group[0]):
                aliases[ref] = representative
                entities[representative].category_ids = _unique(
                    entities[representative].category_ids + entities[ref].category_ids
                )
            first, other = entities[group[0][0]], entities[member[0]]
            if hasattr(first, "reading_order_index") and first.reading_order_index != other.reading_order_index:
                report("order_conflict", "Equivalent panels have different order indices; retained the first index and original observations.", first.ref)

    def resolve(ref):
        return aliases.get(ref, ref)

    result.regions = [region for region in result.regions if region.ref not in aliases]
    for region in result.regions:
        region.elements = [element for element in region.elements if element.ref not in aliases]
        for element in region.elements:
            element.parent_ref = resolve(element.parent_ref)
            element.child_refs = _unique(resolve(ref) for ref in element.child_refs)
            element.source_ref = resolve(element.source_ref)

    # Collapse only identical claims after remapping. Cross-category flags and
    # source provenance are additive. Conflicting statuses are made explicit.
    relations = {}
    relation_sources = {}
    relation_aliases = {}
    for relation in result.relations:
        relation.source_ref = resolve(relation.source_ref)
        relation.target_ref = resolve(relation.target_ref)
        key = (relation.source_ref, relation.target_ref, relation.relation, relation.label, relation.directed)
        if key not in relations:
            relations[key] = relation
            relation_sources[key] = dict.fromkeys(relation.source_refs)
            continue
        first = relations[key]
        if first.ref is None or relation.ref is None:
            # Unnamed claims still share one canonical representative.
            if first.ref is None:
                first.ref = relation.ref
        else:
            relation_aliases[relation.ref] = first.ref
        relation_sources[key].update(dict.fromkeys(relation.source_refs))
        first.cross_category = first.cross_category or relation.cross_category
        if first.evidence_status != relation.evidence_status:
            first.evidence_status = "ambiguous_or_conflicting"
            report("relation_status_conflict", "Equivalent relation claims have conflicting support statuses; preserved as ambiguous.", first.ref or "scene")
    result.relations = list(relations.values())
    aliases.update(relation_aliases)
    for key, relation in relations.items():
        relation.source_refs = _unique(resolve(ref) for ref in relation_sources[key])
    for region in result.regions:
        for element in region.elements:
            element.source_ref = resolve(element.source_ref)
    structure = result.structural_description
    for field in ("component_refs", "relation_refs", "reading_order_refs"):
        setattr(structure, field, _unique(resolve(ref) for ref in getattr(structure, field)))
    for item in result.uncertainties:
        item.target_ref = resolve(item.target_ref)
    for diagnostic in result.diagnostics:
        diagnostic.target_id = resolve(diagnostic.target_id)
    exact = []
    seen_exact = set()
    for item in result.exact_content:
        item.source_ref = resolve(item.source_ref)
        key = (item.text, item.status, item.notation, item.source_ref)
        if item.source_ref in entities and key in seen_exact:
            continue
        seen_exact.add(key)
        exact.append(item)
    result.exact_content = exact
    if aliases or len(result.relations) < len(observation.relations) or len(exact) < len(observation.exact_content):
        removed_elements = sum(len(region.elements) for region in observation.regions) - sum(
            len(region.elements) for region in result.regions
        )
        report("consolidated", (
            f"Consolidated {len(observation.regions) - len(result.regions)} regions, "
            f"{removed_elements} elements, {len(observation.relations) - len(result.relations)} relations "
            f"and {len(observation.exact_content) - len(exact)} exact-content records; "
            "retained original evidence and alias provenance."
        ))
    return result, aliases, unresolved
