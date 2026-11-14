"""Bounded focused readings of precision-sensitive evidence using the same VLM.

The overview stays archived. Focused readings update only their named canonical
records; conflicts stay explicit and no scientific correction is inferred.
"""

from copy import deepcopy
import json
import re
import time

from PIL import Image

from stage_tracing import trace_stage
from visual_evidence.models import VisualEvidence, CanonicalCategory
from visual_evidence.notation import equivalent_math_spelling


MAX_FOCUSED_CHECKS = 3
RADIAL_SCHEMA = {
    "type": "object", "properties": {
        "pattern": {"enum": ["radial", "other", "unclear"]},
        "arrowhead_location": {"enum": ["near_center", "outer_ends", "mixed", "unclear"]},
        "direction": {"enum": ["toward_center", "away_from_center", "mixed", "unclear"]},
        "center_description": {"type": "string"},
        "arrow_color": {"type": "string"},
        "visible_evidence": {"type": "string"},
    },
    "required": ["pattern", "arrowhead_location", "direction", "center_description", "arrow_color", "visible_evidence"],
    "additionalProperties": False,
}
EXACT_SCHEMA = {
    "type": "object", "properties": {
        "readable": {"type": "boolean"}, "text": {"type": "string"},
        "uncertainty": {"type": "string"},
    }, "required": ["readable", "text", "uncertainty"], "additionalProperties": False,
}
EDGE_SCHEMA = {
    "type": "object", "properties": {
        "connected": {"type": "boolean"},
        "arrowhead_at": {"enum": ["source", "target", "both", "unclear"]},
        "visible_evidence": {"type": "string"},
    }, "required": ["connected", "arrowhead_at", "visible_evidence"], "additionalProperties": False,
}
REGION_SCHEMA = {
    "type": "object", "properties": {
        "description": {"type": "string"},
        "categories": {"type": "array", "minItems": 1, "maxItems": 2,
                       "items": {"enum": [f"CT-{i:02d}" for i in range(1, 13)]}},
        "uncertainty": {"type": "string"},
    }, "required": ["description", "categories", "uncertainty"], "additionalProperties": False,
}


def _direction(text):
    inward = bool(re.search(r"\binward|\btoward|\bconverg", text, re.I))
    outward = bool(re.search(r"\boutward|\baway|\bemanat|\bradiat", text, re.I))
    return ("toward_center" if inward else "away_from_center") if inward != outward else None


def _union(boxes):
    boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    return ([min(box[0] for box in boxes), min(box[1] for box in boxes),
             max(box[2] for box in boxes), max(box[3] for box in boxes)] if boxes else None)


def _targets(evidence):
    radial, exact, edges, seen = [], [], [], set()
    nodes = {node["id"]: node for region in evidence.get("regions", []) for node in region.get("elements", [])}
    region_ids = {node["id"]: region["id"] for region in evidence.get("regions", []) for node in region.get("elements", [])}
    for region in evidence.get("regions", []):
        region_box = _union([region.get("bbox_normalized"),
                             *[node.get("bbox_normalized") for node in region.get("elements", [])]])
        for node in region.get("elements", []):
            description = str(node.get("description") or "")
            if re.search(r"\bradial\b|\brays\b|\bradiating\b", description, re.I):
                box = region_box or node.get("bbox_normalized")
                if box and (box[2] - box[0]) * (box[3] - box[1]) < 0.85:
                    key = ("radial", tuple(round(part, 3) for part in box))
                    if key not in seen:
                        radial.append({"kind": "radial", "ref": node["id"], "box": box, "region": region["id"]})
                        seen.add(key)
            records = [record for record in evidence.get("exact_content", []) if record.get("source_ref") == node["id"]]
            for record in records:
                precision = record.get("notation") in {"math", "code", "chemical"}
                numeric = bool(re.fullmatch(r"[+\-−]?\d+(?:[.,]\d+)?\s*[%°A-Za-z/²³]*", str(record.get("text") or "").strip()))
                box = node.get("bbox_normalized")
                small_label = bool(box and (box[2] - box[0]) * (box[3] - box[1]) < 0.02
                                   and len(str(record.get("text") or "").strip()) >= 3)
                if (precision or numeric or small_label) and box:
                    key = ("exact", node["id"])
                    if key not in seen:
                        text = str(record.get("text") or "")
                        operations = len(re.findall(r"[=+*/\u2212\u222b\u2211\u03a3\u221a^]|\\(?:frac|int|sum)", text))
                        tier = (0 if record.get("notation") in {"code", "chemical"}
                                else 1 if record.get("notation") == "math" and operations > 1 else 2)
                        exact.append({"kind": "exact", "ref": node["id"], "box": node["bbox_normalized"],
                                      "region": region["id"], "notation": record.get("notation", "plain"),
                                      "record": record, "priority": (tier, -operations, -len(text))})
                        seen.add(key)
    for edge in evidence.get("relations", []):
        source, target = nodes.get(edge["source_id"]), nodes.get(edge["target_id"])
        if (not edge.get("directed") or edge.get("relation") not in {"points_to", "leads_to"}
                or not source or not target or not source.get("bbox_normalized") or not target.get("bbox_normalized")):
            continue
        # Radial groups have their own geometry-specific check. Avoid spending
        # another slot on the same arrows using a less specific question.
        if any(item["ref"] in {source["id"], target["id"]} for item in radial):
            continue
        box = _union([source["bbox_normalized"], target["bbox_normalized"]])
        if (box[2] - box[0]) * (box[3] - box[1]) < 0.85:
            edges.append({"kind": "edge", "ref": edge["id"], "box": box,
                          "region": region_ids[source["id"]], "record": edge,
                          "source_box": source["bbox_normalized"], "target_box": target["bbox_normalized"]})
    candidates = radial + edges + sorted(exact, key=lambda target: target["priority"])
    # Cover distinct meaningful panels before taking more checks in one panel.
    first, rest, regions = [], [], set()
    for target in candidates:
        (rest if target["region"] in regions else first).append(target)
        regions.add(target["region"])
    return first + rest


def _pixel_box(image, box):
    # Include nearby labels/arrow tips without inventing an ungrounded quadrant.
    left, top, right, bottom = box
    pad_x, pad_y = max(4 / image.width, (right - left) * 0.06), max(4 / image.height, (bottom - top) * 0.06)
    return (max(0, int((left - pad_x) * image.width)), max(0, int((top - pad_y) * image.height)),
            min(image.width, int((right + pad_x) * image.width) + 1),
            min(image.height, int((bottom + pad_y) * image.height) + 1))


def _crop(image, box):
    cropped = image.crop(_pixel_box(image, box))
    scale = 1344 / max(cropped.size)
    return cropped.resize((max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))), Image.Resampling.LANCZOS)


def refine_critical_details(snapshot, frame_path, interface, *, ownership_check=None):
    """Inspect small radial-arrow groups and exact-content blocks, at most three calls.

    Frames without grounded crop bounds remain unchanged. Ownership is checked
    between calls and after responses so cancellation cannot publish a late result.
    """
    try:
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError):
        return snapshot
    evidence = deepcopy(snapshot.get("visual_evidence") or {})
    targets = _targets(evidence)
    from .region_grounding import uncovered_panels
    panels = uncovered_panels(image, evidence)
    declared_ids = {region["id"] for region in evidence.get("regions", [])}
    declared_ids.update(node["id"] for region in evidence.get("regions", []) for node in region.get("elements", []))
    declared_ids.update(edge["id"] for edge in evidence.get("relations", []))
    for index, box in enumerate(panels[:1]):
        suffix = index + 1
        while f"focused_panel_{suffix}" in declared_ids or f"focused_group_{suffix}" in declared_ids:
            suffix += 1
        ref = f"focused_panel_{suffix}"
        targets.insert(0, {"kind": "region", "ref": ref, "region": ref, "box": box,
                           "element_ref": f"focused_group_{suffix}"})
    if not targets:
        return snapshot
    checks = []
    nodes = {node["id"]: node for region in evidence["regions"] for node in region.get("elements", [])}
    nodes.update({edge["id"]: edge for edge in evidence.get("relations", [])})
    nodes.update({target["ref"]: {"id": target["ref"]} for target in targets if target["kind"] == "region"})
    for target in targets[:MAX_FOCUSED_CHECKS]:
        if ownership_check is not None:
            ownership_check()
        node = nodes[target["ref"]]
        started = time.perf_counter()
        check = {"kind": target["kind"], "target_ref": target["ref"], "crop_bounds": target["box"],
                 "original_description": node.get("description")}
        if target["kind"] == "region":
            prompt = (
                "Describe the visible structure of this panel, cropped along an observed divider. "
                "Identify grouping, repeated patterns, spatial arrangement and what is visibly connected. "
                "Do not infer its scientific purpose, motion, history, causes, or teacher intent. "
                "Do not invent tiny text or identities: state which details are too small to read. "
                "Describe the whole panel rather than a single example. If the fine layout inside repeated blocks "
                "is unclear, describe their repetition without guessing which side contains text. "
                "Choose at most two confidently supported broad categories, not every category in the schema. "
                "A grid is not a data table without readable headers or cell associations. When fine content "
                "cannot be identified, use the observational image category for the visible pattern. "
                "Category meanings: " + "; ".join(f"{item.value} = {item.name.lower().replace('_', ' ')}" for item in CanonicalCategory)
            )
            schema, limit = REGION_SCHEMA, 768
        elif target["kind"] == "radial":
            prompt = (
                "Inspect ONLY the visible arrow geometry in this magnified region. Ignore titles and textbook expectations. "
                "Is there a radial group around a central object? Are the actual arrow tips near the center, at the outer ends, "
                "mixed, or unclear? State direction from those tips, not from the way shafts radiate. "
                "Give a short static physical description of the center, the arrow color and the visible evidence. "
                "Do not infer orbiting, motion, forces or energy flow. Choose unclear when tips cannot be read."
            )
            schema, limit = RADIAL_SCHEMA, 384
        elif target["kind"] == "edge":
            def relative(box):
                crop = _pixel_box(image, target["box"])
                return [round((box[0] * image.width - crop[0]) / (crop[2] - crop[0]), 4),
                        round((box[1] * image.height - crop[1]) / (crop[3] - crop[1]), 4),
                        round((box[2] * image.width - crop[0]) / (crop[2] - crop[0]), 4),
                        round((box[3] * image.height - crop[1]) / (crop[3] - crop[1]), 4)]
            prompt = (
                "Inspect only the visible connector between the two named endpoints. "
                f"Within this crop, source normalized bounds are {relative(target['source_box'])} "
                f"and target normalized bounds are {relative(target['target_box'])}. "
                "Are they visibly connected by an arrow, and at which endpoint is its actual pointed tip? "
                "Use source or target as positional names, not a presumed direction. Do not infer flow from labels or topic. "
                "Use unclear if either endpoint or arrowhead cannot be read."
            )
            schema, limit = EDGE_SCHEMA, 384
        else:
            prompt = (
                f"Read the visible {target['notation']} content in this magnified region verbatim. "
                "Transcribe the complete visible expression or text block, not merely its final result. "
                "The image is authoritative; do not convert Unicode into LaTeX or change formatting. "
                "Preserve signs, subscripts, superscripts, grouping, units, capitalization, line breaks and code indentation. "
                "Do not solve, balance, complete or correct the source. 'readable' means the visible characters can be read, "
                "not that their meaning, scientific validity or context is known. Vector arrows and styled symbols are readable notation. "
                "Use readable=true when characters are clear, text for their transcription, and uncertainty='' when none is visually unclear. "
                "Use readable=false only when essential characters are visually illegible. Keep uncertainty to one short visual reason."
            )
            schema, limit = EXACT_SCHEMA, 2048
        # Decoder constraints do not teach the model the response format. Give
        # it the same contract explicitly so it can finish all required fields.
        prompt += " Return only a JSON object matching this schema: " + json.dumps(schema, separators=(",", ":"))
        try:
            response = interface.analyze(_crop(image, target["box"]), prompt, {
                "max_new_tokens": limit, "temperature": 0.0, "_response_schema": schema,
            })
            check.update(raw_response=response.text, model=response.model,
                         finish_reason=response.finish_reason, completion_tokens=response.eval_count,
                         latency_ms=response.total_duration_ms)
            if response.finish_reason not in {None, "stop"} or (response.finish_reason is None and response.eval_count >= limit):
                raise ValueError("Focused response was incomplete")
            result = json.loads(response.text)
            if not isinstance(result, dict):
                raise ValueError("Focused response is not an object")
        except Exception as exc:
            result = {}
            check["error_type"] = type(exc).__name__
        if ownership_check is not None:
            ownership_check()
        if target["kind"] == "region":
            categories = result.get("categories")
            usable = (isinstance(result.get("description"), str) and bool(result["description"].strip())
                      and isinstance(categories, list) and 1 <= len(categories) <= 2
                      and all(isinstance(value, str) and value in {f"CT-{i:02d}" for i in range(1, 13)} for value in categories))
            if usable:
                position = "left panel" if target["box"][0] == 0 else "right panel"
                evidence["regions"].append({
                    "id": target["ref"], "category_ids": list(dict.fromkeys(categories)), "position": position,
                    "bbox_normalized": target["box"], "evidence_status": "partially_supported",
                    "attributes": {"structural_description": result["description"]},
                    "elements": [{"id": target["element_ref"], "kind": "visual_group",
                                  "category_ids": list(dict.fromkeys(categories)), "parent_id": target["ref"],
                                  "description": result["description"], "bbox_normalized": target["box"],
                                  "evidence_status": "partially_supported"}],
                })
                evidence["category_ids"] = list(dict.fromkeys(evidence.get("category_ids", []) + categories))
                check["original_global_layout"] = evidence.get("global_layout")
                check["original_structure_overview"] = (evidence.get("structural_description") or {}).get("overview")
                evidence["global_layout"] = "Separate left and right panels divided by a visible vertical line."
                evidence.setdefault("structural_description", {})["overview"] = evidence["global_layout"]
                evidence.setdefault("uncertainties", []).append({"target_id": target["ref"],
                    "reason": result.get("uncertainty") or "An omitted panel was recovered by a separate focused observation.",
                    "impact": "Panel-level structure is available; exact small content is not independently established."})
            evidence["extraction_status"] = "partial"
            evidence["evidence_status"] = "partially_supported"
            evidence.setdefault("validation_diagnostics", []).append({
                "code": "focused.recovered_panel" if usable else "focused.unverified_panel",
                "message": "A visible divided panel was not covered by the overview; focused observation " + ("recovered its structure." if usable else "did not establish its content."),
                "target_id": target["ref"] if usable else "scene",
            })
            check.update(usable=usable, changed=usable)
            checks.append(check)
            continue
        if check.get("error_type"):
            # An unavailable second reader says nothing about source legibility.
            # Preserve the observation with partial support instead of deleting
            # it or claiming that the visible source became unreadable.
            if node.get("evidence_status") == "supported":
                node["evidence_status"] = "partially_supported"
            if target["kind"] == "exact" and target["record"].get("status") == "supported":
                target["record"]["status"] = "partially_supported"
            evidence["extraction_status"] = "partial"
            evidence["evidence_status"] = "partially_supported"
            evidence.setdefault("uncertainties", []).append({"target_id": node["id"],
                "reason": "Independent focused verification did not complete; the overview reading remains unverified.",
                "impact": "Do not present this detail as independently confirmed."})
            check.update(usable=False, changed=False, verification_failed=True)
            checks.append(check)
            continue
        if target["kind"] == "radial":
            direction = {"near_center": "toward_center", "outer_ends": "away_from_center"}.get(result.get("arrowhead_location"))
            usable = result.get("pattern") == "radial" and direction is not None and result.get("direction") == direction
            old_direction = _direction(str(node.get("description") or "") + " " + str(node.get("attributes", {}).get("direction") or ""))
            changed = usable and old_direction is not None and old_direction != direction
            if usable:
                color = str(result.get("arrow_color") or "").strip()
                center = str(result.get("center_description") or "central object").strip()
                location, pointing = ("near the center", "toward the center") if direction == "toward_center" else ("at the outer ends", "away from the center")
                node["description"] = f"Radial arrows ({color}) around {center}. Their arrowheads are {location}, indicating direction {pointing}."
                if changed:
                    node["description"] = f"Radial arrows ({color}) around {center}. Their arrowheads appear {location}, suggesting direction {pointing}; the direction remains uncertain because the overview and focused readings disagree."
                node.setdefault("attributes", {})["direction"] = direction
                node["evidence_status"] = "partially_supported" if changed else "supported"
                # Bind the crop's center appearance only when the existing
                # geometry identifies exactly one object. Do not leave a stale
                # overview color/shape beside the newly read center description.
                box = node.get("bbox_normalized")
                if box and result.get("center_description"):
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    candidates = [candidate for region in evidence["regions"] if region["id"] == target["region"]
                                  for candidate in region.get("elements", [])
                                  if candidate.get("kind") == "object" and candidate.get("bbox_normalized")
                                  and candidate["bbox_normalized"][0] <= cx <= candidate["bbox_normalized"][2]
                                  and candidate["bbox_normalized"][1] <= cy <= candidate["bbox_normalized"][3]]
                    if len(candidates) == 1:
                        central = candidates[0]
                        check["center_ref"] = central["id"]
                        check["original_center_description"] = central.get("description")
                        central["description"] = f"The central object appears as {center}."
                        central["evidence_status"] = "partially_supported"
            else:
                node["description"] = "Visible line or arrow group; its direction could not be verified."
                node.setdefault("attributes", {}).pop("direction", None)
                node["evidence_status"] = "ambiguous_or_conflicting"
            if changed or not usable:
                # Direction does not establish the overview's endpoint claims.
                for relation in evidence.get("relations", []):
                    if node["id"] in {relation["source_id"], relation["target_id"]} and (
                        relation.get("directed") or relation.get("relation") == "points_to"
                    ):
                        relation["evidence_status"] = "ambiguous_or_conflicting"
                for region in evidence["regions"]:
                    if region["id"] == target["region"]:
                        text = str(region.get("attributes", {}).get("structural_description") or "")
                        if _direction(text):
                            region.get("attributes", {}).pop("structural_description", None)
                overview = evidence.get("structural_description", {})
                if _direction(str(overview.get("overview") or "")):
                    overview["overview"] = f"The frame contains {len(evidence['regions'])} visual regions."
                if _direction(str(evidence.get("global_layout") or "")):
                    evidence.pop("global_layout", None)
        elif target["kind"] == "edge":
            usable = result.get("connected") is True and result.get("arrowhead_at") == "target"
            changed = result.get("connected") is True and result.get("arrowhead_at") in {"source", "both"}
            if not usable:
                node["evidence_status"] = "ambiguous_or_conflicting"
        else:
            usable = result.get("readable") is True and isinstance(result.get("text"), str) and bool(result["text"].strip())
            old_text = target["record"]["text"]
            check["original_text"] = old_text
            same_spelling = (equivalent_math_spelling(old_text, result.get("text", ""))
                             if target["notation"] == "math" else
                             result.get("text", "").replace("\r\n", "\n") == old_text.replace("\r\n", "\n"))
            changed = usable and not same_spelling
            target["record"]["status"] = "ambiguous_or_conflicting" if changed else "supported" if usable else "unavailable_or_unreadable"
            if changed or not usable:
                # A second transcription is not an authority on a disagreement.
                # Retain the original and alternative in the audit, but do not
                # publish either conflicting value as the verified exact text.
                node["evidence_status"] = "ambiguous_or_conflicting"
                node.pop("exact_text", None)
                node.pop("label", None)
                node["attributes"] = {}
            if changed or not usable:
                # A free-form overview description can paraphrase the old
                # operator/value. Never leave it competing with the exact
                # reading; the original stays in this check's audit record.
                kind = str(node.get("kind") or "visual element").replace("_", " ")
                node["description"] = (
                    f"Visible {kind}; its exact content differs between readings."
                    if changed else f"Visible {kind}; its exact content could not be verified."
                )
        check.update(usable=usable, changed=changed)
        checks.append(check)
        if changed or not usable:
            evidence["extraction_status"] = "partial"
            evidence["evidence_status"] = "partially_supported"
            reason = ("Overview and focused reading disagree; retain uncertainty about this detail."
                      if changed else "The focused view did not establish this detail clearly.")
            evidence.setdefault("uncertainties", []).append({"target_id": node["id"], "reason": reason,
                                                            "impact": "Affects precision or directional interpretation."})
            evidence.setdefault("validation_diagnostics", []).append({
                "code": "focused.conflict" if changed else "focused.unreadable", "message": reason, "target_id": node["id"],
            })
        trace_stage("qwen_focused_check", "complete", started_at=started, kind=target["kind"], usable=usable, changed=changed)
    if len(targets) > MAX_FOCUSED_CHECKS:
        evidence["extraction_status"] = "partial"
        evidence["evidence_status"] = "partially_supported"
        evidence.setdefault("validation_diagnostics", []).append({
            "code": "focused.check_limit", "message": "Additional precision-sensitive details were not independently checked.", "target_id": "scene",
        })
    result = dict(snapshot)
    if "_structured_observation" not in snapshot:
        result["_overview_visual_evidence"] = deepcopy(snapshot.get("visual_evidence"))
    result["visual_evidence"] = VisualEvidence.model_validate(evidence).to_schema_dict()
    result["_focused_checks"] = checks
    result["_vlm_metrics"] = dict(snapshot.get("_vlm_metrics") or {}, focused_check_count=len(checks),
                                  focused_check_latency_ms=sum(check.get("latency_ms", 0) for check in checks))
    return result
