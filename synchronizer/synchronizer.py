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
    "points_to": ["points to", "arrow to", "leader line"],
    "leads_to": ["leads to", "flows to", "then"],
    "branches_to": ["branches to", "branch", "yes", "no"],
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
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> List[str]:
    return re.findall(
        r">=|<=|==|!=|[=<>≤≥≠+*/^\-]|[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?",
        _normalized_text(value),
    )


def _content_tokens(value: Any) -> List[str]:
    return [token for token in _tokens(value) if token not in _STOP_WORDS]


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


def extract_atomic_visual_facts(
    visual_evidence: Dict[str, Any], *, max_facts: int = 64
) -> tuple[List[AtomicVisualFact], Dict[str, Any]]:
    """Extract bounded atomic facts from canonical evidence without inference."""

    if not isinstance(visual_evidence, dict):
        return [], {"truncated": False, "omitted_fact_count": 0}

    root_status = visual_evidence.get("evidence_status", "unavailable_or_unreadable")
    extraction_status = visual_evidence.get("extraction_status", "failed")
    root_categories = list(visual_evidence.get("category_ids") or [])
    unusable_root = extraction_status in {"failed", "not_attempted"}
    uncertainty_by_target: Dict[str, List[str]] = {}
    for item in visual_evidence.get("uncertainties") or []:
        if isinstance(item, dict):
            uncertainty_by_target.setdefault(item.get("target_id", "scene"), []).append(
                str(item.get("reason") or "Unspecified visual uncertainty")
            )

    candidates: List[AtomicVisualFact] = []
    names: Dict[str, str] = {}
    categories_by_ref: Dict[str, List[str]] = {}
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
        if unusable_root or status == "unavailable_or_unreadable":
            reliability = "unavailable"
        elif status == "ambiguous_or_conflicting":
            reliability = "ambiguous"
        elif status == "partially_supported" or extraction_status == "partial":
            reliability = "partial"
        else:
            reliability = "supported"
        candidates.append(AtomicVisualFact(
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
        ))
        next_number += 1

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
            add_fact(
                "component", f"Visible {kind}: {name}", [element_id], categories,
                "important", status, subject_groups=[_content_tokens(name)],
            )
            if element.get("position"):
                add_fact(
                    "spatial", f"{name} is {element['position']}", [element_id], categories,
                    "critical", status, subject_groups=[_content_tokens(name)],
                    relation_cues=_content_tokens(element["position"]),
                )
            if element.get("exact_text"):
                notation = "code" if "CT-08" in categories else "math" if "CT-02" in categories else "exact"
                add_fact(
                    "exact_content", f"{notation}: {element['exact_text']}", [element_id],
                    categories, "critical", status,
                    subject_groups=[_content_tokens(element["exact_text"])],
                    precision_sensitive=True,
                )
            for key, value in (element.get("attributes") or {}).items():
                if value is None or isinstance(value, (dict, list)):
                    continue
                importance = "critical" if key in {
                    "unit", "value", "symbol", "state", "indentation_level",
                    "row_index", "column_index"
                } else "supporting"
                add_fact(
                    "attribute", f"{name} {key.replace('_', ' ')} is {value}",
                    [element_id], categories, importance, status,
                    subject_groups=[_content_tokens(name)],
                    relation_cues=[key.replace("_", " ")],
                    precision_sensitive=importance == "critical",
                )
            if element.get("parent_id"):
                parent_id = element["parent_id"]
                add_fact(
                    "hierarchy", f"{name} is part of {names.get(parent_id, parent_id)}",
                    [element_id, parent_id], categories, "critical", status,
                    subject_groups=[_content_tokens(name), _content_tokens(names.get(parent_id, parent_id))],
                    relation_cues=["part of", "within"],
                )

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
            subject_groups=[_content_tokens(source_name), _content_tokens(target_name)],
            relation_cues=cues,
            directional=bool(relation.get("directed")),
        )

    for item in visual_evidence.get("exact_content") or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        source_ref = item.get("source_ref") or "scene"
        notation = item.get("notation") or "exact"
        add_fact(
            "exact_content", f"{notation}: {item['text']}", [source_ref],
            categories_by_ref.get(source_ref, root_categories), "critical",
            item.get("status") or root_status,
            subject_groups=[_content_tokens(item["text"])], precision_sensitive=True,
        )

    structure = visual_evidence.get("structural_description") or {}
    for key in ("hierarchy_notes", "topology_notes"):
        for note in structure.get(key) or []:
            add_fact(
                "structure", note, list(structure.get("component_ids") or []),
                root_categories, "critical", root_status,
                subject_groups=[_content_tokens(note)],
            )
    reading_order = list(structure.get("reading_order") or [])
    if len(reading_order) > 1:
        readable_order = [names.get(ref, ref) for ref in reading_order]
        add_fact(
            "reading_order", "Reading order: " + " then ".join(readable_order),
            reading_order, root_categories, "important", root_status,
            subject_groups=[_content_tokens(name) for name in readable_order],
            relation_cues=["then", "followed by", "before"],
        )

    for detail_name, detail_value in (visual_evidence.get("category_details") or {}).items():
        if detail_value is None:
            continue
        for description in _flatten_detail(detail_name.replace("_", " "), detail_value):
            detail_payload = description.split(":", 1)[-1]
            add_fact(
                "category_detail", description, ["scene"], root_categories,
                "important", root_status, subject_groups=[_content_tokens(detail_payload)],
                precision_sensitive=detail_name in {"mathematics", "programming", "table"},
            )

    # Remove exact duplicates while retaining the earliest source-grounded fact.
    unique: List[AtomicVisualFact] = []
    seen = set()
    for fact in candidates:
        key = (fact.fact_type, _normalized_text(fact.description), tuple(fact.source_refs))
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    unique.sort(key=lambda fact: (_IMPORTANCE_ORDER[fact.importance], fact.fact_id))
    omitted = max(0, len(unique) - max_facts)
    selected = unique[:max_facts]
    for index, fact in enumerate(selected, start=1):
        fact.fact_id = f"fact-{index:03d}"
    return selected, {
        "truncated": omitted > 0,
        "omitted_fact_count": omitted,
        "total_fact_count": len(unique),
    }


def _group_is_present(group: List[str], transcript: str, transcript_tokens: set[str]) -> bool:
    meaningful = [token for token in group if token not in _STOP_WORDS]
    if not meaningful:
        return False
    phrase = " ".join(meaningful)
    return phrase in transcript or all(token in transcript_tokens for token in meaningful)


def _match_atomic_fact(fact: AtomicVisualFact, transcript_text: str) -> tuple[str, Optional[str], str]:
    transcript = _normalized_text(transcript_text)
    transcript_tokens = set(_tokens(transcript))
    meaningful_transcript = transcript_tokens - _DEICTIC_WORDS - _STOP_WORDS
    if fact.evidence_reliability in {"unavailable", "ambiguous"}:
        return "not_usable", None, ""
    if not meaningful_transcript:
        return "unspoken", None, fact.description

    description = _normalized_text(fact.description)
    groups_present = [
        _group_is_present(group, transcript, transcript_tokens)
        for group in fact.subject_groups if group
    ]
    literal_tokens = {
        token for token in _tokens(description)
        if re.fullmatch(r">=|<=|==|!=|[=<>≤≥≠+*/^\-]|\d+(?:\.\d+)?", token)
    }

    if fact.precision_sensitive:
        exact_payload = description.split(":", 1)[-1].strip()
        if exact_payload and exact_payload in transcript:
            status = "spoken_explicit"
        elif literal_tokens and literal_tokens.issubset(transcript_tokens):
            words = set(_content_tokens(description)) - literal_tokens
            overlap = len(words & transcript_tokens) / max(1, len(words))
            if overlap >= 0.8:
                status = "spoken_explicit"
            elif overlap >= 0.5:
                status = "uncertain_match"
            else:
                status = "unspoken"
        else:
            status = "unspoken"
        if fact.evidence_reliability == "partial" and status == "spoken_explicit":
            status = "uncertain_match"
        return (
            status,
            transcript_text if status != "unspoken" else None,
            "" if status == "spoken_explicit" else fact.description,
        )

    cues_present = any(_normalized_text(cue) in transcript for cue in fact.relation_cues)
    if fact.fact_type in {"relation", "hierarchy", "spatial", "reading_order"}:
        direction_supported = True
        if fact.directional and len(fact.subject_groups) >= 2:
            source_phrase = " ".join(fact.subject_groups[0])
            target_phrase = " ".join(fact.subject_groups[1])
            source_index = transcript.find(source_phrase)
            target_index = transcript.find(target_phrase)
            direction_supported = (
                source_index >= 0 and target_index >= 0 and source_index < target_index
            )
        if groups_present and all(groups_present) and cues_present and direction_supported:
            status = "spoken_explicit"
        elif groups_present and all(groups_present) and cues_present and not direction_supported:
            status = "uncertain_match"
        elif groups_present and all(groups_present):
            status = "spoken_partial"
        elif fact.fact_type == "spatial" and groups_present and any(groups_present):
            status = "spoken_partial"
        else:
            status = "unspoken"
    elif groups_present and all(groups_present):
        status = "spoken_explicit"
    else:
        fact_tokens = set(_content_tokens(description))
        overlap = len(fact_tokens & transcript_tokens) / max(1, len(fact_tokens))
        if overlap >= 0.7:
            status = "spoken_partial"
        elif overlap >= 0.4:
            status = "uncertain_match"
        else:
            status = "unspoken"

    if fact.evidence_reliability == "partial" and status == "spoken_explicit":
        status = "uncertain_match"
    missing = "" if status == "spoken_explicit" else fact.description
    return status, transcript_text if status in {"spoken_explicit", "spoken_partial", "uncertain_match"} else None, missing


def reconcile_visual_evidence(
    vlm_snapshot: Dict[str, Any], transcript_text: str, *, max_facts: int = 64
) -> Dict[str, Any]:
    """Create a compact post-VLM/pre-synthesis transcript coverage plan."""

    evidence = vlm_snapshot.get("visual_evidence") if isinstance(vlm_snapshot, dict) else None
    frame_id = evidence.get("frame_id") if isinstance(evidence, dict) else None
    facts, bounds = extract_atomic_visual_facts(evidence or {}, max_facts=max_facts)
    plan_facts = []
    for fact in facts:
        coverage, span, missing = _match_atomic_fact(fact, transcript_text)
        assert coverage in _COVERAGE_LABELS
        item = asdict(fact)
        item.pop("subject_groups", None)
        item.pop("relation_cues", None)
        item.pop("precision_sensitive", None)
        item.pop("directional", None)
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
        "method": "deterministic_conservative_v1",
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


