"""Backward-compatible normalization for visual category metadata.

The extractor historically emitted free-form labels (``graph``, ``notes``,
``slide`` and so on).  The public taxonomy is represented by CT-01 through
CT-12, but raw labels remain part of the persisted contract.  This module is
deliberately conservative: labels that are not specific enough are reported as
ambiguous instead of being silently assigned to an unsupported category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CANONICAL_CATEGORIES: dict[str, str] = {
    "CT-01": "Charts and graphs",
    "CT-02": "Mathematical notation",
    "CT-03": "Scientific and technical structures",
    "CT-04": "Flowcharts and processes",
    "CT-05": "Tables and matrices",
    "CT-06": "Structured instructional text",
    "CT-07": "Geometry",
    "CT-08": "Programming text",
    "CT-09": "Software interfaces",
    "CT-10": "Maps",
    "CT-11": "Observational images",
    "CT-12": "Physical illustrations and demonstrations",
}


# Only unambiguous legacy labels are mapped automatically.  In particular,
# ``diagram`` and ``slide`` may contain several different approved categories.
_UNAMBIGUOUS_LABELS: dict[str, str] = {
    "graph": "CT-01",
    "chart": "CT-01",
    "equation": "CT-02",
    "formula": "CT-02",
    "circuit": "CT-03",
    "biology": "CT-03",
    "chemistry": "CT-03",
    "physics": "CT-03",
    "flowchart": "CT-04",
    "process": "CT-04",
    "table": "CT-05",
    "matrix": "CT-05",
    "text": "CT-06",
    "handwritten_notes": "CT-06",
    "handwritten": "CT-06",
    "notes": "CT-06",
    "geometry": "CT-07",
    "code": "CT-08",
    "programming": "CT-08",
    "software_interface": "CT-09",
    "interface": "CT-09",
    "ui": "CT-09",
    "map": "CT-10",
    "photo": "CT-11",
    "photograph": "CT-11",
    "observational_image": "CT-11",
    "physical_illustration": "CT-12",
    "demonstration": "CT-12",
}

_AMBIGUOUS_LABELS = {
    "diagram": "A diagram may be scientific, procedural, geometric, or illustrative.",
    "slide": "A slide is a container and may contain several approved categories.",
    "educational": "The legacy label does not identify a visual category.",
    "unknown": "No category was available.",
    "non-educational": "The frame was not classified as educational.",
}


@dataclass(frozen=True)
class CategoryNormalization:
    """Lossless result of normalizing one or more raw labels."""

    raw_labels: tuple[str, ...]
    canonical_categories: tuple[str, ...]
    ambiguous_labels: tuple[str, ...]
    unmapped_labels: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.ambiguous_labels or self.unmapped_labels:
            return "ambiguous" if self.canonical_categories else "unmapped"
        return "mapped" if self.canonical_categories else "unmapped"

    @property
    def notes(self) -> tuple[str, ...]:
        notes = [_AMBIGUOUS_LABELS[label] for label in self.ambiguous_labels]
        notes.extend(f"No approved mapping for legacy label '{label}'." for label in self.unmapped_labels)
        return tuple(notes)


def normalize_category_labels(raw_labels: str | Iterable[str] | None) -> CategoryNormalization:
    """Normalize raw labels without inventing categories.

    The returned raw labels are de-duplicated while preserving order.  A label
    can contribute one canonical category, while ambiguous/unmapped labels are
    retained in the result for diagnostics and future review.
    """

    if raw_labels is None:
        values: list[str] = []
    elif isinstance(raw_labels, str):
        values = [raw_labels]
    else:
        values = [value for value in raw_labels if isinstance(value, str)]

    raw: list[str] = []
    seen: set[str] = set()
    canonical: list[str] = []
    ambiguous: list[str] = []
    unmapped: list[str] = []
    for value in values:
        original = value.strip()
        label = original.lower()
        if not label or label in seen:
            continue
        seen.add(label)
        raw.append(original)
        category = _UNAMBIGUOUS_LABELS.get(label)
        if category:
            if category not in canonical:
                canonical.append(category)
        elif label in _AMBIGUOUS_LABELS:
            ambiguous.append(label)
        else:
            unmapped.append(label)

    return CategoryNormalization(
        raw_labels=tuple(raw),
        canonical_categories=tuple(canonical),
        ambiguous_labels=tuple(ambiguous),
        unmapped_labels=tuple(unmapped),
    )


def prompt_category_for(raw_label: str | None) -> str:
    """Return the existing SnapshotEngine directive key for a raw label.

    This preserves the current VLM prompt contract while fixing the old
    confidence-based replacement of valid labels with ``unknown``.  Unknown or
    ambiguous labels remain unchanged so the VLM can still inspect the frame.
    """

    value = (raw_label or "unknown").strip().lower()
    if value == "notes":
        return "handwritten_notes"
    return value or "unknown"

