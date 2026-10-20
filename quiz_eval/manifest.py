"""
The dataset manifest: which lectures, which review slots, and what a human has approved.

Validation returns a Report (errors, warnings). Errors mean the file cannot be trusted; warnings are
deviations from the contract's protocol shape or things a reviewer should look at. Nothing is
repaired and nothing is invented: a missing value stays missing and is reported as pending.

An item can be "approved" only when everything a human must have decided is present. That is what
stops a half-filled row from looking like a human approval.
"""

import csv
import os
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quiz_eval import transcript as tx
from quiz_eval.constants import (
    FROZEN_CONFIG_FIELDS, FROZEN_CONFIG_HASH_FIELDS, GATE, SHA256_RE, is_placeholder,
    EXCLUSION_REASONS, LECTURE_ID_RE, MANIFEST_STATUSES, MANIFEST_VERSION, PROTOCOL, QUESTION_TYPES,
    SAMPLE_ID_RE, SAMPLE_REVIEW_STATUSES, SPLITS, SUBJECTS, VIDEO_ID_RE, WORD_COUNT_STATUSES,
)


def canonical_sha256(data: Any) -> str:
    """Hash of a manifest as a canonical JSON document (sorted keys, no spaces). Any edit changes it, which is
    how a scorecard proves it was produced against exactly the manifest that was frozen."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


def _is_int(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _check_question(q: Any, qtype: Any, where: str, report: Report) -> None:
    if not isinstance(q, dict):
        report.error(where, "question must be an object")
        return
    if not _nonblank(q.get("prompt")):
        report.error(where, "question.prompt is required")
    if qtype == "mcq":
        choices = q.get("choices")
        if not isinstance(choices, list) or not 3 <= len(choices) <= 4:
            report.error(where, "an MCQ needs 3 or 4 choices")
            return
        ids = []
        for c in choices:
            if not isinstance(c, dict) or not _nonblank(c.get("id")) or not _nonblank(c.get("text")):
                report.error(where, "each choice needs an id and text")
                return
            ids.append(c["id"])
        if len(set(ids)) != len(ids):
            report.error(where, "choice ids must be unique")
        if q.get("correct_choice_id") not in ids:
            report.error(where, "correct_choice_id must be one of the choices")
    elif qtype == "tf":
        if not isinstance(q.get("answer"), bool):
            report.error(where, "a true/false question needs a boolean answer")


def _check_evidence(e: Any, where: str, report: Report) -> None:
    if not isinstance(e, dict):
        report.error(where, "evidence must be an object")
        return
    if not _nonblank(e.get("quote")):
        report.error(where, "evidence.quote is required")
    s, t = e.get("start_seconds"), e.get("end_seconds")
    if not (_is_num(s) and _is_num(t) and 0 <= s <= t):
        report.error(where, "evidence needs finite start_seconds <= end_seconds, both >= 0")


def validate_manifest(data: Any) -> Report:
    report = Report()
    if not isinstance(data, dict):
        report.error("manifest", "must be a JSON object")
        return report
    if data.get("manifest_version") != MANIFEST_VERSION:
        report.error("manifest_version", f"must be {MANIFEST_VERSION}")
    if not _nonblank(data.get("name")):
        report.error("name", "is required")
    if data.get("status") not in MANIFEST_STATUSES:
        report.error("status", f"must be one of {MANIFEST_STATUSES}")
    if not isinstance(data.get("is_synthetic"), bool):
        report.error("is_synthetic", "must be true or false (synthetic data must say so)")
    frozen = data.get("status") == "frozen"

    lectures = data.get("lectures")
    samples = data.get("samples")
    if not isinstance(lectures, list):
        report.error("lectures", "must be a list")
        lectures = []
    if not isinstance(samples, list):
        report.error("samples", "must be a list")
        samples = []

    by_id: Dict[str, Dict[str, Any]] = {}
    for i, lec in enumerate(lectures):
        where = f"lectures[{i}]"
        if not isinstance(lec, dict):
            report.error(where, "must be an object")
            continue
        lid = lec.get("lecture_id")
        if not isinstance(lid, str) or not LECTURE_ID_RE.match(lid):
            report.error(where, "lecture_id is required (letters, digits, - _)")
            continue
        where = f"lecture {lid}"
        if lid in by_id:
            report.error(where, "duplicate lecture_id")
        by_id[lid] = lec
        if not (isinstance(lec.get("video_id"), str) and VIDEO_ID_RE.match(lec["video_id"])):
            report.error(where, "video_id must be an 11-character YouTube id")
        if lec.get("subject") not in SUBJECTS:
            report.error(where, f"subject must be one of {SUBJECTS}")
        if lec.get("split") not in SPLITS:
            report.error(where, f"split must be one of {SPLITS}")
        ref = lec.get("transcript_ref")
        if not (isinstance(ref, dict) and _nonblank(ref.get("path")) and _nonblank(ref.get("lecture_id"))):
            report.error(where, "transcript_ref needs path and lecture_id")
        wc = lec.get("expected_word_count")
        if wc is not None and not _is_int(wc, 1):
            report.error(where, "expected_word_count must be a positive integer or null (pending)")
        status = lec.get("word_count_status")
        if status not in WORD_COUNT_STATUSES:
            report.error(where, f"word_count_status must be one of {WORD_COUNT_STATUSES}")
        elif (wc is None) != (status == "pending"):
            report.error(where, "expected_word_count is null exactly when word_count_status is 'pending'")
        if frozen and lec.get("split") == "pending":
            report.error(where, "a frozen manifest must assign every lecture to development or confirmatory")

    seen = set()
    for i, s in enumerate(samples):
        where = f"samples[{i}]"
        if not isinstance(s, dict):
            report.error(where, "must be an object")
            continue
        sid = s.get("sample_id")
        if not isinstance(sid, str) or not SAMPLE_ID_RE.match(sid):
            report.error(where, "sample_id is required (3 to 61 characters: letters, digits, - _)")
            continue
        where = f"sample {sid}"
        if sid in seen:
            report.error(where, "duplicate sample_id")
        seen.add(sid)
        lec = by_id.get(s.get("lecture_id"))
        if lec is None:
            report.error(where, "lecture_id does not match any lecture")
        elif s.get("subject") != lec.get("subject"):
            report.error(where, "subject must equal its lecture's subject")
        if s.get("question_type") not in QUESTION_TYPES and s.get("question_type") != "pending":
            report.error(where, "question_type must be 'mcq', 'tf' or 'pending'")
        review = s.get("review")
        if not isinstance(review, dict) or review.get("status") not in SAMPLE_REVIEW_STATUSES:
            report.error(where, f"review.status must be one of {SAMPLE_REVIEW_STATUSES}")
            continue
        st = review["status"]
        if st == "approved":
            _check_approved(s, review, where, report)
        elif st == "rejected":
            if not (_nonblank(review.get("reviewer")) and _nonblank(review.get("notes"))):
                report.error(where, "a rejected item needs the reviewer and notes")
        elif st == "excluded":
            if review.get("exclusion_reason") not in EXCLUSION_REASONS:
                report.error(where, f"an excluded item needs exclusion_reason from {EXCLUSION_REASONS}")
            if not _nonblank(review.get("notes")):
                report.error(where, "an excluded item needs notes explaining why")
        else:  # pending: whatever is present must still be well-formed
            if s.get("question") is not None:
                _check_question(s.get("question"), s.get("question_type"), where, report)
            if s.get("evidence") is not None:
                _check_evidence(s.get("evidence"), where, report)

    _protocol_warnings(data, lectures, report)
    return report


def _check_approved(s: Dict[str, Any], review: Dict[str, Any], where: str, report: Report) -> None:
    """Everything a human must have decided before an item can count as approved."""
    if s.get("question_type") not in QUESTION_TYPES:
        report.error(where, "an approved item needs question_type 'mcq' or 'tf'")
    _check_question(s.get("question"), s.get("question_type"), where, report)
    if not _nonblank(s.get("key_rationale")):
        report.error(where, "an approved item needs key_rationale")
    _check_evidence(s.get("evidence"), where, report)
    if s.get("answerable_from_evidence") is not True:
        report.error(where, "an approved item must have answerable_from_evidence true")
    if s.get("requires_visual") is not False:
        report.error(where, "an approved item must have requires_visual false (visual questions are excluded)")
    if not (_nonblank(review.get("reviewer")) and _nonblank(review.get("reviewed_at"))):
        report.error(where, "an approved item needs reviewer and reviewed_at")


def _protocol_warnings(data: Dict[str, Any], lectures: List[Any], report: Report) -> None:
    approvers = data.get("authorized_approvers")
    if approvers is not None:
        # Who may grant or deny a subject-level approval. Frozen with the manifest, so it cannot be widened after the run.
        if not isinstance(approvers, list) or not all(isinstance(n, str) and not is_placeholder(n) for n in approvers)                 or len({n.strip().casefold() for n in approvers}) != len(approvers):
            report.error("authorized_approvers", "must be a list of distinct, real names (no blanks or placeholders)")
    if data.get("status") != "frozen":
        return
    if not _nonblank(data.get("selection_rule")):
        report.error("selection_rule", "a frozen manifest must state the rule that chose the questions to review")
    if not _nonblank(data.get("frozen_at")):
        report.error("frozen_at", "a frozen manifest needs the date it was frozen")
    config = data.get("frozen_config")
    if not isinstance(config, dict):
        report.error("frozen_config", "a frozen manifest must pin the prompt, model and configuration the run will use: " + ", ".join(FROZEN_CONFIG_FIELDS))
    else:
        for key in FROZEN_CONFIG_FIELDS:
            value = config.get(key)
            if is_placeholder(value):
                report.error("frozen_config", f"{key} is required and must not be a placeholder")
            elif key in FROZEN_CONFIG_HASH_FIELDS and not SHA256_RE.match(value):
                report.error("frozen_config", f"{key} must be a lowercase 64-character SHA-256")
    slots = [s for s in data.get("samples", []) if isinstance(s, dict)]
    by_lecture = {l.get("lecture_id"): l for l in lectures if isinstance(l, dict)}
    confirmatory = [s for s in slots if (by_lecture.get(s.get("lecture_id")) or {}).get("split") == "confirmatory"]
    if len(confirmatory) != GATE["required_confirmatory_items"]:
        report.warn("protocol", f"{len(confirmatory)} confirmatory review slots are registered; the contract's gate needs {GATE['required_confirmatory_items']}")
    per = defaultdict(Counter)
    for lec in lectures:
        if isinstance(lec, dict):
            per[lec.get("split")][lec.get("subject")] += 1
    for split, key in (("development", "development_per_subject"), ("confirmatory", "confirmatory_per_subject")):
        for subject in SUBJECTS:
            if per[split][subject] != PROTOCOL[key]:
                report.warn("protocol", f"{split}: {subject} has {per[split][subject]} lectures; the contract plans {PROTOCOL[key]}")


def summarize_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    """Counts of lectures and samples by split, subject, review status and type. Pending stays pending."""
    lectures = [l for l in data.get("lectures", []) if isinstance(l, dict)]
    samples = [s for s in data.get("samples", []) if isinstance(s, dict)]
    by_subject: Dict[str, Dict[str, int]] = {}
    for subject in SUBJECTS:
        row = {"lectures": sum(1 for l in lectures if l.get("subject") == subject)}
        for status in SAMPLE_REVIEW_STATUSES:
            row[status] = sum(1 for s in samples if s.get("subject") == subject and (s.get("review") or {}).get("status") == status)
        by_subject[subject] = row
    return {
        "name": data.get("name"), "status": data.get("status"), "is_synthetic": data.get("is_synthetic"),
        "lectures": len(lectures), "samples": len(samples),
        "lectures_by_split": dict(Counter(l.get("split") for l in lectures)),
        "samples_by_status": {st: sum(1 for s in samples if (s.get("review") or {}).get("status") == st) for st in SAMPLE_REVIEW_STATUSES},
        "samples_by_type": dict(Counter(s.get("question_type") for s in samples)),
        "by_subject": by_subject,
        "exclusions_by_reason": dict(Counter((s.get("review") or {}).get("exclusion_reason") for s in samples
                                              if (s.get("review") or {}).get("status") == "excluded")),
        "word_counts_pending": sum(1 for l in lectures if l.get("expected_word_count") is None),
    }


def check_transcripts(data: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare each lecture's expected word count with the transcript, and check each sample's evidence quote
    and times against it. Reports unavailable transcripts as unknown; never as a failure.
    """
    out: Dict[str, Any] = {"lectures": {}, "samples": {}}
    cache: Dict[str, Any] = {}
    for lec in data.get("lectures", []):
        ref = lec.get("transcript_ref") or {}
        segments = tx.load_segments(ref.get("path", ""), ref.get("lecture_id", ""), base_dir)
        cache[lec.get("lecture_id")] = segments
        measured = tx.word_count(segments)
        expected = lec.get("expected_word_count")
        if segments is None or measured is None:
            verdict = "unknown"
        elif expected is None:
            verdict = "pending"
        else:
            verdict = "match" if measured == expected else "mismatch"
        out["lectures"][lec.get("lecture_id")] = {"expected": expected, "measured": measured, "word_count": verdict}
    for s in data.get("samples", []):
        e = s.get("evidence")
        if isinstance(e, dict):
            out["samples"][s.get("sample_id")] = tx.check_evidence(e.get("quote"), e.get("start_seconds"), e.get("end_seconds"), cache.get(s.get("lecture_id")))
    return out


def build_from_dataset(dataset_dir: str, name: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    A DRAFT manifest from an existing dataset directory (lecture_manifest.csv + transcript_segments.jsonl),
    read-only. It lists the lectures with their real metadata and measured word counts, assigns NO split
    (that is a human, pre-registered decision) and creates NO samples. Nothing is invented.
    """
    base = base_dir or tx.PROJECT_ROOT
    csv_path = tx.resolve_path(os.path.join(dataset_dir, "lecture_manifest.csv"), base)
    seg_rel = os.path.join(dataset_dir, "transcript_segments.jsonl").replace("\\", "/")
    lectures = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lecture_id = row["lecture_id"]
            segments = tx.load_segments(seg_rel, lecture_id, base)
            csv_words = row.get("n_words", "").strip()
            expected = int(csv_words) if csv_words.isdigit() and int(csv_words) > 0 else tx.word_count(segments)
            lectures.append({
                "lecture_id": lecture_id,
                "video_id": row["video_id"],
                "subject": row["subject"],
                "title": row.get("lecture_title", ""),
                "split": "pending",
                "transcript_ref": {"path": seg_rel, "lecture_id": lecture_id},
                "expected_word_count": expected,
                "word_count_status": "from_dataset_manifest" if csv_words.isdigit() and int(csv_words) > 0 else ("measured" if expected else "pending"),
            })
    return {
        "manifest_version": MANIFEST_VERSION, "name": name, "status": "draft", "is_synthetic": False,
        "notes": "Draft built from the dataset's own lecture_manifest.csv. No split is assigned and no review item exists yet.",
        "lectures": sorted(lectures, key=lambda l: l["lecture_id"]), "samples": [],
    }
