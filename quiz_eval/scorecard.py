"""
The reviewer scorecard and the metrics computed from it.

HUMAN judgments live in `reviews` (and, for disputes, `adjudication`). Everything under `compute_*`
is arithmetic on those judgments, plus a few mechanical transcript checks reported separately.

A verdict FAILS CLOSED. It can say "criteria a to c met, pending d" only when ALL of these hold:

  1. the manifest is validated, frozen (with its recorded freeze state and pinned prompt/model/config),
     and not synthetic;
  2. the scorecard is a recorded real run with complete, non-placeholder provenance, bound to that exact
     manifest by hash, and the run matches the configuration the manifest froze;
  3. every pre-registered confirmatory slot (exactly the contract's 80) has an item, and every one of
     those items has two INDEPENDENT reviews that agree or were adjudicated;
  4. no confirmatory item is pending, single-reviewed, unresolved, excluded or missing;
  5. the scorecard validates with no errors.

Anything else is reported as it is (not_evidence, not_eligible, release_blocked, not_evaluable,
criteria_not_met) and never as a pass. Exclusions, unresolved items and missing slots can never improve
a result: they stop a pass. A release-blocking failure (a key that is a false statement about the subject)
stays visible in every outcome, in every split, whatever the sample size.

The thresholds are the contract's own (constants.GATE) and are not changed here. The tool never says a
feature may be enabled: at most it says criteria a, b and c are met on the full registered sample, and
that criterion d (people checking accessibility) is not measured here.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from quiz_eval import manifest as mf
from quiz_eval import transcript as tx
from quiz_eval.constants import (
    BLOCKING_JUDGMENT, CONTRADICTIONS, EXCLUSION_CAP_CONFIRMATORY, FAILING_VALUES, FROZEN_CONFIG_FIELDS, GATE, JUDGMENTS, NA_ONLY_FOR_TF,
    QUESTION_TYPES, REVIEWS_REQUIRED_FOR_GATE, RUN_HASH_FIELDS, RUN_REQUIRED_FIELDS, RUN_SOURCES,
    SCORECARD_EXCLUSION_REASONS, SCORECARD_VERSION, SHA256_RE, SOURCE_ERROR_JUDGMENT, SPLITS, SUBJECT_APPROVAL_DECISIONS, SUBJECTS,
    is_placeholder,
)
from quiz_eval.manifest import Report, _is_num, _nonblank


def _valid_value(value: Any) -> bool:
    """true, false, null or the string "n/a". By value, not identity: a value read from JSON is a new object."""
    return value is True or value is False or value is None or (isinstance(value, str) and value == "n/a")


def _person(value: Any) -> str:
    return str(value).strip().casefold() if _nonblank(value) else ""


# ----------------------------------------------------------------------------- validation


def _check_judgments(j: Any, qtype: str, where: str, report: Report, require_complete: bool = False) -> None:
    if not isinstance(j, dict):
        report.error(where, "judgments must be an object")
        return
    for key in j:
        if key not in JUDGMENTS:
            report.error(where, f"unknown judgment '{key}'")
    for key in JUDGMENTS:
        value = j.get(key)
        if not _valid_value(value):
            report.error(where, f"{key} must be true, false, 'n/a' or null (not judged)")
            continue
        if value == "n/a" and not (key in NA_ONLY_FOR_TF and qtype == "tf"):
            report.error(where, f"{key} can be 'n/a' only on a true/false question for single_defensible_option or misleading_distractors")
        if qtype == "tf" and key in NA_ONLY_FOR_TF and not (value is None or value == "n/a"):
            report.error(where, f"{key} does not apply to a true/false question; use 'n/a'")
        if require_complete and value is None:
            report.error(where, f"{key} has not been judged")
    for message in contradictions(j):
        report.error(where, f"contradictory judgments: {message}")


def contradictions(j: Dict[str, Any]) -> List[str]:
    """Pairs of recorded judgments that cannot both be true (only where both are actually judged)."""
    out = []
    for ka, va, kb, vb, why in CONTRADICTIONS:
        if j.get(ka) is va and j.get(kb) is vb:
            out.append(f"{ka}={va} with {kb}={vb}: {why}")
    if j.get(BLOCKING_JUDGMENT) is True and j.get("key_correct") is True and j.get(SOURCE_ERROR_JUDGMENT) is False:
        out.append("key_false_about_subject=True with key_correct=True while lecture_factually_incorrect=False: "
                   "a key that follows a correct lecture cannot be false about the subject")
    return out


def _has_source_error(value: Any) -> bool:
    """Evidence for a claimed source-lecture error: the lecture's own words and what is wrong with them."""
    return isinstance(value, dict) and _nonblank(value.get("quote")) and _nonblank(value.get("note"))


def _is_complete(review: Dict[str, Any], qtype: str) -> bool:
    j = review.get("judgments")
    if not (isinstance(j, dict) and _nonblank(review.get("reviewer")) and _nonblank(review.get("completed_at"))):
        return False
    for key in JUDGMENTS:
        v = j.get(key)
        if v is None:
            return False
        if qtype == "tf" and key in NA_ONLY_FOR_TF:
            if v != "n/a":
                return False
        elif not isinstance(v, bool):
            return False
    return True


def _failing(judgments: Any) -> List[str]:
    return [k for k, bad in FAILING_VALUES.items() if isinstance(judgments, dict) and judgments.get(k) is bad]


def _validate_run(data: Dict[str, Any], report: Report) -> None:
    run = data.get("run")
    if not isinstance(run, dict) or run.get("source") not in RUN_SOURCES:
        report.error("run.source", f"must be one of {RUN_SOURCES}")
        return
    source = run["source"]
    if source == "synthetic" and data.get("is_synthetic") is not True:
        report.error("run.source", "a synthetic run must set is_synthetic true")
    if source == "recorded_real_run":
        if data.get("is_synthetic") is True:
            report.error("run.source", "a recorded real run cannot be synthetic")
        for key in RUN_REQUIRED_FIELDS:
            value = run.get(key)
            if is_placeholder(value):
                report.error(f"run.{key}", "is required for a recorded real run and must not be blank or a placeholder")
            elif key in RUN_HASH_FIELDS and not SHA256_RE.match(value):
                report.error(f"run.{key}", "must be a lowercase 64-character SHA-256")
        binding = data.get("manifest_binding")
        if not (isinstance(binding, dict) and not is_placeholder(binding.get("name")) and isinstance(binding.get("sha256"), str) and SHA256_RE.match(binding["sha256"])):
            report.error("manifest_binding", "a recorded real run must name its manifest and give the manifest's SHA-256")


def validate_scorecard(data: Any, manifest: Optional[Dict[str, Any]] = None) -> Report:
    report = Report()
    if not isinstance(data, dict):
        report.error("scorecard", "must be a JSON object")
        return report
    if data.get("scorecard_version") != SCORECARD_VERSION:
        report.error("scorecard_version", f"must be {SCORECARD_VERSION}")
    if not isinstance(data.get("is_synthetic"), bool):
        report.error("is_synthetic", "must be true or false")
    _validate_run(data, report)
    items = data.get("items")
    if not isinstance(items, list):
        report.error("items", "must be a list")
        return report

    lectures = {l.get("lecture_id"): l for l in (manifest or {}).get("lectures", []) if isinstance(l, dict)}
    slots = {s.get("sample_id"): s for s in (manifest or {}).get("samples", []) if isinstance(s, dict)}
    run = data.get("run")
    if manifest and isinstance(run, dict) and run.get("source") == "recorded_real_run":
        binding = data.get("manifest_binding") or {}
        if isinstance(binding.get("sha256"), str) and binding["sha256"] != mf.canonical_sha256(manifest):
            report.error("manifest_binding", "the scorecard was not produced against this manifest (hash differs); a frozen manifest must not change")
    slot_rank: Dict[Any, int] = {}
    per_lecture: Counter = Counter()
    for s_ in (manifest or {}).get("samples", []):
        if isinstance(s_, dict):
            per_lecture[s_.get("lecture_id")] += 1
            slot_rank[s_.get("sample_id")] = per_lecture[s_.get("lecture_id")]
    seen, used_slots = set(), set()
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            report.error(where, "must be an object")
            continue
        iid = item.get("item_id")
        if not _nonblank(iid):
            report.error(where, "item_id is required")
            continue
        where = f"item {iid}"
        if iid in seen:
            report.error(where, "duplicate item_id")
        seen.add(iid)
        qtype = item.get("question_type")
        if qtype not in QUESTION_TYPES:
            report.error(where, f"question_type must be one of {QUESTION_TYPES}")
            qtype = "mcq"
        if item.get("subject") not in SUBJECTS:
            report.error(where, f"subject must be one of {SUBJECTS}")
        if item.get("split") not in SPLITS:
            report.error(where, f"split must be one of {SPLITS}")
        if lectures:
            lec = lectures.get(item.get("lecture_id"))
            if lec is None:
                report.error(where, "lecture_id is not in the manifest")
            else:
                if lec.get("subject") != item.get("subject"):
                    report.error(where, "subject differs from the manifest's lecture")
                if lec.get("split") != item.get("split"):
                    report.error(where, "split differs from the manifest's lecture")
        elif not _nonblank(item.get("lecture_id")):
            report.error(where, "lecture_id is required")
        if slots:
            slot = slots.get(item.get("sample_id"))
            if slot is None:
                report.error(where, "sample_id must name one of the manifest's registered slots (an unregistered extra item is not allowed)")
            else:
                if item.get("sample_id") in used_slots:
                    report.error(where, "two items use the same registered slot")
                used_slots.add(item.get("sample_id"))
                if slot.get("lecture_id") != item.get("lecture_id"):
                    report.error(where, "the registered slot belongs to a different lecture")
                position = item.get("returned_position")
                if type(position) is not int or position != slot_rank.get(item.get("sample_id")):
                    report.error(where, f"returned_position must be {slot_rank.get(item.get('sample_id'))}: slots are filled by the model's returned "
                                        "order within the lecture, in the manifest's slot order, never by choice")
        ev = item.get("evidence")
        if not isinstance(ev, list) or not 1 <= len(ev) <= 3:
            report.error(where, "evidence must list 1 to 3 items as the server returned them")
        else:
            for e in ev:
                if not (isinstance(e, dict) and _nonblank(e.get("quote")) and _is_num(e.get("start_seconds")) and _is_num(e.get("end_seconds"))):
                    report.error(where, "each evidence item needs quote, start_seconds and end_seconds")
        if not isinstance(item.get("question"), dict) or not _nonblank((item.get("question") or {}).get("prompt")):
            report.error(where, "question (as shown to the reviewer) with a prompt is required")
        _validate_reviews(item, qtype, where, report)
    _validate_approvals(data, manifest, report)
    return report


def _validate_approvals(data: Dict[str, Any], manifest: Optional[Dict[str, Any]], report: Report) -> None:
    approvals = data.get("subject_approvals", [])
    if not isinstance(approvals, list):
        report.error("subject_approvals", "must be a list")
        return
    authorized = {_person(n) for n in (manifest or {}).get("authorized_approvers", []) if _nonblank(n)}
    seen = set()
    for k, a in enumerate(approvals):
        w = f"subject_approvals[{k}]"
        if not isinstance(a, dict):
            report.error(w, "must be an object")
            continue
        if a.get("subject") not in SUBJECTS:
            report.error(w, f"subject must be one of {SUBJECTS}")
        elif a["subject"] in seen:
            report.error(w, "a subject has more than one approval record; keep the audit trail by recording one decision per subject")
        seen.add(a.get("subject"))
        if a.get("decision") not in SUBJECT_APPROVAL_DECISIONS:
            report.error(w, f"decision must be one of {SUBJECT_APPROVAL_DECISIONS}")
        for key in ("approver", "rationale", "decided_at"):
            if is_placeholder(a.get(key)):
                report.error(w, f"{key} is required and must not be blank or a placeholder")
        if not (isinstance(a.get("basis_sha256"), str) and SHA256_RE.match(a["basis_sha256"])):
            report.error(w, "basis_sha256 must be the SHA-256 the report printed for this subject's figures")
        if manifest is not None and not is_placeholder(a.get("approver")) and _person(a["approver"]) not in authorized:
            report.error(w, "the approver is not in the frozen manifest's authorized_approvers")


def _validate_reviews(item: Dict[str, Any], qtype: str, where: str, report: Report) -> None:
    reviews = item.get("reviews", [])
    exclusion = item.get("exclusion")
    if not isinstance(reviews, list) or len(reviews) > 2:
        report.error(where, "reviews must be a list of at most two reviews")
        return
    names = []
    for k, review in enumerate(reviews):
        w = f"{where} review {k + 1}"
        if not isinstance(review, dict):
            report.error(w, "must be an object")
            continue
        _check_judgments(review.get("judgments"), qtype, w, report)
        sf = review.get("subject_failure")
        if sf is not None and not (isinstance(sf, dict) and (sf.get("category") is None or _nonblank(sf.get("category")))):
            report.error(w, "subject_failure needs a category (or null) and notes")
        rj = review.get("judgments")
        if isinstance(rj, dict) and rj.get(SOURCE_ERROR_JUDGMENT) is True and not _has_source_error(review.get("source_error")):
            report.error(w, "lecture_factually_incorrect is true: record source_error {quote, note} with the lecture's own words and what is wrong")
        if _person(review.get("reviewer")):
            names.append(_person(review.get("reviewer")))
    if len(names) == 2 and names[0] == names[1]:
        report.error(where, "the two reviews were made by the same reviewer; they must be independent people")

    if exclusion is not None:
        if item.get("adjudication") is not None:
            report.error(where, "contradictory record: an item cannot be both excluded and adjudicated")
        ok = isinstance(exclusion, dict) and exclusion.get("reason") in SCORECARD_EXCLUSION_REASONS and _nonblank(exclusion.get("notes"))
        if not ok:
            report.error(where, f"an exclusion needs reason from {SCORECARD_EXCLUSION_REASONS} and notes (a wrong, unanswerable or visual question is judged, not excluded)")
        else:
            a, b = _person(exclusion.get("excluded_by")), _person(exclusion.get("confirmed_by"))
            if not (a and b and _nonblank(exclusion.get("decided_at"))):
                report.error(where, "an exclusion needs excluded_by, confirmed_by and decided_at")
            elif a == b:
                report.error(where, "an exclusion must be confirmed by a second person")
        for k, review in enumerate(reviews):
            failing = _failing(review.get("judgments") if isinstance(review, dict) else None)
            if failing:
                report.error(where, f"review {k + 1} records a failure ({', '.join(failing)}); a recorded failure can never be excluded")
        return

    adj = item.get("adjudication")
    if adj is None:
        return
    w = f"{where} adjudication"
    if len(reviews) != 2 or not all(isinstance(r, dict) and _is_complete(r, qtype) for r in reviews):
        report.error(w, "adjudication applies only after two COMPLETE reviews")
        return
    if not (isinstance(adj, dict) and _nonblank(adj.get("adjudicator")) and isinstance(adj.get("judgments"), dict)
            and _nonblank(adj.get("notes")) and _nonblank(adj.get("decided_at"))):
        report.error(w, "needs adjudicator, judgments, notes and decided_at")
        return
    if _person(adj["adjudicator"]) in names:
        report.error(w, "the adjudicator cannot be one of the item's reviewers")
    a, b = reviews[0]["judgments"], reviews[1]["judgments"]
    disputed = [k for k in JUDGMENTS if a[k] != b[k]]
    if not disputed:
        report.error(w, "the reviewers agree on everything: there is nothing to adjudicate")
        return
    extra = [k for k in adj["judgments"] if k not in disputed]
    if extra:
        report.error(w, f"supplies judgments that are not in dispute ({', '.join(extra)}); they would be ignored")
    for k, value in adj["judgments"].items():
        if k in disputed and not (isinstance(value, bool) or (value == "n/a" and k in NA_ONLY_FOR_TF and qtype == "tf")):
            report.error(w, f"{k} must be true or false")
    missing = [k for k in disputed if k not in adj["judgments"]]
    if missing:
        report.error(w, f"does not resolve every disagreement (missing: {', '.join(missing)})")
    else:
        merged = {k: (adj["judgments"][k] if k in disputed else a[k]) for k in JUDGMENTS}
        for message in contradictions(merged):
            report.error(w, f"the adjudicated result is contradictory: {message}")


# ----------------------------------------------------------------------------- resolution


def resolve_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    How an item stands: excluded | pending | single_review | agreed | adjudicated | unresolved.
    `final` holds the judgments that count (None when the item counts nowhere). Only `agreed` and
    `adjudicated` (two independent reviews) are gate-ready.
    """
    if item.get("exclusion") is not None:
        return {"state": "excluded", "final": None, "disagreements": [], "reason": (item["exclusion"] or {}).get("reason")}
    qtype = item.get("question_type")
    reviews = [r for r in item.get("reviews", []) if isinstance(r, dict)]
    complete = [r for r in reviews if _is_complete(r, qtype)]
    if not complete:
        return {"state": "pending", "final": None, "disagreements": []}
    if len(reviews) > len(complete):
        return {"state": "pending", "final": None, "disagreements": [], "note": "a second review is started but not complete"}
    if len(complete) == 1:
        return {"state": "single_review", "final": dict(complete[0]["judgments"]), "disagreements": []}
    if _person(complete[0]["reviewer"]) == _person(complete[1]["reviewer"]):
        return {"state": "single_review", "final": dict(complete[0]["judgments"]), "disagreements": [], "note": "both reviews are by the same person"}
    a, b = complete[0]["judgments"], complete[1]["judgments"]
    disputed = [k for k in JUDGMENTS if a[k] != b[k]]
    if not disputed:
        return {"state": "agreed", "final": dict(a), "disagreements": []}
    adj = item.get("adjudication") or {}
    values = adj.get("judgments", {}) if isinstance(adj, dict) else {}
    adjudicator = _person(adj.get("adjudicator")) if isinstance(adj, dict) else ""
    independent = adjudicator and adjudicator not in (_person(complete[0]["reviewer"]), _person(complete[1]["reviewer"]))
    covered = all(isinstance(values.get(k), bool) or values.get(k) == "n/a" for k in disputed)
    if independent and covered and _nonblank(adj.get("decided_at")) and not [k for k in values if k not in disputed]:
        final = {k: (values[k] if k in disputed else a[k]) for k in JUDGMENTS}
        if not contradictions(final):
            return {"state": "adjudicated", "final": final, "disagreements": disputed}
    return {"state": "unresolved", "final": None, "disagreements": disputed}


GATE_READY = ("agreed", "adjudicated")


# ----------------------------------------------------------------------------- statistics


def wilson(successes: int, n: int, z: float = GATE["z"]) -> Optional[Tuple[float, float]]:
    """95% Wilson score interval (low, high), or None when n is 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> Dict[str, Any]:
    interval = wilson(k, n)
    return {
        "count": k, "n": n,
        "observed": (k / n) if n else None,
        "wilson_low": interval[0] if interval else None, "wilson_high": interval[1] if interval else None,
    }


def cohens_kappa(pairs: List[Tuple[bool, bool]]) -> Optional[float]:
    """Cohen's kappa for two raters on a yes/no judgment; None when undefined (no pairs, or no variation)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


# ----------------------------------------------------------------------------- metrics


def _rows(items: List[Dict[str, Any]], split: Optional[str], gate_only: bool) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out = []
    for item in items:
        res = resolve_item(item)
        ok = res["state"] in GATE_READY if gate_only else res["final"] is not None
        if ok and (split is None or item.get("split") == split):
            out.append((item, res["final"]))
    return out


def _criteria(rows: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    mcq = [(i, f) for i, f in rows if i.get("question_type") == "mcq"]
    bad = [(i, f) for i, f in rows if f["key_correct"] is False or f["unsupported_claims"] is True or f["misleading_distractors"] is True]
    c_mcq = _rate(sum(1 for _, f in mcq if f["single_defensible_option"] is True), len(mcq))
    c_mcq["measurable"] = len(mcq) > 0      # no MCQ in the sample: NOT MEASURABLE, never a pass or a failure
    mis = _rate(sum(1 for _, f in mcq if f["misleading_distractors"] is True), len(mcq))
    mis["measurable"] = len(mcq) > 0
    return {
        "a_answerable_and_key_correct": _rate(sum(1 for _, f in rows if f["answerable_from_transcript"] and f["key_correct"]), len(rows)),
        "b_wrong_key_or_misleading": _rate(len(bad), len(rows)),
        "b_key_false_about_subject": {"count": sum(1 for _, f in rows if f[BLOCKING_JUDGMENT] is True), "n": len(rows)},
        "c_range_contains_support": _rate(sum(1 for _, f in rows if f["evidence_range_contains_support"]), len(rows)),
        "c_mcq_single_defensible_option": c_mcq,
        "evidence_quote_valid": _rate(sum(1 for _, f in rows if f["evidence_quote_valid"]), len(rows)),
        "evidence_relevant": _rate(sum(1 for _, f in rows if f["evidence_relevant"]), len(rows)),
        "lecture_factually_incorrect": _rate(sum(1 for _, f in rows if f[SOURCE_ERROR_JUDGMENT]), len(rows)),
        "requires_visual": _rate(sum(1 for _, f in rows if f["requires_visual"]), len(rows)),
        "unsupported_claims": _rate(sum(1 for _, f in rows if f["unsupported_claims"]), len(rows)),
        "misleading_distractors": mis,
    }


def _blockers(items: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    """
    Release-blocking failures: a key that is a false statement about the subject. `confirmed` = the resolved
    judgment says so. `flagged_not_confirmed` = a reviewer recorded it and the item is not (yet) resolved to
    that. Counted from the reviews themselves, so no state (pending, unresolved, excluded) can hide one.
    """
    confirmed, flagged, by_cause = [], [], {}
    for item in items:
        if item.get("split") != split:
            continue
        res = resolve_item(item)
        marks = [r for r in item.get("reviews", []) if isinstance(r, dict) and isinstance(r.get("judgments"), dict)
                 and r["judgments"].get(BLOCKING_JUDGMENT) is True]
        if res["state"] in GATE_READY and res["final"].get(BLOCKING_JUDGMENT) is True:
            confirmed.append(item.get("item_id"))
            cause = "source_error_reproduced" if res["final"][SOURCE_ERROR_JUDGMENT] is True and not _model_error(res["final"]) else "model_error"
            by_cause.setdefault(cause, []).append(item.get("item_id"))
        elif marks:
            flagged.append(item.get("item_id"))
    return {"confirmed": confirmed, "flagged_not_confirmed": flagged, "confirmed_by_cause": by_cause}


def gate_eligibility(scorecard: Dict[str, Any], manifest: Optional[Dict[str, Any]]) -> List[str]:
    """Reasons this scorecard may NOT be given a release verdict. Empty means eligible. Missing evidence fails closed."""
    reasons: List[str] = []
    if scorecard.get("is_synthetic") is not False:
        reasons.append("the scorecard is synthetic or does not state is_synthetic false")
    run = scorecard.get("run") if isinstance(scorecard.get("run"), dict) else {}
    if run.get("source") != "recorded_real_run":
        reasons.append(f"run.source is '{run.get('source')}', not 'recorded_real_run'")
    probe = Report()
    _validate_run(scorecard, probe)
    reasons.extend(f"run provenance: {e}" for e in probe.errors)
    if manifest is None:
        reasons.append("no manifest was supplied, so the registered sample and its freeze state cannot be checked")
        return reasons
    mreport = mf.validate_manifest(manifest)
    if not mreport.ok:
        reasons.append(f"the manifest has {len(mreport.errors)} validation error(s)")
    if manifest.get("status") != "frozen":
        reasons.append(f"the manifest status is '{manifest.get('status')}', not 'frozen'")
    if manifest.get("is_synthetic") is not False:
        reasons.append("the manifest is synthetic or does not state is_synthetic false")
    binding = scorecard.get("manifest_binding") if isinstance(scorecard.get("manifest_binding"), dict) else {}
    if binding.get("sha256") != mf.canonical_sha256(manifest):
        reasons.append("the scorecard's manifest hash does not match this manifest")
    config = manifest.get("frozen_config") if isinstance(manifest.get("frozen_config"), dict) else {}
    for key in FROZEN_CONFIG_FIELDS:
        if run.get(key) != config.get(key) or is_placeholder(config.get(key)):
            reasons.append(f"run.{key} does not equal the frozen configuration")
    sreport = validate_scorecard(scorecard, manifest)
    if not sreport.ok:
        reasons.append(f"the scorecard has {len(sreport.errors)} validation error(s)")
    required = _confirmatory_slots(manifest)
    if len(required) != GATE["required_confirmatory_items"]:
        reasons.append(f"{len(required)} confirmatory slots are registered; the contract's gate needs exactly {GATE['required_confirmatory_items']}")
    return reasons


def _confirmatory_slots(manifest: Optional[Dict[str, Any]]) -> List[str]:
    lectures = {l.get("lecture_id"): l for l in (manifest or {}).get("lectures", []) if isinstance(l, dict)}
    return [s.get("sample_id") for s in (manifest or {}).get("samples", []) if isinstance(s, dict)
            and (lectures.get(s.get("lecture_id")) or {}).get("split") == "confirmatory"]


def _slot_accounting(items: List[Dict[str, Any]], manifest: Optional[Dict[str, Any]], subject: Optional[str] = None) -> Dict[str, Any]:
    """Every registered confirmatory slot is accounted for: fully reviewed, or one of the ways it is not."""
    by_slot = {i.get("sample_id"): i for i in items}
    out: Counter = Counter()
    required = _confirmatory_slots(manifest)
    if subject is not None:
        subject_of = {s.get("sample_id"): s.get("subject") for s in (manifest or {}).get("samples", []) if isinstance(s, dict)}
        required = [sid for sid in required if subject_of.get(sid) == subject]
    for sid in required:
        item = by_slot.get(sid)
        if item is None:
            out["missing"] += 1
            continue
        state = resolve_item(item)["state"]
        out["fully_reviewed" if state in GATE_READY else state] += 1
    keys = ("fully_reviewed", "pending", "single_review", "unresolved", "excluded", "missing")
    return {"required": len(required), **{k: out.get(k, 0) for k in keys}}


def _status(criterion: Dict[str, Any], passed: bool) -> str:
    if criterion.get("measurable") is False or criterion.get("n") == 0 or criterion.get("observed") is None:
        return "not_measurable"
    return "met" if passed else "not_met"


def _per_criterion(criteria: Dict[str, Any], use_wilson: bool = True) -> Dict[str, str]:
    """
    The contract's criteria a, b, c (thresholds unchanged) applied to one set of counted items.
    `use_wilson` is False for a single subject: the contract calls per-subject intervals wide and descriptive (section 11.2),
    so only the observed-rate parts of criterion a are applied there. (POLICY_DECISIONS.md D; needs confirmation.)
    """
    g = GATE
    a, b = criteria["a_answerable_and_key_correct"], criteria["b_wrong_key_or_misleading"]
    c1, c2 = criteria["c_range_contains_support"], criteria["c_mcq_single_defensible_option"]
    return {
        "a": _status(a, a["observed"] is not None and a["observed"] >= g["a_observed_min"]
                       and (not use_wilson or a["wilson_low"] >= g["a_wilson_lower_min"])),
        "b": _status(b, b["observed"] is not None and b["observed"] <= g["b_wrong_or_misleading_max"] and criteria["b_key_false_about_subject"]["count"] == 0),
        "c_range": _status(c1, c1["observed"] is not None and c1["observed"] >= g["c_range_contains_support_min"]),
        "c_mcq_single": _status(c2, c2["observed"] is not None and c2["observed"] >= g["c_mcq_single_defensible_min"]),
    }


def _model_error(f: Dict[str, Any]) -> bool:
    """An error the MODEL created: a wrong key, an unsupported claim or a misleading distractor (criterion b's composition)."""
    return f["key_correct"] is False or f["unsupported_claims"] is True or f["misleading_distractors"] is True


def _attribution(rows: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, int]:
    """
    Source-lecture errors and model-created errors, kept apart:
      clean                          lecture correct, nothing the model created is wrong
      model_error_only               lecture correct, the model created an error
      source_error_reproduced        the lecture is wrong and the model followed it, adding nothing false of its own
      source_error_and_model_error   the lecture is wrong AND the model created a separate error
    """
    out = {"clean": 0, "model_error_only": 0, "source_error_reproduced": 0, "source_error_and_model_error": 0}
    for _, f in rows:
        source, model = f[SOURCE_ERROR_JUDGMENT] is True, _model_error(f)
        out["source_error_and_model_error" if source and model else "source_error_reproduced" if source else "model_error_only" if model else "clean"] += 1
    return out


def _subject_basis(subject: str, criteria: Dict[str, Any], slots: Dict[str, Any], blockers: List[Any]) -> str:
    """Hash of exactly the figures an approver is shown, so an approval cannot outlive a change in the evidence."""
    keep = ("a_answerable_and_key_correct", "b_wrong_key_or_misleading", "c_range_contains_support", "c_mcq_single_defensible_option")
    figures = {k: [criteria[k]["count"], criteria[k]["n"]] for k in keep}
    figures["key_false_about_subject"] = criteria["b_key_false_about_subject"]["count"]
    return mf.canonical_sha256({"subject": subject, "figures": figures, "slots": slots, "blockers": sorted(map(str, blockers))})


def _subject_review_people(items: List[Dict[str, Any]], subject: str) -> set:
    people = set()
    for item in items:
        if item.get("subject") != subject:
            continue
        for r in item.get("reviews", []):
            if isinstance(r, dict):
                people.add(_person(r.get("reviewer")))
        adj = item.get("adjudication")
        if isinstance(adj, dict):
            people.add(_person(adj.get("adjudicator")))
    people.discard("")
    return people


def _subject_report(subject: str, rows, slots: Dict[str, Any], blockers: List[Any], items, scorecard, manifest) -> Dict[str, Any]:
    """Criteria status, blockers, slot accounting and approval state for one subject. No subject-level threshold is invented:
    the contract's own criteria are applied unchanged to the subject's own counted items, and ANY shortfall needs an approval."""
    criteria = _criteria(rows)
    per = _per_criterion(criteria, use_wilson=False)
    if slots["required"] == 0 or slots["fully_reviewed"] == 0:
        status = "no_evidence"
    elif slots["fully_reviewed"] != slots["required"]:
        status = "incomplete"
    elif blockers or "not_met" in per.values():
        status = "not_met"
    elif "not_measurable" in per.values():
        status = "not_measurable"
    else:
        status = "met"
    basis = _subject_basis(subject, criteria, slots, blockers)
    approval: Dict[str, Any] = {"state": "not_required", "record": None, "problems": []}
    if status != "met":
        approval["state"] = "pending"
        records = [a for a in (scorecard.get("subject_approvals") or []) if isinstance(a, dict) and a.get("subject") == subject]
        if records:
            a = records[0]
            why: List[str] = []
            authorized = {_person(n) for n in (manifest or {}).get("authorized_approvers", []) if _nonblank(n)}
            if a.get("decision") not in SUBJECT_APPROVAL_DECISIONS:
                why.append("the decision is not granted or denied")
            if is_placeholder(a.get("rationale")) or is_placeholder(a.get("approver")) or is_placeholder(a.get("decided_at")):
                why.append("approver, rationale and decided_at must all be recorded")
            if _person(a.get("approver")) not in authorized:
                why.append("the approver is not in the frozen manifest's authorized_approvers")
            if _person(a.get("approver")) in _subject_review_people(items, subject):
                why.append("the approver reviewed or adjudicated items of this subject and cannot also approve it")
            if a.get("basis_sha256") != basis:
                why.append("the approval was made on different figures than the current ones (basis hash differs)")
            if not why:
                approval = {"state": a["decision"], "record": {k: a.get(k) for k in ("approver", "rationale", "decided_at", "decision")}, "problems": []}
            else:
                approval["problems"] = why
    shown = ("a_answerable_and_key_correct", "b_wrong_key_or_misleading", "b_key_false_about_subject", "c_range_contains_support",
             "c_mcq_single_defensible_option", "lecture_factually_incorrect")
    return {"criteria": {k: v for k, v in criteria.items() if k in shown},
            "per_criterion": per, "status": status, "slots": slots, "release_blocking_confirmed": list(blockers),
            "error_attribution": _attribution(rows), "approval": approval, "basis_sha256": basis}


def _verdict(criteria: Dict[str, Any], accounting: Dict[str, Any], blockers: Dict[str, Any], reasons: List[str], synthetic: bool,
             subjects: Optional[Dict[str, Any]] = None, lecture_gaps: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    g = GATE
    per = _per_criterion(criteria)
    subjects = subjects or {}
    incomplete = accounting["required"] == 0 or accounting["fully_reviewed"] != accounting["required"]
    n_blockers = len(blockers["confirmed"])
    denied = [s for s, r in subjects.items() if r["approval"]["state"] == "denied"]
    waiting = [s for s, r in subjects.items() if r["status"] != "met" and r["approval"]["state"] != "granted"]
    pooled_met = None if incomplete or "not_measurable" in per.values() else all(v == "met" for v in per.values())
    blocking: List[str] = []
    if accounting["excluded"] > EXCLUSION_CAP_CONFIRMATORY:
        blocking.append(f"{accounting['excluded']} excluded confirmatory slot(s); the approved exclusion cap is {EXCLUSION_CAP_CONFIRMATORY} "
                        "and an excluded slot is neither replaced nor removed from the 80")
    for key, label in (("missing", "missing (no item for a registered slot)"), ("pending", "pending (no complete review)"),
                       ("single_review", "single-reviewed (never fills a slot)"), ("unresolved", "unresolved (disagreement without a valid adjudication)")):
        if accounting[key]:
            blocking.append(f"{accounting[key]} confirmatory slot(s) {label}")
    if n_blockers:
        blocking.append(f"{n_blockers} confirmed release-blocking failure(s): a key that is a false statement about the subject")
    blocking.extend(f"subject approval denied: {s}" for s in denied)
    if synthetic:
        overall = "not_evidence"
    elif reasons:
        overall = "not_eligible"
    elif n_blockers or denied:
        overall = "release_blocked"
    elif incomplete or "not_measurable" in per.values():
        overall = "not_evaluable"
    elif not all(v == "met" for v in per.values()):
        overall = "criteria_not_met"
    elif waiting:
        overall = "awaiting_subject_approval"
    else:
        overall = "criteria_a_to_c_met_pending_d"
    show = overall in ("criteria_a_to_c_met_pending_d", "awaiting_subject_approval", "criteria_not_met", "release_blocked")
    return {
        "overall": overall,
        "ineligible_because": reasons,
        "blocking_reasons": blocking,
        "evaluation_complete": (not incomplete) and not reasons,
        "pooled_criteria_met": pooled_met,
        "per_criterion": per if show else None,
        "subject_criteria": {s: r["status"] for s, r in subjects.items()},
        "subject_approvals": {s: r["approval"]["state"] for s, r in subjects.items()},
        "subjects_awaiting_approval": waiting,
        "shortfall_by_lecture": lecture_gaps or {},
        "release_blocking_failures_confirmatory": n_blockers,
        "release_blocking_failures_flagged_unconfirmed": len(blockers["flagged_not_confirmed"]),
        "gate_slots": accounting,
        "reviewed_confirmatory_items": accounting["fully_reviewed"], "needed_for_a_verdict": g["required_confirmatory_items"],
        "d_human_accessibility_checks": "not_measured_here",
        "meaning": "Never an approval to enable the feature: that needs criterion d (recorded human checks) and an explicit decision.",
    }


def _agreement(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    agreement: Dict[str, Any] = {}
    for key in JUDGMENTS:
        pairs = []
        for item in items:
            rs = [r for r in item.get("reviews", []) if isinstance(r, dict) and _is_complete(r, item.get("question_type"))]
            if len(rs) == 2 and all(isinstance(r["judgments"][key], bool) for r in rs):
                pairs.append((rs[0]["judgments"][key], rs[1]["judgments"][key]))
        agreement[key] = {"pairs": len(pairs), "percent_agreement": (sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs else None, "kappa": cohens_kappa(pairs)}
    return agreement


def compute_metrics(scorecard: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    items = [i for i in scorecard.get("items", []) if isinstance(i, dict)]
    resolutions = [resolve_item(i) for i in items]
    states = Counter(r["state"] for r in resolutions)
    synthetic = scorecard.get("is_synthetic") is not False

    confirmatory = _rows(items, "confirmatory", gate_only=True)
    if manifest:
        registered = set(_confirmatory_slots(manifest))
        if registered:      # a manifest with no registered slots cannot narrow anything; the verdict is ineligible anyway
            confirmatory = [(i, f) for i, f in confirmatory if i.get("sample_id") in registered]
    criteria = _criteria(confirmatory)
    development = _rows(items, "development", gate_only=False)
    blockers = {sp: _blockers(items, sp) for sp in ("confirmatory", "development", "pending")}
    accounting = _slot_accounting(items, manifest)
    reasons = gate_eligibility(scorecard, manifest)

    by_subject = {}
    for s in SUBJECTS:
        rows = [(i, f) for i, f in confirmatory if i.get("subject") == s]
        confirmed = [x for x in blockers["confirmatory"]["confirmed"] if any(i.get("item_id") == x and i.get("subject") == s for i in items)]
        report = _subject_report(s, rows, _slot_accounting(items, manifest, s), confirmed, items, scorecard, manifest)
        report["items_by_state"] = dict(Counter(resolve_item(i)["state"] for i in items if i.get("subject") == s and i.get("split") == "confirmatory"))
        by_subject[s] = {**report["criteria"], **{k: v for k, v in report.items() if k != "criteria"}}

    return {
        "is_synthetic": synthetic,
        "run": scorecard.get("run"),
        "items": len(items),
        "states": {k: states.get(k, 0) for k in ("pending", "single_review", "agreed", "adjudicated", "unresolved", "excluded")},
        "state_meaning": {
            "pending": "no complete review yet", "single_review": "one complete review: insufficient for the gate",
            "agreed": "two independent reviews agree (gate-ready)", "adjudicated": "two independent reviews, disputes decided by a third person (gate-ready)",
            "unresolved": "two reviews disagree and no valid adjudication", "excluded": "recorded exclusion (never counts toward a pass; the approved cap is 0)",
        },
        "gate_slots": accounting,
        "exclusions_by_reason": dict(Counter(r.get("reason") for r in resolutions if r["state"] == "excluded")),
        "splits_counted": {"development": len(development), "confirmatory": len(confirmatory)},
        "development_note": "Development-split items are reported apart and are never used for a claim or the gate.",
        "gate_basis": f"confirmatory split, gate-ready items only ({REVIEWS_REQUIRED_FOR_GATE} independent reviews)",
        "criteria_confirmatory": criteria,
        "error_attribution_confirmatory": _attribution(confirmatory),
        "criteria_development_descriptive": _criteria(development),
        "by_subject_descriptive": by_subject,
        "release_blockers": blockers,
        "question_type_mix_reviewed": dict(Counter(i.get("question_type") for i, _ in confirmatory)),
        "reviewer_agreement": _agreement(items),
        "subject_failures": _subject_failures(items),
        "verdict": _verdict(criteria, accounting, blockers["confirmatory"], reasons, synthetic, by_subject, _lecture_gaps(items, manifest)),
        "not_measured_by_this_tool": NOT_MEASURED,
    }


def _lecture_gaps(items: List[Dict[str, Any]], manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """For each confirmatory lecture that could not fill all its registered slots: how many, and in what state."""
    if not manifest:
        return {}
    slot_ids = set(_confirmatory_slots(manifest))
    lectures = sorted({s.get("lecture_id") for s in manifest.get("samples", []) if isinstance(s, dict) and s.get("sample_id") in slot_ids})
    gaps = {}
    for lecture_id in lectures:
        sub = {**manifest, "samples": [s for s in manifest["samples"] if isinstance(s, dict) and s.get("lecture_id") == lecture_id]}
        acc = _slot_accounting([i for i in items if i.get("lecture_id") == lecture_id], sub)
        if acc["fully_reviewed"] != acc["required"]:
            gaps[lecture_id] = acc
    return gaps


def _subject_failures(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures: Counter = Counter()
    for item in items:
        if resolve_item(item)["final"] is None:
            continue
        for r in item.get("reviews", []):
            sf = (r or {}).get("subject_failure") or {}
            if sf.get("category"):
                failures[(item.get("subject"), sf["category"])] += 1
                break
    return [{"subject": s, "category": c, "count": n} for (s, c), n in sorted(failures.items())]


NOT_MEASURED = [
    "Criterion d: keyboard, NVDA and 200% zoom checks by a person (docs/QUIZ_CONTRACT.md section 10).",
    "How many candidates screening rejected, and why: needs the 'omitted' counts of real responses, which exist only after a real run.",
    "Whether a correct-looking question hides a caption error: automatic captions are the transcript, so a copied error looks valid to every check here.",
    "Distractor plausibility, hint quality and explanation accuracy beyond what reviewers record in the judgments and notes.",
    "Latency, token use, cost and structured-output adherence of the real model.",
    "Learning outcomes for students.",
    "Whether the recorded provenance is TRUE: the tool checks that it is present, well-formed and consistent, not that a run really happened.",
]


# ----------------------------------------------------------------------------- mechanical evidence checks


def automatic_evidence_checks(scorecard: Dict[str, Any], manifest: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Mechanical checks of every item's evidence against the transcript, kept SEPARATE from the human
    judgments. Also lists items where the two disagree, for a reviewer to look at again (the human
    judgment still stands; a mismatch is a prompt to re-check, not an override).
    """
    segments: Dict[str, Any] = {}
    for lec in manifest.get("lectures", []):
        ref = lec.get("transcript_ref") or {}
        segments[lec.get("lecture_id")] = tx.load_segments(ref.get("path", ""), ref.get("lecture_id", ""), base_dir)
    counts: Dict[str, Counter] = defaultdict(Counter)
    flags: List[Dict[str, Any]] = []
    for item in scorecard.get("items", []):
        segs = segments.get(item.get("lecture_id"))
        final = resolve_item(item)["final"]
        for k, e in enumerate(item.get("evidence", [])):
            res = tx.check_evidence(e.get("quote"), e.get("start_seconds"), e.get("end_seconds"), segs)
            for name, value in res.items():
                counts[name][{True: "true", False: "false", None: "unknown"}[value]] += 1
            if final is not None:
                if res["quote_in_transcript"] is False and final["evidence_quote_valid"] is True:
                    flags.append({"item_id": item.get("item_id"), "evidence": k, "flag": "reviewer says the quote is valid but it is not in the transcript"})
                if res["quote_in_range"] is False and res["quote_elsewhere"] and final["evidence_range_contains_support"] is True:
                    flags.append({"item_id": item.get("item_id"), "evidence": k, "flag": "reviewer says the range contains the support but the quote is elsewhere"})
    return {"counts": {k: dict(v) for k, v in counts.items()}, "review_flags": flags,
            "note": "'unknown' means no transcript was available to check; it is not a failure."}
