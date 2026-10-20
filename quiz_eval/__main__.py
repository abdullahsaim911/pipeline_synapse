"""
Command line for the model-free evaluation tooling.

    python -m quiz_eval check MANIFEST.json [--scorecard SCORECARD.json] [--transcripts] [--json]
    python -m quiz_eval build --dataset dataset/synapse_stem25 --name NAME --out MANIFEST.json [--force]

`check` validates, summarises, and (with a scorecard) computes metrics from reviewed items only.
It reads files; it never calls a model, a provider or any network service. Exit codes: 0 clean,
1 validation errors, 2 bad usage.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from quiz_eval import manifest as mf
from quiz_eval import scorecard as sc


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(rate: Dict[str, Any]) -> str:
    if rate.get("n") == 0 or rate.get("observed") is None:
        return "no items judged"
    low = rate.get("wilson_low")
    return f"{rate['count']}/{rate['n']} = {rate['observed']:.1%} (95% lower bound {low:.1%})"


def _text(summary: Dict[str, Any], manifest_report: mf.Report, metrics: Optional[Dict[str, Any]], scorecard_report: Optional[mf.Report],
          transcripts: Optional[Dict[str, Any]], auto: Optional[Dict[str, Any]]) -> str:
    out: List[str] = []
    tag = "  [SYNTHETIC: test data, not evaluation evidence]" if summary.get("is_synthetic") else ""
    out.append(f"MANIFEST {summary['name']} (status: {summary['status']}){tag}")
    out.append(f"  lectures: {summary['lectures']}  by split: {summary['lectures_by_split']}  word counts still pending: {summary['word_counts_pending']}")
    out.append(f"  samples: {summary['samples']}  by review status: {summary['samples_by_status']}  by type: {summary['samples_by_type']}")
    out.append("  by subject (lectures / pending / approved / rejected / excluded):")
    for subject, row in summary["by_subject"].items():
        out.append(f"    {subject:<17} {row['lectures']:>3} / {row['pending']} / {row['approved']} / {row['rejected']} / {row['excluded']}")
    if summary["exclusions_by_reason"]:
        out.append(f"  exclusions: {summary['exclusions_by_reason']}")
    if transcripts:
        counts = {}
        for row in transcripts["lectures"].values():
            counts[row["word_count"]] = counts.get(row["word_count"], 0) + 1
        out.append(f"  transcript word counts vs expected: {counts}  ('unknown' = transcript not available, 'pending' = no expected count)")
    for label, report in (("manifest", manifest_report), ("scorecard", scorecard_report)):
        if report is None:
            continue
        out.append(f"{label.upper()} VALIDATION: {'ok' if report.ok else str(len(report.errors)) + ' error(s)'}, {len(report.warnings)} warning(s)")
        out.extend(f"  ERROR   {e}" for e in report.errors[:50])
        out.extend(f"  warning {w}" for w in report.warnings[:50])
    if metrics:
        m = metrics
        out.append("")
        out.append(f"SCORECARD (run: {(m['run'] or {}).get('source')}){'  [SYNTHETIC: not evidence]' if m['is_synthetic'] else ''}")
        out.append(f"  items: {m['items']}  states: {m['states']}")
        if m["exclusions_by_reason"]:
            out.append(f"  exclusions: {m['exclusions_by_reason']}")
        out.append(f"  gate slots (confirmatory, registered): {m['gate_slots']}")
        out.append(f"  counted by split: {m['splits_counted']}  (gate basis: {m['gate_basis']}; development is never used for a claim)")
        out.append("  HUMAN-JUDGED rates, confirmatory items that have a complete review:")
        c = m["criteria_confirmatory"]
        out.append(f"    a  answerable and key correct : {_pct(c['a_answerable_and_key_correct'])}   (contract: >= 90% observed, lower bound >= 80%)")
        out.append(f"    b  wrong key or misleading    : {_pct(c['b_wrong_key_or_misleading'])}   (contract: <= 5%)")
        out.append(f"       key false about the subject: {c['b_key_false_about_subject']['count']} of {c['b_key_false_about_subject']['n']}   (contract: any blocks release)")
        out.append(f"    c  range contains support     : {_pct(c['c_range_contains_support'])}   (contract: >= 90%)")
        out.append(f"       MCQ with one defensible    : {_pct(c['c_mcq_single_defensible_option'])}   (contract: >= 85%)")
        out.append(f"    mix of reviewed items: {m['question_type_mix_reviewed']}  (recorded, no ratio is required)")
        mis = c["misleading_distractors"]
        if mis.get("measurable") is False:
            out.append("       (no MCQ among the counted items: MCQ criteria are NOT MEASURABLE, neither passed nor failed)")
        rb = m["release_blockers"]
        out.append("  RELEASE BLOCKERS (key false about the subject; shown whatever the sample size):")
        for split, row in rb.items():
            out.append(f"    {split:<13} confirmed: {row['confirmed']}  flagged by a reviewer, not confirmed: {row['flagged_not_confirmed']}")
        out.append("  per subject (confirmatory, gate-ready items; descriptive):")
        for subject, row in m["by_subject_descriptive"].items():
            out.append(f"    {subject:<17} a: {_pct(row['a_answerable_and_key_correct'])}; b: {_pct(row['b_wrong_key_or_misleading'])}; "
                       f"false-about-subject: {row['b_key_false_about_subject']['count']}; c: {_pct(row['c_range_contains_support'])}; "
                       f"one-defensible: {_pct(row['c_mcq_single_defensible_option'])}; states: {row['items_by_state']}")
            out.append(f"      status: {row['status']}; slots: {row['slots']}; approval: {row['approval']['state']}"
                       f"{' (' + '; '.join(row['approval']['problems']) + ')' if row['approval']['problems'] else ''}; "
                       f"errors by origin: {row['error_attribution']}; basis for an approval: {row['basis_sha256']}")
        d = m["criteria_development_descriptive"]
        out.append(f"  development split (descriptive only, never used for the gate): a {_pct(d['a_answerable_and_key_correct'])}")
        out.append(f"  errors by origin (confirmatory): {m['error_attribution_confirmatory']}  "
                   "(source_error_* = the lecture itself is wrong; model_error_only = the model created it)")
        v = m["verdict"]
        for reason in v["blocking_reasons"]:
            out.append(f"  BLOCKING: {reason}")
        if v["shortfall_by_lecture"]:
            out.append(f"  shortfall by lecture: {v['shortfall_by_lecture']}")
        out.append(f"  evaluation complete: {v['evaluation_complete']}; pooled criteria met: {v['pooled_criteria_met']}; "
                   f"subject criteria: {v['subject_criteria']}; subject approvals: {v['subject_approvals']}")
        for reason in v["ineligible_because"]:
            out.append(f"  NOT ELIGIBLE: {reason}")
        if v.get("per_criterion"):
            out.append(f"  per criterion: {v['per_criterion']}")
        out.append(f"  VERDICT: {v['overall']}  ({v['reviewed_confirmatory_items']} reviewed confirmatory items; {v['needed_for_a_verdict']} needed for a verdict)")
        out.append(f"           criterion d (human accessibility checks): {v['d_human_accessibility_checks']}. {v['meaning']}")
        if auto:
            out.append(f"  AUTOMATIC evidence checks (not human judgments): {auto['counts']}")
            out.extend(f"    re-check: {f['item_id']}: {f['flag']}" for f in auto["review_flags"][:20])
        out.append("  Not measured by this tool:")
        out.extend(f"    - {line}" for line in m["not_measured_by_this_tool"])
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="quiz_eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate and summarise a manifest, optionally with a scorecard")
    check.add_argument("manifest")
    check.add_argument("--scorecard")
    check.add_argument("--transcripts", action="store_true", help="also read the referenced transcript files (read-only)")
    check.add_argument("--base-dir", help="directory that relative transcript paths are resolved against (default: the project root)")
    check.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    build = sub.add_parser("build", help="write a DRAFT manifest from a dataset directory (no splits, no samples)")
    build.add_argument("--dataset", required=True)
    build.add_argument("--name", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--base-dir")
    build.add_argument("--force", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0

    try:
        if args.command == "build":
            if os.path.exists(args.out) and not args.force:
                print(f"{args.out} exists; pass --force to overwrite it", file=sys.stderr)
                return 2
            data = mf.build_from_dataset(args.dataset, args.name, args.base_dir)
            with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            print(f"wrote {args.out}: {len(data['lectures'])} lectures, no splits assigned, no samples")
            return 0

        manifest = _load(args.manifest)
        manifest_report = mf.validate_manifest(manifest)
        summary = mf.summarize_manifest(manifest) if isinstance(manifest, dict) else {}
        transcripts = mf.check_transcripts(manifest, args.base_dir) if (args.transcripts and manifest_report.ok) else None
        metrics = scorecard_report = auto = None
        if args.scorecard:
            card = _load(args.scorecard)
            scorecard_report = sc.validate_scorecard(card, manifest if manifest_report.ok else None)
            if isinstance(card, dict) and isinstance(card.get("items"), list):
                # Metrics are shown even when validation failed, so a blocker is never hidden; the verdict then
                # says the scorecard is not eligible.
                metrics = sc.compute_metrics(card, manifest if manifest_report.ok else None)
                if scorecard_report.ok and args.transcripts and manifest_report.ok:
                    auto = sc.automatic_evidence_checks(card, manifest, args.base_dir)
        ok = manifest_report.ok and (scorecard_report is None or scorecard_report.ok)  # a verdict never changes the exit code
        if args.json:
            print(json.dumps({
                "ok": ok, "manifest_summary": summary, "manifest_errors": manifest_report.errors, "manifest_warnings": manifest_report.warnings,
                "scorecard_errors": scorecard_report.errors if scorecard_report else None, "metrics": metrics, "transcript_checks": transcripts, "automatic_evidence_checks": auto,
            }, indent=2, ensure_ascii=False))
        elif summary:
            print(_text(summary, manifest_report, metrics, scorecard_report, transcripts, auto))
        else:
            print("\n".join(manifest_report.errors))
        return 0 if ok else 1
    except (OSError, ValueError) as exc:
        print(f"could not read input: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
