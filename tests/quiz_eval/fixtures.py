"""
SYNTHETIC fixtures for the evaluation tooling tests. Everything here is invented to exercise code paths:
no lecture, quote, question or judgment below is real, and every document built here says is_synthetic true.
"""

import copy
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quiz_eval.constants import JUDGMENTS, NA_ONLY_FOR_TF, SUBJECTS  # noqa: E402

TRANSCRIPT_LINES = [
    "synthetic caption one talks about alpha and beta",
    "synthetic caption two says that gamma follows from beta",
    "synthetic caption three introduces delta and epsilon",
    "synthetic caption four explains that epsilon is larger than delta",
    "synthetic caption five wraps up the topic",
]


def write_transcript(directory, lectures=("SYN01",), name="segments.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        for lecture_id in lectures:
            for i, text in enumerate(TRANSCRIPT_LINES):
                handle.write(json.dumps({"lecture_id": lecture_id, "start_seconds": i * 10.0, "end_seconds": i * 10.0 + 10.0, "text": text}) + "\n")
    return path


def words_in_transcript():
    return sum(len(t.split()) for t in TRANSCRIPT_LINES)


def lecture(lecture_id="SYN01", subject="Mathematics", split="confirmatory", path="segments.jsonl", words=None, video_id="AAAAAAAAAAA"):
    return {
        "lecture_id": lecture_id, "video_id": video_id, "subject": subject, "split": split,
        "transcript_ref": {"path": path, "lecture_id": lecture_id},
        "expected_word_count": words, "word_count_status": "pending" if words is None else "from_dataset_manifest",
    }


def sample(sample_id="SYN01-S1", lecture_id="SYN01", subject="Mathematics", status="pending", **over):
    base = {
        "sample_id": sample_id, "lecture_id": lecture_id, "subject": subject, "question_type": "pending",
        "requires_visual": None, "answerable_from_evidence": None, "question": None, "key_rationale": None, "evidence": None,
        "review": {"status": status, "reviewer": None, "reviewed_at": None, "notes": None, "exclusion_reason": None},
    }
    base.update(over)
    return base


def approved_sample(sample_id="SYN01-S1", **over):
    base = dict(
        status="approved", question_type="mcq", requires_visual=False, answerable_from_evidence=True,
        question={"prompt": "Which follows from beta?", "choices": [{"id": "a", "text": "gamma"}, {"id": "b", "text": "delta"}, {"id": "c", "text": "zeta"}], "correct_choice_id": "a"},
        key_rationale="Caption two says gamma follows from beta.",
        evidence={"quote": "gamma follows from beta", "start_seconds": 10.0, "end_seconds": 20.0},
        review={"status": "approved", "reviewer": "reviewer-A", "reviewed_at": "2000-01-01", "notes": "synthetic", "exclusion_reason": None},
    )
    base.update(over)
    return sample(sample_id, **base)


def manifest(lectures=None, samples=None, **over):
    data = {
        "manifest_version": 1, "name": "SYNTHETIC test manifest", "status": "draft", "is_synthetic": True,
        "lectures": lectures if lectures is not None else [lecture()], "samples": samples if samples is not None else [],
    }
    data.update(over)
    return data


def judgments(question_type="mcq", **over):
    j = {k: True for k in JUDGMENTS}
    j.update({"unsupported_claims": False, "misleading_distractors": False, "requires_visual": False, "key_false_about_subject": False,
              "lecture_factually_incorrect": False})
    if question_type == "tf":
        j.update({k: "n/a" for k in NA_ONLY_FOR_TF})
    j.update(over)
    return j


def review(question_type="mcq", reviewer="reviewer-A", **over):
    return {"reviewer": reviewer, "completed_at": "2000-01-01", "judgments": judgments(question_type, **over), "subject_failure": {"category": None, "notes": None}, "notes": None}


def item(item_id="G1", subject="Mathematics", split="confirmatory", question_type="mcq", reviews=None, **over):
    """By default an item carries two agreeing reviews by two different people (gate-ready)."""
    base = {
        "item_id": item_id, "sample_id": None, "lecture_id": "SYN01", "subject": subject, "split": split, "question_type": question_type,
        "question": {"prompt": "Synthetic question?", "choices": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}], "correct_choice_id": "a"},
        "evidence": [{"start_seconds": 10.0, "end_seconds": 20.0, "quote": "gamma follows from beta"}],
        "reviews": [review(question_type), review(question_type, reviewer="reviewer-B")] if reviews is None else reviews, "adjudication": None, "exclusion": None,
    }
    base.update(over)
    return base


def scorecard(items=None, synthetic=True, source="synthetic", **over):
    data = {
        "scorecard_version": 2, "is_synthetic": synthetic, "manifest_name": "SYNTHETIC test manifest",
        "run": {"run_id": "synthetic-run", "source": source, "prompt_version": "none", "model": "none", "generated_at": None},
        "items": items if items is not None else [],
    }
    data.update(over)
    return data


FAKE_SHA = "ab" * 32
FAKE_SHA_2 = "cd" * 32
FROZEN_CONFIG = {"prompt_version": "TEST-FIXTURE-prompt", "prompt_sha256": FAKE_SHA, "model": "TEST-FIXTURE-model", "config_sha256": FAKE_SHA_2}


def frozen_world(n=80, wrong=0, **overrides):
    """
    A frozen, non-synthetic manifest with n confirmatory slots and a scorecard that says it is a recorded
    real run of it. EVERYTHING here is an invented test fixture (ids start TEST-FIXTURE): it is not a real run,
    and no real run, review or approval exists. Returns (manifest, scorecard).
    """
    lectures = [lecture(f"L{k}", subject=SUBJECTS[k], split="confirmatory", video_id=f"{k:011d}") for k in range(5)]
    lectures += [lecture(f"D{k}", subject=SUBJECTS[k], split="development", video_id=f"{k + 5:011d}") for k in range(5)]
    samples, items = [], []
    for i in range(n):
        k = i % 5
        sid = f"S{i + 1:03d}"
        samples.append(sample(sid, lecture_id=f"L{k}", subject=SUBJECTS[k]))
        position = i // 5 + 1          # slots of one lecture are filled in the model's returned order
        j = {"key_correct": False} if i < wrong else {}
        items.append(item(f"G{i + 1}", subject=SUBJECTS[k], sample_id=sid, lecture_id=f"L{k}", returned_position=position,
                          reviews=[review("mcq", **j), review("mcq", reviewer="reviewer-B", **j)]))
    m = manifest(lectures=lectures, samples=samples, status="frozen", is_synthetic=False, name="TEST-FIXTURE manifest",
                 authorized_approvers=["Approver One"],
                 selection_rule="TEST-FIXTURE rule", frozen_at="2000-01-01", frozen_config=dict(FROZEN_CONFIG))
    m.update(overrides)
    from quiz_eval.manifest import canonical_sha256
    card = scorecard(items, synthetic=False, source="recorded_real_run", manifest_name=m["name"])
    card["manifest_binding"] = {"name": m["name"], "sha256": canonical_sha256(m)}
    card["run"] = {"run_id": "TEST-FIXTURE-NOT-A-RUN", "source": "recorded_real_run", "generated_at": "2000-01-01T00:00:00Z",
                   "code_version": "TEST-FIXTURE", "raw_responses_sha256": "ef" * 32, **FROZEN_CONFIG}
    return m, card


def rebind(m, card):
    """Re-bind a scorecard to a manifest after a test edited the manifest."""
    from quiz_eval.manifest import canonical_sha256
    card["manifest_binding"] = {"name": m["name"], "sha256": canonical_sha256(m)}
    return card


def subjects_cycle(n):
    return [SUBJECTS[i % len(SUBJECTS)] for i in range(n)]


def many_items(n, wrong=0, subject_cycle=True, split="confirmatory", question_type="mcq"):
    """n fully reviewed items; the first `wrong` have a wrong key."""
    out = []
    for i in range(n):
        j = {"key_correct": False} if i < wrong else {}
        out.append(item(f"G{i + 1}", subject=SUBJECTS[i % 5] if subject_cycle else "Mathematics", split=split, question_type=question_type,
                        reviews=[review(question_type, **j), review(question_type, reviewer="reviewer-B", **j)]))
    return out


def deep(data):
    return copy.deepcopy(data)
