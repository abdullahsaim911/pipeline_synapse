"""Manifest validation, summaries, transcript cross-checks and the draft builder (synthetic data only)."""

import csv
import json
import os
import tempfile
import unittest

from tests.quiz_eval import fixtures as fx
from quiz_eval import manifest as mf
from quiz_eval.constants import EXCLUSION_REASONS, SUBJECTS


def errors(data):
    return mf.validate_manifest(data).errors


class StructureTests(unittest.TestCase):
    def test_a_minimal_manifest_is_valid(self):
        self.assertEqual(errors(fx.manifest()), [])

    def test_basic_shape_errors(self):
        for bad in (None, [], "x", 5):
            self.assertTrue(errors(bad), repr(bad))
        for key, value in (("manifest_version", 2), ("name", ""), ("status", "final"), ("is_synthetic", "yes"), ("lectures", {}), ("samples", None)):
            data = fx.manifest()
            data[key] = value
            self.assertTrue(errors(data), key)

    def test_is_synthetic_must_be_stated(self):
        data = fx.manifest()
        del data["is_synthetic"]
        self.assertTrue(any("is_synthetic" in e for e in errors(data)))

    def test_lecture_rules(self):
        cases = {
            "duplicate id": [fx.lecture(), fx.lecture()],
            "bad video id": [fx.lecture(video_id="short")],
            "subject": [dict(fx.lecture(), subject="History")],
            "split": [dict(fx.lecture(), split="train")],
            "transcript_ref": [dict(fx.lecture(), transcript_ref={"path": "x"})],
            "negative words": [fx.lecture(words=-5)],
            "string words": [dict(fx.lecture(), expected_word_count="12", word_count_status="measured")],
            "null count but not pending": [dict(fx.lecture(), expected_word_count=None, word_count_status="measured")],
            "count but pending": [dict(fx.lecture(words=10), word_count_status="pending")],
            "not an object": ["x"],
            "bad id": [dict(fx.lecture(), lecture_id="")],
        }
        for name, lectures in cases.items():
            self.assertTrue(errors(fx.manifest(lectures=lectures)), name)

    def test_word_count_pending_is_valid_not_a_failure(self):
        self.assertEqual(errors(fx.manifest(lectures=[fx.lecture(words=None)])), [])

    def test_sample_ids_and_links(self):
        s = fx.sample()
        self.assertTrue(errors(fx.manifest(samples=[s, fx.sample()])))                                   # duplicate id
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(lecture_id="NOPE")])))                    # unknown lecture
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(subject="Physics")])))                    # subject mismatch
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(sample_id="a")])))                        # id too short
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(question_type="short")])))               # unsupported type
        self.assertTrue(errors(fx.manifest(samples=[dict(fx.sample(), review={"status": "maybe"})])))  # status
        self.assertEqual(errors(fx.manifest(samples=[s])), [])


class ReviewStatusTests(unittest.TestCase):
    def test_a_complete_approved_item_is_valid(self):
        self.assertEqual(errors(fx.manifest(samples=[fx.approved_sample()])), [])

    def test_an_approval_cannot_be_half_filled(self):
        base = fx.approved_sample()
        breakages = {
            "no question": {"question": None}, "no rationale": {"key_rationale": " "}, "no evidence": {"evidence": None},
            "not answerable": {"answerable_from_evidence": False}, "answerable unknown": {"answerable_from_evidence": None},
            "visual": {"requires_visual": True}, "visual unknown": {"requires_visual": None}, "type pending": {"question_type": "pending"},
            "no reviewer": {"review": dict(base["review"], reviewer=None)}, "no date": {"review": dict(base["review"], reviewed_at="")},
            "bad quote": {"evidence": dict(base["evidence"], quote="")}, "end before start": {"evidence": dict(base["evidence"], start_seconds=30.0)},
            "negative time": {"evidence": dict(base["evidence"], start_seconds=-1.0)}, "nan time": {"evidence": dict(base["evidence"], end_seconds=float("nan"))},
            "key not a choice": {"question": dict(base["question"], correct_choice_id="z")},
            "two choices": {"question": dict(base["question"], choices=base["question"]["choices"][:2])},
            "five choices": {"question": dict(base["question"], choices=[{"id": c, "text": c} for c in "abcde"])},
            "duplicate choice ids": {"question": dict(base["question"], choices=[{"id": "a", "text": "x"}] * 3)},
        }
        for name, over in breakages.items():
            self.assertTrue(errors(fx.manifest(samples=[dict(base, **over)])), name)

    def test_true_false_needs_a_boolean_answer(self):
        tf = fx.approved_sample(question_type="tf") if False else dict(fx.approved_sample(), question_type="tf", question={"prompt": "Beta precedes gamma?", "answer": True})
        self.assertEqual(errors(fx.manifest(samples=[tf])), [])
        self.assertTrue(errors(fx.manifest(samples=[dict(tf, question={"prompt": "p", "answer": "true"})])))

    def test_rejected_and_excluded_items_record_why(self):
        rejected = fx.sample(status="rejected")
        self.assertTrue(errors(fx.manifest(samples=[rejected])))
        rejected["review"].update(reviewer="reviewer-A", notes="key was wrong")
        self.assertEqual(errors(fx.manifest(samples=[rejected])), [])
        excluded = fx.sample(status="excluded")
        self.assertTrue(errors(fx.manifest(samples=[excluded])))
        excluded["review"].update(exclusion_reason="requires_visual")
        self.assertTrue(errors(fx.manifest(samples=[excluded])))     # still needs notes
        excluded["review"].update(notes="needs the slide")
        self.assertEqual(errors(fx.manifest(samples=[excluded])), [])
        excluded["review"]["exclusion_reason"] = "bored"
        self.assertTrue(errors(fx.manifest(samples=[excluded])))
        self.assertIn("requires_visual", EXCLUSION_REASONS)

    def test_a_pending_item_may_be_empty_but_what_is_present_must_be_well_formed(self):
        self.assertEqual(errors(fx.manifest(samples=[fx.sample()])), [])
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(question_type="mcq", question={"prompt": "p", "choices": []})])))
        self.assertTrue(errors(fx.manifest(samples=[fx.sample(evidence={"quote": "q", "start_seconds": 5, "end_seconds": 1})])))


class FrozenTests(unittest.TestCase):
    def frozen(self, **over):
        lectures = [fx.lecture(f"L{i}", subject=SUBJECTS[i % 5], split="development" if i < 5 else "confirmatory", video_id=f"{i:011d}") for i in range(25)]
        data = fx.manifest(lectures=lectures, status="frozen", selection_rule="every 3rd question by id", frozen_at="2000-01-01",
                          frozen_config=dict(fx.FROZEN_CONFIG))
        data.update(over)
        return data

    def test_a_contract_shaped_frozen_manifest_has_no_errors_or_warnings(self):
        # 5 development (one per subject) and 20 confirmatory would be 4 per subject: 25 lectures in total
        data = self.frozen()
        data["samples"] = [fx.sample(f"S{i:03d}", lecture_id=f"L{5 + i % 20}", subject=SUBJECTS[(5 + i % 20) % 5]) for i in range(80)]
        report = mf.validate_manifest(data)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_a_frozen_manifest_without_eighty_confirmatory_slots_warns(self):
        report = mf.validate_manifest(self.frozen())
        self.assertEqual(report.errors, [])
        self.assertTrue(any("80" in w for w in report.warnings))

    def test_frozen_requires_a_rule_a_date_and_every_split(self):
        self.assertTrue(errors(self.frozen(selection_rule="")))
        self.assertTrue(errors(self.frozen(frozen_at=None)))
        data = self.frozen()
        data["lectures"][0]["split"] = "pending"
        self.assertTrue(errors(data))
        self.assertEqual(errors(fx.manifest(lectures=[fx.lecture(split="pending")], status="draft")), [])

    def test_a_frozen_manifest_must_pin_prompt_model_and_config(self):
        self.assertTrue(errors(self.frozen(frozen_config=None)))
        self.assertTrue(errors(self.frozen(frozen_config={})))
        for key in fx.FROZEN_CONFIG:
            for bad in ("", "PENDING", None):
                cfg = dict(fx.FROZEN_CONFIG, **{key: bad})
                self.assertTrue(errors(self.frozen(frozen_config=cfg)), (key, bad))
        for key in ("prompt_sha256", "config_sha256"):
            self.assertTrue(errors(self.frozen(frozen_config=dict(fx.FROZEN_CONFIG, **{key: "XYZ"}))), key)
        self.assertEqual(errors(fx.manifest(status="draft")), [])      # a draft needs none of it

    def test_the_canonical_hash_ignores_key_order_and_changes_with_content(self):
        a = {"x": 1, "y": [1, 2], "z": {"p": 1, "q": 2}}
        b = {"z": {"q": 2, "p": 1}, "y": [1, 2], "x": 1}
        self.assertEqual(mf.canonical_sha256(a), mf.canonical_sha256(b))
        self.assertNotEqual(mf.canonical_sha256(a), mf.canonical_sha256(dict(a, x=2)))
        self.assertRegex(mf.canonical_sha256(a), r"^[0-9a-f]{64}$")

    def test_departures_from_the_contract_shape_are_warnings_not_errors(self):
        data = self.frozen()
        data["lectures"][0]["split"] = "confirmatory"
        report = mf.validate_manifest(data)
        self.assertEqual(report.errors, [])
        self.assertTrue(any("development" in w for w in report.warnings))
        self.assertTrue(any("confirmatory" in w for w in report.warnings))

    def test_no_protocol_warnings_before_freezing(self):
        self.assertEqual(mf.validate_manifest(fx.manifest()).warnings, [])


class SummaryTests(unittest.TestCase):
    def test_pending_stays_pending_and_is_counted_by_subject(self):
        lectures = [fx.lecture("A1", "Mathematics", video_id="A" * 11), fx.lecture("B1", "Physics", video_id="B" * 11)]
        samples = [
            fx.approved_sample("A1-S1", lecture_id="A1"),
            fx.sample("A1-S2", lecture_id="A1"),
            fx.sample("B1-S1", lecture_id="B1", subject="Physics", status="excluded"),
        ]
        samples[2]["review"].update(exclusion_reason="requires_visual", notes="slide")
        summary = mf.summarize_manifest(fx.manifest(lectures=lectures, samples=samples))
        self.assertEqual(summary["samples_by_status"], {"pending": 1, "approved": 1, "rejected": 0, "excluded": 1})
        self.assertEqual(summary["by_subject"]["Mathematics"], {"lectures": 1, "pending": 1, "approved": 1, "rejected": 0, "excluded": 0})
        self.assertEqual(summary["by_subject"]["Physics"]["excluded"], 1)
        self.assertEqual(summary["by_subject"]["Chemistry"]["lectures"], 0)
        self.assertEqual(summary["exclusions_by_reason"], {"requires_visual": 1})
        self.assertEqual(summary["word_counts_pending"], 2)

    def test_empty_manifest_summarises_to_zeros(self):
        summary = mf.summarize_manifest(fx.manifest(lectures=[]))
        self.assertEqual((summary["lectures"], summary["samples"]), (0, 0))


class TranscriptCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fx.write_transcript(self.tmp.name)

    def lectures_result(self, words):
        data = fx.manifest(lectures=[fx.lecture(words=words)])
        return mf.check_transcripts(data, self.tmp.name)["lectures"]["SYN01"]

    def test_word_counts_match_mismatch_pending_and_unknown_are_different_things(self):
        total = fx.words_in_transcript()
        self.assertEqual(self.lectures_result(total)["word_count"], "match")
        self.assertEqual(self.lectures_result(total + 1)["word_count"], "mismatch")
        self.assertEqual(self.lectures_result(None)["word_count"], "pending")
        data = fx.manifest(lectures=[fx.lecture(words=10, path="missing.jsonl")])
        self.assertEqual(mf.check_transcripts(data, self.tmp.name)["lectures"]["SYN01"]["word_count"], "unknown")

    def test_a_sample_quote_is_checked_against_its_own_range(self):
        good = fx.approved_sample()
        wrong_range = fx.approved_sample("SYN01-S2", evidence={"quote": "gamma follows from beta", "start_seconds": 30.0, "end_seconds": 40.0})
        invented = fx.approved_sample("SYN01-S3", evidence={"quote": "something nobody said", "start_seconds": 10.0, "end_seconds": 20.0})
        out = mf.check_transcripts(fx.manifest(samples=[good, wrong_range, invented]), self.tmp.name)["samples"]
        self.assertEqual((out["SYN01-S1"]["quote_in_range"], out["SYN01-S1"]["quote_elsewhere"]), (True, False))
        self.assertEqual((out["SYN01-S2"]["quote_in_range"], out["SYN01-S2"]["quote_elsewhere"]), (False, True))     # real quote, wrong time
        self.assertEqual((out["SYN01-S3"]["quote_in_transcript"], out["SYN01-S3"]["quote_elsewhere"]), (False, False))  # made-up quote


class BuilderTests(unittest.TestCase):
    def make_dataset(self, directory, n_words="9"):
        os.makedirs(os.path.join(directory, "ds"))
        fx.write_transcript(os.path.join(directory, "ds"), lectures=("SYN01", "SYN02"), name="transcript_segments.jsonl")
        with open(os.path.join(directory, "ds", "lecture_manifest.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["lecture_id", "subject", "video_id", "lecture_title", "n_words"])
            writer.writeheader()
            writer.writerow({"lecture_id": "SYN01", "subject": "Mathematics", "video_id": "AAAAAAAAAAA", "lecture_title": "t", "n_words": n_words})
            writer.writerow({"lecture_id": "SYN02", "subject": "Physics", "video_id": "BBBBBBBBBBB", "lecture_title": "t", "n_words": ""})

    def test_builds_a_draft_with_no_split_and_no_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_dataset(tmp)
            data = mf.build_from_dataset("ds", "SYNTHETIC draft", tmp)
        self.assertEqual((data["status"], data["is_synthetic"], data["samples"]), ("draft", False, []))
        self.assertEqual([l["split"] for l in data["lectures"]], ["pending", "pending"])
        first, second = data["lectures"]
        self.assertEqual((first["expected_word_count"], first["word_count_status"]), (9, "from_dataset_manifest"))
        self.assertEqual(second["word_count_status"], "measured")                      # no count in the CSV: measured from the transcript
        self.assertEqual(second["expected_word_count"], fx.words_in_transcript())
        self.assertEqual(mf.validate_manifest(data).errors, [])

    def test_the_builder_only_reads_its_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_dataset(tmp)
            before = sorted(os.listdir(os.path.join(tmp, "ds")))
            mf.build_from_dataset("ds", "x", tmp)
            self.assertEqual(sorted(os.listdir(os.path.join(tmp, "ds"))), before)


class ShippedFilesTests(unittest.TestCase):
    ROOT = fx.PROJECT_ROOT

    def load(self, *parts):
        with open(os.path.join(self.ROOT, "quiz_eval", *parts), encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_template_is_valid_and_marked_synthetic(self):
        data = self.load("templates", "manifest.template.json")
        self.assertEqual(mf.validate_manifest(data).errors, [])
        self.assertTrue(data["is_synthetic"])
        self.assertEqual(data["status"], "template")

    def test_the_scorecard_template_is_valid_synthetic_and_can_never_pass(self):
        from quiz_eval import scorecard as sc
        card = self.load("templates", "scorecard.template.json")
        self.assertEqual(sc.validate_scorecard(card).errors, [])
        self.assertTrue(card["is_synthetic"])
        self.assertEqual(card["run"]["source"], "pending")
        self.assertEqual(sc.compute_metrics(card, self.load("templates", "manifest.template.json"))["verdict"]["overall"], "not_evidence")

    def test_the_draft_manifest_has_no_invented_review_data(self):
        data = self.load("manifests", "stem25_draft.json")
        self.assertEqual(mf.validate_manifest(data).errors, [])
        self.assertEqual((data["status"], data["samples"]), ("draft", []))
        self.assertTrue(all(l["split"] == "pending" for l in data["lectures"]))
        self.assertEqual(len(data["lectures"]), 25)
        counts = {s: sum(1 for l in data["lectures"] if l["subject"] == s) for s in SUBJECTS}
        self.assertEqual(set(counts.values()), {5})

    def test_the_draft_word_counts_match_the_real_transcripts_when_the_dataset_is_present(self):
        path = os.path.join(self.ROOT, "dataset", "synapse_stem25", "transcript_segments.jsonl")
        if not os.path.isfile(path):
            self.skipTest("dataset not present")
        result = mf.check_transcripts(self.load("manifests", "stem25_draft.json"))
        self.assertEqual({v["word_count"] for v in result["lectures"].values()}, {"match"})


if __name__ == "__main__":
    unittest.main()
