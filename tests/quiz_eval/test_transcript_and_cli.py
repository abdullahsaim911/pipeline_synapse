"""Transcript helpers, the command line, and the guarantee that the tooling is model-free."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests.quiz_eval import fixtures as fx
from quiz_eval import __main__ as cli
from quiz_eval import transcript as tx


class LoadSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_missing_file_is_none_but_an_unknown_lecture_is_empty(self):
        self.assertIsNone(tx.load_segments("nope.jsonl", "SYN01", self.tmp.name))
        fx.write_transcript(self.tmp.name)
        self.assertEqual(tx.load_segments("segments.jsonl", "OTHER", self.tmp.name), [])

    def test_only_the_requested_lecture_and_only_usable_rows(self):
        path = os.path.join(self.tmp.name, "s.jsonl")
        rows = [
            {"lecture_id": "A", "start_seconds": 0, "end_seconds": 5, "text": "good"},
            {"lecture_id": "B", "start_seconds": 0, "end_seconds": 5, "text": "other lecture"},
            {"lecture_id": "A", "start_seconds": 5, "end_seconds": 4, "text": "end before start"},
            {"lecture_id": "A", "start_seconds": -1, "end_seconds": 4, "text": "negative"},
            {"lecture_id": "A", "start_seconds": 6, "end_seconds": 9, "text": "   "},
            {"lecture_id": "A", "start_seconds": "6", "end_seconds": 9, "text": "string time"},
            {"lecture_id": "A", "start_seconds": True, "end_seconds": 9, "text": "bool time"},
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(r) for r in rows) + "\nnot json\n\n[1, 2]\n")
        segments = tx.load_segments("s.jsonl", "A", self.tmp.name)
        self.assertEqual([s["text"] for s in segments], ["good"])

    def test_segments_are_sorted_and_counted_by_whitespace(self):
        path = os.path.join(self.tmp.name, "s.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for start, text in ((10, "three little words"), (0, "two words")):
                handle.write(json.dumps({"lecture_id": "A", "start_seconds": start, "end_seconds": start + 5, "text": text}) + "\n")
        segments = tx.load_segments("s.jsonl", "A", self.tmp.name)
        self.assertEqual([s["start"] for s in segments], [0.0, 10.0])
        self.assertEqual(tx.word_count(segments), 5)
        self.assertIsNone(tx.word_count(None))
        self.assertIsNone(tx.word_count([]))
        self.assertEqual(tx.lecture_end(segments), 15.0)

    def test_the_file_is_only_read(self):
        path = fx.write_transcript(self.tmp.name)
        before = os.stat(path).st_mtime_ns, read_bytes(path)
        tx.load_segments("segments.jsonl", "SYN01", self.tmp.name)
        self.assertEqual((os.stat(path).st_mtime_ns, read_bytes(path)), before)


class CheckEvidenceTests(unittest.TestCase):
    def segs(self):
        return [{"start": i * 10.0, "end": i * 10.0 + 10.0, "text": t} for i, t in enumerate(fx.TRANSCRIPT_LINES)]

    def test_in_range_elsewhere_and_absent(self):
        r = tx.check_evidence("gamma follows from beta", 10, 20, self.segs())
        self.assertEqual((r["quote_in_range"], r["quote_elsewhere"], r["quote_in_transcript"]), (True, False, True))
        r = tx.check_evidence("gamma follows from beta", 40, 50, self.segs())
        self.assertEqual((r["quote_in_range"], r["quote_elsewhere"], r["quote_in_transcript"]), (False, True, True))
        r = tx.check_evidence("an invented sentence", 10, 20, self.segs())
        self.assertEqual((r["quote_in_range"], r["quote_elsewhere"], r["quote_in_transcript"]), (False, False, False))

    def test_unknown_is_not_false(self):
        r = tx.check_evidence("gamma", 10, 20, None)
        self.assertEqual((r["quote_in_range"], r["quote_in_transcript"], r["range_inside_lecture"]), (None, None, None))
        self.assertIs(r["range_well_formed"], True)       # needs no transcript
        self.assertIsNone(tx.check_evidence("gamma", 10, 20, [])["quote_in_range"])

    def test_matching_ignores_case_punctuation_and_caption_markers(self):
        self.assertTrue(tx.check_evidence("GAMMA follows -- from beta!", 10, 20, self.segs())["quote_in_range"])
        self.assertTrue(tx.check_evidence("Synthetic Caption Two says that gamma", 10, 20, self.segs())["quote_in_range"])

    def test_range_checks(self):
        self.assertTrue(tx.check_evidence("alpha", 0, 50, self.segs())["range_inside_lecture"])
        self.assertFalse(tx.check_evidence("alpha", 0, 500, self.segs())["range_inside_lecture"])
        for bad in ((20, 10), (-1, 5), (float("nan"), 5), (0, float("inf")), ("0", 5), (True, 5), (None, None)):
            self.assertFalse(tx.check_evidence("alpha", bad[0], bad[1], self.segs())["range_well_formed"], str(bad))
            self.assertIsNone(tx.check_evidence("alpha", bad[0], bad[1], self.segs())["quote_in_range"])

    def test_blank_or_non_string_quotes_are_false_not_unknown(self):
        for quote in ("", "   ", None, 5, "!!!"):
            self.assertFalse(tx.check_evidence(quote, 10, 20, self.segs())["quote_in_transcript"], repr(quote))

    def test_the_time_tolerance_is_small_and_explicit(self):
        near = tx.check_evidence("gamma follows from beta", 21.5, 25, self.segs(), tolerance=2.0)   # caption two ends at 20
        far = tx.check_evidence("gamma follows from beta", 23.0, 25, self.segs(), tolerance=2.0)
        self.assertTrue(near["quote_in_range"])
        self.assertFalse(far["quote_in_range"])


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fx.write_transcript(self.tmp.name)

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return path

    def test_check_summarises_a_valid_manifest(self):
        path = self.write("m.json", fx.manifest(lectures=[fx.lecture(words=fx.words_in_transcript())], samples=[fx.approved_sample()]))
        code, out, _ = run_cli("check", path, "--transcripts", "--base-dir", self.tmp.name)
        self.assertEqual(code, 0)
        self.assertIn("SYNTHETIC", out)
        self.assertIn("approved", out)
        self.assertIn("'match': 1", out)

    def test_check_fails_on_an_invalid_manifest(self):
        bad = fx.manifest()
        bad["status"] = "final"
        code, out, _ = run_cli("check", self.write("bad.json", bad))
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out)

    def test_check_with_a_scorecard_prints_metrics_and_a_verdict_that_is_not_evidence_for_synthetic_data(self):
        m = self.write("m.json", fx.manifest(lectures=[fx.lecture()]))
        s = self.write("s.json", fx.scorecard(fx.many_items(6, subject_cycle=False)))
        code, out, _ = run_cli("check", m, "--scorecard", s, "--transcripts", "--base-dir", self.tmp.name)
        self.assertEqual(code, 0)
        self.assertIn("HUMAN-JUDGED", out)
        self.assertIn("6/6", out)
        self.assertIn("VERDICT: not_evidence", out)
        self.assertIn("AUTOMATIC evidence checks", out)
        self.assertIn("Not measured by this tool", out)

    def test_an_invalid_scorecard_exits_1_and_is_never_eligible_for_a_verdict(self):
        m = self.write("m.json", fx.manifest())
        bad = fx.scorecard([fx.item(), fx.item()])
        code, out, _ = run_cli("check", m, "--scorecard", self.write("s.json", bad))
        self.assertEqual(code, 1)
        self.assertIn("duplicate item_id", out)
        self.assertNotIn("criteria_a_to_c_met", out)         # metrics may still be shown (blockers stay visible), never a pass

    def test_json_output_is_machine_readable(self):
        m = self.write("m.json", fx.manifest())
        s = self.write("s.json", fx.scorecard(fx.many_items(3, subject_cycle=False)))
        code, out, _ = run_cli("check", m, "--scorecard", s, "--json")
        data = json.loads(out)
        self.assertEqual((code, data["ok"]), (0, True))
        self.assertEqual(data["metrics"]["verdict"]["overall"], "not_evidence")

    def test_unreadable_input_exits_2_without_a_traceback(self):
        code, _, err = run_cli("check", os.path.join(self.tmp.name, "missing.json"))
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err)
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertEqual(run_cli("check", bad)[0], 2)

    def test_bad_usage_exits_2(self):
        self.assertEqual(run_cli("frobnicate")[0], 2)
        self.assertEqual(run_cli()[0], 2)

    def test_build_writes_a_draft_and_refuses_to_overwrite(self):
        ds = os.path.join(self.tmp.name, "ds")
        os.makedirs(ds)
        fx.write_transcript(ds, name="transcript_segments.jsonl")
        with open(os.path.join(ds, "lecture_manifest.csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("lecture_id,subject,video_id,lecture_title,n_words\nSYN01,Mathematics,AAAAAAAAAAA,t,\n")
        out_path = os.path.join(self.tmp.name, "draft.json")
        code, out, _ = run_cli("build", "--dataset", "ds", "--base-dir", self.tmp.name, "--name", "SYNTHETIC", "--out", out_path)
        self.assertEqual(code, 0)
        self.assertIn("no splits assigned, no samples", out)
        data = read_json(out_path)
        self.assertEqual((data["status"], data["samples"], data["lectures"][0]["split"]), ("draft", [], "pending"))
        again = run_cli("build", "--dataset", "ds", "--base-dir", self.tmp.name, "--name", "x", "--out", out_path)
        self.assertEqual(again[0], 2)
        self.assertEqual(read_json(out_path)["name"], "SYNTHETIC")      # untouched
        self.assertEqual(run_cli("build", "--dataset", "ds", "--base-dir", self.tmp.name, "--name", "x", "--out", out_path, "--force")[0], 0)


class ModelFreeTests(unittest.TestCase):
    PACKAGE = os.path.join(fx.PROJECT_ROOT, "quiz_eval")

    def test_no_source_file_imports_a_model_a_provider_the_endpoint_or_a_network_library(self):
        import ast
        banned = {"llm_interface", "google", "ollama", "requests", "httpx", "urllib", "urllib3", "socket", "http", "fastapi", "api", "qa", "subprocess"}
        for name in sorted(os.listdir(self.PACKAGE)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(self.PACKAGE, name), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])
            self.assertEqual(imported & banned, set(), name)
            attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertNotIn("environ", attributes, name)      # no credentials, no configuration read

    def test_importing_the_package_loads_no_provider_or_web_stack(self):
        probe = ("import sys; import quiz_eval, quiz_eval.manifest, quiz_eval.scorecard, quiz_eval.transcript, quiz_eval.__main__ as m; "
                 "print(sorted(n for n in sys.modules if n.split('.')[0] in ('llm_interface','requests','httpx','fastapi','google','api','qa')))")
        done = subprocess.run([sys.executable, "-c", probe], cwd=fx.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(done.stdout.strip(), "[]", done.stderr[-300:])

    def test_the_pinned_dataset_and_code_are_not_written_by_the_tooling(self):
        # building a draft from the real dataset must leave it byte-identical
        import hashlib
        target = os.path.join(fx.PROJECT_ROOT, "dataset", "synapse_stem25", "lecture_manifest.csv")
        if not os.path.isfile(target):
            self.skipTest("dataset not present")
        digest = lambda: hashlib.sha256(read_bytes(target)).hexdigest()
        before = digest()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_cli("build", "--dataset", "dataset/synapse_stem25", "--name", "x", "--out", os.path.join(tmp, "d.json"))[0], 0)
        self.assertEqual(digest(), before)


if __name__ == "__main__":
    unittest.main()
