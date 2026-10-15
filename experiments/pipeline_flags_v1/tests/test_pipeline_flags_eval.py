"""Offline tests for pipeline_flags_eval.py. No model call, no model loading, no network.

    python -B -m unittest discover -s experiments/pipeline_flags_v1/tests -p "test_*.py"
"""
import copy, importlib.util, os, sys, tempfile, unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import pipeline_flags_eval as P  # noqa: E402

spec = importlib.util.spec_from_file_location("frozen_bm25_eval_t", os.path.join(REPO, "eval", "bm25_eval.py"))
BM25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BM25)
Q = P._qa()


def cue(i, start, end, text, lecture="LX"):
    return {"segment_id": f"{lecture}_S{i:04d}", "start_seconds": start, "end_seconds": end, "text": text}


def rec(qid, lec="L1", **kv):
    base = {"question_id": qid, "lecture_id": lec, "group": "PRIMARY_ANSWERABLE", "hit_rate_at_5": 0.0, "mrr_at_10": 0.0, "evidence_hit": 0.0, "strict_containment": 0.0,
            "full_interval_coverage": 0.0}
    base.update(kv)
    return base


class DecisionRuleTests(unittest.TestCase):
    def test_three_way_rule_and_boundaries(self):
        self.assertEqual(P.decide(0.01, 0.2), "improvement")
        self.assertEqual(P.decide(-0.2, -0.01), "decline")
        self.assertEqual(P.decide(-0.05, 0.05), "no_demonstrated_difference")
        self.assertEqual(P.decide(0.0, 0.1), "no_demonstrated_difference")
        self.assertEqual(P.decide(-0.1, 0.0), "no_demonstrated_difference")

    def test_paired_uses_first_minus_second_and_counts_questions(self):
        a = [rec(f"Q{i}", f"L{i % 4}", hit_rate_at_5=1.0, mrr_at_10=0.5) for i in range(40)]
        b = [rec(f"Q{i}", f"L{i % 4}", hit_rate_at_5=0.0, mrr_at_10=0.25) for i in range(40)]
        r = P.paired(BM25, a, b, ("hit_rate_at_5", "mrr_at_10"), 300, 20261013, 0.95)
        self.assertEqual((r["hit_rate_at_5"]["difference"], r["hit_rate_at_5"]["decision"]), (1.0, "improvement"))
        self.assertEqual((r["mrr_at_10"]["difference"], r["mrr_at_10"]["first_better"], r["mrr_at_10"]["tied"]), (0.25, 40, 0))
        rev = P.paired(BM25, b, a, ("hit_rate_at_5",), 300, 20261013, 0.95)
        self.assertEqual((rev["hit_rate_at_5"]["difference"], rev["hit_rate_at_5"]["decision"]), (-1.0, "decline"))

    def test_identical_arms_have_no_demonstrated_difference(self):
        a = [rec(f"Q{i}", hit_rate_at_5=float(i % 2)) for i in range(30)]
        r = P.paired(BM25, a, [dict(x) for x in a], ("hit_rate_at_5",), 200, 1, 0.95)
        self.assertEqual((r["hit_rate_at_5"]["decision"], r["hit_rate_at_5"]["ci_low"], r["hit_rate_at_5"]["ci_high"]), ("no_demonstrated_difference", 0.0, 0.0))

    def test_misaligned_arms_are_rejected(self):
        with self.assertRaises(P.EvalError):
            P.paired(BM25, [rec("A"), rec("B")], [rec("B"), rec("A")], ("hit_rate_at_5",), 10, 1, 0.95)

    def test_clustered_contrast(self):
        a = [rec(f"Q{i}", f"L{i % 5}", evidence_hit=1.0) for i in range(25)]
        b = [rec(f"Q{i}", f"L{i % 5}", evidence_hit=0.0) for i in range(25)]
        self.assertEqual(P.clustered(BM25, a, b, ("evidence_hit",), 200, 1, 0.95)["evidence_hit"]["decision"], "improvement")

    def test_latency_stats(self):
        s = P.latency_stats([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.assertEqual((s["median"], s["max"], s["n"]), (55.0, 100, 10))
        self.assertIsNone(P.latency_stats([]))


class RecommendationRuleTests(unittest.TestCase):
    def contrast(self, hit="no_demonstrated_difference", mrr="no_demonstrated_difference"):
        return {"hit_rate_at_5": {"decision": hit}, "mrr_at_10": {"decision": mrr}}

    def evid(self, hit="no_demonstrated_difference", strict="no_demonstrated_difference"):
        return {"evidence_hit": {"decision": hit}, "strict_containment": {"decision": strict}}

    def test_all_criteria_met(self):
        r = P.recommend("f", self.contrast(hit="improvement"), self.evid(), 100.0, 0.0, 500.0, 2.0)
        self.assertTrue(r["verdict"].startswith("RECOMMEND"))
        self.assertTrue(all(r["criteria"].values()))

    def test_no_ranking_improvement_blocks_it(self):
        r = P.recommend("f", self.contrast(), self.evid(), 10.0, 0.0, 500.0)
        self.assertEqual(r["verdict"], "DO NOT ENABLE YET")
        self.assertFalse(r["criteria"]["ranking_improvement"])

    def test_a_decline_on_either_ranking_metric_blocks_it_even_with_an_improvement(self):
        r = P.recommend("f", self.contrast(hit="improvement", mrr="decline"), self.evid(), 10.0, 0.0, 500.0)
        self.assertFalse(r["criteria"]["ranking_improvement"])

    def test_evidence_decline_or_more_gate_failures_block_it(self):
        self.assertFalse(P.recommend("f", self.contrast(hit="improvement"), self.evid(strict="decline"), 10.0, 0.0, 500.0)["criteria"]["evidence_no_decline"])
        self.assertFalse(P.recommend("f", self.contrast(hit="improvement"), self.evid(), 10.0, 0.02, 500.0)["criteria"]["evidence_no_decline"])

    def test_latency_budget_and_cold_start(self):
        slow = P.recommend("f", self.contrast(hit="improvement"), self.evid(), 900.0, 0.0, 500.0, 1.0)
        self.assertFalse(slow["criteria"]["warm_latency_within_budget"])
        cold = P.recommend("f", self.contrast(hit="improvement"), self.evid(), 100.0, 0.0, 500.0, 21.0)
        self.assertFalse(cold["criteria"]["cold_start_ok"])
        self.assertTrue(any("warm-up" in x for x in cold["reasons"]))
        self.assertEqual(cold["verdict"], "DO NOT ENABLE YET")


class MappingTests(unittest.TestCase):
    CUES = [cue(1, 0.0, 6.0, "welcome um to the anti derivative"), cue(2, 6.0, 12.0, "uh"), cue(3, 12.0, 18.0, "the acronym li8 helps choose u"),
            cue(4, 18.0, 24.0, "then integrate dv")]

    def setUp(self):
        self.ec = Q.R.EvidenceConfig()
        self.base = BM25.build_lecture_context(self.CUES, Q.R, Q.normalize, self.ec)

    def test_asr_context_keeps_cue_ids_by_start_time_and_drops_empty_captions(self):
        a = P.asr_context(Q, self.base, self.ec)
        self.assertEqual([s.start for s in a["segs"]], [0.0, 6.0, 12.0, 18.0][:0] or [0.0, 12.0, 18.0])
        union = set().union(*a["chunk_cues"])
        self.assertEqual(union, {"LX_S0001", "LX_S0003", "LX_S0004"})            # LX_S0002 ("uh") is gone
        self.assertEqual(self.base["dropped"], set())                              # baseline keeps it

    def test_asr_cleaned_text_is_what_is_ranked(self):
        a = P.asr_context(Q, self.base, self.ec)
        self.assertIn("LIATE", " ".join(s.text for s in a["segs"]))
        ranked_base, _, _ = BM25.rank_question(Q.R, "What is LIATE?", self.base)
        ranked_asr, _, _ = BM25.rank_question(Q.R, "What is LIATE?", a)
        self.assertTrue(ranked_base and ranked_asr)

    def test_cue_ids_of_plan_maps_blocks_and_checks_the_text(self):
        plan = Q.R.select_evidence("What is the acronym?", self.base["segs"], config=Q.R.EvidenceConfig(max_chars=2000, full_transcript_chars=5000))
        ids = P.cue_ids_of_plan(plan, self.base["segs"], self.base["start_to_id"])
        self.assertEqual(ids, {"LX_S0001", "LX_S0002", "LX_S0003", "LX_S0004"})
        with self.assertRaises(P.EvalError):                                         # the wrong segment list must be detected
            P.cue_ids_of_plan(plan, P.asr_context(Q, self.base, self.ec)["segs"], self.base["start_to_id"])

    def test_failed_gate_means_empty_evidence(self):
        self.assertEqual(P.cue_ids_of_plan(Q.R.EvidencePlan("no_evidence"), self.base["segs"], self.base["start_to_id"]), set())

    def test_evidence_metrics(self):
        ivs = [{"A1", "A2"}, {"B1"}]
        m = P.evidence_metrics({"A1", "B1"}, ivs)
        self.assertEqual((m["evidence_hit"], m["interval_coverage"], m["full_interval_coverage"], m["strict_containment"]), (1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(m["gold_cue_fraction"], 2 / 3)
        self.assertEqual(P.evidence_metrics(set(), ivs)["evidence_hit"], 0.0)
        self.assertIsNone(P.evidence_metrics({"A1"}, []))


class ArmSettingsTests(unittest.TestCase):
    def test_arms_have_exactly_the_registered_flags(self):
        arms = P.make_evidence_arms(Q)
        self.assertEqual(list(arms), list(P.EVID_ARMS))
        self.assertEqual({a: v[2] for a, v in arms.items()}, {"A0_baseline": False, "A1_hybrid": False, "A2_asr": True, "A12_hybrid_asr": True})
        self.assertEqual({a: v[0].config.asr_sanitize for a, v in arms.items()}, {"A0_baseline": False, "A1_hybrid": False, "A2_asr": True, "A12_hybrid_asr": True})
        self.assertIs(arms["A0_baseline"][1].inner, Q.R.select_evidence)
        self.assertIsInstance(arms["A1_hybrid"][1].inner, Q.hybrid.HybridSelector)
        self.assertIsNot(arms["A1_hybrid"][1].inner, arms["A12_hybrid_asr"][1].inner)       # separate caches
        self.assertEqual((arms["A1_hybrid"][1].inner.config.candidates, arms["A1_hybrid"][1].inner.config.rerank_top_k), (12, 4))
        self.assertIsNone(arms["A1_hybrid"][1].inner.embedder._model)                       # nothing loaded when arms are built

    def test_capture_remembers_the_last_plan_and_times_the_selector(self):
        calls = []

        def inner(*a, **k):
            calls.append(1)
            return "plan%d" % len(calls)
        cap = P.Capture(inner)
        cap("q", [])
        cap("q", [])
        self.assertEqual((cap.plan, cap.calls), ("plan2", 2))
        self.assertGreaterEqual(cap.seconds, 0.0)
        cap.reset()
        self.assertEqual((cap.plan, cap.calls, cap.seconds), (None, 0, 0.0))

    def test_instant_model_never_leaves_the_process(self):
        m = P.InstantModel(Q.Result)
        self.assertEqual(m.generate("p").text, "The instructor says this.")
        self.assertEqual(m.calls, 1)


class FaithfulnessPilotHelpersTests(unittest.TestCase):
    def test_perturb_numbers_changes_every_number_and_counts_them(self):
        text, n = P.perturb_numbers("there are 46 chromosomes and 23 pairs, not 7")
        self.assertEqual(n, 3)
        for old in ("46", "23", " 7"):
            self.assertNotIn(old, text)
        self.assertEqual(P.perturb_numbers("no digits here"), ("no digits here", 0))

    def test_fabricated_sentence_is_unrelated_to_the_benchmark_lectures_and_unsupported_by_a_typical_passage(self):
        checker = Q.faith.LexicalSupportChecker(Q.faith.FaithfulnessConfig(mode="shadow"))
        r = checker.check(P.FABRICATED_SENTENCE, ["the derivative of x squared is two x and the integral reverses it"])
        self.assertEqual(r.score, 0.0)


class RegistrationGateTests(unittest.TestCase):
    def test_wrong_hash_is_refused(self):
        with self.assertRaises(P.EvalError):
            P.load_registration(expected="0" * 64)

    def test_thresholds_and_budgets_are_the_documented_ones(self):
        self.assertEqual((P.PASSES, P.COLD_START_BUDGET_S, P.LATENCY_BUDGET_MS["A1_hybrid"], P.LATENCY_BUDGET_MS["A2_asr"]), (3, 5.0, 500.0, 50.0))
        self.assertEqual(P.THRESHOLDS, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))

    def test_ask_variables_abort_verification(self):
        reg = {"pins": {g: {} for g in ("frozen_config", "frozen_harness", "saved_baseline", "saved_m3", "code", "dataset")}, "models": {}}
        os.environ["ASK_HYBRID_RETRIEVAL"] = "1"
        try:
            with self.assertRaises(P.EvalError):
                P.verify_all(reg, check_models=False)
        finally:
            del os.environ["ASK_HYBRID_RETRIEVAL"]

    def test_a_changed_pin_is_reported(self):
        reg = {"pins": {g: {} for g in ("frozen_config", "frozen_harness", "saved_baseline", "saved_m3", "code", "dataset")}, "models": {}}
        reg["pins"]["code"] = {"qa/hybrid.py": "f" * 64}
        with self.assertRaises(P.EvalError) as cm:
            P.verify_all(reg, check_models=False)
        self.assertIn("qa/hybrid.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
