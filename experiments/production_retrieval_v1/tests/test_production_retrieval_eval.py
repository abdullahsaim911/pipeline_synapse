"""Tests for production_retrieval_eval.py (P1). Synthetic lectures only; the language model is never called.

    python -B -m unittest discover -s experiments/production_retrieval_v1/tests -p "test_*.py"
"""
import copy, os, sys, tempfile, unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import production_retrieval_eval as P  # noqa: E402
from qa import retrieval as R, service as S  # noqa: E402
from qa.transcript import normalize_segments  # noqa: E402
from llm_interface.providers import GenerationResult  # noqa: E402

P1 = P.load_config()
BM25 = P.load_frozen_harness(P1)


def cue(i, start, end, text, lecture='LX'):
    return {'segment_id': f'{lecture}_S{i:04d}', 'start_seconds': start, 'end_seconds': end, 'text': text}


def lecture(n=60, words='alpha beta gamma delta'):
    """A synthetic lecture of n 10-second cues; cue k mentions the unique word wordK."""
    return [cue(k + 1, k * 10.0, k * 10.0 + 10.0, f'{words} word{k} filler talk about topic{k}') for k in range(n)]


def context(cues):
    return BM25.build_lecture_context(cues, R, normalize_segments, R.EvidenceConfig())


class TestRegistration(unittest.TestCase):
    def test_wrong_config_hash_fails_loudly(self):
        with self.assertRaises(P.P1Error):
            P.load_config(expected='0' * 64)

    def test_registered_files_match_disk(self):
        self.assertGreater(P.verify_registered_files(P1), 8)

    def test_changed_registered_file_is_reported(self):
        bad = copy.deepcopy(P1)
        bad['frozen_inputs']['bm25_harness']['eval/bm25_eval.py'] = 'f' * 64
        with self.assertRaises(P.P1Error):
            P.verify_registered_files(bad)
        with self.assertRaises(P.P1Error):
            P.load_frozen_harness(bad)

    def test_ask_environment_variable_aborts(self):
        os.environ['ASK_MAX_PROMPT_TOKENS'] = '123'
        try:
            with self.assertRaises(P.P1Error):
                P.verify_inputs()
        finally:
            del os.environ['ASK_MAX_PROMPT_TOKENS']

    def test_inputs_verify_in_clean_environment(self):
        p1, cfg, bm25, pins = P.verify_inputs()
        self.assertEqual(cfg['freeze']['protocol_version'], 'bm25_v1.0')
        self.assertGreater(pins['frozen_config_pins'], 10)


class TestEvidenceMetrics(unittest.TestCase):
    IVS = [{'A1', 'A2'}, {'B1'}]

    def test_hand_calculated_cases(self):
        m = P.evidence_metrics({'A1', 'B1'}, self.IVS)
        self.assertEqual((m['evidence_hit'], m['interval_coverage'], m['full_interval_coverage'], m['strict_containment']), (1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(m['gold_cue_fraction'], 2 / 3)
        m = P.evidence_metrics({'A1', 'A2', 'B1', 'Z'}, self.IVS)
        self.assertEqual((m['strict_containment'], m['gold_cue_fraction']), (1.0, 1.0))
        m = P.evidence_metrics({'A2'}, self.IVS)
        self.assertEqual((m['evidence_hit'], m['interval_coverage'], m['full_interval_coverage']), (1.0, 0.5, 0.0))
        m = P.evidence_metrics(set(), self.IVS)
        self.assertEqual((m['evidence_hit'], m['interval_coverage'], m['gold_cue_fraction']), (0.0, 0.0, 0.0))

    def test_no_gold_means_unscoreable(self):
        self.assertIsNone(P.evidence_metrics({'A1'}, []))


class TestProductionCall(unittest.TestCase):
    def run_question(self, cues, question):
        ctx = context(cues)
        plan, outcome, calls = P.production_plan(S, GenerationResult, question, ctx['segs'])
        return ctx, plan, outcome, calls

    def test_select_evidence_is_restored_after_the_call(self):
        original = S.select_evidence
        self.run_question(lecture(300), 'What is said about word30?')
        self.assertIs(S.select_evidence, original)

    def test_stand_in_replaces_the_model_and_is_counted(self):
        ctx, plan, outcome, calls = self.run_question(lecture(300), 'What is said about word30?')
        self.assertEqual(calls, 1)                                     # exactly one (fake) model call
        self.assertEqual(outcome.status, 'answered')
        self.assertTrue(plan.ok)

    def test_matching_question_returns_evidence_containing_the_matching_cue(self):
        ctx, plan, outcome, calls = self.run_question(lecture(300), 'What is said about word40?')
        ev = P.evidence_cue_ids(plan, ctx)
        self.assertIn('LX_S0041', ev)                                  # cue k=40 is cue number 41
        self.assertEqual(plan.mode, 'retrieval')
        self.assertLess(len(ev), 80)                                   # a strict subset of the lecture is sent

    def test_gate_failure_gives_empty_evidence_and_no_model_call(self):
        ctx, plan, outcome, calls = self.run_question(lecture(300), 'Who won the football world cup in Brazil?')
        self.assertFalse(plan.ok)
        self.assertEqual((calls, outcome.status), (0, 'insufficient_context'))
        self.assertEqual(P.evidence_cue_ids(plan, ctx), set())

    def test_short_lecture_is_sent_whole(self):
        cues = lecture(8, words='tiny')
        ctx, plan, outcome, calls = self.run_question(cues, 'What is said about word3?')
        self.assertEqual((plan.mode, plan.complete), ('full', True))
        self.assertEqual(P.evidence_cue_ids(plan, ctx), {c['segment_id'] for c in cues})

    def test_cue_ids_follow_start_times_when_a_cue_is_dropped(self):
        cues = lecture(300)
        cues[5]['text'] = '[music]'                                    # cleaned to empty -> dropped
        ctx = context(cues)
        self.assertEqual(ctx['dropped'], {'LX_S0006'})
        plan, _, _ = P.production_plan(S, GenerationResult, 'What is said about word30?', ctx['segs'])
        ev = P.evidence_cue_ids(plan, ctx)
        self.assertNotIn('LX_S0006', ev)
        self.assertIn('LX_S0031', ev)

    def test_evaluate_one_scores_a_hit_and_a_miss(self):
        cues = lecture(300)
        ctx = context(cues)
        row = {'question_id': 'LX_Q1', 'lecture_id': 'LX', 'group': 'PRIMARY_ANSWERABLE'}
        qrow = {'subject': 'S', 'question_category': 'direct_keyword', 'question_text': 'What is said about word40?'}
        hit = P.evaluate_one(BM25, R, S, GenerationResult, row, qrow, {'gold_intervals': [{'segment_ids': ['LX_S0041']}]}, ctx)
        self.assertEqual((hit['evidence_hit'], hit['strict_containment'], hit['gate_failure'], hit['model_called']), (1.0, 1.0, 0, True))
        self.assertEqual(hit['raw_hit_at_5'], 1.0)
        miss = P.evaluate_one(BM25, R, S, GenerationResult, row, qrow, {'gold_intervals': [{'segment_ids': ['LX_S0003']}]}, ctx)
        self.assertEqual(miss['evidence_hit'], 0.0)
        self.assertEqual(miss['model_called'], True)                    # the model is still called, with the wrong evidence
        fail = P.evaluate_one(BM25, R, S, GenerationResult, row, dict(qrow, question_text='football world cup in Brazil'), {'gold_intervals': [{'segment_ids': ['LX_S0041']}]}, ctx)
        self.assertEqual((fail['evidence_hit'], fail['gate_failure'], fail['model_called']), (0.0, 1, False))

    def test_unanswerable_question_is_run_but_not_scored(self):
        ctx = context(lecture(300))
        row = {'question_id': 'LX_Q2', 'lecture_id': 'LX', 'group': 'PRIMARY_UNANSWERABLE'}
        rec = P.evaluate_one(BM25, R, S, GenerationResult, row, {'subject': 'S', 'question_category': 'unanswerable', 'question_text': 'football world cup in Brazil'}, {'gold_intervals': []}, ctx)
        self.assertFalse(rec['scoreable'])
        self.assertNotIn('evidence_hit', rec)


class TestAggregationAndOutputs(unittest.TestCase):
    def recs(self):
        out = []
        for i in range(40):
            out.append({'question_id': f'Q{i}', 'lecture_id': f'L{i % 5}', 'group': 'PRIMARY_ANSWERABLE', 'subject': 'S', 'question_category': 'c', 'scoreable': True, 'mode': 'retrieval', 'decision': 'ok',
                        'model_called': True, 'complete': False, 'gate_failure': 0, 'n_blocks': 3, 'evidence_chars': 1000, 'transcript_chars': 10000, 'evidence_share': 0.1, 'estimated_prompt_tokens': 1500,
                        'evidence_hit': 1.0 if i % 10 else 0.0, 'interval_coverage': 1.0, 'full_interval_coverage': 1.0, 'strict_containment': 0.0, 'gold_cue_fraction': 0.5,
                        'raw_hit_at_5': 1.0, 'raw_hit_at_6': 1.0})
        return out

    def test_summarize_means_flags_and_interval_order(self):
        s = P.summarize(BM25, self.recs(), ['evidence_hit'], 500, 20261013, 0.95)
        e = s['metrics']['evidence_hit']
        self.assertAlmostEqual(e['estimate'], 36 / 40)
        self.assertEqual(s['sample_flag'], 'OK')
        self.assertLessEqual(e['ci_low'], e['estimate'])
        self.assertGreaterEqual(e['ci_high'], e['estimate'])

    def test_very_small_strata_have_no_interval(self):
        s = P.summarize(BM25, self.recs()[:5], ['evidence_hit'], 500, 20261013, 0.95)
        self.assertEqual((s['sample_flag'], s['metrics']['evidence_hit']['ci_low']), ('VERY_SMALL', None))

    def test_records_csv_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, 'a.csv'), os.path.join(d, 'b.csv')
            P.write_records(a, self.recs())
            P.write_records(b, copy.deepcopy(self.recs()))
            self.assertEqual(P.sha256_file(a), P.sha256_file(b))

    def test_raw_consistency_gate_detects_a_changed_value(self):
        import csv
        saved = list(csv.DictReader(open(P.SAVED_PQ, encoding='utf-8', newline='')))
        good = [{'question_id': r['question_id'], 'group': r['group'], 'scoreable': True, 'raw_hit_at_5': float(r['hit_rate_at_5'])} for r in saved if r['hit_rate_at_5'] != '']
        self.assertEqual(P.check_raw_consistency(good), len(good))
        bad = copy.deepcopy(good)
        bad[0]['raw_hit_at_5'] = 1.0 - bad[0]['raw_hit_at_5']
        with self.assertRaises(P.P1Error):
            P.check_raw_consistency(bad)

    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            old = P.RESULTS
            P.RESULTS = d
            try:
                os.makedirs(os.path.join(d, 'runs', 'run_Z'))
                self.assertEqual(P.p1_run('Z'), 2)
            finally:
                P.RESULTS = old


if __name__ == '__main__':
    unittest.main()
