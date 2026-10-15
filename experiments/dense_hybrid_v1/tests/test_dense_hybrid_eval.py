"""Tests for experiments/dense_hybrid_v1/dense_hybrid_eval.py (phase 1: M1 only). Synthetic data; no model, no encoding.

    HF_HUB_OFFLINE=1 python -B -m unittest discover -s experiments/dense_hybrid_v1/tests -p "test_*.py"
"""
import copy, csv, json, os, sys, tempfile, unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import dense_hybrid_eval as X  # noqa: E402
from qa import retrieval as R  # noqa: E402
from qa.transcript import normalize_segments  # noqa: E402

PREREG = X.load_prereg()
BM25 = X.load_frozen_module(PREREG)


def cue(i, start, end, text, lecture='LX'):
    return {'segment_id': f'{lecture}_S{i:04d}', 'start_seconds': start, 'end_seconds': end, 'text': text}


LECTURE = [cue(1, 0.0, 10.0, 'alpha beta'), cue(2, 10.0, 20.0, '[music]'), cue(3, 20.0, 30.0, 'gamma delta'),
           cue(4, 45.0, 55.0, 'epsilon zeta'), cue(5, 55.0, 70.0, 'eta theta'), cue(6, 100.0, 110.0, 'iota kappa')]


class TestPins(unittest.TestCase):
    def test_wrong_prereg_hash_fails_loudly(self):
        with self.assertRaises(X.PreregError):
            X.load_prereg(expected_sha='0' * 64)

    def test_prereg_pinned_files_match_disk(self):
        self.assertEqual(X.verify_prereg_files(PREREG), 8)   # frozen config + 3 harness files + 4 saved result files

    def test_changed_pinned_file_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'f.txt'), 'w') as f:
                f.write('changed')
            fake = {'frozen_inputs': {'frozen_bm25_config': {'path': 'f.txt', 'sha256': '1' * 64},
                                      'harness_code': {'files': {'missing.py': '2' * 64}}, 'saved_bm25_results': {'files': {}}}}
            with self.assertRaises(X.PreregError) as cm:
                X.verify_prereg_files(fake, d)
            msg = str(cm.exception)
            self.assertIn('f.txt', msg)
            self.assertIn('MISSING', msg)

    def test_frozen_module_refused_when_hash_differs(self):
        bad = copy.deepcopy(PREREG)
        bad['frozen_inputs']['harness_code']['files']['eval/bm25_eval.py'] = 'f' * 64
        with self.assertRaises(X.PreregError):
            X.load_frozen_module(bad)

    def test_all_real_inputs_verify(self):
        prereg, cfg, bm25, pins = X.verify_all_inputs()
        self.assertEqual(cfg['freeze']['protocol_version'], 'bm25_v1.0')
        self.assertTrue(all(t['ok'] for t in pins['frozen_config_pins']))

    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(X.m1_validate(d), 2)


class TestChunkIdentityAndOrdering(unittest.TestCase):
    def setUp(self):
        self.ec = R.EvidenceConfig()
        self.ctx = BM25.build_lecture_context(LECTURE, R, normalize_segments, self.ec)

    def test_dropped_cue_is_identified_by_id_not_position(self):
        self.assertEqual(self.ctx['dropped'], {'LX_S0002'})
        self.assertEqual(len(self.ctx['segs']), 5)

    def test_chunks_partition_retained_cues_in_order(self):
        flat = [i for a, b in self.ctx['chunks'] for i in range(a, b + 1)]
        self.assertEqual(flat, list(range(5)))
        union = set().union(*self.ctx['chunk_cues'])
        self.assertEqual(union, {'LX_S0001', 'LX_S0003', 'LX_S0004', 'LX_S0005', 'LX_S0006'})
        self.assertEqual(sum(len(c) for c in self.ctx['chunk_cues']), 5)          # each cue in exactly one chunk
        starts = [i['start'] for i in self.ctx['chunk_info']]
        self.assertEqual(starts, sorted(starts))

    def test_chunk_boundaries_follow_40s_rule(self):
        # cues 1,3 (0-30 s) share a chunk; cue 4 starts a new one because 55-0 > 40
        self.assertEqual(self.ctx['chunk_cues'][0], {'LX_S0001', 'LX_S0003'})
        self.assertIn('LX_S0004', self.ctx['chunk_cues'][1])

    def test_check_chunk_identity_passes_and_detects_count_change(self):
        cfg = {'chunking': {'chunk_counts_per_lecture': {'LX': len(self.ctx['chunks'])}}}
        self.assertEqual(X.check_chunk_identity(cfg, {'LX': self.ctx}), {'LX': len(self.ctx['chunks'])})
        cfg['chunking']['chunk_counts_per_lecture']['LX'] += 1
        with self.assertRaises(X.ReproductionError):
            X.check_chunk_identity(cfg, {'LX': self.ctx})

    def test_check_chunk_identity_detects_broken_partition(self):
        broken = copy.deepcopy(self.ctx)
        broken['chunks'] = [(0, 1), (1, 4)]                                         # cue 1 in two chunks
        broken['chunk_info'] = broken['chunk_info'][:2]
        with self.assertRaises(X.ReproductionError):
            X.check_chunk_identity({'chunking': {'chunk_counts_per_lecture': {}}}, {'LX': broken})

    def test_real_chunk_total_matches_frozen_config(self):
        prereg, cfg, bm25, _ = X.verify_all_inputs()
        _, _, gold, cues = X.load_data(cfg)
        ec = R.EvidenceConfig()
        total = 0
        for lid, n in cfg['chunking']['chunk_counts_per_lecture'].items():
            ctx = bm25.build_lecture_context(cues[lid], R, normalize_segments, ec)
            self.assertEqual(len(ctx['chunks']), n, lid)
            total += n
        self.assertEqual(total, PREREG['chunks']['expected_total_chunks'])


class TestRankingTieBreakAndRelevance(unittest.TestCase):
    def test_ties_go_to_lower_chunk_index(self):
        self.assertEqual(BM25.rank_from_scores([1.0, 3.0, 3.0, 2.0, 3.0]), [1, 2, 4, 3, 0])

    def test_zero_scores_are_never_ranked(self):
        self.assertEqual(BM25.rank_from_scores([0.0, 2.0, 0.0]), [1])
        self.assertEqual(BM25.rank_from_scores([0.0, 0.0]), [])

    def test_relevance_is_any_gold_cue_and_dropped_cues_are_removed(self):
        rec = {'gold_intervals': [{'segment_ids': ['LX_S0002']}, {'segment_ids': ['LX_S0003', 'LX_S0004']}]}
        ivs = BM25.gold_intervals(rec, {'LX_S0002'})
        self.assertEqual(ivs, [{'LX_S0003', 'LX_S0004'}])                            # all-dropped interval disappears
        chunk_cues = [{'LX_S0001'}, {'LX_S0004', 'LX_S0005'}]
        m = BM25.question_metrics([0, 1], chunk_cues, ivs)
        self.assertEqual(m['first_relevant_rank'], 2)
        self.assertEqual(m['hit_rate_at_1'], 0.0)
        self.assertEqual(m['hit_rate_at_3'], 1.0)

    def test_evaluate_ranking_marks_relevant_chunks_and_serializes_scores(self):
        ctx = BM25.build_lecture_context(LECTURE, R, normalize_segments, R.EvidenceConfig())
        row = {'question_id': 'LX_Q01', 'lecture_id': 'LX', 'group': 'PRIMARY_ANSWERABLE', 'n_gold_intervals': '1'}
        qrow = {'subject': 'S', 'question_category': 'c'}
        gold = {'gold_intervals': [{'segment_ids': ['LX_S0005']}]}
        n = len(ctx['chunks'])
        scores = [0.0] * n
        scores[1] = 2.5
        scores[0] = 2.5
        ranked = BM25.rank_from_scores(scores)
        rec = X.evaluate_ranking(BM25, row, qrow, gold, ctx, ranked, scores, 3)
        self.assertEqual([t['chunk_index'] for t in rec['top10']], [0, 1])
        self.assertEqual([t['relevant'] for t in rec['top10']], [False, True])
        self.assertEqual(rec['mrr_at_10'], 0.5)
        self.assertEqual(rec['n_ranked_chunks'], 2)


class TestMetricCalculations(unittest.TestCase):
    CH = [{'A1', 'A2'}, {'B1'}, {'C1', 'C2'}, {'D1'}]

    def test_interval_recall_hit_and_full_coverage(self):
        ivs = [{'A1'}, {'C2'}]
        m = BM25.question_metrics([1, 0, 2, 3], self.CH, ivs)
        self.assertEqual((m['interval_recall_at_1'], m['hit_rate_at_1'], m['full_evidence_coverage_at_1']), (0.0, 0.0, 0.0))
        self.assertEqual((m['interval_recall_at_3'], m['hit_rate_at_3'], m['full_evidence_coverage_at_3']), (1.0, 1.0, 1.0))
        self.assertEqual(m['mrr_at_10'], 0.5)

    def test_short_lists_are_not_padded(self):
        m = BM25.question_metrics([], self.CH, [{'A1'}])
        self.assertEqual((m['hit_rate_at_5'], m['mrr_at_10']), (0.0, 0.0))
        self.assertIsNone(m['first_relevant_rank'])

    def test_mrr_zero_beyond_depth_10(self):
        ch = [{f'X{i}'} for i in range(12)]
        m = BM25.question_metrics(list(range(12)), ch, [{'X11'}])
        self.assertEqual(m['mrr_at_10'], 0.0)
        self.assertEqual(m['first_relevant_rank'], 12)

    def test_unscoreable_without_gold(self):
        self.assertIsNone(BM25.question_metrics([0], self.CH, []))


class TestConsistencyChecks(unittest.TestCase):
    HEADER = ['question_id', 'hit_rate_at_5', 'mrr_at_10', 'top10_json', 'n_ranked_chunks']

    def _write(self, d, name, rows):
        p = os.path.join(d, name)
        with open(p, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.HEADER, lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        return p

    ROW = {'question_id': 'Q1', 'hit_rate_at_5': '1.0', 'mrr_at_10': '0.5', 'top10_json': '[]', 'n_ranked_chunks': '0'}

    def test_identical_csv_has_no_differences(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = self._write(d, 'a.csv', [self.ROW]), self._write(d, 'b.csv', [self.ROW])
            self.assertEqual(X.compare_csv(a, b), [])

    def test_changed_cell_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, 'a.csv', [self.ROW])
            b = self._write(d, 'b.csv', [dict(self.ROW, mrr_at_10='0.25')])
            diffs = X.compare_csv(a, b)
            self.assertEqual(len(diffs), 1)
            self.assertIn('mrr_at_10', diffs[0])

    def test_row_count_and_order_differences_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, 'a.csv', [self.ROW, dict(self.ROW, question_id='Q2')])
            b = self._write(d, 'b.csv', [dict(self.ROW, question_id='Q2'), self.ROW])
            self.assertTrue(any('order' in x for x in X.compare_csv(a, b)))
            c = self._write(d, 'c.csv', [self.ROW])
            self.assertTrue(any('row count' in x for x in X.compare_csv(a, c)))

    def test_csv_without_question_id_column_is_compared_by_row(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, val in (('a.csv', '0.5'), ('b.csv', '0.5'), ('c.csv', '0.6')):
                p = os.path.join(d, name)
                with open(p, 'w', encoding='utf-8', newline='') as f:
                    f.write('set,metric,estimate\nPRIMARY,hit_rate_at_5,' + val + '\n')
                paths.append(p)
            self.assertEqual(X.compare_csv(paths[0], paths[1]), [])
            diffs = X.compare_csv(paths[0], paths[2])
            self.assertEqual(len(diffs), 1)
            self.assertIn('row 1', diffs[0])

    def test_record_level_comparison_detects_changed_ranking(self):
        top = [{'rank': 1, 'chunk_index': 0, 'score': 1.0}]
        rec = {'question_id': 'Q1', 'top10': top, 'n_ranked_chunks': 1}
        for k in ('interval_recall_at_1', 'interval_recall_at_3', 'interval_recall_at_5', 'hit_rate_at_1', 'hit_rate_at_3', 'hit_rate_at_5',
                  'full_evidence_coverage_at_1', 'full_evidence_coverage_at_3', 'full_evidence_coverage_at_5', 'mrr_at_10'):
            rec[k] = 1.0
        header = ['question_id', 'n_ranked_chunks', 'top10_json', 'interval_recall_at_1', 'interval_recall_at_3', 'interval_recall_at_5', 'hit_rate_at_1', 'hit_rate_at_3',
                  'hit_rate_at_5', 'full_evidence_coverage_at_1', 'full_evidence_coverage_at_3', 'full_evidence_coverage_at_5', 'mrr_at_10']
        row = {k: '1.0' for k in header}
        row.update({'question_id': 'Q1', 'n_ranked_chunks': '1', 'top10_json': json.dumps(top, separators=(',', ':'))})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 's.csv')
            with open(p, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                w.writeheader()
                w.writerow(row)
            self.assertEqual(X.compare_records_to_saved([rec], p), [])
            changed = dict(rec, top10=[dict(top[0], chunk_index=1)])
            self.assertTrue(X.compare_records_to_saved([changed], p))
            self.assertTrue(X.compare_records_to_saved([dict(rec, mrr_at_10=0.5)], p))


class TestEnvironmentSettings(unittest.TestCase):
    def test_offline_flags_set_by_module(self):
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')

    def test_determinism_report_has_required_fields(self):
        r = X.determinism_report()
        self.assertIn('env', r)
        self.assertIn('torch', r)


if __name__ == '__main__':
    unittest.main()
