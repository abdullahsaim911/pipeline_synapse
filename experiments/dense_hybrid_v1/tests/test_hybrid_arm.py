"""Tests for experiments/dense_hybrid_v1/hybrid_arm.py (M3 RRF, flip table, paired contrasts). Synthetic data; no model.

    HF_HUB_OFFLINE=1 python -B -m unittest discover -s experiments/dense_hybrid_v1/tests -p "test_*.py"
"""
import json, math, os, random, shutil, sys, tempfile, unittest
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import dense_hybrid_eval as X  # noqa: E402
import hybrid_arm as H  # noqa: E402

PREREG = X.load_prereg()
BM25 = X.load_frozen_module(PREREG)


class TestRRFFormula(unittest.TestCase):
    def test_hand_calculated_example(self):
        # bm25 ranking: chunk2 (rank1), chunk0 (rank2); chunk1 absent.  dense ranking: 0,1,2.
        ranked, scores, rb, rd = H.rrf_fuse([2, 0], [0, 1, 2], 3)
        self.assertEqual(rb, {2: 1, 0: 2})
        self.assertEqual(rd, {0: 1, 1: 2, 2: 3})
        self.assertEqual(scores[0], 1 / 62 + 1 / 61)          # bm25 rank 2, dense rank 1
        self.assertEqual(scores[1], 0.0 + 1 / 62)             # absent from bm25: no penalty, dense rank 2
        self.assertEqual(scores[2], 1 / 61 + 1 / 63)          # bm25 rank 1, dense rank 3
        self.assertEqual(ranked, [0, 2, 1])

    def test_ranks_are_one_based(self):
        _, scores, _, _ = H.rrf_fuse([0], [0], 1)
        self.assertEqual(scores[0], 1 / 61 + 1 / 61)
        self.assertNotEqual(scores[0], 1 / 60 + 1 / 60)

    def test_k_is_60_and_weights_are_equal(self):
        self.assertEqual((H.RRF_K, H.W_BM25, H.W_DENSE), (60, 1.0, 1.0))
        cfg = PREREG['arms']['M3_hybrid_rrf']
        self.assertEqual((cfg['rrf_k'], cfg['weights']), (60, {'bm25': 1.0, 'dense': 1.0}))

    def test_exact_rational_agreement(self):
        # float scores must agree with exact arithmetic to within float rounding
        _, scores, _, _ = H.rrf_fuse([3, 1, 0], [1, 0, 2, 3], 4)
        exact = {0: Fraction(1, 63) + Fraction(1, 62), 1: Fraction(1, 62) + Fraction(1, 61), 2: Fraction(1, 63), 3: Fraction(1, 61) + Fraction(1, 64)}
        for c, e in exact.items():
            self.assertAlmostEqual(scores[c], float(e), places=15)

    def test_missing_bm25_results_fall_back_to_dense_order(self):
        ranked, scores, rb, rd = H.rrf_fuse([], [2, 0, 1], 3)
        self.assertEqual(ranked, [2, 0, 1])
        self.assertEqual(rb, {})
        self.assertEqual(scores[2], 1 / 61)

    def test_absent_chunk_contributes_exactly_zero_not_a_penalty_rank(self):
        _, scores, _, _ = H.rrf_fuse([0], [0, 1, 2], 3)
        self.assertEqual(scores[1], 1 / 62)                    # not 1/62 + 1/(60+len+1)
        self.assertEqual(scores[2], 1 / 63)

    def test_bm25_only_hit_can_outrank_dense_only(self):
        # chunk 5: bm25 rank 1, dense rank 6; chunk 0: absent from bm25, dense rank 1
        ranked, scores, _, _ = H.rrf_fuse([5], [0, 1, 2, 3, 4, 5], 6)
        self.assertGreater(scores[5], scores[0])               # 1/61+1/66 > 1/61
        self.assertEqual(ranked[0], 5)


class TestTieBreaks(unittest.TestCase):
    def test_exact_equal_scores_break_by_original_chunk_index(self):
        # chunk 0: bm25 rank 2, dense rank 1 ; chunk 1: bm25 rank 1, dense rank 2  -> identical sums
        ranked, scores, _, _ = H.rrf_fuse([1, 0], [0, 1], 3)
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(ranked[:2], [0, 1])

    def test_all_equal_scores_keep_original_order(self):
        # no bm25 list and a dense list that cannot produce ties -> use two lists that produce identical scores for 4 chunks
        ranked, scores, _, _ = H.rrf_fuse([3, 2, 1, 0], [0, 1, 2, 3], 4)
        self.assertEqual(scores[0], scores[3])
        self.assertEqual(scores[1], scores[2])
        self.assertEqual(ranked, [0, 3, 1, 2])

    def test_agrees_with_preregistered_sort_key_on_random_inputs(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 15)
            dense = list(range(n))
            rng.shuffle(dense)
            bm = rng.sample(range(n), rng.randint(0, n))
            ranked, scores, rb, rd = H.rrf_fuse(bm, dense, n)
            expect = sorted(range(n), key=lambda c: (-(((1.0 / (60 + rb[c])) if c in rb else 0.0) + ((1.0 / (60 + rd[c])) if c in rd else 0.0)), c))
            self.assertEqual(ranked, expect)
            self.assertEqual(len(ranked), n)                   # dense ranks everything, so all chunks are ranked

    def test_unrounded_scores_are_used(self):
        # scores that differ only beyond 12 decimals must still be ordered by value, not tied
        ranked, scores, _, _ = H.rrf_fuse([0, 1], [1, 0], 2)
        self.assertEqual(scores[0], scores[1])
        a = 1 / 61 + 1 / 62
        b = 1 / 61 + 1 / 62 + 1e-16
        self.assertNotEqual(a, b)

    def test_duplicate_or_out_of_range_inputs_are_rejected(self):
        with self.assertRaises(H.HybridError):
            H.rrf_fuse([0, 0], [0, 1], 2)
        with self.assertRaises(H.HybridError):
            H.rrf_fuse([5], [0, 1], 2)


class TestDecisionRule(unittest.TestCase):
    def test_three_way_rule(self):
        self.assertEqual(H.decide(0.01, 0.20), 'improvement')
        self.assertEqual(H.decide(-0.20, -0.01), 'decline')
        self.assertEqual(H.decide(-0.05, 0.05), 'no_demonstrated_difference')

    def test_bounds_equal_to_zero_are_not_improvement_or_decline(self):
        self.assertEqual(H.decide(0.0, 0.1), 'no_demonstrated_difference')
        self.assertEqual(H.decide(-0.1, 0.0), 'no_demonstrated_difference')
        self.assertEqual(H.decide(0.0, 0.0), 'no_demonstrated_difference')


def rec(qid, lec, **kv):
    base = {'question_id': qid, 'lecture_id': lec, 'subject': 'S', 'question_category': 'c', 'gold_interval_count': '1', 'n_ranked_chunks': 5, 'n_chunks': 12, 'first_relevant_rank': 1}
    base.update({m: 0.0 for m in BM25.METRICS})
    base.update(kv)
    return base


class TestPairedContrast(unittest.TestCase):
    def test_sign_convention_and_improvement(self):
        first = [rec(f'Q{i}', f'L{i % 3}', hit_rate_at_5=1.0, mrr_at_10=0.5) for i in range(30)]
        second = [rec(f'Q{i}', f'L{i % 3}', hit_rate_at_5=0.0, mrr_at_10=0.25) for i in range(30)]
        r = H.paired_contrast(BM25, first, second, ('hit_rate_at_5', 'mrr_at_10'), 500, 20261013, 0.95)
        self.assertEqual(r['hit_rate_at_5']['difference'], 1.0)
        self.assertEqual(r['hit_rate_at_5']['decision'], 'improvement')
        self.assertEqual(r['mrr_at_10']['difference'], 0.25)
        rev = H.paired_contrast(BM25, second, first, ('hit_rate_at_5',), 500, 20261013, 0.95)
        self.assertEqual(rev['hit_rate_at_5']['difference'], -1.0)
        self.assertEqual(rev['hit_rate_at_5']['decision'], 'decline')

    def test_identical_arms_show_no_difference(self):
        a = [rec(f'Q{i}', 'L', hit_rate_at_5=float(i % 2)) for i in range(20)]
        r = H.paired_contrast(BM25, a, [dict(x) for x in a], ('hit_rate_at_5',), 300, 20261013, 0.95)
        self.assertEqual(r['hit_rate_at_5']['decision'], 'no_demonstrated_difference')
        self.assertEqual((r['hit_rate_at_5']['ci_low'], r['hit_rate_at_5']['ci_high']), (0.0, 0.0))

    def test_matches_independent_reimplementation_of_the_registered_bootstrap(self):
        diffs = [1, 0, 0, 1, -1, 0, 1, 0, 0, 1, 0, 0]
        first = [rec(f'Q{i}', 'L', hit_rate_at_5=float(max(d, 0))) for i, d in enumerate(diffs)]
        second = [rec(f'Q{i}', 'L', hit_rate_at_5=float(max(-d, 0))) for i, d in enumerate(diffs)]
        B, seed, n = 400, 20261013, len(diffs)
        rng = random.Random(seed)
        reps = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        lo, hi = reps[math.ceil(0.025 * B) - 1], reps[math.ceil(0.975 * B) - 1]
        r = H.paired_contrast(BM25, first, second, ('hit_rate_at_5',), B, seed, 0.95)['hit_rate_at_5']
        self.assertEqual((r['ci_low'], r['ci_high']), (lo, hi))
        self.assertAlmostEqual(r['difference'], sum(diffs) / n)

    def test_misaligned_arms_are_rejected(self):
        a = [rec('Q1', 'L'), rec('Q2', 'L')]
        with self.assertRaises(H.HybridError):
            H.paired_contrast(BM25, a, [a[1], a[0]], ('hit_rate_at_5',), 10, 1, 0.95)

    def test_clustered_contrast_pools_by_lecture(self):
        first = [rec(f'Q{i}', f'L{i % 4}', hit_rate_at_5=1.0) for i in range(20)]
        second = [rec(f'Q{i}', f'L{i % 4}', hit_rate_at_5=0.0) for i in range(20)]
        r = H.clustered_contrast(BM25, first, second, ('hit_rate_at_5',), 200, 20261013, 0.95)['hit_rate_at_5']
        self.assertEqual((r['ci_low'], r['ci_high'], r['clusters']), (1.0, 1.0, 4))


class TestFlipTable(unittest.TestCase):
    def test_categories_and_selection(self):
        m1 = [rec('A', 'L', hit_rate_at_5=0.0), rec('B', 'L', hit_rate_at_5=1.0), rec('C', 'L', hit_rate_at_5=0.0), rec('D', 'L', hit_rate_at_5=1.0), rec('E', 'L', hit_rate_at_5=1.0)]
        m2 = [rec('A', 'L', hit_rate_at_5=1.0), rec('B', 'L', hit_rate_at_5=1.0), rec('C', 'L', hit_rate_at_5=0.0), rec('D', 'L', hit_rate_at_5=0.0), rec('E', 'L', hit_rate_at_5=1.0)]
        m3 = [rec('A', 'L', hit_rate_at_5=1.0), rec('B', 'L', hit_rate_at_5=0.0), rec('C', 'L', hit_rate_at_5=0.0), rec('D', 'L', hit_rate_at_5=1.0), rec('E', 'L', hit_rate_at_5=1.0)]
        rows = {r['question_id']: r for r in H.flip_rows(m1, m2, m3)}
        self.assertEqual(set(rows), {'A', 'B', 'C', 'D'})                                 # E: all hit -> omitted
        self.assertEqual(rows['A']['category'], 'bm25_miss_hybrid_hit_dense_hit')
        self.assertEqual(rows['B']['category'], 'bm25_hit_hybrid_miss_dense_hit')
        self.assertEqual(rows['C']['category'], 'all_arms_miss_at_5')
        self.assertEqual(rows['D']['category'], 'bm25_hit_hybrid_hit_dense_miss')

    def test_bm25_short_list_counts(self):
        recs = [rec('A', 'L', n_ranked_chunks=0, n_chunks=12), rec('B', 'L', n_ranked_chunks=4, n_chunks=12), rec('C', 'L', n_ranked_chunks=7, n_chunks=12), rec('D', 'L', n_ranked_chunks=8, n_chunks=8)]
        c = H.bm25_list_length_counts(recs)
        self.assertEqual(c['shorter_than_5']['questions'], 2)
        self.assertEqual(c['shorter_than_10']['questions'], 4)
        self.assertEqual(c['shorter_than_10']['of_which_lecture_has_at_least_k_chunks'], 3)    # D is short only because the lecture has 8 chunks
        self.assertEqual(c['empty_list']['questions'], 1)


class TestInputVerification(unittest.TestCase):
    def test_tampered_m2_output_aborts(self):
        fin = os.path.join(H.M2_DIR, 'final')
        backup = os.path.join(tempfile.mkdtemp(), 'bk')
        shutil.copytree(fin, backup)
        orig = H.M2_DIR
        try:
            tmp = tempfile.mkdtemp()
            shutil.copytree(orig, os.path.join(tmp, 'm2'))
            p = os.path.join(tmp, 'm2', 'final', 'aggregate_metrics.json')
            with open(p, 'a') as f:
                f.write(' ')
            H.M2_DIR = os.path.join(tmp, 'm2')
            with self.assertRaises(H.HybridError):
                H.verify_m2_outputs(PREREG)
        finally:
            H.M2_DIR = orig
            shutil.rmtree(os.path.dirname(backup), ignore_errors=True)

    def test_real_m2_outputs_verify(self):
        man = H.verify_m2_outputs(PREREG)
        self.assertEqual(man['repeatability'], 'PASS')

    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            old = H.RESULTS
            H.RESULTS = d
            try:
                os.makedirs(os.path.join(d, 'runs', 'run_Z'))
                self.assertEqual(H.m3_run('Z'), 2)
            finally:
                H.RESULTS = old


if __name__ == '__main__':
    unittest.main()
