"""Tests for experiments/dense_hybrid_v1/dense_arm.py (M2). Fake embeddings for logic; ONE smoke test loads the cached model and encodes
a few synthetic sentences (no benchmark data).

    HF_HUB_OFFLINE=1 python -B -m unittest discover -s experiments/dense_hybrid_v1/tests -p "test_*.py"
"""
import copy, json, os, sys, tempfile, unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import dense_arm as D  # noqa: E402
import dense_hybrid_eval as X  # noqa: E402
from qa import retrieval as R  # noqa: E402
from qa.transcript import normalize_segments  # noqa: E402

PREREG = X.load_prereg()
BM25 = X.load_frozen_module(PREREG)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def cue(i, start, end, text, lecture='LX'):
    return {'segment_id': f'{lecture}_S{i:04d}', 'start_seconds': start, 'end_seconds': end, 'text': text}


LECTURE = [cue(1, 0.0, 10.0, 'alpha beta'), cue(2, 10.0, 20.0, '[music]'), cue(3, 20.0, 30.0, 'gamma delta'), cue(4, 45.0, 55.0, 'epsilon zeta'), cue(5, 100.0, 110.0, 'iota kappa')]


class TestRankingAndScores(unittest.TestCase):
    def test_all_chunks_are_ranked_including_zero_and_negative_scores(self):
        self.assertEqual(D.rank_by_scores([0.0, -0.2, 0.5]), [2, 0, 1])           # nothing filtered (unlike BM25)

    def test_ties_go_to_lower_chunk_index(self):
        self.assertEqual(D.rank_by_scores([0.5, 0.9, 0.5, 0.9, 0.1]), [1, 3, 0, 2, 4])

    def test_cosine_equals_dot_of_unit_vectors_and_picks_nearest(self):
        q = unit([1, 0, 0, 0])
        C = np.stack([unit([0, 1, 0, 0]), unit([1, 1, 0, 0]), unit([1, 0, 0, 0])])
        s = D.cosine_scores(C, q)
        self.assertEqual(D.rank_by_scores(s)[0], 2)
        self.assertAlmostEqual(s[1], 1 / np.sqrt(2), places=6)
        self.assertTrue(all(isinstance(x, float) for x in s))

    def test_identical_vectors_tie_exactly(self):
        v = unit([1, 2, 3, 4])
        s = D.cosine_scores(np.stack([v, v, v]), v)
        self.assertEqual(len(set(s)), 1)
        self.assertEqual(D.rank_by_scores(s), [0, 1, 2])


class TestEmbeddingChecks(unittest.TestCase):
    def test_valid_embeddings_pass(self):
        e = np.stack([unit(np.random.RandomState(i).randn(384)) for i in range(5)])
        self.assertLess(D.check_embeddings(e), 1e-5)

    def test_wrong_dimension_nan_inf_and_non_unit_are_rejected(self):
        good = np.stack([unit(np.ones(384))])
        with self.assertRaises(D.DenseError):
            D.check_embeddings(good[:, :100])
        for bad in (np.nan, np.inf):
            e = good.copy()
            e[0, 0] = bad
            with self.assertRaises(D.DenseError):
                D.check_embeddings(e)
        with self.assertRaises(D.DenseError):
            D.check_embeddings(good * 2)

    def test_array_hash_is_stable_and_sensitive_to_one_bit(self):
        a = np.stack([unit(np.arange(1, 385))])
        b = a.copy()
        self.assertEqual(D.array_sha256(a), D.array_sha256(b))
        b.view(np.uint32)[0, 0] ^= 1
        self.assertNotEqual(D.array_sha256(a), D.array_sha256(b))


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.ctx = BM25.build_lecture_context(LECTURE, R, normalize_segments, R.EvidenceConfig())
        self.n = len(self.ctx['chunks'])

    def test_chunk_text_is_cleaned_cues_joined_by_one_space(self):
        self.assertEqual(D.chunk_text(self.ctx, 0), 'alpha beta gamma delta')         # [music] cue dropped, others joined

    def test_lecture_slices_and_records_align_with_questions(self):
        ctx2 = BM25.build_lecture_context([cue(i, s, e, t, 'LY') for i, s, e, t in [(1, 0.0, 10.0, 'one two'), (2, 60.0, 70.0, 'three four')]], R, normalize_segments, R.EvidenceConfig())
        ctxs = {'LX': self.ctx, 'LY': ctx2}
        keys = [(lid, p) for lid in sorted(ctxs) for p in range(len(ctxs[lid]['chunks']))]
        sl = D.lecture_slices(keys)
        self.assertEqual(sl['LX'], (0, self.n))
        self.assertEqual(sl['LY'], (self.n, self.n + len(ctx2['chunks'])))
        dim = 8
        ce = np.stack([unit(np.eye(dim)[i % dim] + 0.01 * (i + 1)) for i in range(len(keys))])
        # question 1 (LX) is closest to LX chunk 1; question 2 (LY) closest to LY chunk 0
        qe = np.stack([ce[1], ce[self.n]])
        qlist = [({'question_id': 'LX_Q1', 'lecture_id': 'LX', 'group': 'PRIMARY_ANSWERABLE', 'n_gold_intervals': '1'}, {'subject': 'S', 'question_category': 'c'}, {'gold_intervals': [{'segment_ids': ['LX_S0004']}]}),
                 ({'question_id': 'LY_Q1', 'lecture_id': 'LY', 'group': 'PRIMARY_ANSWERABLE', 'n_gold_intervals': '1'}, {'subject': 'S', 'question_category': 'c'}, {'gold_intervals': [{'segment_ids': ['LY_S0001']}]})]
        recs, scores = D.records_from_embeddings(BM25, ctxs, qlist, keys, ce, qe)
        self.assertEqual(recs[0]['top10'][0]['chunk_index'], 1)
        self.assertEqual(recs[1]['top10'][0]['chunk_index'], 0)
        self.assertEqual(recs[0]['n_ranked_chunks'], self.n)                           # all chunks ranked
        self.assertEqual(len(scores[0]), self.n)
        self.assertEqual(recs[0]['mrr_at_10'], 1.0)

    def test_misaligned_rows_are_rejected(self):
        keys = [('LX', p) for p in range(self.n)]
        ce = np.stack([unit(np.ones(8))] * (self.n - 1))
        with self.assertRaises(D.DenseError):
            D.records_from_embeddings(BM25, {'LX': self.ctx}, [], keys, ce, np.zeros((0, 8), dtype=np.float32))

    def test_non_contiguous_chunk_order_is_rejected(self):
        with self.assertRaises(D.DenseError):
            D.lecture_slices([('LX', 1), ('LX', 2)])


class TestSerialization(unittest.TestCase):
    def _records(self):
        ctx = BM25.build_lecture_context(LECTURE, R, normalize_segments, R.EvidenceConfig())
        row = {'question_id': 'LX_Q1', 'lecture_id': 'LX', 'group': 'PRIMARY_ANSWERABLE', 'n_gold_intervals': '1'}
        scores = [0.3125, 0.25, 0.25][:len(ctx['chunks'])] + [0.0] * max(0, len(ctx['chunks']) - 3)
        ranked = D.rank_by_scores(scores)
        return [X.evaluate_ranking(BM25, row, {'subject': 'S', 'question_category': 'c'}, {'gold_intervals': [{'segment_ids': ['LX_S0003']}]}, ctx, ranked, scores, 0)], ctx, scores

    def test_csv_and_rankings_are_byte_deterministic(self):
        recs, ctx, scores = self._records()
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, 'a.csv'), os.path.join(d, 'b.csv')
            D.write_records_csv(a, BM25, recs)
            D.write_records_csv(b, BM25, copy.deepcopy(recs))
            self.assertEqual(X.sha256_file(a), X.sha256_file(b))
            qlist = [({'question_id': 'LX_Q1'}, None, None)]
            r1, r2 = os.path.join(d, 'r1.csv'), os.path.join(d, 'r2.csv')
            D.write_full_rankings(r1, qlist, {}, [scores])
            D.write_full_rankings(r2, qlist, {}, [list(scores)])
            self.assertEqual(X.sha256_file(r1), X.sha256_file(r2))

    def test_scores_serialized_unrounded(self):
        s = [0.1234567890123456]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'r.csv')
            D.write_full_rankings(p, [({'question_id': 'Q'}, None, None)], {}, [s])
            self.assertEqual(float(X.read_csv_rows(p)[0]['score_repr']), s[0])

    def test_compare_runs_detects_a_single_bit_and_ranking_change(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = []
            for name in ('A', 'B'):
                p = os.path.join(d, name)
                os.makedirs(p)
                dirs.append(p)
                ce = np.stack([unit(np.arange(1, 385)), unit(np.arange(2, 386))]).astype(np.float32)
                np.save(os.path.join(p, 'chunk_embeddings.npy'), ce)
                np.save(os.path.join(p, 'question_embeddings.npy'), ce[:1])
                with open(os.path.join(p, 'full_rankings.csv'), 'w', newline='') as f:
                    f.write('question_id,rank,chunk_index,score_repr\nQ,1,0,0.5\nQ,2,1,0.25\n')
                for n in ('per_question_results.csv', 'chunk_index.csv'):
                    with open(os.path.join(p, n), 'w') as f:
                        f.write('x\n')
            self.assertEqual(D.compare_runs(*dirs)['result'], 'PASS')
            ce = np.load(os.path.join(dirs[1], 'chunk_embeddings.npy'))
            ce.view(np.uint32)[0, 0] ^= 1
            np.save(os.path.join(dirs[1], 'chunk_embeddings.npy'), ce)
            rep = D.compare_runs(*dirs)
            self.assertEqual(rep['result'], 'FAIL')
            self.assertEqual(rep['checks']['chunk_embeddings.npy']['n_differing_values'], 1)
            with open(os.path.join(dirs[1], 'full_rankings.csv'), 'w', newline='') as f:
                f.write('question_id,rank,chunk_index,score_repr\nQ,1,1,0.5\nQ,2,0,0.25\n')
            self.assertGreater(D.compare_runs(*dirs)['checks']['full_rankings']['ranking_order_differences'], 0)


class TestAggregationSelfConsistency(unittest.TestCase):
    def test_aggregation_function_reproduces_saved_baseline_aggregate(self):
        prereg, cfg, bm25, _ = X.verify_all_inputs()
        records, _ = X.run_m1(cfg, bm25, R, normalize_segments)
        groups = bm25.split_groups(X.load_data(cfg)[0])
        agg = D.aggregate_records(bm25, cfg, records, groups)
        with open(X.SAVED_DIR + '/aggregate_metrics.json', encoding='utf-8') as f:
            self.assertEqual(json.loads(json.dumps(agg)), json.load(f))


class TestRealModelSmoke(unittest.TestCase):
    """Loads the cached snapshot offline and encodes a few synthetic sentences (not benchmark data)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.snap, cls.fp, _ = D.load_model(PREREG)

    def test_dimension_finite_unit_norm_and_repeatable(self):
        texts = ['The derivative measures the rate of change.', 'Mitochondria produce ATP.', 'What is a derivative?']
        a, b = D.encode(self.model, texts), D.encode(self.model, texts)
        self.assertEqual(a.shape, (3, 384))
        self.assertEqual(a.dtype, np.float32)
        D.check_embeddings(a)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_related_sentence_ranks_first(self):
        chunks = ['Mitochondria produce ATP in the cell.', 'The derivative measures the rate of change of a function.', 'The French revolution began in 1789.']
        q = D.encode(self.model, ['What does a derivative measure?'])
        s = D.cosine_scores(D.encode(self.model, chunks), q[0])
        self.assertEqual(D.rank_by_scores(s)[0], 1)

    def test_settings_match_preregistration(self):
        self.assertEqual(self.model.max_seq_length, 256)
        self.assertEqual(self.model.get_embedding_dimension(), 384)
        self.assertEqual(self.fp, PREREG['dense_model']['local_snapshot']['tree_fingerprint_sha256'])
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')

    def test_tampered_fingerprint_is_refused(self):
        bad = copy.deepcopy(PREREG)
        bad['dense_model']['local_snapshot']['tree_fingerprint_sha256'] = '0' * 64
        with self.assertRaises(D.DenseError):
            D.load_model(bad)


if __name__ == '__main__':
    unittest.main()
