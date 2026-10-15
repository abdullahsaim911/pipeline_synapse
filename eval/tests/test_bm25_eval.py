"""Focused tests for eval/bm25_eval.py (synthetic data only; no dataset run, no model/API calls).

    python -B -m unittest discover -s eval/tests -p "test_*.py"
"""
import hashlib, json, os, random, sys, tempfile, unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
REPO = os.path.dirname(EVAL)
sys.path.insert(0, EVAL)
sys.path.insert(0, REPO)

import bm25_eval as H  # noqa: E402
from qa import retrieval as R  # noqa: E402
from qa.transcript import normalize_segments  # noqa: E402


def cue(i, start, end, text, lecture='LX'):
    return {'segment_id': f'{lecture}_S{i:04d}', 'start_seconds': start, 'end_seconds': end, 'text': text}


class TestMetricDefinitions(unittest.TestCase):
    # chunks 0..3 with cue sets; ranked order passed explicitly
    CH = [{'A1', 'A2'}, {'B1'}, {'C1', 'C2'}, {'D1'}]

    def test_single_interval_recall_hit_full_equal(self):
        m = H.question_metrics([2, 0, 1], self.CH, [{'A2', 'A3'}])   # interval covered by chunk 0 (rank 2)
        self.assertEqual((m['interval_recall_at_1'], m['hit_rate_at_1'], m['full_evidence_coverage_at_1']), (0.0, 0.0, 0.0))
        for k in (3, 5):
            self.assertEqual((m[f'interval_recall_at_{k}'], m[f'hit_rate_at_{k}'], m[f'full_evidence_coverage_at_{k}']), (1.0, 1.0, 1.0))
        self.assertEqual(m['mrr_at_10'], 0.5)
        self.assertEqual(m['first_relevant_rank'], 2)

    def test_multi_interval_partial_then_full(self):
        # two distinct intervals: one in chunk 0, one in chunk 2
        ivs = [{'A1'}, {'C2'}]
        m = H.question_metrics([0, 1, 2, 3], self.CH, ivs)
        self.assertEqual(m['interval_recall_at_1'], 0.5)          # one of two intervals
        self.assertEqual(m['hit_rate_at_1'], 1.0)                 # at least one interval
        self.assertEqual(m['full_evidence_coverage_at_1'], 0.0)   # not every interval
        self.assertEqual(m['interval_recall_at_3'], 1.0)          # chunk 2 is rank 3
        self.assertEqual(m['full_evidence_coverage_at_3'], 1.0)
        self.assertEqual(m['mrr_at_10'], 1.0)

    def test_one_chunk_can_cover_two_intervals(self):
        m = H.question_metrics([0], [{'A1', 'B1'}], [{'A1'}, {'B1'}])
        self.assertEqual((m['interval_recall_at_1'], m['full_evidence_coverage_at_1']), (1.0, 1.0))

    def test_touching_one_cue_covers_the_interval_not_complete_recall(self):
        m = H.question_metrics([0], [{'A1'}], [{'A1', 'A2', 'A3', 'A4'}])
        self.assertEqual(m['interval_recall_at_1'], 1.0)   # one shared cue is enough: NOT complete-evidence recall
        self.assertEqual(m['full_evidence_coverage_at_1'], 1.0)

    def test_mrr_depth_and_beyond_depth(self):
        chunks = [{f'X{i}'} for i in range(15)]
        ranked = list(range(15))
        m = H.question_metrics(ranked, chunks, [{'X9'}])           # rank 10 -> counted
        self.assertAlmostEqual(m['mrr_at_10'], 0.1)
        m = H.question_metrics(ranked, chunks, [{'X10'}])          # rank 11 -> beyond depth 10 -> 0, but rank still recorded
        self.assertEqual(m['mrr_at_10'], 0.0)
        self.assertEqual(m['first_relevant_rank'], 11)

    def test_invariants(self):
        rng = random.Random(7)
        for _ in range(200):
            nchunks = rng.randint(1, 12)
            chunks = [{f'C{c}_{j}' for j in range(rng.randint(1, 3))} for c in range(nchunks)]
            allcues = sorted(set().union(*chunks))
            ivs = [set(rng.sample(allcues, rng.randint(1, min(3, len(allcues))))) for _ in range(rng.randint(1, 3))]
            ivs = [iv for i, iv in enumerate(ivs) if not any(iv & other for other in ivs[:i])]   # distinct (disjoint) intervals
            ranked = rng.sample(range(nchunks), rng.randint(0, nchunks))
            m = H.question_metrics(ranked, chunks, ivs)
            for k in H.KS:
                self.assertGreaterEqual(m[f'hit_rate_at_{k}'], m[f'full_evidence_coverage_at_{k}'])
                self.assertTrue(0 <= m[f'interval_recall_at_{k}'] <= 1)
                self.assertEqual(m[f'hit_rate_at_{k}'] == 1.0, m[f'interval_recall_at_{k}'] > 0)
                if len(ivs) == 1:
                    self.assertEqual(m[f'hit_rate_at_{k}'], m[f'full_evidence_coverage_at_{k}'])
                    self.assertEqual(m[f'hit_rate_at_{k}'], m[f'interval_recall_at_{k}'])
            for a, b in zip(H.KS, H.KS[1:]):
                for name in ('interval_recall_at_', 'hit_rate_at_', 'full_evidence_coverage_at_'):
                    self.assertLessEqual(m[f'{name}{a}'], m[f'{name}{b}'])
            self.assertGreaterEqual(m['mrr_at_10'], m['hit_rate_at_1'])

    def test_no_gold_interval_is_unscoreable(self):
        self.assertIsNone(H.question_metrics([0], self.CH, []))

    def test_dropped_gold_cues_leave_the_denominator(self):
        gold = {'gold_intervals': [{'segment_ids': ['LX_S0002']}, {'segment_ids': ['LX_S0003', 'LX_S0002']}, {'segment_ids': ['LX_S0004']}]}
        ivs = H.gold_intervals(gold, dropped={'LX_S0002'})
        self.assertEqual(ivs, [{'LX_S0003'}, {'LX_S0004'}])        # the all-dropped interval disappears; partially dropped keeps its other cue
        self.assertIsNone(H.question_metrics([0], self.CH, H.gold_intervals({'gold_intervals': [{'segment_ids': ['LX_S0002']}]}, {'LX_S0002'})))


class TestRankingTiesAndZeroScores(unittest.TestCase):
    def test_positive_only_stable_descending_earlier_chunk_first(self):
        scores = [1.0, 2.0, 2.0, 0.0, 1.0, 3.0]
        self.assertEqual(H.rank_from_scores(scores), [5, 1, 2, 0, 4])   # ties: lower chunk index first; zero never ranked

    def test_matches_production_stable_reverse_sort(self):
        rng = random.Random(3)
        for _ in range(100):
            scores = [rng.choice([0.0, 0.5, 1.0, 1.5]) for _ in range(rng.randint(0, 25))]
            prod = [i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
            self.assertEqual(H.rank_from_scores(scores), prod)

    def test_all_zero_or_empty_ranking(self):
        self.assertEqual(H.rank_from_scores([0.0, 0.0, 0.0]), [])
        self.assertEqual(H.rank_from_scores([]), [])
        m = H.question_metrics([], [{'A1'}, {'B1'}], [{'A1'}])
        self.assertTrue(all(m[k] == 0.0 for k in m if k != 'first_relevant_rank'))
        self.assertIsNone(m['first_relevant_rank'])

    def test_fewer_positive_chunks_than_k_are_not_padded(self):
        chunks = [{'A1'}, {'B1'}, {'C1'}]
        m = H.question_metrics([1], chunks, [{'A1'}, {'B1'}])      # only one positive chunk
        self.assertEqual(m['interval_recall_at_5'], 0.5)           # zero-score chunks must not be added to fill k=5
        self.assertEqual(m['hit_rate_at_5'], 1.0)

    def test_end_to_end_bm25_positive_only(self):
        ctx = _three_topic_lecture()
        ranked, scores, n_terms = H.rank_question(R, 'How does chlorophyll absorb light?', ctx)
        self.assertEqual(ranked[0], 0)
        self.assertTrue(all(scores[i] > 0 for i in ranked))
        self.assertTrue(all(i in ranked for i in range(len(scores)) if scores[i] > 0))
        ranked2, _, _ = H.rank_question(R, 'quantum entanglement', ctx)
        self.assertEqual(ranked2, [])                              # no query term occurs anywhere -> empty ranking
        ranked3, _, n3 = H.rank_question(R, 'what is the', ctx)    # only stopwords -> no terms -> empty ranking
        self.assertEqual((ranked3, n3), ([], 0))


def _three_topic_lecture():
    rows = [cue(1, 0.0, 45.0, 'photosynthesis uses chlorophyll to absorb light energy in the chloroplast'),
            cue(2, 45.0, 90.0, 'mitochondria perform respiration and burn glucose for energy'),
            cue(3, 90.0, 135.0, 'ribosome reads messenger rna during translation to build protein; light chains too')]
    return H.build_lecture_context(rows, R, normalize_segments, R.EvidenceConfig())


class TestCueToTimeMapping(unittest.TestCase):
    def rows(self):
        return [cue(1, 0.0, 10.0, 'alpha beta'), cue(2, 10.0, 20.0, '[LAUGHTER]'), cue(3, 20.0, 30.0, 'gamma delta'),
                cue(4, 30.0, 40.0, 'epsilon'), cue(5, 40.0, 50.0, 'zeta eta'), cue(6, 50.0, 60.0, 'theta')]

    def test_ids_come_from_start_time_not_position(self):
        ctx = H.build_lecture_context(self.rows(), R, normalize_segments, R.EvidenceConfig())
        self.assertEqual(ctx['dropped'], {'LX_S0002'})                  # the cue that cleans to empty
        self.assertEqual(len(ctx['segs']), 5)
        self.assertEqual([ctx['start_to_id'][s.start] for s in ctx['segs']], ['LX_S0001', 'LX_S0003', 'LX_S0004', 'LX_S0005', 'LX_S0006'])
        # chunks (40 s limit): segs 0..2 then 3..4
        self.assertEqual(ctx['chunk_cues'], [{'LX_S0001', 'LX_S0003', 'LX_S0004'}, {'LX_S0005', 'LX_S0006'}])
        self.assertEqual((ctx['chunk_info'][0]['start'], ctx['chunk_info'][0]['end']), (0.0, 40.0))
        self.assertEqual((ctx['chunk_info'][1]['start'], ctx['chunk_info'][1]['end']), (40.0, 60.0))
        self.assertEqual((ctx['chunk_info'][1]['first_cue'], ctx['chunk_info'][1]['last_cue']), ('LX_S0005', 'LX_S0006'))
        # the position-based mapping the protocol forbids would be WRONG after the dropped cue
        naive = {f"LX_S{i + 1:04d}" for i in range(0, 3)}
        self.assertNotEqual(ctx['chunk_cues'][0], naive)

    def test_gold_on_a_dropped_cue_is_not_retrievable(self):
        ctx = H.build_lecture_context(self.rows(), R, normalize_segments, R.EvidenceConfig())
        gold = {'gold_intervals': [{'segment_ids': ['LX_S0002']}, {'segment_ids': ['LX_S0005']}]}
        ivs = H.gold_intervals(gold, ctx['dropped'])
        self.assertEqual(ivs, [{'LX_S0005'}])
        m = H.question_metrics([1], ctx['chunk_cues'], ivs)
        self.assertEqual(m['interval_recall_at_1'], 1.0)

    def test_duplicate_start_times_are_refused(self):
        rows = [cue(1, 0.0, 5.0, 'a b'), cue(2, 0.0, 6.0, 'c d')]
        with self.assertRaises(H.ProtocolError):
            H.build_lecture_context(rows, R, normalize_segments, R.EvidenceConfig())

    def test_multi_interval_question_through_real_bm25(self):
        ctx = _three_topic_lecture()
        gold = {'gold_intervals': [{'segment_ids': ['LX_S0001']}, {'segment_ids': ['LX_S0003']}]}
        row = {'question_id': 'LX_Q01', 'lecture_id': 'LX', 'group': 'PRIMARY_ANSWERABLE', 'n_gold_intervals': '2'}
        q = {'question_text': 'Which part mentions light?', 'subject': 'Biology', 'question_category': 'multi_segment'}
        rec = H.evaluate_question(R, row, q, gold, ctx)
        self.assertEqual(rec['n_gold_intervals_scored'], 2)
        self.assertEqual(rec['interval_recall_at_1'], 0.5)             # only one chunk fits in the top 1, two distinct intervals
        self.assertEqual(rec['hit_rate_at_1'], 1.0)
        self.assertEqual(rec['full_evidence_coverage_at_1'], 0.0)
        self.assertEqual(rec['interval_recall_at_3'], 1.0)
        self.assertEqual(rec['full_evidence_coverage_at_3'], 1.0)
        self.assertEqual([t['rank'] for t in rec['top10']], [1, 2])    # only the two chunks containing "light"
        self.assertTrue(all(t['relevant'] for t in rec['top10']))


class TestBootstrap(unittest.TestCase):
    def reference(self, vals, B, seed, level):
        rng = random.Random(seed)
        n = len(vals)
        reps = []
        for _ in range(B):
            idx = [rng.randrange(n) for _ in range(n)]
            reps.append(sum(vals[i] for i in idx) / n)
        reps.sort()
        lo, hi = -(-B * 25 // 1000), -(-B * 975 // 1000)       # integer ceil(0.025*B), ceil(0.975*B): no float rounding
        return reps[lo - 1], reps[hi - 1]

    def test_percentile_positions(self):
        self.assertEqual(H.percentile_ranks(10000, 0.95), (250, 9750))
        self.assertEqual(H.percentile_ranks(40, 0.95), (1, 39))
        self.assertEqual(H.percentile_ranks(1000, 0.95), (25, 975))

    def test_matches_independent_reference_and_is_reproducible(self):
        vals = [((i * 37) % 101) / 100 for i in range(40)]      # many distinct values, so different seeds give different intervals
        got = H.bootstrap_ci({'m': vals, 'other': [1.0 - v for v in vals]}, 500, 20261013)
        self.assertEqual(got['m'], self.reference(vals, 500, 20261013, 0.95))
        self.assertEqual(got, H.bootstrap_ci({'m': vals, 'other': [1.0 - v for v in vals]}, 500, 20261013))
        self.assertNotEqual(got['m'], H.bootstrap_ci({'m': vals, 'other': vals}, 500, 20261014)['m'])

    def test_fresh_rng_per_call_so_order_does_not_matter(self):
        a, b = [0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        first = H.bootstrap_ci({'m': a}, 100, 5)['m']
        H.bootstrap_ci({'m': b}, 100, 5)
        self.assertEqual(first, H.bootstrap_ci({'m': a}, 100, 5)['m'])

    def test_constant_values_give_degenerate_interval(self):
        self.assertEqual(H.bootstrap_ci({'m': [0.5] * 12}, 50, 1)['m'], (0.5, 0.5))

    def test_clustered_matches_independent_reference(self):
        clusters = {'B': {'m': [1.0, 0.0]}, 'A': {'m': [0.0, 0.0, 1.0]}, 'C': {'m': [1.0]}}
        got = H.clustered_bootstrap_ci(clusters, 120, 99)['m']
        names = sorted(clusters)
        rng = random.Random(99)
        reps = []
        for _ in range(120):
            idx = [rng.randrange(len(names)) for _ in range(len(names))]
            tot = sum(len(clusters[names[i]]['m']) for i in idx)
            reps.append(sum(sum(clusters[names[i]]['m']) for i in idx) / tot)    # pooled mean
        reps.sort()
        self.assertEqual(got, (reps[2], reps[116]))                              # ranks ceil(.025*120)=3 and ceil(.975*120)=117 (1-based)
        self.assertEqual(got, H.clustered_bootstrap_ci(clusters, 120, 99)['m'])

    def test_clustered_equal_sized_clusters_reduce_to_mean_of_means(self):
        clusters = {c: {'m': v} for c, v in {'A': [1.0, 0.0], 'B': [1.0, 1.0], 'C': [0.0, 0.0]}.items()}
        lo, hi = H.clustered_bootstrap_ci(clusters, 300, 4)['m']
        self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_small_sample_policy(self):
        self.assertEqual([H.flag_n(n) for n in (9, 10, 29, 30)], ['VERY_SMALL', 'SMALL', 'SMALL', 'OK'])
        base = {m: 0.5 for m in H.METRICS}
        mk = lambda n: [dict(base, **{'mrr_at_10': (i % 2) * 1.0}) for i in range(n)]
        very = H.summarize(mk(9), 50, 1, 0.95, True)
        small = H.summarize(mk(12), 50, 1, 0.95, True)
        ok = H.summarize(mk(30), 50, 1, 0.95, True)
        self.assertIsNone(very['metrics']['mrr_at_10']['ci_low'])                # very small: point estimate and n only
        self.assertEqual(very['n'], 9)
        self.assertIsNotNone(small['metrics']['mrr_at_10']['ci_low'])
        self.assertTrue(small['metrics']['mrr_at_10']['instability_warning'])    # small: interval WITH warning
        self.assertFalse(ok['metrics']['mrr_at_10']['instability_warning'])
        self.assertIsNone(H.summarize(mk(40), 50, 1, 0.95, False)['metrics']['mrr_at_10']['ci_low'])   # sensitivity: no CI requested


class TestGroupsAndSettings(unittest.TestCase):
    def test_only_answerable_groups_are_evaluated_and_never_mixed(self):
        rows = [{'question_id': f'Q{i}', 'group': g} for i, g in enumerate(
            ['PRIMARY_ANSWERABLE', 'PRIMARY_UNANSWERABLE', 'SENSITIVITY_ANSWERABLE', 'SENSITIVITY_UNANSWERABLE', 'EXCLUDED', 'PRIMARY_ANSWERABLE'])]
        g = H.split_groups(rows)
        self.assertEqual([r['question_id'] for r in g[H.HEADLINE_GROUP]], ['Q0', 'Q5'])
        self.assertEqual([r['question_id'] for r in g[H.SENSITIVITY_GROUP]], ['Q2'])
        self.assertEqual(H.EVALUATED_GROUPS, ('PRIMARY_ANSWERABLE', 'SENSITIVITY_ANSWERABLE'))

    def test_frozen_settings_gate(self):
        with open(os.path.join(EVAL, 'config_bm25_v1.json'), encoding='utf-8') as fh:
            cfg = json.load(fh)
        H.check_frozen_settings(cfg, R.EvidenceConfig())                          # passes on the real frozen config
        bad = json.loads(json.dumps(cfg)); bad['status'] = 'PROPOSED'
        with self.assertRaises(H.ProtocolError):
            H.check_frozen_settings(bad)
        bad = json.loads(json.dumps(cfg)); bad['chunking']['chunk_seconds'] = 30.0
        with self.assertRaises(H.ProtocolError):
            H.check_frozen_settings(bad, R.EvidenceConfig())
        bad = json.loads(json.dumps(cfg)); bad['retrieval']['full_transcript_chars'] = 7000
        with self.assertRaises(H.ProtocolError):
            H.check_frozen_settings(bad)


class TestHashVerification(unittest.TestCase):
    def make(self, td):
        files = {'qa/retrieval.py': b'r', 'qa/transcript.py': b't', 'qa/__init__.py': b'i', 'ds/questions.csv': b'q', 'audit/flags.csv': b'a',
                 'dataset/review_v1/review_items.csv': b'ri', 'rl.csv': b'rl', 'man.csv': b'm'}
        for rel, data in files.items():
            p = os.path.join(td, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'wb') as fh:
                fh.write(data)
        h = lambda b: hashlib.sha256(b).hexdigest()
        return {'code_sha256': {k: h(files[k]) for k in ('qa/retrieval.py', 'qa/transcript.py', 'qa/__init__.py')},
                'dataset': {'path': 'ds', 'file_sha256': {'questions.csv': h(b'q')}, 'audit_flags_file': 'audit/flags.csv', 'audit_flags_sha256': h(b'a'),
                            'review_items_sha256': h(b'ri'), 'review_log': 'rl.csv', 'review_log_sha256_snapshot': h(b'rl')},
                'question_manifest': {'path': 'man.csv', 'sha256': h(b'm')}}

    def test_match_passes_and_lists_every_target(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.make(td)
            table = H.verify_pinned_hashes(cfg, td)
            self.assertEqual(len(table), 8)
            self.assertTrue(all(t['ok'] for t in table))

    def test_mismatch_raises_naming_the_file(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.make(td)
            for rel, data in (('qa/retrieval.py', b'changed'), ('man.csv', b'changed too')):
                with open(os.path.join(td, rel), 'wb') as fh:
                    fh.write(data)
            with self.assertRaises(H.HashMismatchError) as cm:
                H.verify_pinned_hashes(cfg, td)
            msg = str(cm.exception)
            self.assertIn('code:qa/retrieval.py', msg)
            self.assertIn('question_manifest', msg)
            self.assertNotIn('code:qa/transcript.py', msg)                         # only the mismatching files are listed

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.make(td)
            os.remove(os.path.join(td, 'rl.csv'))
            with self.assertRaises(H.HashMismatchError) as cm:
                H.verify_pinned_hashes(cfg, td)
            self.assertIn('MISSING', str(cm.exception))

    def test_main_refuses_when_a_pinned_file_changed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self.make(td)
            cfg_path = os.path.join(td, 'cfg.json')
            with open(cfg_path, 'w') as fh:
                json.dump(cfg, fh)
            out = os.path.join(td, 'out')
            old_repo = H.REPO
            H.REPO = td
            try:
                with open(os.path.join(td, 'ds/questions.csv'), 'wb') as fh:
                    fh.write(b'tampered')
                rc = H.main(['--config', cfg_path, '--out', out, '--skip-validator'])
            finally:
                H.REPO = old_repo
            self.assertEqual(rc, 3)
            self.assertFalse(os.path.exists(out))                                   # nothing was evaluated or written

    def test_main_refuses_to_overwrite_results(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(H.main(['--out', td]), 2)


if __name__ == '__main__':
    unittest.main()
