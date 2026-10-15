#!/usr/bin/env python3
"""exp_dense_hybrid_v1 - PHASE 3: hybrid RRF arm (M3), flip table and paired contrasts C1-C4.

    python -B experiments/dense_hybrid_v1/hybrid_arm.py m3-run --run-id A      # no model needed
    python -B experiments/dense_hybrid_v1/hybrid_arm.py m3-run --run-id B
    python -B experiments/dense_hybrid_v1/hybrid_arm.py m3-finalize

M3 (preregistered, nothing tuned):  RRF(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)  with ONE-based ranks and equal weights.
A chunk absent from a list contributes exactly 0 for that list (no penalty rank). BM25 ranks positive-score chunks only; dense ranks all chunks.
Final ties: original chunk index ascending on the unrounded float score. Inputs are the validated M1 (frozen BM25) and M2 (stored embeddings) arms.
No model is loaded here: M2 embeddings are read from the repeat-verified run directories.
"""
import csv, json, os, sys

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dense_hybrid_eval as X  # noqa: E402
import dense_arm as D  # noqa: E402

REPO = X.REPO
RESULTS = os.path.join(HERE, 'results', 'm3_hybrid_v1')
M2_DIR = os.path.join(HERE, 'results', 'm2_dense_v1')
RRF_K = 60
W_BM25, W_DENSE = 1.0, 1.0
PRIMARY_METRICS = ('hit_rate_at_5', 'mrr_at_10')
CONTRASTS = [  # id, first arm, second arm, registered (decision-rule) metrics
    ('C1', 'M3_hybrid_rrf', 'M1_bm25', ('hit_rate_at_5',)),
    ('C2', 'M3_hybrid_rrf', 'M1_bm25', ('mrr_at_10',)),
    ('C3', 'M2_dense', 'M1_bm25', PRIMARY_METRICS),
    ('C4', 'M3_hybrid_rrf', 'M2_dense', PRIMARY_METRICS)]


class HybridError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------- pure functions
def rrf_fuse(bm25_ranked, dense_ranked, n_chunks, k=RRF_K, w_bm25=W_BM25, w_dense=W_DENSE):
    """Reciprocal Rank Fusion. Returns (ranked chunk indices, per-chunk score list, bm25 rank map, dense rank map).

    Ranks are ONE-based positions in each (strictly ordered) list. A chunk missing from a list contributes 0 from it.
    Sort key: (-rrf_score, original chunk index). Chunks that appear in neither list are not ranked.
    """
    rb = {c: r for r, c in enumerate(bm25_ranked, 1)}
    rd = {c: r for r, c in enumerate(dense_ranked, 1)}
    if len(rb) != len(bm25_ranked) or len(rd) != len(dense_ranked):
        raise HybridError('a chunk appears twice in an input ranking')
    if any(not 0 <= c < n_chunks for c in list(rb) + list(rd)):
        raise HybridError('chunk index outside the lecture')
    scores = [0.0] * n_chunks
    for c in range(n_chunks):
        sb = w_bm25 * (1.0 / (k + rb[c])) if c in rb else 0.0
        sd = w_dense * (1.0 / (k + rd[c])) if c in rd else 0.0
        scores[c] = sb + sd
    present = sorted(set(rb) | set(rd))
    ranked = sorted(present, key=lambda c: (-scores[c], c))
    return ranked, scores, rb, rd


def decide(lo, hi):
    """Registered three-way rule on the paired 95% interval of (first arm minus second arm)."""
    if lo > 0:
        return 'improvement'
    if hi < 0:
        return 'decline'
    return 'no_demonstrated_difference'


def paired_contrast(bm25, first, second, metrics, resamples, seed, level):
    """first/second: per-question records in the SAME order. Interval of mean(first - second) from the frozen question-level bootstrap
    (same resampled indices for both arms because the resampling is applied to the per-question differences)."""
    if [r['question_id'] for r in first] != [r['question_id'] for r in second]:
        raise HybridError('paired arms are not aligned by question')
    diffs = {m: [a[m] - b[m] for a, b in zip(first, second)] for m in metrics}
    ci = bm25.bootstrap_ci(diffs, resamples, seed, level)
    n = len(first)
    out = {}
    for m in metrics:
        est = sum(diffs[m]) / n
        lo, hi = ci[m]
        out[m] = {'difference': est, 'ci_low': lo, 'ci_high': hi, 'n': n, 'decision': decide(lo, hi),
                  'n_questions_first_better': sum(1 for d in diffs[m] if d > 0), 'n_questions_second_better': sum(1 for d in diffs[m] if d < 0), 'n_tied': sum(1 for d in diffs[m] if d == 0)}
    return out


def clustered_contrast(bm25, first, second, metrics, resamples, seed, level):
    by_lec = {}
    for a, b in zip(first, second):
        by_lec.setdefault(a['lecture_id'], {m: [] for m in metrics})
        for m in metrics:
            by_lec[a['lecture_id']][m].append(a[m] - b[m])
    ci = bm25.clustered_bootstrap_ci(by_lec, resamples, seed, level)
    return {m: {'ci_low': ci[m][0], 'ci_high': ci[m][1], 'clusters': len(by_lec), 'decision_if_applied': decide(*ci[m]), 'role': 'secondary sensitivity check only'} for m in metrics}


def flip_rows(m1, m2, m3):
    """Primary-set questions where any arm's Hit@5 differs from BM25's. rank = first relevant rank (None = not within the arm's list)."""
    rows = []
    for a, b, c in zip(m1, m2, m3):
        h1, h2, h3 = a['hit_rate_at_5'], b['hit_rate_at_5'], c['hit_rate_at_5']
        if h1 == 1.0 and h2 == 1.0 and h3 == 1.0:
            continue
        if h1 == h2 == h3:
            kind = 'all_arms_miss_at_5'
        else:
            kind = ('bm25_miss_' if h1 == 0 else 'bm25_hit_') + 'hybrid_' + ('hit' if h3 == 1 else 'miss') + '_dense_' + ('hit' if h2 == 1 else 'miss')
        rows.append({'question_id': a['question_id'], 'lecture_id': a['lecture_id'], 'subject': a['subject'], 'question_category': a['question_category'], 'gold_interval_count': a['gold_interval_count'],
                     'bm25_hit_at_5': int(h1), 'dense_hit_at_5': int(h2), 'hybrid_hit_at_5': int(h3),
                     'bm25_first_relevant_rank': a['first_relevant_rank'], 'dense_first_relevant_rank': b['first_relevant_rank'], 'hybrid_first_relevant_rank': c['first_relevant_rank'],
                     'bm25_n_ranked': a['n_ranked_chunks'], 'category': kind})
    return rows


def bm25_list_length_counts(records):
    """Questions whose BM25 list is shorter than 5 / 10 entries, and how many of those are short because chunks scored 0 (not because the lecture is small)."""
    out = {}
    for k in (5, 10):
        short = [r for r in records if r['n_ranked_chunks'] < k]
        out[f'shorter_than_{k}'] = {'questions': len(short), 'of_which_lecture_has_at_least_k_chunks': sum(1 for r in short if r['n_chunks'] >= k), 'question_ids': [r['question_id'] for r in short]}
    out['empty_list'] = {'questions': sum(1 for r in records if r['n_ranked_chunks'] == 0), 'question_ids': [r['question_id'] for r in records if r['n_ranked_chunks'] == 0]}
    out['n_questions'] = len(records)
    return out


# ------------------------------------------------------------------------------------------------- input verification
def verify_m2_outputs(prereg):
    """M2 outputs must be the validated ones: final manifest hashes, repeat report PASS, run A/B embeddings equal, run files unchanged."""
    fin = os.path.join(M2_DIR, 'final')
    man = json.load(open(os.path.join(fin, 'run_manifest.json'), encoding='utf-8'))
    for name, h in man['results_sha256'].items():
        if X.sha256_file(os.path.join(fin, name)) != h:
            raise HybridError(f'M2 output {name} differs from its manifest hash')
    rep = json.load(open(os.path.join(fin, 'repeatability_report.json'), encoding='utf-8'))
    if rep['result'] != 'PASS':
        raise HybridError('M2 repeatability report is not PASS')
    info = json.load(open(os.path.join(M2_DIR, 'runs', 'run_A', 'run_info.json'), encoding='utf-8'))
    for n, h in info['file_sha256'].items():
        if X.sha256_file(os.path.join(M2_DIR, 'runs', 'run_A', n)) != h:
            raise HybridError(f'M2 run A file {n} differs from run_info')
    if info['model_tree_fingerprint'] != prereg['dense_model']['local_snapshot']['tree_fingerprint_sha256'] or info['preregistration_sha256'] != X.PREREG_SHA256:
        raise HybridError('M2 run was not made under the preregistered model / preregistration')
    for n in ('chunk_embeddings.npy', 'question_embeddings.npy'):
        if X.sha256_file(os.path.join(M2_DIR, 'runs', 'run_A', n)) != X.sha256_file(os.path.join(M2_DIR, 'runs', 'run_B', n)):
            raise HybridError('M2 run A and B embeddings differ')
    return man


def build_arms(verbose=True):
    """Verify every input, then return everything needed: records and full rankings for M1, M2, M3 (aligned by question)."""
    import numpy as np
    prereg, cfg, bm25, pins = X.verify_all_inputs()
    D.verify_m1_validation_unchanged()
    m2_manifest = verify_m2_outputs(prereg)
    sys.path.insert(0, REPO)
    from qa import retrieval
    from qa.transcript import normalize_segments
    ctxs, qlist, chunk_keys, chunk_texts, groups = D.build_inputs(cfg, bm25, retrieval, normalize_segments)
    manifest, questions, gold, cues = X.load_data(cfg)
    # M1: frozen BM25, full positive-score list per question
    bm_ranked, m1 = [], []
    for row, qrow, grec in qlist:
        ctx = ctxs[row['lecture_id']]
        ranked, scores, n_terms = X.m1_rank(bm25, retrieval, qrow['question_text'], ctx)
        bm_ranked.append(ranked)
        m1.append(X.evaluate_ranking(bm25, row, qrow, grec, ctx, ranked, scores, n_terms))
    diffs = X.compare_records_to_saved(m1, os.path.join(X.SAVED_DIR, 'per_question_results.csv'))
    if diffs:
        raise HybridError('M1 records differ from the saved baseline: ' + '; '.join(diffs[:5]))
    # M2: stored embeddings
    ce = np.load(os.path.join(M2_DIR, 'runs', 'run_A', 'chunk_embeddings.npy'))
    qe = np.load(os.path.join(M2_DIR, 'runs', 'run_A', 'question_embeddings.npy'))
    D.check_embeddings(ce)
    D.check_embeddings(qe)
    m2, m2_scores = D.records_from_embeddings(bm25, ctxs, qlist, chunk_keys, ce, qe)
    tmp = os.path.join(RESULTS, '_m2_check.csv')
    os.makedirs(RESULTS, exist_ok=True)
    D.write_records_csv(tmp, bm25, m2)
    same = X.sha256_file(tmp) == m2_manifest['results_sha256']['per_question_results.csv']
    os.remove(tmp)
    if not same:
        raise HybridError('M2 records recomputed from stored embeddings differ from the validated M2 per_question_results.csv')
    dense_ranked = [D.rank_by_scores(s) for s in m2_scores]
    # M3: RRF
    m3, m3_scores, ranks = [], [], []
    for (row, qrow, grec), br, dr in zip(qlist, bm_ranked, dense_ranked):
        ctx = ctxs[row['lecture_id']]
        ranked, scores, rb, rd = rrf_fuse(br, dr, len(ctx['chunks']))
        m3.append(X.evaluate_ranking(bm25, row, qrow, grec, ctx, ranked, scores, 0))
        m3_scores.append(scores)
        ranks.append((rb, rd))
    return {'prereg': prereg, 'cfg': cfg, 'bm25': bm25, 'pins': pins, 'groups': groups, 'qlist': qlist, 'ctxs': ctxs, 'm1': m1, 'm2': m2, 'm3': m3,
            'bm_ranked': bm_ranked, 'dense_ranked': dense_ranked, 'm3_scores': m3_scores, 'ranks': ranks, 'm2_manifest': m2_manifest}


# ------------------------------------------------------------------------------------------------- outputs
def write_m3_rankings(path, qlist, m3, m3_scores, ranks):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['question_id', 'rank', 'chunk_index', 'rrf_score_repr', 'bm25_rank', 'dense_rank'])
        for (row, _, _), scores, (rb, rd) in zip(qlist, m3_scores, ranks):
            order = sorted(set(rb) | set(rd), key=lambda c: (-scores[c], c))
            for r, c in enumerate(order, 1):
                w.writerow([row['question_id'], r, c, repr(scores[c]), rb.get(c, ''), rd.get(c, '')])


def write_flip_csv(path, rows):
    cols = ['question_id', 'lecture_id', 'subject', 'question_category', 'gold_interval_count', 'bm25_hit_at_5', 'dense_hit_at_5', 'hybrid_hit_at_5',
            'bm25_first_relevant_rank', 'dense_first_relevant_rank', 'hybrid_first_relevant_rank', 'bm25_n_ranked', 'category']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r[k] is None else r[k]) for k in cols})


def m3_run(run_id):
    out = os.path.join(RESULTS, 'runs', f'run_{run_id}')
    if os.path.exists(out):
        print(f'REFUSING to overwrite {out}', file=sys.stderr)
        return 2
    try:
        a = build_arms()
    except RuntimeError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    os.makedirs(out)
    D.write_records_csv(os.path.join(out, 'per_question_results_m3.csv'), a['bm25'], a['m3'])
    write_m3_rankings(os.path.join(out, 'full_rankings_m3.csv'), a['qlist'], a['m3'], a['m3_scores'], a['ranks'])
    files = {n: X.sha256_file(os.path.join(out, n)) for n in ('per_question_results_m3.csv', 'full_rankings_m3.csv')}
    json.dump({'run_id': run_id, 'file_sha256': files, 'python': sys.version.split()[0]}, open(os.path.join(out, 'run_info.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    print(f'run {run_id}: {files["per_question_results_m3.csv"][:16]} {files["full_rankings_m3.csv"][:16]}')
    return 0


def finalize_contrasts(bm25, cfg, prim, sens):
    st = cfg['statistics']['confidence_interval']
    B, seed, level = st['resamples'], st['seed'], st['level']
    out = {'settings': {'method': 'paired question-level percentile bootstrap of the mean per-question difference (first minus second); same resampled indices for both arms',
                        'level': level, 'resamples': B, 'seed': seed, 'reseeded_per_contrast': True, 'interval': 'nearest-rank 250th / 9750th',
                        'sign_convention': 'difference = first arm minus second arm; positive = first arm better',
                        'decision_rule': 'improvement iff the ENTIRE interval is above 0; decline iff the ENTIRE interval is below 0; otherwise no demonstrated difference (a bound equal to 0 counts as containing 0)',
                        'multiplicity': 'no correction (preregistered): 2 primary contrasts x 2 metrics'},
           'contrasts': []}
    all_m = bm25.METRICS
    s1 = cfg['statistics']['secondary_sensitivity_analysis']
    for cid, fa, sa, reg in CONTRASTS:
        f, s = prim[fa], prim[sa]
        full = paired_contrast(bm25, f, s, all_m, B, seed, level)
        entry = {'id': cid, 'contrast': f'{fa} minus {sa}', 'set': 'PRIMARY_ANSWERABLE', 'n': len(f), 'role': 'PRIMARY' if cid in ('C1', 'C2') else 'SECONDARY',
                 'registered_metrics': {m: full[m] for m in reg},
                 'descriptive_other_metrics': {m: {k: v for k, v in full[m].items() if k != 'decision'} for m in all_m if m not in reg},
                 'descriptive_other_metrics_note': 'point differences and intervals shown for completeness; NO decision rule applied and no inferential claim made',
                 'lecture_clustered_secondary': clustered_contrast(bm25, f, s, reg, s1['resamples'], s1['seed'], s1['level'])}
        out['contrasts'].append(entry)
    sens_out = {'note': 'SENSITIVITY_ANSWERABLE n=6: DESCRIPTIVE ONLY. No intervals, no decisions, not confirmatory.', 'n': len(sens['M1_bm25']), 'arms': {}, 'differences': {}}
    for arm, recs in sens.items():
        sens_out['arms'][arm] = {m: sum(r[m] for r in recs) / len(recs) for m in all_m}
    for cid, fa, sa, reg in CONTRASTS:
        sens_out['differences'][f'{cid}: {fa} minus {sa}'] = {m: sens_out['arms'][fa][m] - sens_out['arms'][sa][m] for m in all_m}
    out['sensitivity_descriptive'] = sens_out
    return out


def m3_finalize():
    da, db = (os.path.join(RESULTS, 'runs', f'run_{r}') for r in 'AB')
    final = os.path.join(RESULTS, 'final')
    if os.path.exists(final):
        print(f'REFUSING to overwrite {final}', file=sys.stderr)
        return 2
    try:
        a = build_arms()
    except RuntimeError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    bm25, cfg = a['bm25'], a['cfg']
    rep = {'tolerance_rule': 'bitwise (byte) identity of the two independent M3 runs', 'files': {}}
    for n in ('per_question_results_m3.csv', 'full_rankings_m3.csv'):
        rep['files'][n] = {'sha256_a': X.sha256_file(os.path.join(da, n)), 'sha256_b': X.sha256_file(os.path.join(db, n))}
        rep['files'][n]['byte_identical'] = rep['files'][n]['sha256_a'] == rep['files'][n]['sha256_b']
    os.makedirs(final)
    # the in-process M3 result must equal the files written by the two runs
    tmp = os.path.join(final, '_chk.csv')
    D.write_records_csv(tmp, bm25, a['m3'])
    rep['files']['per_question_results_m3.csv']['equals_in_process_recomputation'] = X.sha256_file(tmp) == rep['files']['per_question_results_m3.csv']['sha256_a']
    os.remove(tmp)
    rep['result'] = 'PASS' if all(v['byte_identical'] for v in rep['files'].values()) and rep['files']['per_question_results_m3.csv']['equals_in_process_recomputation'] else 'FAIL'
    json.dump(rep, open(os.path.join(final, 'repeatability_report.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    print('M3 repeatability:', rep['result'])
    if rep['result'] != 'PASS':
        print('STOP: M3 runs differ. No metrics written.', file=sys.stderr)
        return 5
    # aggregation self-check (M1) and arm aggregates
    saved = json.load(open(os.path.join(X.SAVED_DIR, 'aggregate_metrics.json'), encoding='utf-8'))
    agg1 = D.aggregate_records(bm25, cfg, a['m1'], a['groups'])
    if json.loads(json.dumps(agg1)) != saved:
        raise HybridError('aggregation self-check failed against the saved baseline')
    agg3 = D.aggregate_records(bm25, cfg, a['m3'], a['groups'])
    agg3['arm'] = 'M3_hybrid_rrf (RRF k=60, equal weights, one-based ranks)'
    json.dump(agg3, open(os.path.join(final, 'aggregate_metrics_m3.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    D.flat_aggregate_csv(os.path.join(final, 'aggregate_metrics_m3.csv'), agg3)
    D.write_records_csv(os.path.join(final, 'per_question_results_m3.csv'), bm25, a['m3'])
    write_m3_rankings(os.path.join(final, 'full_rankings_m3.csv'), a['qlist'], a['m3'], a['m3_scores'], a['ranks'])
    # split by set
    arms = {'M1_bm25': a['m1'], 'M2_dense': a['m2'], 'M3_hybrid_rrf': a['m3']}
    prim = {k: [r for r in v if r['group'] == bm25.HEADLINE_GROUP and r['scoreable']] for k, v in arms.items()}
    sens = {k: [r for r in v if r['group'] == bm25.SENSITIVITY_GROUP and r['scoreable']] for k, v in arms.items()}
    if not (len(prim['M1_bm25']) == len(prim['M2_dense']) == len(prim['M3_hybrid_rrf']) == 133 and len(sens['M1_bm25']) == 6):
        raise HybridError('unexpected set sizes')
    contr = finalize_contrasts(bm25, cfg, prim, sens)
    json.dump(contr, open(os.path.join(final, 'paired_contrasts.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    # flip table (primary set)
    rows = flip_rows(prim['M1_bm25'], prim['M2_dense'], prim['M3_hybrid_rrf'])
    write_flip_csv(os.path.join(final, 'flip_table.csv'), rows)
    misses = [r for r in rows if r['bm25_hit_at_5'] == 0]
    flip = {'n_primary': 133, 'bm25_hit_at_5_misses': len(misses),
            'hybrid_rescued': [r['question_id'] for r in misses if r['hybrid_hit_at_5'] == 1], 'hybrid_not_rescued': [r['question_id'] for r in misses if r['hybrid_hit_at_5'] == 0],
            'dense_rescued': [r['question_id'] for r in misses if r['dense_hit_at_5'] == 1], 'dense_not_rescued': [r['question_id'] for r in misses if r['dense_hit_at_5'] == 0],
            'bm25_hits_lost_by_hybrid': [r['question_id'] for r in rows if r['bm25_hit_at_5'] == 1 and r['hybrid_hit_at_5'] == 0],
            'bm25_hits_lost_by_dense': [r['question_id'] for r in rows if r['bm25_hit_at_5'] == 1 and r['dense_hit_at_5'] == 0],
            'questions_missed_by_all_three': [r['question_id'] for r in rows if r['category'] == 'all_arms_miss_at_5']}
    json.dump(flip, open(os.path.join(final, 'flip_summary.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    # BM25 list-length diagnostics
    diag = {'PRIMARY_ANSWERABLE': bm25_list_length_counts(prim['M1_bm25']), 'SENSITIVITY_ANSWERABLE': bm25_list_length_counts(sens['M1_bm25']),
            'dense_truncation': json.load(open(os.path.join(M2_DIR, 'runs', 'run_A', 'run_info.json'), encoding='utf-8'))['truncation']}
    json.dump(diag, open(os.path.join(final, 'diagnostics.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    # three-arm comparison table (point estimates only)
    with open(os.path.join(final, 'arm_comparison.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['set', 'arm', 'n'] + bm25.METRICS)
        for setname, d in (('PRIMARY_ANSWERABLE', prim), ('SENSITIVITY_ANSWERABLE', sens)):
            for arm, recs in d.items():
                w.writerow([setname, arm, len(recs)] + [repr(sum(r[m] for r in recs) / len(recs)) for m in bm25.METRICS])
    names = ['per_question_results_m3.csv', 'full_rankings_m3.csv', 'aggregate_metrics_m3.json', 'aggregate_metrics_m3.csv', 'paired_contrasts.json', 'flip_table.csv', 'flip_summary.json', 'diagnostics.json', 'arm_comparison.csv', 'repeatability_report.json']
    manifest = {'experiment': 'exp_dense_hybrid_v1', 'arm': 'M3_hybrid_rrf + contrasts', 'preregistration_sha256': X.PREREG_SHA256,
                'input_verification': {'frozen_config_pins_ok': len(a['pins']['frozen_config_pins']), 'prereg_file_pins_ok': a['pins']['prereg_file_pins_checked'],
                                       'm1_validation_equals_saved_baseline': True, 'm2_final_hashes_verified': True, 'm2_runs_A_B_embeddings_identical': True,
                                       'm2_records_recomputed_equal_validated': True, 'm1_records_equal_saved_per_question': True, 'aggregation_self_check_on_m1': True},
                'model_tree_fingerprint': a['m2_manifest']['run_A']['model_tree_fingerprint'], 'offline': {'HF_HUB_OFFLINE': os.environ['HF_HUB_OFFLINE'], 'TRANSFORMERS_OFFLINE': os.environ['TRANSFORMERS_OFFLINE']},
                'rrf': {'k': RRF_K, 'weights': {'bm25': W_BM25, 'dense': W_DENSE}, 'ranks': 'one-based', 'missing_list': 'contributes 0', 'tie_break': '-score, original chunk index'},
                'input_hashes': {'m2_chunk_embeddings': D.array_sha256(__import__('numpy').load(os.path.join(M2_DIR, 'runs', 'run_A', 'chunk_embeddings.npy'))),
                                 'm2_question_embeddings': D.array_sha256(__import__('numpy').load(os.path.join(M2_DIR, 'runs', 'run_A', 'question_embeddings.npy'))),
                                 'm2_results': a['m2_manifest']['results_sha256']},
                'code_sha256': {n: X.sha256_file(os.path.join(HERE, n)) for n in ('dense_hybrid_eval.py', 'dense_arm.py', 'hybrid_arm.py')},
                'results_sha256': {n: X.sha256_file(os.path.join(final, n)) for n in names},
                'limitations': ['Gold labels are machine-drafted from captions; only 1 of 176 primary questions was checked against the video.', 'n=133 in 25 lectures; questions within a lecture are correlated.',
                                'Alternate valid passages outside gold count as misses for every arm.', 'Offline retrieval benchmark only: no claim about production answer quality or user benefit.']}
    json.dump(manifest, open(os.path.join(final, 'run_manifest.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    for c in contr['contrasts']:
        for m, e in c['registered_metrics'].items():
            print(f"{c['id']} {c['contrast']:38s} {m:14s} diff {e['difference']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}] -> {e['decision']}")
    print('flip:', {k: (v if isinstance(v, int) else len(v)) for k, v in flip.items()})
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['m3-run'] and '--run-id' in argv:
        return m3_run(argv[argv.index('--run-id') + 1])
    if argv[:1] == ['m3-finalize']:
        return m3_finalize()
    print('usage: hybrid_arm.py m3-run --run-id A|B | m3-finalize', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
