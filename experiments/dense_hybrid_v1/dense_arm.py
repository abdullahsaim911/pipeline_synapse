#!/usr/bin/env python3
"""exp_dense_hybrid_v1 - PHASE 2: dense-only arm (M2). M3/RRF is NOT implemented here.

    # two independent runs (separate processes), then the comparison + metrics
    python -B experiments/dense_hybrid_v1/dense_arm.py m2-run --run-id A
    python -B experiments/dense_hybrid_v1/dense_arm.py m2-run --run-id B
    python -B experiments/dense_hybrid_v1/dense_arm.py m2-finalize

Offline only (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are forced to 1 and the model is loaded from the local snapshot directory with
local_files_only=True). Nothing is tuned: text construction, query, model, batch size and ranking rule are the preregistered ones.

Repeatability rule (the preregistration declares identical rankings/metrics but NO numerical tolerance for embeddings or scores):
the strictest rule is applied - embeddings, scores and rankings must be BITWISE identical between the two runs. Any difference is
reported (max absolute difference, count) and the finalize step stops without writing metrics.
"""
import csv, glob, hashlib, json, math, os, platform, sys, time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dense_hybrid_eval as X  # noqa: E402

REPO = X.REPO
RESULTS = os.path.join(HERE, 'results', 'm2_dense_v1')
BATCH_SIZE = 32
EXPECTED_DIM = 384
EXPECTED_MAX_SEQ = 256
HF_MODEL_DIR = os.path.expanduser('~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2')

RECORD_COLS = ['question_id', 'lecture_id', 'group', 'subject', 'question_category', 'gold_interval_count', 'n_gold_intervals', 'n_gold_intervals_scored',
               'n_gold_cues_retrievable', 'n_chunks', 'n_ranked_chunks', 'scoreable']


class DenseError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- pure helpers (testable without a model)
def rank_by_scores(scores):
    """M2 ranking: ALL chunks, ordered by (-score, original chunk index). No positive-score filter."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def cosine_scores(chunk_emb, query_emb):
    """Dot product of L2-normalised vectors, accumulated in float64 from the float32 embeddings. Returns Python floats."""
    import numpy as np
    return [float(x) for x in np.dot(chunk_emb.astype(np.float64), query_emb.astype(np.float64))]


def check_embeddings(emb, dim=EXPECTED_DIM, norm_tol=1e-5):
    """Shape, finiteness and unit norm. Raises DenseError. Returns max |norm-1|."""
    import numpy as np
    if emb.ndim != 2 or emb.shape[1] != dim:
        raise DenseError(f'embedding shape {emb.shape}, expected (n, {dim})')
    if not np.isfinite(emb).all():
        raise DenseError('non-finite values in embeddings')
    dev = float(np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1.0).max())
    if dev > norm_tol:
        raise DenseError(f'embeddings are not unit-norm: max |norm-1| = {dev}')
    return dev


def chunk_text(ctx, chunk_pos):
    """Preregistered chunk text: the member cues' cleaned texts joined with a single space, in order."""
    a, b = ctx['chunks'][chunk_pos]
    return ' '.join(ctx['segs'][i].text for i in range(a, b + 1))


def array_sha256(arr):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_inputs(cfg, bm25, retrieval, normalize_segments):
    """Contexts, ordered chunk list (lecture id sorted, chunk index ascending) and ordered question list (headline group first)."""
    manifest, questions, gold, cues = X.load_data(cfg)
    ec = retrieval.EvidenceConfig()
    bm25.check_frozen_settings(cfg, ec)
    groups = bm25.split_groups(manifest)
    ctxs, qlist = {}, []
    for gname in bm25.EVALUATED_GROUPS:
        for row in groups[gname]:
            lid = row['lecture_id']
            if lid not in ctxs:
                ctxs[lid] = bm25.build_lecture_context(cues[lid], retrieval, normalize_segments, ec, cfg['text_field']['value'])
            qlist.append((row, questions[row['question_id']], gold[row['question_id']]))
    X.check_chunk_identity(cfg, ctxs)
    chunk_keys, chunk_texts = [], []
    for lid in sorted(ctxs):
        for pos in range(len(ctxs[lid]['chunks'])):
            t = chunk_text(ctxs[lid], pos)
            if not t.strip():
                raise DenseError(f'empty chunk text {lid}#{pos}')
            chunk_keys.append((lid, pos))
            chunk_texts.append(t)
    return ctxs, qlist, chunk_keys, chunk_texts, groups


def lecture_slices(chunk_keys):
    sl, start = {}, 0
    for i, (lid, pos) in enumerate(chunk_keys):
        if i == 0 or chunk_keys[i - 1][0] != lid:
            start = i
            if pos != 0:
                raise DenseError('chunk order is not contiguous per lecture')
        sl[lid] = (start, i + 1)
    return sl


def records_from_embeddings(bm25, ctxs, qlist, chunk_keys, chunk_emb, query_emb):
    """Per-question records for M2 plus the full score vectors (alignment is asserted)."""
    if len(qlist) != len(query_emb) or len(chunk_keys) != len(chunk_emb):
        raise DenseError('embedding rows do not match questions/chunks')
    sl = lecture_slices(chunk_keys)
    records, all_scores = [], []
    for qi, (row, qrow, grec) in enumerate(qlist):
        lid = row['lecture_id']
        a, b = sl[lid]
        if b - a != len(ctxs[lid]['chunks']) or [k[1] for k in chunk_keys[a:b]] != list(range(b - a)):
            raise DenseError(f'chunk/lecture alignment broken for {lid}')
        scores = cosine_scores(chunk_emb[a:b], query_emb[qi])
        ranked = rank_by_scores(scores)
        rec = X.evaluate_ranking(bm25, row, qrow, grec, ctxs[lid], ranked, scores, 0)
        records.append(rec)
        all_scores.append(scores)
    return records, all_scores


# --------------------------------------------------------------------------------------------- aggregation (same procedure as the frozen harness)
def aggregate_records(bm25, cfg, records, groups):
    """Replicates bm25_eval.run_evaluation's aggregation on given records (summarize + clustered bootstrap + strata check)."""
    import collections, statistics
    st = cfg['statistics']
    B, seed, level = st['confidence_interval']['resamples'], st['confidence_interval']['seed'], st['confidence_interval']['level']
    agg = {'counts': cfg['question_manifest']['counts'], 'sets': {}, 'not_evaluated': {k: len(v) for k, v in groups.items() if k not in bm25.EVALUATED_GROUPS}}
    agg['unscoreable_question_ids'] = [r['question_id'] for r in records if not r['scoreable']]
    for gname in bm25.EVALUATED_GROUPS:
        recs = [r for r in records if r['group'] == gname and r['scoreable']]
        is_head = gname == bm25.HEADLINE_GROUP
        block = {'role': 'headline' if is_head else 'sensitivity (reported separately, never merged)', 'overall': bm25.summarize(recs, B, seed, level, with_ci=is_head)}
        if is_head:
            dims = list(st['breakdowns']['dimensions'])
            br = {d: {lab: bm25.summarize([r for r in recs if r[d] == lab], B, seed, level, with_ci=True) for lab in sorted({r[d] for r in recs})} for d in dims}
            block['breakdowns'] = br
            by_lec = collections.defaultdict(list)
            for r in recs:
                by_lec[r['lecture_id']].append(r)
            s1 = st['secondary_sensitivity_analysis']
            cci = bm25.clustered_bootstrap_ci({lid: {m: [r[m] for r in rs] for m in bm25.METRICS} for lid, rs in by_lec.items()}, s1['resamples'], s1['seed'], s1['level'])
            block['secondary_lecture_clustered'] = {
                'role': 'SECONDARY sensitivity analysis; the question-level interval is primary', 'clusters': len(by_lec),
                'questions_per_cluster_min_median_max': [min(len(v) for v in by_lec.values()), statistics.median(len(v) for v in by_lec.values()), max(len(v) for v in by_lec.values())],
                'metrics': {m: {'estimate': block['overall']['metrics'][m]['estimate'], 'ci_low': cci[m][0], 'ci_high': cci[m][1], 'ci_method': 'lecture-clustered percentile bootstrap'} for m in bm25.METRICS}}
            derived = st['breakdowns']['derived_sample_sizes_PRIMARY_ANSWERABLE']
            for d in dims:
                got = {lab: {'n': v['n'], 'sample_flag': v['sample_flag']} for lab, v in br[d].items()}
                if got != derived[d]:
                    raise DenseError(f'breakdown {d} strata differ from the frozen config')
        agg['sets'][gname] = block
    return agg


# --------------------------------------------------------------------------------------------- serialization
def write_records_csv(path, bm25, records):
    cols = RECORD_COLS + bm25.METRICS + ['first_relevant_rank', 'top10_json']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in records:
            row = {k: r.get(k, '') for k in cols if k != 'top10_json'}
            row['first_relevant_rank'] = '' if r.get('first_relevant_rank') is None else r['first_relevant_rank']
            row['top10_json'] = json.dumps(r['top10'], separators=(',', ':'))
            w.writerow(row)


def write_full_rankings(path, qlist, ctxs, all_scores):
    """All-chunk ranking with unrounded scores (repr), for the repeat comparison and later hybrid use."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['question_id', 'rank', 'chunk_index', 'score_repr'])
        for (row, _, _), scores in zip(qlist, all_scores):
            for rank, c in enumerate(rank_by_scores(scores), 1):
                w.writerow([row['question_id'], rank, c, repr(scores[c])])


def flat_aggregate_csv(path, agg):
    flat = []
    def add(set_name, level, dim, label, blk):
        for m, e in blk['metrics'].items():
            flat.append({'set': set_name, 'level': level, 'dimension': dim, 'stratum': label, 'n': blk.get('n', ''), 'sample_flag': blk.get('sample_flag', ''), 'metric': m,
                         'estimate': e['estimate'], 'ci_low': e.get('ci_low'), 'ci_high': e.get('ci_high'), 'ci_method': e.get('ci_method'), 'instability_warning': e.get('instability_warning')})
    for sname, blk in agg['sets'].items():
        add(sname, 'overall', '', '', blk['overall'])
        for d, labs in blk.get('breakdowns', {}).items():
            for lab, b in labs.items():
                add(sname, 'stratum', d, lab, b)
        if 'secondary_lecture_clustered' in blk:
            add(sname, 'overall_lecture_clustered', '', '', {'n': blk['overall']['n'], 'sample_flag': blk['overall']['sample_flag'], 'metrics': blk['secondary_lecture_clustered']['metrics']})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(flat)


# --------------------------------------------------------------------------------------------- model
def model_tree_fingerprint(snapshot_dir):
    files = {}
    for root, _, names in os.walk(snapshot_dir):
        for n in names:
            p = os.path.join(root, n)
            files[os.path.relpath(p, snapshot_dir).replace(os.sep, '/')] = X.sha256_file(p)
    return hashlib.sha256('\n'.join(f'{k}\t{v}' for k, v in sorted(files.items())).encode()).hexdigest(), files


def locate_snapshot(prereg):
    want = prereg['dense_model']['local_snapshot']['snapshot_commit_directory']
    path = os.path.join(HF_MODEL_DIR, 'snapshots', want)
    if not os.path.isdir(path):
        raise DenseError(f'preregistered snapshot directory missing: {path}')
    return path


def load_model(prereg):
    """Verify the snapshot against the preregistered fingerprint, then load it offline from the local directory."""
    snap = locate_snapshot(prereg)
    fp, files = model_tree_fingerprint(snap)
    if fp != prereg['dense_model']['local_snapshot']['tree_fingerprint_sha256']:
        raise DenseError(f'model snapshot fingerprint differs from the preregistration: {fp}')
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(snap, device='cpu', local_files_only=True)
    if model.max_seq_length != EXPECTED_MAX_SEQ or model.get_embedding_dimension() != EXPECTED_DIM:
        raise DenseError(f'unexpected model settings: max_seq_length={model.max_seq_length}, dim={model.get_embedding_dimension()}')
    return model, snap, fp, files


def encode(model, texts):
    import numpy as np
    import torch
    with torch.inference_mode():
        emb = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def count_truncated(model, texts):
    ids = model.tokenizer(texts, truncation=False, add_special_tokens=True)['input_ids']
    lens = [len(x) for x in ids]
    return lens, sum(1 for n in lens if n > model.max_seq_length)


# --------------------------------------------------------------------------------------------- CLI: one independent run
def snapshot_unchanged(snap, prereg):
    fp, _ = model_tree_fingerprint(snap)
    return fp == prereg['dense_model']['local_snapshot']['tree_fingerprint_sha256']


def m2_run(run_id):
    import numpy as np
    out = os.path.join(RESULTS, 'runs', f'run_{run_id}')
    if os.path.exists(out):
        print(f'REFUSING to overwrite {out}', file=sys.stderr)
        return 2
    try:
        prereg, cfg, bm25, pins = X.verify_all_inputs()
        verify_m1_validation_unchanged()
    except RuntimeError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    sys.path.insert(0, REPO)
    from qa import retrieval
    from qa.transcript import normalize_segments
    ctxs, qlist, chunk_keys, chunk_texts, groups = build_inputs(cfg, bm25, retrieval, normalize_segments)
    if len(chunk_texts) != prereg['chunks']['expected_total_chunks']:
        raise DenseError(f'{len(chunk_texts)} chunks, preregistered {prereg["chunks"]["expected_total_chunks"]}')
    print(f'inputs verified; {len(chunk_texts)} chunks, {len(qlist)} questions')
    t0 = time.time()
    model, snap, fp, _ = load_model(prereg)
    t_load = time.time() - t0
    q_texts = [q[1]['question_text'] for q in qlist]                       # raw question text, no prefix, no preprocessing
    t1 = time.time()
    chunk_emb = encode(model, chunk_texts)
    query_emb = encode(model, q_texts)
    t_enc = time.time() - t1
    dev_c, dev_q = check_embeddings(chunk_emb), check_embeddings(query_emb)
    ch_len, ch_trunc = count_truncated(model, chunk_texts)
    q_len, q_trunc = count_truncated(model, q_texts)
    records, all_scores = records_from_embeddings(bm25, ctxs, qlist, chunk_keys, chunk_emb, query_emb)
    os.makedirs(out)
    np.save(os.path.join(out, 'chunk_embeddings.npy'), chunk_emb)
    np.save(os.path.join(out, 'question_embeddings.npy'), query_emb)
    write_records_csv(os.path.join(out, 'per_question_results.csv'), bm25, records)
    write_full_rankings(os.path.join(out, 'full_rankings.csv'), qlist, ctxs, all_scores)
    with open(os.path.join(out, 'chunk_index.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['row', 'lecture_id', 'chunk_index', 'n_chars', 'n_tokens_with_special', 'truncated_at_256'])
        for i, ((lid, pos), t, n) in enumerate(zip(chunk_keys, chunk_texts, ch_len)):
            w.writerow([i, lid, pos, len(t), n, int(n > EXPECTED_MAX_SEQ)])
    import torch, numpy, sentence_transformers, transformers
    info = {'run_id': run_id, 'preregistration_sha256': X.PREREG_SHA256, 'model_snapshot': snap.replace('\\', '/'), 'model_tree_fingerprint': fp,
            'model_unchanged_after_run': snapshot_unchanged(snap, prereg),
            'n_chunks': len(chunk_texts), 'n_questions': len(qlist), 'batch_size': BATCH_SIZE, 'device': 'cpu', 'dtype_embeddings': 'float32',
            'score_accumulation': 'float64 dot product of the float32 normalised embeddings (np.dot)',
            'encode_call': 'SentenceTransformer.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True) under torch.inference_mode()',
            'query_text': 'question_text exactly as in questions.csv (no prefix, no preprocessing)', 'chunk_text': 'member cues cleaned text joined with one space',
            'chunk_embeddings_sha256': array_sha256(chunk_emb), 'question_embeddings_sha256': array_sha256(query_emb), 'max_abs_norm_deviation': {'chunks': dev_c, 'questions': dev_q},
            'truncation': {'max_seq_length': EXPECTED_MAX_SEQ, 'chunks_truncated': ch_trunc, 'chunks_total': len(ch_len), 'max_chunk_tokens': max(ch_len),
                           'questions_truncated': q_trunc, 'max_question_tokens': max(q_len), 'note': 'diagnostic only; truncation is the model default and was not altered'},
            'timing_seconds': {'model_load': round(t_load, 2), 'encode_all': round(t_enc, 2)},
            'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__, 'sentence_transformers': sentence_transformers.__version__,
                        'transformers': transformers.__version__, 'numpy': numpy.__version__, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                        'torch_deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
                        'determinism_note': 'torch deterministic-algorithms mode was NOT enabled; repeatability is verified empirically (bitwise comparison of two runs), not assumed.',
                        'env': {k: os.environ.get(k) for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
            'file_sha256': {n: X.sha256_file(os.path.join(out, n)) for n in ('chunk_embeddings.npy', 'question_embeddings.npy', 'per_question_results.csv', 'full_rankings.csv', 'chunk_index.csv')}}
    with open(os.path.join(out, 'run_info.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(info, f, indent=2)
    print(f'run {run_id}: chunk emb sha {info["chunk_embeddings_sha256"][:16]}, question emb sha {info["question_embeddings_sha256"][:16]}; truncated chunks {ch_trunc}/{len(ch_len)}; encode {t_enc:.1f}s; model unchanged {info["model_unchanged_after_run"]}')
    return 0


def verify_m1_validation_unchanged():
    d = os.path.join(HERE, 'results', 'm1_validation')
    for name in X.COMPARED_FILES:
        if X.sha256_file(os.path.join(d, name)) != X.sha256_file(os.path.join(X.SAVED_DIR, name)):
            raise X.ReproductionError(f'M1 validation output {name} no longer equals the saved baseline')


# --------------------------------------------------------------------------------------------- CLI: compare + metrics
def compare_runs(dir_a, dir_b):
    import numpy as np
    rep = {'tolerance_rule': 'ZERO tolerance (bitwise identity) - the preregistration declares identical rankings/metrics and no numerical tolerance; the strictest rule is applied', 'checks': {}}
    for name in ('chunk_embeddings.npy', 'question_embeddings.npy'):
        a, b = np.load(os.path.join(dir_a, name)), np.load(os.path.join(dir_b, name))
        same = a.shape == b.shape and a.tobytes() == b.tobytes()
        diff = None if a.shape != b.shape else float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        rep['checks'][name] = {'bitwise_identical': same, 'shape': list(a.shape), 'sha256_a': array_sha256(a), 'sha256_b': array_sha256(b), 'max_abs_difference': diff,
                               'n_differing_values': None if a.shape != b.shape else int((a != b).sum())}
    ra, rb = X.read_csv_rows(os.path.join(dir_a, 'full_rankings.csv')), X.read_csv_rows(os.path.join(dir_b, 'full_rankings.csv'))
    rank_diff = [(x['question_id'], x['rank']) for x, y in zip(ra, rb) if (x['question_id'], x['rank'], x['chunk_index']) != (y['question_id'], y['rank'], y['chunk_index'])]
    score_diff = [(x['question_id'], x['rank']) for x, y in zip(ra, rb) if x['score_repr'] != y['score_repr']]
    maxd = max((abs(float(x['score_repr']) - float(y['score_repr'])) for x, y in zip(ra, rb)), default=0.0)
    rep['checks']['full_rankings'] = {'rows': [len(ra), len(rb)], 'ranking_order_differences': len(rank_diff), 'score_differences': len(score_diff), 'max_abs_score_difference': maxd,
                                     'first_ranking_differences': rank_diff[:10]}
    for name in ('per_question_results.csv', 'full_rankings.csv', 'chunk_index.csv'):
        rep['checks'][name + '_bytes'] = {'byte_identical': X.sha256_file(os.path.join(dir_a, name)) == X.sha256_file(os.path.join(dir_b, name))}
    ok = (all(v.get('bitwise_identical', True) and v.get('byte_identical', True) for v in rep['checks'].values()) and not rank_diff and not score_diff and len(ra) == len(rb))
    rep['result'] = 'PASS' if ok else 'FAIL'
    return rep


def m2_finalize():
    da, db = (os.path.join(RESULTS, 'runs', f'run_{r}') for r in 'AB')
    final = os.path.join(RESULTS, 'final')
    if os.path.exists(final):
        print(f'REFUSING to overwrite {final}', file=sys.stderr)
        return 2
    prereg, cfg, bm25, pins = X.verify_all_inputs()
    verify_m1_validation_unchanged()
    rep = compare_runs(da, db)
    infos = [json.load(open(os.path.join(d, 'run_info.json'), encoding='utf-8')) for d in (da, db)]
    rep['runs'] = {i['run_id']: {'timing_seconds': i['timing_seconds'], 'model_unchanged_after_run': i['model_unchanged_after_run'], 'torch_threads': i['runtime']['torch_threads']} for i in infos}
    os.makedirs(final)
    with open(os.path.join(final, 'repeatability_report.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(rep, f, indent=2)
    print('repeatability:', rep['result'])
    if rep['result'] != 'PASS':
        print('STOP: the two dense runs are not bitwise identical. Metrics are NOT produced. See repeatability_report.json', file=sys.stderr)
        return 5
    sys.path.insert(0, REPO)
    from qa import retrieval
    from qa.transcript import normalize_segments
    import numpy as np
    ctxs, qlist, chunk_keys, chunk_texts, groups = build_inputs(cfg, bm25, retrieval, normalize_segments)
    ce, qe = np.load(os.path.join(da, 'chunk_embeddings.npy')), np.load(os.path.join(da, 'question_embeddings.npy'))
    records, _ = records_from_embeddings(bm25, ctxs, qlist, chunk_keys, ce, qe)
    # the CSV recomputed from the stored embeddings must equal the one written by run A
    write_records_csv(os.path.join(final, 'per_question_results.csv'), bm25, records)
    if X.sha256_file(os.path.join(final, 'per_question_results.csv')) != X.sha256_file(os.path.join(da, 'per_question_results.csv')):
        raise DenseError('per-question results recomputed from stored embeddings differ from the run output')
    # self-check of the aggregation code: applied to the M1 records it must reproduce the saved baseline aggregate exactly
    m1_records, _ = X.run_m1(cfg, bm25, retrieval, normalize_segments)
    m1_agg = aggregate_records(bm25, cfg, m1_records, bm25.split_groups(X.load_data(cfg)[0]))
    with open(os.path.join(X.SAVED_DIR, 'aggregate_metrics.json'), encoding='utf-8') as f:
        saved_agg = json.load(f)
    if json.loads(json.dumps(m1_agg)) != saved_agg:
        raise DenseError('aggregation self-check failed: M1 aggregate differs from the saved baseline aggregate')
    print('aggregation self-check on M1 records: identical to saved bm25_v1.0 aggregate')
    agg = aggregate_records(bm25, cfg, records, groups)
    agg['arm'] = 'M2_dense (all-MiniLM-L6-v2, cosine, ALL chunks ranked)'
    with open(os.path.join(final, 'aggregate_metrics.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(agg, f, indent=2)
    flat_aggregate_csv(os.path.join(final, 'aggregate_metrics.csv'), agg)
    manifest = {'experiment': 'exp_dense_hybrid_v1', 'arm': 'M2_dense', 'preregistration_sha256': X.PREREG_SHA256, 'repeatability': rep['result'],
                'aggregation_self_check_on_m1': 'identical to saved bm25_v1.0 aggregate_metrics.json',
                'run_A': infos[0], 'harness_sha256': {n: X.sha256_file(os.path.join(HERE, n)) for n in ('dense_hybrid_eval.py', 'dense_arm.py')},
                'results_sha256': {n: X.sha256_file(os.path.join(final, n)) for n in ('per_question_results.csv', 'aggregate_metrics.json', 'aggregate_metrics.csv', 'repeatability_report.json')},
                'notes': ['n_query_terms is omitted: it is a BM25 notion.', 'Sensitivity set (n=6) is descriptive: point estimates only.', 'No M1/M3 numbers or contrasts are produced in this stage.']}
    with open(os.path.join(final, 'run_manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2)
    head = agg['sets'][bm25.HEADLINE_GROUP]['overall']
    print(f"M2 dense, PRIMARY_ANSWERABLE n={head['n']}")
    for m in bm25.METRICS:
        e = head['metrics'][m]
        print(f"  {m:32s} {e['estimate']:.4f}  95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['m2-run'] and '--run-id' in argv:
        return m2_run(argv[argv.index('--run-id') + 1])
    if argv[:1] == ['m2-finalize']:
        return m2_finalize()
    print('usage: dense_arm.py m2-run --run-id A|B | m2-finalize', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
