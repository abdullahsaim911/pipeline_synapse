#!/usr/bin/env python3
"""exp_dense_hybrid_v1 harness - PHASE 1: frozen BM25 arm (M1) only.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B experiments/dense_hybrid_v1/dense_hybrid_eval.py m1-validate [--out DIR]

Implemented here: pinned-input verification, the M1 (raw BM25) arm, its comparison with the saved bm25_v1.0 results, and an environment
determinism report. NOT implemented (waits for approval): M2 dense, M3 hybrid RRF, paired contrasts. No model is loaded and nothing is encoded.

The frozen harness eval/bm25_eval.py is IMPORTED (after its SHA-256 is checked against the preregistration), never copied or edited.
"""
import csv, hashlib, importlib.util, json, os, platform, subprocess, sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PREREG_PATH = os.path.join(HERE, 'config_exp1.json')
PREREG_SHA256 = 'ba06467957935190b50ab2e8d7cb09c5e392bbda40f4219325677af4ccc09dc6'
DEFAULT_M1_OUT = os.path.join(HERE, 'results', 'm1_validation')
SAVED_DIR = os.path.join(REPO, 'eval', 'results', 'bm25_v1.0')
COMPARED_FILES = ('per_question_results.csv', 'aggregate_metrics.json', 'aggregate_metrics.csv')


class PreregError(RuntimeError):
    """The preregistration or a pinned input differs from what was registered."""


class ReproductionError(RuntimeError):
    """The M1 arm does not reproduce the saved bm25_v1.0 results."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ pinned inputs
def load_prereg(path=PREREG_PATH, expected_sha=PREREG_SHA256):
    actual = sha256_file(path)
    if actual != expected_sha:
        raise PreregError(f'preregistration changed: expected {expected_sha}, got {actual}')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def verify_prereg_files(prereg, root=REPO):
    """Harness / saved-result / frozen-config hashes recorded in the preregistration. Raises PreregError listing every difference."""
    fi, bad = prereg['frozen_inputs'], []
    targets = [('frozen_bm25_config', fi['frozen_bm25_config']['path'], fi['frozen_bm25_config']['sha256'])]
    targets += [('harness:' + p, p, h) for p, h in fi['harness_code']['files'].items()]
    targets += [('saved_result:' + p, p, h) for p, h in fi['saved_bm25_results']['files'].items()]
    for name, rel, expected in targets:
        path = os.path.join(root, rel)
        actual = sha256_file(path) if os.path.exists(path) else None
        if actual != expected:
            bad.append(f'  {name} ({rel}): expected {expected[:16]}..., got {"MISSING" if actual is None else actual[:16] + "..."}')
    if bad:
        raise PreregError('pinned input differs from the preregistration:\n' + '\n'.join(bad))
    return len(targets)


def load_frozen_module(prereg, root=REPO):
    """Import eval/bm25_eval.py read-only, only after its hash matches the preregistration."""
    rel = 'eval/bm25_eval.py'
    expected = prereg['frozen_inputs']['harness_code']['files'][rel]
    path = os.path.join(root, rel)
    if sha256_file(path) != expected:
        raise PreregError(f'{rel} does not match the preregistered hash; refusing to import it')
    spec = importlib.util.spec_from_file_location('frozen_bm25_eval', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_all_inputs(prereg=None, root=REPO):
    """Prereg hash, preregistered file hashes, every pin of the frozen config, config-vs-prereg consistency. Returns (prereg, cfg, bm25, table)."""
    prereg = prereg or load_prereg()
    n_prereg = verify_prereg_files(prereg, root)
    bm25 = load_frozen_module(prereg, root)
    with open(os.path.join(root, prereg['frozen_inputs']['frozen_bm25_config']['path']), encoding='utf-8') as f:
        cfg = json.load(f)
    table = bm25.verify_pinned_hashes(cfg, root)              # raises HashMismatchError listing every mismatch
    fi = prereg['frozen_inputs']
    problems = []
    if fi['dataset']['file_sha256'] != cfg['dataset']['file_sha256']:
        problems.append('dataset file hashes in the preregistration differ from the frozen config')
    if fi['production_code'] != cfg['code_sha256']:
        problems.append('production code hashes in the preregistration differ from the frozen config')
    if fi['question_manifest']['sha256'] != cfg['question_manifest']['sha256']:
        problems.append('question manifest hash differs')
    with open(os.path.join(root, 'eval/results/bm25_v1.0/run_manifest.json'), encoding='utf-8') as f:
        rm = json.load(f)
    if rm['config_sha256'] != fi['frozen_bm25_config']['sha256']:
        problems.append('saved run_manifest config hash differs from the frozen config')
    if problems:
        raise PreregError('inconsistent pins:\n  ' + '\n  '.join(problems))
    return prereg, cfg, bm25, {'prereg_file_pins_checked': n_prereg, 'frozen_config_pins': table}


# ------------------------------------------------------------------------------------------------ data + chunks
def load_data(cfg, root=REPO):
    ds = os.path.join(root, cfg['dataset']['path'])

    def rd(p):
        with open(p, encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    manifest = rd(os.path.join(root, cfg['question_manifest']['path']))
    questions = {r['question_id']: r for r in rd(os.path.join(ds, 'questions.csv'))}
    with open(os.path.join(ds, 'gold_evidence.json'), encoding='utf-8') as f:
        gold = {g['question_id']: g for g in json.load(f)['questions']}
    cues = {}
    with open(os.path.join(ds, 'transcript_segments.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            cues.setdefault(r['lecture_id'], []).append(r)
    return manifest, questions, gold, cues


def check_chunk_identity(cfg, ctxs):
    """Chunks must be exactly those of bm25_v1.0: counts per lecture, a partition of the retained cues, in lecture order."""
    problems = []
    expected = cfg['chunking']['chunk_counts_per_lecture']
    for lid, ctx in sorted(ctxs.items()):
        if lid in expected and len(ctx['chunks']) != expected[lid]:
            problems.append(f'{lid}: {len(ctx["chunks"])} chunks, frozen config says {expected[lid]}')
        flat = [i for a, b in ctx['chunks'] for i in range(a, b + 1)]
        if flat != list(range(len(ctx['segs']))):
            problems.append(f'{lid}: chunks are not a contiguous, ordered partition of the retained cues')
        starts = [info['start'] for info in ctx['chunk_info']]
        if starts != sorted(starts):
            problems.append(f'{lid}: chunk start times are not in lecture order')
    if problems:
        raise ReproductionError('chunk identity differs from bm25_v1.0:\n  ' + '\n  '.join(problems))
    return {lid: len(c['chunks']) for lid, c in ctxs.items()}


# ------------------------------------------------------------------------------------------------ M1 arm
def m1_rank(bm25, retrieval, question_text, ctx):
    """M1 = the frozen raw BM25 ranking: positive scores only, ordered by (-score, chunk index)."""
    return bm25.rank_question(retrieval, question_text, ctx)


def evaluate_ranking(bm25, row, qrow, gold_record, ctx, ranked, scores, n_terms):
    """Per-question record for any arm's ranked list. Same shape as the frozen harness record so the same writers apply."""
    ivs = bm25.gold_intervals(gold_record, ctx['dropped'])
    gold_all = set().union(*ivs) if ivs else set()
    m = bm25.question_metrics(ranked, ctx['chunk_cues'], ivs)
    top = []
    for rank, c in enumerate(ranked[:bm25.MRR_DEPTH], 1):
        info = ctx['chunk_info'][c]
        top.append({'rank': rank, 'chunk_index': c, 'score': scores[c], 'start': info['start'], 'end': info['end'], 'first_cue': info['first_cue'], 'last_cue': info['last_cue'],
                    'relevant': bool(ctx['chunk_cues'][c] & gold_all), 'intervals_covered': [i for i, iv in enumerate(ivs) if ctx['chunk_cues'][c] & iv]})
    rec = {'question_id': row['question_id'], 'lecture_id': row['lecture_id'], 'group': row['group'], 'subject': qrow['subject'], 'question_category': qrow['question_category'],
           'gold_interval_count': '1' if int(row['n_gold_intervals']) == 1 else '2+', 'n_gold_intervals': int(row['n_gold_intervals']), 'n_gold_intervals_scored': len(ivs),
           'n_gold_cues_retrievable': len(gold_all), 'n_query_terms': n_terms, 'n_chunks': len(ctx['chunks']), 'n_ranked_chunks': len(ranked), 'scoreable': m is not None, 'top10': top}
    if m is not None:
        rec.update(m)
    return rec


def run_m1(cfg, bm25, retrieval, normalize_segments, root=REPO):
    """M1 per-question records for the two evaluated groups (headline first), using experiment code over the frozen primitives."""
    manifest, questions, gold, cues = load_data(cfg, root)
    ec = retrieval.EvidenceConfig()
    bm25.check_frozen_settings(cfg, ec)
    groups = bm25.split_groups(manifest)
    ctxs, records = {}, []
    for gname in bm25.EVALUATED_GROUPS:
        for row in groups[gname]:
            lid = row['lecture_id']
            if lid not in ctxs:
                ctxs[lid] = bm25.build_lecture_context(cues[lid], retrieval, normalize_segments, ec, cfg['text_field']['value'])
            ranked, scores, n_terms = m1_rank(bm25, retrieval, questions[row['question_id']]['question_text'], ctxs[lid])
            records.append(evaluate_ranking(bm25, row, questions[row['question_id']], gold[row['question_id']], ctxs[lid], ranked, scores, n_terms))
    return records, ctxs


# ------------------------------------------------------------------------------------------------ comparison with saved results
def read_csv_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def compare_csv(saved_path, new_path):
    """Exact comparison: header, row count, every cell. Returns a list of human-readable differences (empty = identical)."""
    a, b = read_csv_rows(saved_path), read_csv_rows(new_path)
    diffs = []
    with open(saved_path, encoding='utf-8', newline='') as f1, open(new_path, encoding='utf-8', newline='') as f2:
        ha, hb = next(csv.reader(f1)), next(csv.reader(f2))
    if ha != hb:
        diffs.append(f'header differs: {ha} vs {hb}')
    if len(a) != len(b):
        diffs.append(f'row count differs: saved {len(a)} vs new {len(b)}')
    for n, (ra, rb) in enumerate(zip(a, b), 1):
        label = ra.get('question_id') or f'row {n}'
        for k in ra:
            if ra[k] != rb.get(k):
                diffs.append(f'{label} column {k}: saved {ra[k][:80]!r} vs new {str(rb.get(k))[:80]!r}')
    if a and 'question_id' in a[0] and [r.get('question_id') for r in a] != [r.get('question_id') for r in b]:
        diffs.append('question order/ids differ')
    return diffs


def compare_records_to_saved(records, saved_csv_path):
    """Independent value-level check of the experiment's own records against the saved CSV (before any serialization)."""
    saved = {r['question_id']: r for r in read_csv_rows(saved_csv_path)}
    diffs = []
    if sorted(saved) != sorted(r['question_id'] for r in records):
        diffs.append('question id sets differ')
    for r in records:
        s = saved.get(r['question_id'])
        if s is None:
            continue
        if json.dumps(r['top10'], separators=(',', ':')) != s['top10_json']:
            diffs.append(f'{r["question_id"]}: top-10 list (chunk order, scores, relevance) differs')
        for k in ('interval_recall_at_1', 'interval_recall_at_3', 'interval_recall_at_5', 'hit_rate_at_1', 'hit_rate_at_3', 'hit_rate_at_5',
                  'full_evidence_coverage_at_1', 'full_evidence_coverage_at_3', 'full_evidence_coverage_at_5', 'mrr_at_10'):
            if str(r.get(k, '')) != s[k]:
                diffs.append(f'{r["question_id"]}: {k} saved {s[k]} vs new {r.get(k)}')
        if str(r['n_ranked_chunks']) != s['n_ranked_chunks']:
            diffs.append(f'{r["question_id"]}: n_ranked_chunks differs')
    return diffs


# ------------------------------------------------------------------------------------------------ environment
def determinism_report():
    """Report only (nothing is changed, nothing is encoded). Importing torch is allowed here; no model is loaded."""
    out = {'python': platform.python_version(), 'executable': sys.executable, 'platform': platform.platform(),
           'env': {k: os.environ.get(k) for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
           'sys_flags': {'dont_write_bytecode': sys.dont_write_bytecode, 'hash_randomization': sys.flags.hash_randomization, 'isolated': sys.flags.isolated, 'utf8_mode': sys.flags.utf8_mode},
           'bm25_arm_uses': 'pure Python (random.Random with an explicit seed, sorted() with explicit keys); no numpy/torch in the M1 path'}
    try:
        import torch, numpy
        out['torch'] = {'version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads(),
                        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(), 'default_dtype': str(torch.get_default_dtype()), 'default_device_cpu_only': not torch.cuda.is_available()}
        out['numpy'] = numpy.__version__
    except Exception as e:                                       # reported, not hidden
        out['torch'] = f'not importable here: {type(e).__name__}: {e}'
    out['notes'] = ['Dense encoding (later phase) will run on CPU, float32, fixed batch size; repeat-run ranking identity is the preregistered check.',
                    'torch.use_deterministic_algorithms is NOT enabled in this phase; it matters only for dense inference and will be decided with the dense phase.']
    return out


# ------------------------------------------------------------------------------------------------ main
def m1_validate(out_dir=DEFAULT_M1_OUT, skip_validator=False):
    if os.path.exists(out_dir):
        print(f'REFUSING to overwrite existing directory: {out_dir}', file=sys.stderr)
        return 2
    try:
        prereg, cfg, bm25, pins = verify_all_inputs()
    except (PreregError, RuntimeError) as e:                      # HashMismatchError is a RuntimeError
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    print(f'preregistration hash OK; {pins["prereg_file_pins_checked"]} preregistered files OK; {len(pins["frozen_config_pins"])} frozen-config pins OK')
    if not skip_validator:
        v = subprocess.run([sys.executable, '-B', os.path.join(REPO, 'eval', 'validate_protocol.py')], capture_output=True, text=True, cwd=REPO)
        if v.returncode != 0:
            print('validate_protocol.py FAILED:\n' + v.stdout[-3000:], file=sys.stderr)
            return 4
        print('protocol validator: ' + v.stdout.splitlines()[0])
    sys.path.insert(0, REPO)
    from qa import retrieval
    from qa.transcript import normalize_segments
    records, ctxs = run_m1(cfg, bm25, retrieval, normalize_segments)
    chunk_counts = check_chunk_identity(cfg, ctxs)
    print(f'chunk identity OK: {sum(chunk_counts.values())} chunks in {len(chunk_counts)} lectures')
    diffs = compare_records_to_saved(records, os.path.join(SAVED_DIR, 'per_question_results.csv'))
    print(f'M1 records vs saved per-question results: {len(records)} questions, {len(diffs)} differences')
    # Aggregates (bootstrap included) come from the frozen harness function on the same inputs and are serialized by the frozen writer.
    recs2, agg, ctxs2 = bm25.run_evaluation(cfg, REPO, retrieval, normalize_segments)
    if [json.dumps(r, sort_keys=True, default=str) for r in recs2] != [json.dumps(r, sort_keys=True, default=str) for r in records]:
        diffs.append('experiment M1 records differ from the frozen harness records computed in the same process')
    hash_table = pins['frozen_config_pins']
    cfg_path = os.path.join(REPO, 'eval', 'config_bm25_v1.json')
    bm25.write_outputs(out_dir, records, agg, cfg, hash_table, cfg_path=cfg_path, extra_manifest={'note': 'm1 validation output of exp_dense_hybrid_v1; not a final experiment result'})
    report = {'preregistration_sha256': PREREG_SHA256, 'files': {}}
    for name in COMPARED_FILES:
        saved, new = os.path.join(SAVED_DIR, name), os.path.join(out_dir, name)
        sa, sn = sha256_file(saved), sha256_file(new)
        entry = {'saved_sha256': sa, 'new_sha256': sn, 'byte_identical': sa == sn}
        if name.endswith('.csv'):
            entry['cell_differences'] = compare_csv(saved, new)
        report['files'][name] = entry
        print(f'{name}: byte-identical={sa == sn}')
    report['record_level_differences'] = diffs
    report['chunk_counts_total'] = sum(chunk_counts.values())
    report['questions_compared'] = len(records)
    report['result'] = 'PASS' if (not diffs and all(e['byte_identical'] for e in report['files'].values())) else 'FAIL'
    report['determinism'] = determinism_report()
    with open(os.path.join(out_dir, 'm1_validation_report.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2)
    print('M1 reproduction:', report['result'])
    for d in diffs[:20]:
        print('  DIFF', d, file=sys.stderr)
    return 0 if report['result'] == 'PASS' else 5


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != 'm1-validate':
        print('usage: dense_hybrid_eval.py m1-validate [--out DIR] [--skip-validator]   (dense/hybrid phases are not implemented yet)', file=sys.stderr)
        return 1
    out = os.path.abspath(argv[argv.index('--out') + 1]) if '--out' in argv else DEFAULT_M1_OUT
    return m1_validate(out, skip_validator='--skip-validator' in argv)


if __name__ == '__main__':
    sys.exit(main())
