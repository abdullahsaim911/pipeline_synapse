#!/usr/bin/env python3
"""prod_retrieval_p1 - what evidence does the PRODUCTION /ask pipeline hand to the language model? (frozen-protocol item P1)

    python -B experiments/production_retrieval_v1/production_retrieval_eval.py p1-run --run-id A
    python -B experiments/production_retrieval_v1/production_retrieval_eval.py p1-run --run-id B
    python -B experiments/production_retrieval_v1/production_retrieval_eval.py p1-finalize

Runs the real qa.service.AskService.answer with a recording stand-in for the model (no network, no API key) and scores the passages it would
send. Nothing in qa/ is edited: the evidence plan is captured by wrapping qa.service.select_evidence for the duration of the call.
Definitions are in config_p1.json (registered before the first run).
"""
import collections, csv, hashlib, importlib.util, json, os, platform, statistics, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CONFIG_PATH = os.path.join(HERE, 'config_p1.json')
CONFIG_SHA256 = '719251ff7f48c965b8914530e6b8496a509544af4991c0c17b9facd57616fba1'
RESULTS = os.path.join(HERE, 'results', 'p1_v1')
SAVED_PQ = os.path.join(REPO, 'eval', 'results', 'bm25_v1.0', 'per_question_results.csv')

ANSWERABLE = ('PRIMARY_ANSWERABLE', 'SENSITIVITY_ANSWERABLE')
UNANSWERABLE = ('PRIMARY_UNANSWERABLE', 'SENSITIVITY_UNANSWERABLE')
HEADLINE = ('evidence_hit', 'full_interval_coverage')
EV_METRICS = ('evidence_hit', 'interval_coverage', 'full_interval_coverage', 'strict_containment', 'gold_cue_fraction')


class P1Error(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ pinned inputs
def load_config(path=CONFIG_PATH, expected=CONFIG_SHA256):
    if sha256_file(path) != expected:
        raise P1Error('config_p1.json differs from the registered hash')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def verify_registered_files(p1, root=REPO):
    fi, bad, n = p1['frozen_inputs'], [], 0
    pins = [('frozen config', fi['frozen_bm25_config']['path'], fi['frozen_bm25_config']['sha256'])]
    pins += [('harness', p, h) for p, h in fi['bm25_harness'].items()]
    pins += [('saved results', p, h) for p, h in fi['saved_bm25_per_question_results'].items()]
    pins += [('production code', p, h) for p, h in fi['production_code_at_registration'].items()]
    for name, rel, expected in pins:
        n += 1
        path = os.path.join(root, rel)
        actual = sha256_file(path) if os.path.exists(path) else None
        if actual != expected:
            bad.append(f'  {name} {rel}: expected {expected[:16]}..., got {"MISSING" if actual is None else actual[:16] + "..."}')
    if bad:
        raise P1Error('registered input changed:\n' + '\n'.join(bad))
    return n


def load_frozen_harness(p1, root=REPO):
    rel = 'eval/bm25_eval.py'
    path = os.path.join(root, rel)
    if sha256_file(path) != p1['frozen_inputs']['bm25_harness'][rel]:
        raise P1Error('eval/bm25_eval.py differs from the registered hash; not importing it')
    spec = importlib.util.spec_from_file_location('frozen_bm25_eval', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_inputs(p1=None):
    p1 = p1 or load_config()
    n = verify_registered_files(p1)
    bm25 = load_frozen_harness(p1)
    with open(os.path.join(REPO, p1['frozen_inputs']['frozen_bm25_config']['path']), encoding='utf-8') as f:
        cfg = json.load(f)
    table = bm25.verify_pinned_hashes(cfg, REPO)
    ask_vars = sorted(k for k in os.environ if k.startswith('ASK_'))
    if ask_vars:
        raise P1Error(f'ASK_* environment variables are set ({ask_vars}); production defaults must be measured')
    return p1, cfg, bm25, {'registered_pins': n, 'frozen_config_pins': len(table)}


# ------------------------------------------------------------------------------------------------ the production call
class RecordingProvider:
    """Stands in for the language model: counts calls, returns a fixed well-formed reply, touches no network."""
    name = 'recording-standin'
    model = 'none'

    def __init__(self, result_cls):
        self.calls = 0
        self._result_cls = result_cls

    def generate(self, prompt, settings=None):
        self.calls += 1
        return self._result_cls(text='The instructor says this.', provider=self.name, model=self.model, latency_ms=0, usage=None)


def production_plan(service_mod, provider_result_cls, question, segments):
    """Run AskService.answer unmodified and return (final EvidencePlan or None, outcome, model_calls)."""
    captured = {}
    original = service_mod.select_evidence

    def recording(*args, **kwargs):
        plan = original(*args, **kwargs)
        captured['plan'] = plan
        return plan

    provider = RecordingProvider(provider_result_cls)
    service = service_mod.AskService(provider=provider, config=service_mod.AskConfig.from_env())
    service_mod.select_evidence = recording
    try:
        outcome = service.answer(question=question, segments=segments, current_time=None, moment=None)
    finally:
        service_mod.select_evidence = original
    return captured.get('plan'), outcome, provider.calls


# ------------------------------------------------------------------------------------------------ pure scoring
def evidence_cue_ids(plan, ctx):
    """Cue IDs inside the passages the model would receive (empty when the gate failed)."""
    if plan is None or not plan.ok:
        return set()
    ids = set()
    segs = ctx['segs']
    for b in plan.blocks:
        for i in range(b.first_index, b.last_index + 1):
            ids.add(ctx['start_to_id'][segs[i].start])
    return ids


def evidence_metrics(evidence, intervals):
    """Metrics of an evidence cue set against gold intervals (sets of retrievable cue IDs). None when there is no gold interval."""
    if not intervals:
        return None
    gold = set().union(*intervals)
    covered = sum(1 for iv in intervals if evidence & iv)
    return {'evidence_hit': 1.0 if covered >= 1 else 0.0,
            'interval_coverage': covered / len(intervals),
            'full_interval_coverage': 1.0 if covered == len(intervals) else 0.0,
            'strict_containment': 1.0 if gold <= evidence else 0.0,
            'gold_cue_fraction': len(gold & evidence) / len(gold)}


def evaluate_one(bm25, retrieval, service_mod, result_cls, row, qrow, grec, ctx):
    plan, outcome, calls = production_plan(service_mod, result_cls, qrow['question_text'], ctx['segs'])
    ev = evidence_cue_ids(plan, ctx)
    ivs = bm25.gold_intervals(grec, ctx['dropped'])
    m = evidence_metrics(ev, ivs)
    ranked, _, _ = bm25.rank_question(retrieval, qrow['question_text'], ctx)
    raw = bm25.question_metrics(ranked, ctx['chunk_cues'], ivs, ks=(5, 6)) if ivs else None
    total_chars = sum(len(s.text) + 1 for s in ctx['segs'])
    ev_chars = sum(len(b.text) for b in plan.blocks) if (plan is not None and plan.ok) else 0
    rec = {'question_id': row['question_id'], 'lecture_id': row['lecture_id'], 'group': row['group'], 'subject': qrow['subject'], 'question_category': qrow['question_category'],
           'n_gold_intervals_scored': len(ivs), 'decision': plan.decision if plan is not None else 'no_plan', 'mode': (plan.mode if plan is not None else '') or '',
           'complete': bool(plan.complete) if plan is not None else False, 'model_called': calls > 0, 'gate_failure': 0 if (plan is not None and plan.ok) else 1,
           'n_blocks': len(plan.blocks) if plan is not None else 0, 'evidence_chars': ev_chars, 'transcript_chars': total_chars,
           'evidence_share': round(ev_chars / total_chars, 6) if total_chars else 0.0, 'estimated_prompt_tokens': outcome.estimated_prompt_tokens,
           'outcome_status': outcome.status, 'scoreable': m is not None}
    if m is not None:
        rec.update(m)
        rec['raw_hit_at_5'] = raw['hit_rate_at_5']
        rec['raw_hit_at_6'] = raw['hit_rate_at_6']
    return rec


# ------------------------------------------------------------------------------------------------ run
COLS = ['question_id', 'lecture_id', 'group', 'subject', 'question_category', 'n_gold_intervals_scored', 'decision', 'mode', 'complete', 'model_called', 'gate_failure', 'n_blocks',
        'evidence_chars', 'transcript_chars', 'evidence_share', 'estimated_prompt_tokens', 'outcome_status', 'scoreable'] + list(EV_METRICS) + ['raw_hit_at_5', 'raw_hit_at_6']


def run_all(p1, cfg, bm25):
    sys.path.insert(0, REPO)
    from qa import retrieval, service as service_mod
    from qa.transcript import normalize_segments
    from llm_interface.providers import GenerationResult
    ds = os.path.join(REPO, cfg['dataset']['path'])

    def rd(p):
        with open(p, encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    manifest = rd(os.path.join(REPO, cfg['question_manifest']['path']))
    questions = {r['question_id']: r for r in rd(os.path.join(ds, 'questions.csv'))}
    with open(os.path.join(ds, 'gold_evidence.json'), encoding='utf-8') as f:
        gold = {g['question_id']: g for g in json.load(f)['questions']}
    cues = collections.defaultdict(list)
    with open(os.path.join(ds, 'transcript_segments.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            cues[r['lecture_id']].append(r)
    ec = retrieval.EvidenceConfig()
    bm25.check_frozen_settings(cfg, ec)
    groups = bm25.split_groups(manifest)
    ctxs, records = {}, []
    for gname in ANSWERABLE + UNANSWERABLE:
        for row in groups[gname]:
            lid = row['lecture_id']
            if lid not in ctxs:
                ctxs[lid] = bm25.build_lecture_context(cues[lid], retrieval, normalize_segments, ec, cfg['text_field']['value'])
            records.append(evaluate_one(bm25, retrieval, service_mod, GenerationResult, row, questions[row['question_id']], gold[row['question_id']], ctxs[lid]))
    return records, groups, service_mod.AskConfig.from_env()


def check_raw_consistency(records):
    saved = {r['question_id']: r for r in csv.DictReader(open(SAVED_PQ, encoding='utf-8', newline=''))}
    bad = [r['question_id'] for r in records if r['scoreable'] and r['group'] in ANSWERABLE and str(r['raw_hit_at_5']) != saved[r['question_id']]['hit_rate_at_5']]
    if bad:
        raise P1Error(f'raw BM25 Hit@5 differs from the saved baseline for {len(bad)} question(s): {bad[:5]}')
    return sum(1 for r in records if r['scoreable'] and r['group'] in ANSWERABLE)


def write_records(path, records):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, '') for k in COLS})


# ------------------------------------------------------------------------------------------------ aggregation
def summarize(bm25, recs, metrics, B, seed, level, with_ci=True):
    n = len(recs)
    flag = bm25.flag_n(n)
    vals = {m: [r[m] for r in recs] for m in metrics}
    ci = bm25.bootstrap_ci(vals, B, seed, level) if (with_ci and n and flag != 'VERY_SMALL') else None
    out = {'n': n, 'sample_flag': flag, 'metrics': {}}
    for m in metrics:
        e = {'estimate': (sum(vals[m]) / n) if n else None, 'ci_low': None, 'ci_high': None}
        if ci:
            e['ci_low'], e['ci_high'] = ci[m]
            e['instability_warning'] = flag == 'SMALL'
        out['metrics'][m] = e
    return out


def describe(xs):
    xs = sorted(xs)
    return {'mean': sum(xs) / len(xs), 'median': statistics.median(xs), 'max': xs[-1], 'min': xs[0]} if xs else None


def aggregate(bm25, p1, cfg, records, groups):
    st = cfg['statistics']['confidence_interval']
    B, seed, level = st['resamples'], st['seed'], st['level']
    prim = [r for r in records if r['group'] == 'PRIMARY_ANSWERABLE' and r['scoreable']]
    sens = [r for r in records if r['group'] == 'SENSITIVITY_ANSWERABLE' and r['scoreable']]
    all_m = list(EV_METRICS) + ['gate_failure']
    agg = {'headline_metrics': list(HEADLINE), 'sets': {}}
    agg['sets']['PRIMARY_ANSWERABLE'] = {'role': 'headline', 'overall': summarize(bm25, prim, all_m, B, seed, level)}
    by_lec = collections.defaultdict(list)
    for r in prim:
        by_lec[r['lecture_id']].append(r)
    s1 = cfg['statistics']['secondary_sensitivity_analysis']
    cci = bm25.clustered_bootstrap_ci({lid: {m: [r[m] for r in rs] for m in all_m} for lid, rs in by_lec.items()}, s1['resamples'], s1['seed'], s1['level'])
    ov = agg['sets']['PRIMARY_ANSWERABLE']['overall']['metrics']
    agg['sets']['PRIMARY_ANSWERABLE']['lecture_clustered_secondary'] = {m: {'estimate': ov[m]['estimate'], 'ci_low': cci[m][0], 'ci_high': cci[m][1]} for m in all_m}
    strata = {}
    for dim in ('mode', 'subject', 'question_category'):
        strata[dim] = {lab: summarize(bm25, [r for r in prim if r[dim] == lab], all_m, B, seed, level) for lab in sorted({r[dim] for r in prim})}
    agg['sets']['PRIMARY_ANSWERABLE']['strata_descriptive'] = strata
    agg['sets']['SENSITIVITY_ANSWERABLE'] = {'role': 'descriptive only (n small): point estimates, no intervals', 'overall': summarize(bm25, sens, all_m, B, seed, level, with_ci=False)}
    # paired, descriptive comparison with the raw BM25 ranking
    comp = {}
    for k in (5, 6):
        diffs = {'difference': [r['evidence_hit'] - r[f'raw_hit_at_{k}'] for r in prim]}
        lo, hi = bm25.bootstrap_ci(diffs, B, seed, level)['difference']
        comp[f'evidence_hit_minus_raw_hit_at_{k}'] = {'difference': sum(diffs['difference']) / len(prim), 'ci_low': lo, 'ci_high': hi, 'raw_estimate': sum(r[f'raw_hit_at_{k}'] for r in prim) / len(prim),
                                                      'production_better': sum(1 for d in diffs['difference'] if d > 0), 'raw_better': sum(1 for d in diffs['difference'] if d < 0),
                                                      'role': 'DESCRIPTIVE (no decision rule registered)'}
    agg['sets']['PRIMARY_ANSWERABLE']['paired_vs_raw_ranking'] = comp
    agg['sets']['PRIMARY_ANSWERABLE']['cross_table_raw_hit_at_5_vs_evidence_hit'] = {
        f'raw={int(a)},evidence={int(b)}': sum(1 for r in prim if r['raw_hit_at_5'] == a and r['evidence_hit'] == b) for a in (1.0, 0.0) for b in (1.0, 0.0)}
    agg['sets']['PRIMARY_ANSWERABLE']['failed_questions'] = [{'question_id': r['question_id'], 'decision': r['decision'], 'mode': r['mode'], 'evidence_hit': r['evidence_hit'], 'raw_hit_at_5': r['raw_hit_at_5'],
                                                               'gold_cue_fraction': round(r['gold_cue_fraction'], 3)} for r in prim if r['evidence_hit'] == 0.0]
    agg['sets']['PRIMARY_ANSWERABLE']['not_fully_contained'] = sum(1 for r in prim if r['strict_containment'] == 0.0)
    agg['cost'] = {'estimated_prompt_tokens_when_model_called': describe([r['estimated_prompt_tokens'] for r in prim if r['model_called']]),
                   'evidence_chars_when_model_called': describe([r['evidence_chars'] for r in prim if r['model_called']]),
                   'evidence_share_of_transcript': describe([r['evidence_share'] for r in prim if r['model_called']]),
                   'blocks_per_question': describe([r['n_blocks'] for r in prim if r['model_called']]),
                   'mode_counts_primary': dict(collections.Counter(r['mode'] or r['decision'] for r in prim))}
    un = [r for r in records if r['group'] in UNANSWERABLE]
    agg['unanswerable_gate_behaviour_descriptive'] = {
        'note': 'NOT an abstention evaluation (A1 deferred). Counts only: what the evidence gate did with questions that have no gold answer.',
        'n': len(un), 'by_group': {g: sum(1 for r in un if r['group'] == g) for g in UNANSWERABLE},
        'gate_decision_counts': dict(collections.Counter(r['decision'] for r in un)), 'mode_counts': dict(collections.Counter(r['mode'] or '-' for r in un)),
        'model_would_be_called': sum(1 for r in un if r['model_called']), 'resolved_without_model': sum(1 for r in un if not r['model_called']),
        'complete_transcript_sent': sum(1 for r in un if r['complete'])}
    agg['answerable_gate_decision_counts'] = dict(collections.Counter(r['decision'] for r in records if r['group'] in ANSWERABLE and r['scoreable']))
    return agg


def flat_csv(path, agg):
    rows = []
    def add(set_name, level, dim, label, blk):
        for m, e in blk['metrics'].items():
            rows.append({'set': set_name, 'level': level, 'dimension': dim, 'stratum': label, 'n': blk['n'], 'sample_flag': blk['sample_flag'], 'metric': m, 'estimate': e['estimate'], 'ci_low': e['ci_low'], 'ci_high': e['ci_high']})
    for sname, blk in agg['sets'].items():
        add(sname, 'overall', '', '', blk['overall'])
        for d, labs in blk.get('strata_descriptive', {}).items():
            for lab, b in labs.items():
                add(sname, 'stratum', d, lab, b)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ CLI
def p1_run(run_id):
    out = os.path.join(RESULTS, 'runs', f'run_{run_id}')
    if os.path.exists(out):
        print(f'REFUSING to overwrite {out}', file=sys.stderr)
        return 2
    try:
        p1, cfg, bm25, pins = verify_inputs()
        records, groups, ask_cfg = run_all(p1, cfg, bm25)
        n_checked = check_raw_consistency(records)
    except RuntimeError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    os.makedirs(out)
    write_records(os.path.join(out, 'per_question_results.csv'), records)
    info = {'run_id': run_id, 'config_sha256': CONFIG_SHA256, 'python': platform.python_version(), 'ask_config': {k: getattr(ask_cfg, k) for k in ('max_prompt_tokens', 'full_transcript_max_tokens', 'chars_per_token', 'max_evidence_chars')},
            'pins': pins, 'raw_hit_at_5_checked_against_saved': n_checked, 'questions_run': len(records), 'model_calls_by_stand_in': sum(1 for r in records if r['model_called']),
            'per_question_results_sha256': sha256_file(os.path.join(out, 'per_question_results.csv'))}
    with open(os.path.join(out, 'run_info.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(info, f, indent=2)
    print(f"run {run_id}: {len(records)} questions, raw Hit@5 matches saved baseline for {n_checked}, sha {info['per_question_results_sha256'][:16]}")
    return 0


def p1_finalize():
    da, db = (os.path.join(RESULTS, 'runs', f'run_{r}') for r in 'AB')
    final = os.path.join(RESULTS, 'final')
    if os.path.exists(final):
        print(f'REFUSING to overwrite {final}', file=sys.stderr)
        return 2
    try:
        p1, cfg, bm25, pins = verify_inputs()
    except RuntimeError as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr)
        return 3
    a, b = (sha256_file(os.path.join(d, 'per_question_results.csv')) for d in (da, db))
    os.makedirs(final)
    rep = {'rule': 'byte identity of two independent runs', 'sha256_a': a, 'sha256_b': b, 'result': 'PASS' if a == b else 'FAIL'}
    with open(os.path.join(final, 'repeatability_report.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(rep, f, indent=2)
    print('repeatability:', rep['result'])
    if a != b:
        print('STOP: runs differ; no aggregate written', file=sys.stderr)
        return 5
    records, groups, ask_cfg = run_all(p1, cfg, bm25)
    check_raw_consistency(records)
    write_records(os.path.join(final, 'per_question_results.csv'), records)
    if sha256_file(os.path.join(final, 'per_question_results.csv')) != a:
        raise P1Error('in-process recomputation differs from the run output')
    agg = aggregate(bm25, p1, cfg, records, groups)
    with open(os.path.join(final, 'aggregate_metrics.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(agg, f, indent=2)
    flat_csv(os.path.join(final, 'aggregate_metrics.csv'), agg)
    names = ['per_question_results.csv', 'aggregate_metrics.json', 'aggregate_metrics.csv', 'repeatability_report.json']
    manifest = {'experiment': p1['experiment_id'], 'config_sha256': CONFIG_SHA256, 'pins': pins, 'python': platform.python_version(),
                'ask_config': {k: getattr(ask_cfg, k) for k in ('max_prompt_tokens', 'full_transcript_max_tokens', 'chars_per_token', 'max_evidence_chars')},
                'harness_sha256': sha256_file(os.path.abspath(__file__)), 'results_sha256': {n: sha256_file(os.path.join(final, n)) for n in names},
                'no_model_called': 'the stand-in provider replaced Gemini; no network or API key was used', 'limitations': p1['limitations']}
    with open(os.path.join(final, 'run_manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2)
    ov = agg['sets']['PRIMARY_ANSWERABLE']['overall']
    print(f"PRIMARY_ANSWERABLE n={ov['n']}")
    for m, e in ov['metrics'].items():
        print(f"  {m:24s} {e['estimate']:.4f}  [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['p1-run'] and '--run-id' in argv:
        return p1_run(argv[argv.index('--run-id') + 1])
    if argv[:1] == ['p1-finalize']:
        return p1_finalize()
    print('usage: production_retrieval_eval.py p1-run --run-id A|B | p1-finalize', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
