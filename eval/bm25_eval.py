#!/usr/bin/env python3
"""Deterministic raw BM25 evaluation harness for the FROZEN protocol bm25_v1.0 (eval/config_bm25_v1.json).

    python -B eval/bm25_eval.py [--out DIR] [--config PATH] [--skip-validator]

What it does (and nothing more):
  1. verifies every pinned SHA-256 (code, dataset files, manifest, audit flags, review items/log) and fails on any mismatch,
     BEFORE the qa code is imported;
  2. ranks the chunks of the question's lecture with the production BM25 (qa.retrieval._group / _rank_groups), raw ranking:
     positive scores only, ordered by (-score, chunk index), no gating, padding, merging or reordering;
  3. scores Interval Recall@1/3/5, Hit Rate@1/3/5, Full-evidence coverage@1/3/5 and MRR@10 against gold_evidence.json;
  4. adds the frozen bootstrap intervals (question-level primary, lecture-clustered secondary) and breakdowns;
  5. writes per-question results, aggregates and a run manifest.

Not implemented on purpose (deferred in the protocol): abstention scoring (A1), production-set evaluation (P1). No model/API calls.
Importing this module does not import qa and does nothing else.
"""
import csv, json, math, os, platform, random, statistics, subprocess, sys, hashlib, collections, datetime
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CONFIG = os.path.join(HERE, 'config_bm25_v1.json')
DEFAULT_OUT = os.path.join(HERE, 'results', 'bm25_v1.0')
REVIEW_ITEMS_PATH = 'dataset/review_v1/review_items.csv'

KS = (1, 3, 5)
MRR_DEPTH = 10
METRICS = ([f'interval_recall_at_{k}' for k in KS] + [f'hit_rate_at_{k}' for k in KS] +
           [f'full_evidence_coverage_at_{k}' for k in KS] + [f'mrr_at_{MRR_DEPTH}'])
HEADLINE_GROUP = 'PRIMARY_ANSWERABLE'
SENSITIVITY_GROUP = 'SENSITIVITY_ANSWERABLE'
EVALUATED_GROUPS = (HEADLINE_GROUP, SENSITIVITY_GROUP)


class HashMismatchError(RuntimeError):
    """A pinned input does not match the frozen protocol."""


class ProtocolError(RuntimeError):
    """The config / environment does not match what this harness implements."""


# ----------------------------------------------------------------------------------------- hashes / protocol
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pinned_targets(cfg):
    """(name, relative path, expected sha256) for every hash the frozen config pins."""
    ds = cfg['dataset']
    t = [(f'code:{p}', p, h) for p, h in cfg['code_sha256'].items()]
    t += [(f'dataset:{f}', f"{ds['path']}/{f}", h) for f, h in ds['file_sha256'].items()]
    t += [('audit_flags', ds['audit_flags_file'], ds['audit_flags_sha256']),
          ('review_items', REVIEW_ITEMS_PATH, ds['review_items_sha256']),
          ('review_log_snapshot', ds['review_log'], ds['review_log_sha256_snapshot']),
          ('question_manifest', cfg['question_manifest']['path'], cfg['question_manifest']['sha256'])]
    return t


def verify_pinned_hashes(cfg, root=REPO):
    """Return the verification table; raise HashMismatchError listing EVERY mismatching or missing file."""
    table, bad = [], []
    for name, rel, expected in pinned_targets(cfg):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            actual = None
        else:
            actual = sha256_file(path)
        ok = actual == expected
        table.append({'name': name, 'path': rel, 'expected_sha256': expected, 'actual_sha256': actual, 'ok': ok})
        if not ok:
            bad.append(f"  {name} ({rel}): expected {expected[:16]}..., got {'MISSING' if actual is None else actual[:16] + '...'}")
    if bad:
        raise HashMismatchError('Pinned input hash mismatch - the evaluation will not run:\n' + '\n'.join(bad) +
                                '\nThe frozen protocol bm25_v1.0 pins these files. Restore them, or freeze a new protocol version.')
    return table


def check_frozen_settings(cfg, evidence_config=None):
    """Fail unless the config is the frozen bm25_v1.0 protocol and equals what this harness implements."""
    def need(cond, msg):
        if not cond:
            raise ProtocolError(msg)
    need(cfg.get('status') == 'FROZEN' and cfg.get('freeze', {}).get('protocol_version') == 'bm25_v1.0', 'config is not the FROZEN bm25_v1.0 protocol')
    m, r, s = cfg['metrics'], cfg['retrieval'], cfg['statistics']
    need(m['recall_at_k'] == list(KS) and m['hit_rate_at_k'] == list(KS) and m['full_evidence_coverage_at_k'] == list(KS), 'metric k values differ from the harness (1, 3, 5)')
    need(m['mrr_depth'] == MRR_DEPTH, 'MRR depth differs from the harness (10)')
    need(r['mode'] == 'RAW_BM25_RANKING' and r['full_transcript_chars'] == 0, 'retrieval mode must be RAW_BM25_RANKING with full_transcript_chars=0')
    need(cfg['text_field']['value'] == 'text', 'text field must be "text"')
    ci, s1 = s['confidence_interval'], s['secondary_sensitivity_analysis']
    need(ci['level'] == 0.95 and ci['resamples'] == s1['resamples'] and ci['seed'] == s1['seed'], 'bootstrap settings inconsistent')
    if evidence_config is not None:
        need(cfg['chunking']['chunk_seconds'] == evidence_config.chunk_seconds and cfg['chunking']['chunk_max_chars'] == evidence_config.chunk_max_chars,
             'chunking differs from qa.retrieval.EvidenceConfig defaults')
        need(evidence_config.full_transcript_chars == 0, 'EvidenceConfig default full_transcript_chars is not 0')


# ----------------------------------------------------------------------------------------- ranking (pure)
def rank_from_scores(scores):
    """Raw ranking: chunks with a POSITIVE score only, ordered by (-score, chunk index). Zero-score chunks are never ranked."""
    return [i for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)) if scores[i] > 0]


def flag_n(n):
    return 'VERY_SMALL' if n < 10 else ('SMALL' if n < 30 else 'OK')


# ----------------------------------------------------------------------------------------- metrics (pure)
def question_metrics(ranked, chunk_cues, intervals, ks=KS, depth=MRR_DEPTH):
    """Frozen per-question metrics.

    ranked      chunk indices of the raw ranked list (positive scores only)
    chunk_cues  list (per chunk) of sets of cue IDs
    intervals   list of non-empty sets: the distinct gold intervals' retrievable cue IDs
    A chunk COVERS an interval when their cue sets share at least one cue. Returns None when there is no gold interval.
    """
    if not intervals:
        return None
    gold_all = set().union(*intervals)
    out = {}
    for k in ks:
        top = ranked[:k]                      # fewer than k positive chunks -> use all of them, never pad
        covered = sum(1 for iv in intervals if any(chunk_cues[c] & iv for c in top))
        out[f'interval_recall_at_{k}'] = covered / len(intervals)
        out[f'hit_rate_at_{k}'] = 1.0 if covered >= 1 else 0.0
        out[f'full_evidence_coverage_at_{k}'] = 1.0 if covered == len(intervals) else 0.0
    first_any = next((r for r, c in enumerate(ranked, 1) if chunk_cues[c] & gold_all), None)
    out[f'mrr_at_{depth}'] = (1.0 / first_any) if (first_any is not None and first_any <= depth) else 0.0
    out['first_relevant_rank'] = first_any
    return out


# ----------------------------------------------------------------------------------------- bootstrap (pure)
def percentile_ranks(resamples, level):
    """1-based nearest-rank positions of the lower / upper limits (250th and 9750th for B=10000, level 0.95)."""
    alpha = (1 - Fraction(str(level))) / 2
    return math.ceil(alpha * resamples), math.ceil((1 - alpha) * resamples)


def bootstrap_ci(values, resamples, seed, level=0.95):
    """Question-level percentile bootstrap. values: {metric: [per-question value]} (same order, same length).

    Fresh random.Random(seed); each replicate draws n indices with rng.randrange(n) called n times in order; all metrics share the replicates.
    """
    metrics = list(values)
    n = len(values[metrics[0]])
    rng = random.Random(seed)
    reps = {m: [] for m in metrics}
    getters = {m: values[m].__getitem__ for m in metrics}
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        for m in metrics:
            reps[m].append(sum(map(getters[m], idx)) / n)
    lo, hi = percentile_ranks(resamples, level)
    out = {}
    for m in metrics:
        s = sorted(reps[m])
        out[m] = (s[lo - 1], s[hi - 1])
    return out


def clustered_bootstrap_ci(cluster_values, resamples, seed, level=0.95):
    """Lecture-clustered percentile bootstrap (secondary). cluster_values: {cluster: {metric: [per-question values]}}.

    Clusters are taken in sorted order; each replicate draws G clusters with rng.randrange(G) called G times; the replicate value is the
    POOLED mean over all questions of the drawn clusters (a cluster drawn twice counts twice).
    """
    names = sorted(cluster_values)
    metrics = list(cluster_values[names[0]])
    sums = [{m: sum(cluster_values[c][m]) for m in metrics} for c in names]
    counts = [len(cluster_values[c][metrics[0]]) for c in names]
    G = len(names)
    rng = random.Random(seed)
    reps = {m: [] for m in metrics}
    for _ in range(resamples):
        idx = [rng.randrange(G) for _ in range(G)]
        total = sum(counts[i] for i in idx)
        for m in metrics:
            reps[m].append(sum(sums[i][m] for i in idx) / total)
    lo, hi = percentile_ranks(resamples, level)
    return {m: (sorted(reps[m])[lo - 1], sorted(reps[m])[hi - 1]) for m in metrics}


# ----------------------------------------------------------------------------------------- data preparation
def split_groups(manifest_rows):
    """Questions per manifest group. Only the two ANSWERABLE groups are ever scored; nothing is mixed."""
    g = collections.defaultdict(list)
    for r in manifest_rows:
        g[r['group']].append(r)
    return g


def build_lecture_context(cue_rows, retrieval, normalize_segments, evidence_config, text_field='text'):
    """Segments, chunks and cue-ID sets of one lecture.

    cue_rows: dataset rows {segment_id, start_seconds, end_seconds, text, ...}. Cue IDs are obtained by the segment START TIME
    (verified unique), never by array position, because normalize_segments drops cues whose cleaned text is empty.
    """
    starts = [c['start_seconds'] for c in cue_rows]
    if len(set(starts)) != len(starts):
        raise ProtocolError('cue start times are not unique in a lecture; start-time mapping is unsafe')
    by_start = {c['start_seconds']: c for c in cue_rows}
    segs = normalize_segments([{'start': c['start_seconds'], 'end': c['end_seconds'], 'text': c[text_field]} for c in cue_rows])
    start_to_id = {}
    for s in segs:
        if s.start not in by_start:
            raise ProtocolError(f'segment start {s.start} has no matching cue')
        start_to_id[s.start] = by_start[s.start]['segment_id']
    retained = set(start_to_id.values())
    dropped = {c['segment_id'] for c in cue_rows} - retained
    chunks = retrieval._group(range(len(segs)), segs, evidence_config)
    chunk_cues = [{start_to_id[segs[i].start] for i in range(a, b + 1)} for a, b in chunks]
    chunk_info = [{'first_cue': start_to_id[segs[a].start], 'last_cue': start_to_id[segs[b].start], 'start': segs[a].start, 'end': segs[b].end, 'n_cues': b - a + 1} for a, b in chunks]
    return {'segs': segs, 'chunks': chunks, 'chunk_cues': chunk_cues, 'chunk_info': chunk_info, 'dropped': dropped, 'start_to_id': start_to_id}


def rank_question(retrieval, question_text, ctx):
    """Raw BM25 ranking for one question in one lecture. Returns (ranked chunk indices, per-chunk scores, n_terms)."""
    terms = retrieval.content_terms(retrieval._TIME_TEXT_RE.sub(' ', question_text))
    groups = [retrieval.expand_term(t) for t in terms]
    if not groups:
        return [], [0.0] * len(ctx['chunks']), 0
    ranking = retrieval._rank_groups(groups, ctx['chunks'], ctx['segs'])
    return rank_from_scores(ranking.scores), list(ranking.scores), len(terms)


def gold_intervals(gold_record, dropped):
    """Distinct gold intervals as sets of retrievable cue IDs; intervals whose cues were all dropped are removed."""
    ivs = [set(iv['segment_ids']) - dropped for iv in gold_record['gold_intervals']]
    return [iv for iv in ivs if iv]


def evaluate_question(retrieval, row, qrow, gold_record, ctx):
    ranked, scores, n_terms = rank_question(retrieval, qrow['question_text'], ctx)
    ivs = gold_intervals(gold_record, ctx['dropped'])
    gold_all = set().union(*ivs) if ivs else set()
    m = question_metrics(ranked, ctx['chunk_cues'], ivs)
    top = []
    for rank, c in enumerate(ranked[:MRR_DEPTH], 1):
        info = ctx['chunk_info'][c]
        top.append({'rank': rank, 'chunk_index': c, 'score': scores[c], 'start': info['start'], 'end': info['end'], 'first_cue': info['first_cue'], 'last_cue': info['last_cue'],
                    'relevant': bool(ctx['chunk_cues'][c] & gold_all), 'intervals_covered': [i for i, iv in enumerate(ivs) if ctx['chunk_cues'][c] & iv]})
    rec = {'question_id': row['question_id'], 'lecture_id': row['lecture_id'], 'group': row['group'], 'subject': qrow['subject'], 'question_category': qrow['question_category'],
           'gold_interval_count': '1' if int(row['n_gold_intervals']) == 1 else '2+', 'n_gold_intervals': int(row['n_gold_intervals']), 'n_gold_intervals_scored': len(ivs),
           'n_gold_cues_retrievable': len(gold_all), 'n_query_terms': n_terms, 'n_chunks': len(ctx['chunks']), 'n_ranked_chunks': len(ranked),
           'scoreable': m is not None, 'top10': top}
    if m is not None:
        rec.update(m)
    return rec


# ----------------------------------------------------------------------------------------- aggregation
def mean(xs):
    return sum(xs) / len(xs)


def summarize(records, resamples, seed, level, with_ci):
    """Point estimates (+ bootstrap interval when allowed) for the records of one stratum."""
    n = len(records)
    flag = flag_n(n)
    out = {'n': n, 'sample_flag': flag, 'metrics': {}}
    values = {m: [r[m] for r in records] for m in METRICS}
    ci = bootstrap_ci(values, resamples, seed, level) if (with_ci and flag != 'VERY_SMALL' and n > 0) else None
    for m in METRICS:
        e = {'estimate': mean(values[m]) if n else None}
        if ci is not None:
            e.update({'ci_low': ci[m][0], 'ci_high': ci[m][1], 'ci_method': 'question-level percentile bootstrap', 'instability_warning': flag == 'SMALL'})
        else:
            e.update({'ci_low': None, 'ci_high': None, 'ci_method': None, 'instability_warning': None})
        out['metrics'][m] = e
    if ci is None:
        out['ci_note'] = 'no confidence interval (VERY_SMALL stratum or interval not applicable); point estimate and n only'
    elif flag == 'SMALL':
        out['ci_note'] = 'SMALL sample (10 <= n < 30): interval reported with an instability warning'
    return out


def run_evaluation(cfg, root=REPO, retrieval=None, normalize_segments=None):
    """Scores the two answerable groups. Returns (per-question records, aggregates)."""
    ds = os.path.join(root, cfg['dataset']['path'])
    rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8', newline='')))
    manifest = rd(os.path.join(root, cfg['question_manifest']['path']))
    questions = {r['question_id']: r for r in rd(os.path.join(ds, 'questions.csv'))}
    gold = {g['question_id']: g for g in json.load(open(os.path.join(ds, 'gold_evidence.json'), encoding='utf-8'))['questions']}
    cues = collections.defaultdict(list)
    for line in open(os.path.join(ds, 'transcript_segments.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        cues[r['lecture_id']].append(r)
    ec = retrieval.EvidenceConfig()
    check_frozen_settings(cfg, ec)
    groups = split_groups(manifest)
    ctxs, records = {}, []
    for gname in EVALUATED_GROUPS:                       # unanswerable / excluded questions are never touched
        for row in groups[gname]:
            lid = row['lecture_id']
            if lid not in ctxs:
                ctxs[lid] = build_lecture_context(cues[lid], retrieval, normalize_segments, ec, cfg['text_field']['value'])
            records.append(evaluate_question(retrieval, row, questions[row['question_id']], gold[row['question_id']], ctxs[lid]))
    st = cfg['statistics']
    B, seed, level = st['confidence_interval']['resamples'], st['confidence_interval']['seed'], st['confidence_interval']['level']
    agg = {'counts': cfg['question_manifest']['counts'], 'sets': {}, 'not_evaluated': {k: len(v) for k, v in groups.items() if k not in EVALUATED_GROUPS}}
    unscoreable = [r['question_id'] for r in records if not r['scoreable']]
    agg['unscoreable_question_ids'] = unscoreable
    for gname in EVALUATED_GROUPS:
        recs = [r for r in records if r['group'] == gname and r['scoreable']]
        is_head = gname == HEADLINE_GROUP
        block = {'role': 'headline' if is_head else 'sensitivity (reported separately, never merged)', 'overall': summarize(recs, B, seed, level, with_ci=is_head)}
        if is_head:
            dims = list(st['breakdowns']['dimensions'])
            br = {}
            for d in dims:
                labels = sorted({r[d] for r in recs})
                br[d] = {lab: summarize([r for r in recs if r[d] == lab], B, seed, level, with_ci=True) for lab in labels}
            block['breakdowns'] = br
            # the clustered bootstrap runs on the headline set, overall, headline metrics only
            by_lec = collections.defaultdict(list)
            for r in recs:
                by_lec[r['lecture_id']].append(r)
            s1 = st['secondary_sensitivity_analysis']
            cci = clustered_bootstrap_ci({lid: {m: [r[m] for r in rs] for m in METRICS} for lid, rs in by_lec.items()}, s1['resamples'], s1['seed'], s1['level'])
            block['secondary_lecture_clustered'] = {'role': 'SECONDARY sensitivity analysis; the question-level interval is primary', 'clusters': len(by_lec),
                                                    'questions_per_cluster_min_median_max': [min(len(v) for v in by_lec.values()), statistics.median(len(v) for v in by_lec.values()), max(len(v) for v in by_lec.values())],
                                                    'metrics': {m: {'estimate': block['overall']['metrics'][m]['estimate'], 'ci_low': cci[m][0], 'ci_high': cci[m][1], 'ci_method': 'lecture-clustered percentile bootstrap'} for m in METRICS}}
            derived = st['breakdowns']['derived_sample_sizes_PRIMARY_ANSWERABLE']
            for d in dims:
                got = {lab: {'n': v['n'], 'sample_flag': v['sample_flag']} for lab, v in br[d].items()}
                if got != derived[d]:
                    raise ProtocolError(f'breakdown {d} strata differ from the frozen config: {got} vs {derived[d]}')
        agg['sets'][gname] = block
    return records, agg, ctxs


# ----------------------------------------------------------------------------------------- outputs
def write_outputs(out_dir, records, agg, cfg, hash_table, extra_manifest, cfg_path=DEFAULT_CONFIG):
    os.makedirs(out_dir)
    cols = ['question_id', 'lecture_id', 'group', 'subject', 'question_category', 'gold_interval_count', 'n_gold_intervals', 'n_gold_intervals_scored', 'n_gold_cues_retrievable',
            'n_query_terms', 'n_chunks', 'n_ranked_chunks', 'scoreable'] + METRICS + ['first_relevant_rank', 'top10_json']
    with open(os.path.join(out_dir, 'per_question_results.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in records:
            row = {k: r.get(k, '') for k in cols if k != 'top10_json'}
            row['first_relevant_rank'] = '' if r.get('first_relevant_rank') is None else r['first_relevant_rank']
            row['top10_json'] = json.dumps(r['top10'], separators=(',', ':'))
            w.writerow(row)
    json.dump(agg, open(os.path.join(out_dir, 'aggregate_metrics.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
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
            s = blk['secondary_lecture_clustered']
            add(sname, 'overall_lecture_clustered', '', '', {'n': blk['overall']['n'], 'sample_flag': blk['overall']['sample_flag'], 'metrics': s['metrics']})
    with open(os.path.join(out_dir, 'aggregate_metrics.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(flat)
    results_hashes = {n: sha256_file(os.path.join(out_dir, n)) for n in ('per_question_results.csv', 'aggregate_metrics.json', 'aggregate_metrics.csv')}
    manifest = {'protocol_id': cfg['protocol_id'], 'protocol_version': cfg['freeze']['protocol_version'], 'protocol_frozen_date': cfg['freeze']['frozen_date'], 'protocol_status': cfg['status'],
                'config_path': os.path.relpath(cfg_path, REPO).replace(os.sep, '/'), 'config_sha256': sha256_file(cfg_path),
                'hash_verification': {'all_ok': all(h['ok'] for h in hash_table), 'checked': hash_table},
                'settings': {'ks': list(KS), 'mrr_depth': MRR_DEPTH, 'metrics': METRICS, 'chunk_seconds': cfg['chunking']['chunk_seconds'], 'chunk_max_chars': cfg['chunking']['chunk_max_chars'],
                             'ranking': 'raw BM25 (qa.retrieval._rank_groups), positive scores only, ordered by (-score, chunk index)', 'full_transcript_chars': 0, 'text_field': 'text',
                             'relevance': cfg['relevance']['rule'], 'bootstrap': {'primary': cfg['statistics']['confidence_interval'], 'secondary': cfg['statistics']['secondary_sensitivity_analysis']['name'],
                                                                                    'resamples': cfg['statistics']['confidence_interval']['resamples'], 'seed': cfg['statistics']['confidence_interval']['seed']}},
                'harness_files_sha256': {p: sha256_file(os.path.join(HERE, p)) for p in ('bm25_eval.py', 'tests/test_bm25_eval.py') if os.path.exists(os.path.join(HERE, p))},
                'results_sha256': results_hashes, 'python': platform.python_version(), 'platform': platform.platform(),
                'groups': {'evaluated': list(EVALUATED_GROUPS), 'counts_in_manifest': agg['counts'], 'not_evaluated': agg['not_evaluated'],
                           'not_evaluated_reason': 'unanswerable questions need the deferred abstention evaluation (A1); EXCLUDED questions are visual/uncertain/pending; production-set evaluation (P1) deferred'},
                'limitations': [cfg['verification_statement']['summary'], cfg['verification_statement']['required_wording'],
                                'Raw BM25 ranking describes retrieval quality only; it is not the production evidence set (gated, padded, merged) and says nothing about abstention.',
                                'Six short lectures (' + ', '.join(cfg['abstention']['facts_from_code']['protocol_pin_full_transcript_chars_0_differs_from_production_for_lectures']) + ') are sent whole in production; here they are ranked like the others.',
                                'Interval Recall@k and Full-evidence coverage@k count an interval as covered when ONE of its cues is in a top-k chunk; they are not complete-evidence recall.',
                                'Question-level intervals treat questions as independent; the lecture-clustered interval is the secondary check.',
                                'qa/ is untracked in Git; code hashes are the only version pin.'],
                'deviations_from_frozen_protocol': [], 'not_implemented_deferred': ['A1 abstention scoring', 'P1 production-set evaluation']}
    manifest.update(extra_manifest)
    json.dump(manifest, open(os.path.join(out_dir, 'run_manifest.json'), 'w', encoding='utf-8', newline='\n'), indent=2)
    return results_hashes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = DEFAULT_OUT
    cfg_path = DEFAULT_CONFIG
    skip_validator = '--skip-validator' in argv
    if '--out' in argv:
        out_dir = os.path.abspath(argv[argv.index('--out') + 1])
    if '--config' in argv:
        cfg_path = os.path.abspath(argv[argv.index('--config') + 1])
    if os.path.exists(out_dir):
        print(f'REFUSING to overwrite existing results directory: {out_dir}', file=sys.stderr)
        return 2
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    try:
        table = verify_pinned_hashes(cfg, REPO)                      # BEFORE importing qa or reading any data
    except HashMismatchError as e:
        print(str(e), file=sys.stderr)
        return 3
    print(f'pinned hashes verified: {len(table)} files OK')
    if not skip_validator:
        v = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'validate_protocol.py')], capture_output=True, text=True)
        if v.returncode != 0:
            print('validate_protocol.py FAILED - the evaluation will not run:\n' + v.stdout[-3000:], file=sys.stderr)
            return 4
        print('protocol validator: ' + v.stdout.splitlines()[0])
    sys.path.insert(0, REPO)
    from qa import retrieval
    from qa.transcript import normalize_segments
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    records, agg, ctxs = run_evaluation(cfg, REPO, retrieval, normalize_segments)
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    write_outputs(out_dir, records, agg, cfg, table, cfg_path=cfg_path, extra_manifest={'run_started_utc': started, 'run_finished_utc': finished, 'cues_dropped_by_preprocessing': sorted(set().union(*[c['dropped'] for c in ctxs.values()]))})
    head = agg['sets'][HEADLINE_GROUP]['overall']
    print(f"\nPRIMARY_ANSWERABLE n={head['n']}")
    for m in METRICS:
        e = head['metrics'][m]
        print(f"  {m:32s} {e['estimate']:.4f}  95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
    print('results written to', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
