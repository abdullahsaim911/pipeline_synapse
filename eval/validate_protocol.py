#!/usr/bin/env python3
"""Structural validation of eval/config_bm25_v1.json and eval/question_manifest_bm25_v1.csv.

STRUCTURE ONLY: this is not the evaluation harness. It does not rank, score or call any model. It checks that the
protocol file is well formed, that every pinned value and hash matches the real files, and that the manifest agrees
with the reviewed dataset, the audit flags and the review log. Run from anywhere:  python eval/validate_protocol.py
"""
import csv, json, os, sys, hashlib, collections, dataclasses
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.chdir(REPO); sys.path.insert(0, REPO)
res = []
def chk(c, name, detail=''): res.append(('PASS' if c else 'FAIL', name, detail))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8', newline='')))

cfg = json.load(open('eval/config_bm25_v1.json', encoding='utf-8'))
chk(True, 'config parses')
for k in ['protocol_id', 'status', 'dataset', 'question_manifest', 'text_field', 'preprocessing', 'cue_id_mapping', 'chunking', 'retrieval', 'relevance', 'metrics', 'abstention_note', 'code_sha256', 'unresolved_protocol_questions', 'not_done']:
    chk(k in cfg, f'config key present: {k}')
# ---- pinned values requested in the protocol
from qa import retrieval as R
d = dataclasses.asdict(R.EvidenceConfig())
chk(cfg['metrics']['recall_at_k'] == [1, 3, 5], 'Recall@k = [1, 3, 5]')
chk(cfg['metrics']['mrr_depth'] == 10, 'MRR depth = 10')
chk(cfg['retrieval']['full_transcript_chars'] == 0 and d['full_transcript_chars'] == 0, 'full_transcript_chars pinned 0 and equals EvidenceConfig default')
chk(cfg['chunking']['chunk_seconds'] == d['chunk_seconds'] == 40.0 and cfg['chunking']['chunk_max_chars'] == d['chunk_max_chars'] == 900, 'chunking 40.0 s / 900 chars equals the actual EvidenceConfig defaults', f"code: {d['chunk_seconds']} / {d['chunk_max_chars']}")
chk(cfg['retrieval']['mode'] == 'RAW_BM25_RANKING', 'ranking mode = RAW_BM25_RANKING')
chk('at least one gold-evidence cue ID' in cfg['relevance']['rule'], 'relevance rule = chunk contains >=1 gold cue ID')
chk(cfg['retrieval']['evidence_config_defaults_recorded'] == d, 'recorded EvidenceConfig defaults equal the code')
chk('SEPARATE abstention evaluation' in cfg['abstention_note'], 'abstention note present')
chk(cfg['text_field']['value'] in ('text', 'text_normalized'), 'text_field is a real dataset field', cfg['text_field']['value'])
chk(R._rank_groups.__name__ == '_rank_groups' and R._group.__name__ == '_group' and hasattr(R, '_TIME_TEXT_RE'), 'private functions named in the protocol exist')
# ---- hashes
for f, h in cfg['code_sha256'].items(): chk(sha(f) == h, f'code hash matches: {f}')
chk(set(cfg['code_sha256']) >= {'qa/retrieval.py', 'qa/transcript.py'}, 'retrieval and transcript preprocessing files are both pinned')
for f, h in cfg['dataset']['file_sha256'].items(): chk(sha(f"{cfg['dataset']['path']}/{f}") == h, f'dataset hash matches: {f}')
chk(sha(cfg['dataset']['audit_flags_file']) == cfg['dataset']['audit_flags_sha256'], 'audit flags hash matches')
chk(sha('dataset/review_v1/review_items.csv') == cfg['dataset']['review_items_sha256'], 'review_items.csv hash matches')
chk(sha(cfg['dataset']['review_log']) == cfg['dataset']['review_log_sha256_snapshot'], 'review log unchanged since manifest was derived (else regenerate the manifest)')
chk(sha(cfg['question_manifest']['path']) == cfg['question_manifest']['sha256'], 'question manifest hash matches')
# ---- manifest vs dataset
DS = cfg['dataset']['path']
Q = {r['question_id']: r for r in rd(f'{DS}/questions.csv')}
M = rd(cfg['question_manifest']['path']); MI = {r['question_id']: r for r in M}
G = {g['question_id']: g for g in json.load(open(f'{DS}/gold_evidence.json', encoding='utf-8'))['questions']}
chk(len(M) == len(MI) == len(Q) and set(MI) == set(Q), 'manifest has every question exactly once', f'{len(M)} rows')
counts = collections.Counter(r['group'] for r in M)
chk(dict(counts) == cfg['question_manifest']['counts'], 'group counts equal config counts', str(dict(sorted(counts.items()))))
chk(sum(counts.values()) == cfg['question_manifest']['total_questions'] == 250, 'total = 250')
chk(counts['PRIMARY_ANSWERABLE'] + counts['PRIMARY_UNANSWERABLE'] == cfg['question_manifest']['primary_total'], 'primary total consistent', str(cfg['question_manifest']['primary_total']))
chk(counts['SENSITIVITY_ANSWERABLE'] + counts['SENSITIVITY_UNANSWERABLE'] == cfg['question_manifest']['sensitivity_total'], 'sensitivity total consistent', str(cfg['question_manifest']['sensitivity_total']))
trk = collections.Counter(q['eval_track'] for q in Q.values())
chk(counts['EXCLUDED'] == sum(v for k, v in trk.items() if not k.startswith('A_transcript')), 'EXCLUDED = every question not in the two scored tracks')
chk(counts['PRIMARY_ANSWERABLE'] + counts['SENSITIVITY_ANSWERABLE'] == trk['A_transcript_qa'] and counts['PRIMARY_UNANSWERABLE'] + counts['SENSITIVITY_UNANSWERABLE'] == trk['A_transcript_qa_abstention'], 'scored candidates = A_transcript_qa + A_transcript_qa_abstention', f"{trk['A_transcript_qa']} + {trk['A_transcript_qa_abstention']}")
bad = [r['question_id'] for r in M if r['eval_track'] != Q[r['question_id']]['eval_track'] or r['lecture_id'] != Q[r['question_id']]['lecture_id'] or r['answerable'] != Q[r['question_id']]['answerable']]
chk(not bad, 'manifest eval_track / lecture_id / answerable agree with questions.csv', str(bad[:5]))
chk(all((r['group'].startswith('PRIMARY')) == (r['scored_in_primary'] == 'true') for r in M), 'scored_in_primary flag matches group')
chk(all(r['group'] == 'EXCLUDED' for r in M if r['eval_track'] in ('B_visually_dependent', 'A_or_B_pending_review') or r['eval_track'].startswith('X_')), 'visual/uncertain/X tracks are all EXCLUDED')
chk(MI['CHEM02_Q01']['group'] == 'EXCLUDED' and MI['CHEM02_Q01']['scored_in_primary'] == 'false', 'CHEM02_Q01 is excluded from scoring')
chk(all(int(r['n_gold_cues_retrievable']) >= 1 for r in M if r['group'] in ('PRIMARY_ANSWERABLE', 'SENSITIVITY_ANSWERABLE')), 'every answerable scored/sensitivity question has >=1 retrievable gold cue')
chk(all(int(r['n_gold_cues']) == 0 for r in M if r['group'].endswith('UNANSWERABLE')), 'unanswerable questions have no gold cues')
chk(all(int(r['n_gold_cues']) == sum(len(iv['segment_ids']) for iv in G[r['question_id']]['gold_intervals']) and int(r['n_gold_intervals']) == len(G[r['question_id']]['gold_intervals']) for r in M), 'gold cue/interval counts match gold_evidence.json')
chk(all(s.split('_S')[0] == r['lecture_id'] for r in M for iv in G[r['question_id']]['gold_intervals'] for s in iv['segment_ids']), 'all gold cue IDs belong to the question lecture')
# ---- independent re-derivation of the open/closed rule from the review log
items = {r['item_id']: r for r in rd('dataset/review_v1/review_items.csv')}; log = rd(cfg['dataset']['review_log'])
def state(i):
    t = next(r for r in log if r['item_id'] == i and r['check_type'] == 'transcript'); v = next(r for r in log if r['item_id'] == i and r['check_type'] == 'video')
    if all(x['reviewer'] and x['review_date'] for x in (t, v)) and v['evidence_supported'] == 'yes' and v['proposed_action'] in ('keep_as_is', 'keep_unanswerable'): return 'closed_keep'
    return 'open_logged' if (t['reviewer'] or v['reviewer']) else 'open_unlogged'
exp = {}
for i, it in items.items(): exp[it['question_id']] = (i, state(i))
mism = [q for q, r in MI.items() if (exp.get(q, ('', 'no_review_item'))[1] != r['review_state'] or exp.get(q, ('', ''))[0] != r['review_item_id'])]
chk(not mism, 'review_state / review_item_id re-derived from the log match the manifest', str(mism[:5]))
chk(all((r['group'].startswith('PRIMARY')) == (r['review_state'] in ('closed_keep', 'no_review_item')) for r in M if r['eval_track'].startswith('A_transcript')), 'scored candidates are PRIMARY iff review state is closed or no review item')
# ---- cue mapping facts re-checked (still no ranking)
from qa.transcript import normalize_segments, clean_text
cues = collections.defaultdict(list)
for l in open(f'{DS}/transcript_segments.jsonl', encoding='utf-8'):
    r = json.loads(l); cues[r['lecture_id']].append(r)
tf = cfg['text_field']['value']; dropped = []; ok_unique = ok_map = ok_txt = ok_part = True
for lid, cs in cues.items():
    starts = [c['start_seconds'] for c in cs]; ok_unique &= len(set(starts)) == len(starts)
    segs = normalize_segments([{'start': c['start_seconds'], 'end': c['end_seconds'], 'text': c[tf]} for c in cs]); by = {c['start_seconds']: c for c in cs}
    kept = set()
    for s in segs:
        c = by.get(s.start); ok_map &= c is not None
        if c: kept.add(c['segment_id']); ok_txt &= s.text == clean_text(c[tf])
    dropped += [c['segment_id'] for c in cs if c['segment_id'] not in kept]
    ch = R._group(range(len(segs)), segs, R.EvidenceConfig()); ok_part &= [i for a, b in ch for i in range(a, b + 1)] == list(range(len(segs)))
v = cfg['cue_id_mapping']['verification_performed_at_creation']
chk(ok_unique and ok_map and ok_txt and ok_part, 'cue mapping: unique starts, every segment maps to a cue, text equals clean_text, chunks partition segments')
chk(sorted(dropped) == sorted(x['cue_id'] for x in v['cues_dropped_with_text_field']), 'dropped-cue list matches recomputation', str(dropped))
chk(not any(r['gold_cues_dropped_by_preprocessing'] for r in M if r['group'] != 'EXCLUDED'), 'no scored/sensitivity gold cue is dropped by preprocessing')
chk(sum(cfg['chunking']['chunk_counts_per_lecture'].values()) == sum(len(R._group(range(len(s)), s, R.EvidenceConfig())) for s in [normalize_segments([{'start': c['start_seconds'], 'end': c['end_seconds'], 'text': c[tf]} for c in cs]) for cs in cues.values()]), 'chunk counts per lecture match the code')
chk(cfg['unresolved_protocol_questions'] and all({'id', 'question', 'proposed'} <= set(u) for u in cfg['unresolved_protocol_questions']), 'unresolved protocol questions are listed with proposals', f"{len(cfg['unresolved_protocol_questions'])} open")
chk(set(os.listdir('eval')) <= {'config_bm25_v1.json', 'question_manifest_bm25_v1.csv', 'validate_protocol.py', 'bm25_eval.py', 'tests', 'results', '__pycache__'}, 'eval/ contains only the frozen config, manifest, this validator, the harness (bm25_eval.py), tests/ and results/', str(sorted(os.listdir('eval'))))
# =====================================================================================================
# U1-U7 decisions (added when the protocol resolved its open questions)
# =====================================================================================================
import math
dec = {x['id'] for x in cfg.get('protocol_decisions', [])}
chk(dec == {'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'S1', 'S2', 'F1'}, 'protocol_decisions records U1-U7, S1, S2 and F1')
chk({u['id'] for u in cfg['unresolved_protocol_questions']} == {'A1', 'P1'}, 'remaining open items are exactly A1 and P1')
chk(all(u.get('status') == 'DEFERRED' and u.get('blocks_raw_bm25_baseline') is False for u in cfg['unresolved_protocol_questions']), 'A1 and P1 are explicitly DEFERRED and do not block the raw BM25 baseline')
# --- retained retrieval settings
rt = cfg['retrieval']
chk('lower chunk index' in rt['tie_break'] and 'never ranked' in rt['zero_score_chunks'] and rt['ranked_list'].startswith('all chunks of the lecture with score > 0'), 'retained: positive scores only, stable descending order, earlier-chunk tie-break')
chk(all(g in ' '.join(rt['gates_not_applied']) for g in ('min_coverage', 'min_relative_score', 'top_k', 'context_padding', 'merging', 'chronological')), 'retained: no production gating, padding, merging or reordering in the raw track')
# --- U1 text path
chk(cfg['text_field']['value'] == 'text' and cfg['preprocessing'].get('text_field_used') == 'text' and 'DECIDED (U1)' in cfg['text_field']['status'], 'U1: production-faithful "text" field decided')
tp = cfg['text_processing_path']; chk(len(tp['order']) == 8 and 'clean_text' in tp['order'][1] and 'stem()' in tp['order'][5] and '_TIME_TEXT_RE' in tp['order'][6], 'U1: 8-step text normalization / tokenization path documented')
stop = sorted(R._STOPWORDS); dfc = tp['derived_from_code']
chk(dfc['stopwords_count'] == len(stop) and dfc['stopwords_sha256'] == hashlib.sha256('\n'.join(stop).encode()).hexdigest(), 'U1: recorded stopword count/hash equal the code', f"{len(stop)} stopwords")
chk(dfc['keep_single_letters'] == ''.join(sorted(R._KEEP_SINGLE)) and dfc['asr_digit_sounds'] == R._ASR_DIGIT_SOUNDS and dfc['synonym_groups'] == list(R._SYNONYM_GROUP_WORDS) and dfc['irregular_forms_count'] == len(R._IRREGULAR), 'U1: recorded keep-single / ASR alias / synonym groups / irregular forms equal the code')
import re as _re
lab = _re.compile(r"^\s*(?:-\s*\[Instructor\]\s*|[A-Z][A-Z .']{2,40}:\s*)")
nlab = sum(1 for cs in cues.values() for c in cs if lab.match(c['text']))
chk(tp['known_text_artifacts']['speaker_label_cues_in_text_field'] == nlab, 'U1: speaker-label cue count equals recomputation', str(nlab))
# --- U2 / U3 metric definitions
mt = cfg['metrics']
chk(mt['recall_at_k'] == [1, 3, 5] and mt['hit_rate_at_k'] == [1, 3, 5] and mt['full_evidence_coverage_at_k'] == [1, 3, 5] and mt['mrr_depth'] == 10, 'U2/U3: Recall@k, Hit Rate@k and Full-evidence coverage@k for k=1,3,5; MRR@10')
chk('NOT complete-question evidence recall' in mt['recall_is_not'] and mt['recall_name'] == 'Interval Recall@k', 'U2: interval Recall@k is explicitly not complete-question evidence recall')
chk('EVERY distinct gold interval' in mt['full_evidence_coverage_definition'] and 'does not mean every gold cue' in mt['full_evidence_coverage_definition'], 'U2: Full-evidence coverage@k defined as every interval covered (not every cue)')
chk('at least one distinct gold interval' in mt['hit_rate_definition'], 'U3: Hit Rate@k defined as at least one gold interval represented in the top k')
chk('overlap that interval' in mt['terminology'] and 'at least one shared cue' in mt['terminology'], 'U2: "covers" = cue-ID overlap with the interval gold cues')
chk(len(mt['invariants_for_harness_tests']) >= 5 and 'reconciliation' in mt and 'with_dataset_readme_section_9' in mt['reconciliation'], 'U2: invariants listed and reconciliation with the README recorded')
bad_iv = []
for qid in (r['question_id'] for r in M if r['group'].endswith('ANSWERABLE') and not r['group'].endswith('UNANSWERABLE')):
    ivs = [set(iv['segment_ids']) for iv in G[qid]['gold_intervals']]
    if any(a & b for i, a in enumerate(ivs) for b in ivs[i + 1:]): bad_iv.append(qid)
chk(not bad_iv, 'gold intervals of each answerable question are disjoint (distinct)', str(bad_iv[:5]))
# --- U4 statistics
st = cfg['statistics']; ci = st['confidence_interval']
chk(ci['level'] == 0.95 and ci['resamples'] == 10000 and isinstance(ci['seed'], int) and 'random.Random' in ci['rng'], 'U4: 95% bootstrap, 10000 resamples, pinned integer seed, stdlib RNG', f"seed {ci['seed']}")
chk(math.ceil(0.025 * ci['resamples']) == 250 and math.ceil(0.975 * ci['resamples']) == 9750 and '250th' in ci['interval'] and '9750th' in ci['interval'], 'U4: nearest-rank percentile positions consistent with resample count')
chk('question-level' in ci['method'] and 'independent' in ci['limitation'] and 'secondary_sensitivity_analysis' in ci['limitation'], 'U4: question-level resampling with its independence limitation stated and pointing to the S1 check')
hdr = set(next(iter(Q.values())).keys()); dims = st['breakdowns']['dimensions']
chk('subject' in hdr and 'question_category' in hdr and 'subject' in dims and 'question_category' in dims, 'U4: breakdown dimensions are existing dataset fields')
def flag(n): return 'VERY_SMALL' if n < 10 else ('SMALL' if n < 30 else 'OK')
def strata(rows):
    f = {'subject': lambda r: Q[r['question_id']]['subject'], 'question_category': lambda r: Q[r['question_id']]['question_category'], 'gold_interval_count': lambda r: '1' if int(r['n_gold_intervals']) == 1 else '2+'}
    return {dm: {k: {'n': v, 'sample_flag': flag(v)} for k, v in sorted(collections.Counter(fn(r) for r in rows).items())} for dm, fn in f.items()}
prim = [r for r in M if r['group'] == 'PRIMARY_ANSWERABLE']; sens = [r for r in M if r['group'] == 'SENSITIVITY_ANSWERABLE']
chk(st['breakdowns']['derived_sample_sizes_PRIMARY_ANSWERABLE'] == strata(prim) and st['breakdowns']['derived_sample_sizes_SENSITIVITY_ANSWERABLE'] == strata(sens), 'U4: recorded stratum sizes and small-sample flags equal recomputation from the dataset')
chk(all(sum(v['n'] for v in dv.values()) == len(prim) for dv in strata(prim).values()), 'U4: every breakdown partitions the primary answerable set', f'n={len(prim)}')
chk(st['sample_sizes_always_reported'] is True and 'VERY_SMALL' in st['small_sample_flags']['rule'], 'U4: sample sizes always reported; small-sample rule pinned')
cats = {Q[r['question_id']]['question_category'] for r in prim}; chk(cats <= {'direct_keyword', 'paraphrase', 'technical_term', 'multi_segment'}, 'U4: category labels are those present in the dataset', str(sorted(cats)))
# --- U5 abstention
ab = cfg['abstention']
chk(ab['status'].startswith('NOT_DEFINED') and ab['metrics_defined'] == [], 'U5: no abstention metric is defined or pinned')
chk(any('raw-BM25' in p or 'Raw BM25' in p or 'raw BM25' in p for p in ab['prohibitions'] + [cfg['abstention_note']]) and 'production abstention' in cfg['abstention_note'], 'U5: raw BM25 may not be used to claim production abstention (stated)')
chk({d['id'] for d in ab['required_decisions']} == {'A1a', 'A1b', 'A1c', 'A1d'}, 'U5: required decisions A1a-A1d documented')
chk(ab['sets'] == {'PRIMARY_UNANSWERABLE': 43, 'SENSITIVITY_UNANSWERABLE': 7} and ab['sets']['PRIMARY_UNANSWERABLE'] == sum(1 for r in M if r['group'] == 'PRIMARY_UNANSWERABLE') and ab['sets']['SENSITIVITY_UNANSWERABLE'] == sum(1 for r in M if r['group'] == 'SENSITIVITY_UNANSWERABLE'), 'U5: unanswerable sets stay separate with counts equal to the manifest')
try:
    from qa import service as S
    ac = S.AskConfig(); thr = ac.full_transcript_max_tokens * ac.chars_per_token
    from qa.transcript import normalize_segments as _ns
    shortl = sorted(l for l, cs in cues.items() if sum(len(s.text) + 1 for s in _ns([{'start': c['start_seconds'], 'end': c['end_seconds'], 'text': c['text']} for c in cs])) <= thr)
    chk(sorted(ab['facts_from_code']['protocol_pin_full_transcript_chars_0_differs_from_production_for_lectures']) == shortl, 'U5: short-lecture (production full-transcript bypass) list equals recomputation', f'{shortl} at <= {int(thr)} chars')
    chk(ab['facts_from_code']['status_values'] == [S.STATUS_ANSWERED, S.STATUS_NOT_IN_LECTURE, S.STATUS_INSUFFICIENT] and str(R.EvidenceConfig().min_coverage) in ab['facts_from_code']['retrieval_gate'], 'U5: production status values and gate threshold equal the code')
except Exception as e:
    chk(False, 'U5: could not import qa.service to re-check production facts', repr(e))
# --- U6 verification statement
vs = cfg['verification_statement']
n_closed = sum(1 for r in M if r['group'].startswith('PRIMARY') and r['review_state'] == 'closed_keep'); n_none = sum(1 for r in M if r['group'].startswith('PRIMARY') and r['review_state'] == 'no_review_item')
chk((vs['primary_video_checked'], vs['primary_not_human_reviewed'], vs['primary_total']) == (n_closed, n_none, 176) == (1, 175, 176), 'U6: 1 video-checked + 175 not human-reviewed = 176 primary (recomputed from the manifest)')
chk([r['question_id'] for r in M if r['group'].startswith('PRIMARY') and r['review_state'] == 'closed_keep'] == ['MATH04_Q08'], 'U6: the only video-checked primary question is MATH04_Q08')
chk('NOT fully verified' in vs['summary'] and 'only MATH04_Q08' in vs['summary'] and '175' in vs['summary'] and 'not as results against verified gold' in vs['required_wording'], 'U6: verification statement and required wording present')
chk(cfg['question_manifest']['counts']['PRIMARY_ANSWERABLE'] == 133 and cfg['question_manifest']['counts']['PRIMARY_UNANSWERABLE'] == 43 and cfg['question_manifest']['primary_total'] == 176 and cfg['question_manifest']['sensitivity_total'] == 13, 'U6: 176 primary / 13 sensitivity kept separate (133+43, 6+7)')
# --- U7 later phase
lp = {p['id']: p for p in cfg['later_phases']}
chk('P1' in lp and lp['P1']['status'].startswith('NOT IMPLEMENTED') and 'select_evidence' in lp['P1']['scope'], 'U7: production-set evaluation recorded as a later, unimplemented phase')
chk(any('production-set evaluation not implemented' in x for x in cfg['not_done']) and any('abstention scoring not defined' in x for x in cfg['not_done']), 'not_done lists production-set and abstention scoring')
# =====================================================================================================
# Final protocol edits: S1, S2, deferrals, FROZEN status
# =====================================================================================================
import datetime as _dt, statistics as _stats
# --- S2
sfl = cfg['statistics']['small_sample_flags']
chk(sfl['status'] == 'APPROVED (S2)' and sfl['approved_thresholds'] == {'small': 'n < 30', 'very_small': 'n < 10'}, 'S2: thresholds approved (n<30 small, n<10 very small)')
chk('NO confidence interval' in sfl['ci_policy']['VERY_SMALL'] and 'instability warning' in sfl['ci_policy']['SMALL'] and set(sfl['ci_policy']) == {'VERY_SMALL', 'SMALL', 'OK'}, 'S2: very-small strata get no interval; small strata get an interval with an instability warning')
chk(flag(9) == 'VERY_SMALL' and flag(10) == 'SMALL' and flag(29) == 'SMALL' and flag(30) == 'OK', 'S2: flag function boundaries (9/10/29/30) match the approved thresholds')
# --- S1
s1 = cfg['statistics']['secondary_sensitivity_analysis']
perlec = collections.Counter(r['lecture_id'] for r in prim); szs = sorted(perlec.values())
chk('SECONDARY' in s1['role'] and 'never replaces' in s1['role'] and 'PRIMARY interval' in cfg['statistics']['primary_interval'] and 'question-level' in cfg['statistics']['primary_interval'], 'S1: lecture-clustered bootstrap is secondary; question-level bootstrap stays primary')
chk(s1['cluster_unit'].startswith('lecture_id') and s1['level'] == 0.95 and s1['resamples'] == 10000 and s1['seed'] == ci['seed'] and 'randrange(G)' in s1['resampling'] and '250th and 9750th' in s1['interval'], 'S1: cluster unit, level, 10000 resamples, pinned seed and percentile positions documented')
chk(s1['derived_from_manifest'] == {'lectures_with_primary_answerable_questions': len(perlec), 'questions_per_lecture_min_median_max': [szs[0], _stats.median(szs), szs[-1]], 'primary_answerable_questions': len(prim)}, 'S1: recorded cluster counts equal recomputation from the manifest', f'{len(perlec)} lectures; per-lecture min/median/max = {szs[0]}/{_stats.median(szs)}/{szs[-1]}')
chk('no scoring code' in s1['status'] and 'headline metrics only' in s1['applies_to'], 'S1: documented method only (no scoring code); headline metrics only')
# --- deferrals (A1 / P1)
chk(cfg['abstention']['deferred'] is True and cfg['abstention']['blocks_raw_bm25_baseline'] is False and cfg['abstention']['metrics_defined'] == [], 'A1: abstention deferred, non-blocking, still no metric defined')
chk(lp['P1']['deferred'] is True and lp['P1']['blocks_raw_bm25_baseline'] is False and lp['P1']['status'].startswith('NOT IMPLEMENTED'), 'P1: production-set phase deferred, non-blocking, not implemented')
# --- FROZEN + README supersession + authority
fz = cfg['freeze']
chk(cfg['status'] == 'FROZEN' and fz['status'] == 'FROZEN', 'status is FROZEN')
try: _dt.date.fromisoformat(fz['frozen_date']); okd = True
except ValueError: okd = False
chk(okd and fz['protocol_version'] == 'bm25_v1.0' and cfg['protocol_id'] == 'bm25_v1', 'freeze: protocol version and ISO frozen date recorded', f"{fz['protocol_version']} @ {fz['frozen_date']}")
chk(len(fz['version_history']) >= 3 and fz['version_history'][-1]['version'] == fz['protocol_version'] and fz['version_history'][-1]['date'] == fz['frozen_date'], 'freeze: version history ends with the frozen version and date')
chk('new protocol_id/version' in fz['change_policy'] and any('A1' in x for x in fz['not_frozen_deferred']) and any('P1' in x for x in fz['not_frozen_deferred']) and any('harness' in x for x in fz['not_frozen_deferred']), 'freeze: change policy and deferred/not-frozen items recorded')
rs = cfg['metrics']['reconciliation']['readme_supersession']
chk('MRR@10' in rs and 'e.g. 20' in rs and 'left unchanged' in rs and 'dataset.file_sha256' in rs, 'README: configured MRR@10 supersedes the illustrative cut-off (README left unchanged)')
chk(cfg['dataset']['file_sha256']['dataset_readme.md'] == sha(f"{cfg['dataset']['path']}/dataset_readme.md"), 'README hash still equals the pinned value (README unchanged)')
chk('authoritative frozen protocol' in cfg['authority'] and 'does not claim that the gold labels are fully human-verified' in cfg['authority'], 'authority statement present and does not claim verified labels')
chk(cfg['verification_statement']['primary_video_checked'] == 1 and cfg['verification_statement']['primary_not_human_reviewed'] == 175, 'frozen config still states 1 video-checked / 175 not human-reviewed')
fails = [r for r in res if r[0] == 'FAIL']
print(f"Structural validation of eval protocol: {'FAIL' if fails else 'PASS'} ({len(res) - len(fails)} passed, {len(fails)} failed)")
for s, n, d in res: print(f'  [{s}] {n}' + (f' | {d}' if d else ''))
sys.exit(1 if fails else 0)
