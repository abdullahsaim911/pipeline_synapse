#!/usr/bin/env python3
"""Structural validation of the review_v1 artifacts. READ-ONLY on the datasets and the audit.

Usage:  python validate_review_artifacts.py [--report OUT.md] [--check-log]
  default      validates review_items.csv, review_checklist.md, verification_log.csv (as a BLANK template), baseline hashes
  --check-log  validates a FILLED log (vocabularies, dates, IDs, approval rules) instead of requiring it to be blank
It does not judge whether any question is correct; it only checks that the artifacts are well formed and consistent with the datasets.
"""
import csv, json, os, re, sys, hashlib, datetime, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RV = os.path.dirname(HERE); ROOT = os.path.dirname(RV)
V0, V1, AUD = f'{ROOT}/synapse_stem25', f'{ROOT}/synapse_stem25_reviewed_v1', f'{ROOT}/audit'
check_log = '--check-log' in sys.argv
report = sys.argv[sys.argv.index('--report') + 1] if '--report' in sys.argv else None
res = []  # (status, check, detail)
def ok(c, d=''): res.append(('PASS', c, d))
def bad(c, d=''): res.append(('FAIL', c, d))
def chk(cond, c, d=''): (ok if cond else bad)(c, d)
def rd(p): return list(csv.DictReader(open(p, encoding='utf-8', newline='')))

# ---- 1 files exist / parse
need = ['review_items.csv', 'review_checklist.md', 'verification_log.csv', 'verification_log_schema.md', 'baseline_hashes.json', 'README.md']
for f in need: chk(os.path.exists(f'{RV}/{f}'), f'file exists: {f}')
items = rd(f'{RV}/review_items.csv'); log = rd(f'{RV}/verification_log.csv')
checklist = open(f'{RV}/review_checklist.md', encoding='utf-8').read(); readme = open(f'{RV}/README.md', encoding='utf-8').read()
base = json.load(open(f'{RV}/baseline_hashes.json', encoding='utf-8'))
ok('files parse', f'{len(items)} items, {len(log)} log rows')
Q = {r['question_id']: r for r in rd(f'{V1}/questions.csv')}
Q0 = {r['question_id']: r for r in rd(f'{V0}/questions.csv')}
M = {r['lecture_id']: r for r in rd(f'{V1}/lecture_manifest.csv')}
seg = {}
for l in open(f'{V1}/transcript_segments.jsonl', encoding='utf-8'):
    r = json.loads(l); seg[r['segment_id']] = r
# ---- 2 items
ids = [r['item_id'] for r in items]
chk(len(ids) == len(set(ids)), 'unique item IDs', f'{len(ids)} items')
chk(len({r['question_id'] for r in items}) == len(items), 'one item per question (no duplicate question IDs)')
cnt = collections.Counter(r['group'] for r in items); chk(dict(cnt) == {'G1': 3, 'G2': 2, 'G3': 10, 'G4': 7}, 'group counts G1=3 G2=2 G3=10 G4=7', str(dict(cnt)))
chk(set(r['priority'] for r in items) <= {'P1', 'P2'}, 'priorities in {P1,P2}', str(dict(collections.Counter(r['priority'] for r in items))))
ACT = {'keep_as_is', 'revise_gold_evidence', 'revise_expected_answer_points', 'revise_question_text', 'reclassify_answerable', 'keep_unanswerable', 'exclude_from_transcript_only', 'move_to_visual_track', 'readmit_to_transcript_only', 'needs_further_review'}
errs = []
for r in items:
    q = Q.get(r['question_id'])
    if not q: errs.append((r['item_id'], 'unknown question')); continue
    if q['lecture_id'] != r['lecture_id']: errs.append((r['item_id'], 'lecture mismatch'))
    if M[r['lecture_id']]['video_url'] != r['video_url']: errs.append((r['item_id'], 'video URL differs from manifest'))
    for k_item, k_q in [('current_eval_track', 'eval_track'), ('current_gold_segments', 'gold_evidence_segments'), ('current_gold_start_seconds', 'gold_start_seconds'), ('current_gold_end_seconds', 'gold_end_seconds'), ('expected_answer_points', 'expected_answer_points'), ('question_text', 'question_text'), ('answerable', 'answerable')]:
        exp = q[k_q] if not (k_item == 'current_gold_segments' and q[k_q] == '') else '(none: unanswerable)'
        if r[k_item] != exp: errs.append((r['item_id'], f'{k_item} differs from reviewed_v1'))
    dur = float(M[r['lecture_id']]['duration_seconds'])
    for w in r['video_windows_seconds'].split(';'):
        a, b = map(float, w.split('-'))
        if not (0 <= a < b <= dur + 1): errs.append((r['item_id'], f'window {w} outside video duration {dur}'))
    for lk in r['video_links'].split(' ; '):
        if not lk.startswith(r['video_url'] + '&t=') or not lk.endswith('s'): errs.append((r['item_id'], 'bad video link'))
    if not r['transcript_checks'].strip() or not r['video_checks'].strip(): errs.append((r['item_id'], 'missing transcript or video checks'))
    if not r['decision_needed'].strip() or not r['reason_for_review'].strip(): errs.append((r['item_id'], 'missing decision/reason'))
    acts = [a.strip() for a in r['allowed_actions'].replace('(needs_further_review)', '').split('|')]
    for a in acts:
        a = a.replace('keep excluded', 'needs_further_review').strip()
        if a not in ACT: errs.append((r['item_id'], f'unknown action {a}'))
    # every S-id cited in text exists for this lecture
    txt = ' '.join([r['transcript_checks'], r['video_checks'], r['reason_for_review'], r['decision_needed']])
    for m in re.finditer(r'\bS(\d{4})\b', txt):
        if f"{r['lecture_id']}_S{m.group(1)}" not in seg: errs.append((r['item_id'], f'cited S{m.group(1)} does not exist in {r["lecture_id"]}'))
    # proposed evidence
    pe = [x for x in r['proposed_evidence_NOT_APPLIED'].split(';') if x]
    ps, pn = [x for x in r['proposed_start_seconds'].split(';') if x], [x for x in r['proposed_end_seconds'].split(';') if x]
    if not (len(pe) == len(ps) == len(pn)): errs.append((r['item_id'], 'proposed evidence/time count mismatch'))
    for x, s, e in zip(pe, ps, pn):
        a, b = (x.split('..') + [x])[:2] if '..' in x else (x, x)
        if a not in seg or b not in seg or not a.startswith(r['lecture_id']): errs.append((r['item_id'], f'proposed id {x} invalid')); continue
        if abs(seg[a]['start_seconds'] - float(s)) > 1e-6 or abs(seg[b]['end_seconds'] - float(e)) > 1e-6: errs.append((r['item_id'], f'proposed times for {x} differ from cues'))
    if r['item_status'] != 'open': errs.append((r['item_id'], 'item_status must start as open'))
chk(not errs, 'items consistent with reviewed_v1 / manifest / transcript (IDs, URLs, windows, current gold, proposals)', '; '.join(f'{a}: {b}' for a, b in errs[:8]) or f'{len(items)} items checked')
# every item marked as in expected set
want = {'CHEM02_Q01', 'MATH04_Q08', 'CHEM04_Q05', 'CHEM04_Q03', 'CHEM05_Q07', 'MATH05_Q04', 'BIO04_Q05', 'CHEM02_Q08', 'BIO03_Q07', 'MATH01_Q07', 'CHEM05_Q06', 'MATH03_Q07', 'BIO05_Q05', 'PHYS02_Q02', 'MATH05_Q01',
        'MATH02_Q10', 'MATH03_Q09', 'MATH01_Q09', 'BIO02_Q10', 'PHYS03_Q09', 'PHYS03_Q10', 'CHEM01_Q10'}
chk({r['question_id'] for r in items} == want, 'item set equals the 22 requested questions (3 corrected + 2 blockers + 10 majors + 7 unanswerable)')
# ---- 3 checklist
chk(all(r['item_id'] in checklist and r['question_id'] in checklist for r in items), 'checklist mentions every item ID and question ID')
chk(all(f"L-{r['item_id']}-T" in checklist and f"L-{r['item_id']}-V" in checklist for r in items), 'checklist references both log rows for every item')
chk('PROPOSED - NOT APPLIED' in checklist, 'proposals are labelled "PROPOSED - NOT APPLIED"')
chk(checklist.count('- [ ] T') + checklist.count('- [ ] V') > 0 and '- [x]' not in checklist, 'checklist boxes all unticked')
# ---- 4 log
lids = [r['log_id'] for r in log]; chk(len(lids) == len(set(lids)) and len(log) == 2 * len(items), 'log rows unique; exactly 2 per item', f'{len(log)} rows')
exp = {f"L-{r['item_id']}-{c}" for r in items for c in 'TV'}; chk(set(lids) == exp, 'log IDs match items x {T,V}')
itm = {r['item_id']: r for r in items}
chk(all(r['item_id'] in itm and itm[r['item_id']]['question_id'] == r['question_id'] and itm[r['item_id']]['lecture_id'] == r['lecture_id'] for r in log), 'log rows reference the right question/lecture')
chk(all((r['check_type'] == 'transcript') == r['log_id'].endswith('-T') for r in log), 'check_type matches log_id suffix')
RES = ['reviewer', 'review_date', 'video_timestamp_checked', 'evidence_supported', 'expected_answer_supported', 'answerability_decision', 'visual_content_observed', 'proposed_action',
       'revised_evidence_ids', 'revised_start_seconds', 'revised_end_seconds', 'notes', 'approval_status', 'approved_by', 'approval_date', 'applied_to_dataset']
if not check_log:
    chk(all(r[k] == '' for r in log for k in RES), 'log is BLANK (all result fields empty)', f'{len(log)} rows x {len(RES)} result fields')
else:
    V = dict(evidence_supported={'yes', 'no', 'partial'}, expected_answer_supported={'yes', 'no', 'partial'}, answerability_decision={'answerable', 'unanswerable', 'ambiguous', 'not_applicable'},
             proposed_action=ACT, approval_status={'not_submitted', 'submitted', 'approved', 'rejected'})
    e = []
    for r in log:
        done = any(r[k] for k in RES)
        for k, allowed in V.items():
            if r[k] and r[k] not in allowed: e.append((r['log_id'], f'{k}={r[k]}'))
        for k in ('review_date', 'approval_date'):
            if r[k]:
                try: datetime.date.fromisoformat(r[k])
                except ValueError: e.append((r['log_id'], f'{k} not YYYY-MM-DD'))
        if done and (not r['reviewer'] or not r['review_date']): e.append((r['log_id'], 'filled row needs reviewer and review_date'))
        if r['check_type'] == 'video' and done and r['evidence_supported'] and not r['video_timestamp_checked']: e.append((r['log_id'], 'video row needs video_timestamp_checked'))
        if r['approval_status'] == 'approved' and not (r['approved_by'] and r['approval_date']): e.append((r['log_id'], 'approved needs approved_by/date'))
        if r['applied_to_dataset'].startswith('yes') and r['approval_status'] != 'approved': e.append((r['log_id'], 'applied without approval'))
        if r['applied_to_dataset'] not in ('', 'no') and not r['applied_to_dataset'].startswith('yes:'): e.append((r['log_id'], 'applied_to_dataset must be no or yes:<version>'))
        if r['revised_evidence_ids']:
            parts = [p for p in r['revised_evidence_ids'].split(';') if p]
            for p in parts:
                a, b = p.split('..') if '..' in p else (p, p)
                if a not in seg or b not in seg or a.split('_')[0] != r['lecture_id']: e.append((r['log_id'], f'revised id {p} invalid'))
            if len(parts) != len([x for x in r['revised_start_seconds'].split(';') if x]) or len(parts) != len([x for x in r['revised_end_seconds'].split(';') if x]): e.append((r['log_id'], 'revised ids/times count mismatch'))
    chk(not e, 'filled log passes vocabulary/ID/approval rules', '; '.join(f'{a}: {b}' for a, b in e[:10]) or 'ok')
# ---- 5 read-only guarantee
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
for name, d in [('synapse_stem25', V0), ('synapse_stem25_reviewed_v1', V1), ('audit', AUD)]:
    cur = {f: sha(f'{d}/{f}') for f in sorted(os.listdir(d)) if os.path.isfile(f'{d}/{f}')}
    chk(cur == base[name], f'{name} unchanged since review_v1 was created', f'{len(cur)} files')
# the scored set / exclusion untouched
tr = collections.Counter(r['eval_track'] for r in Q.values())
chk(tr['X_EXCLUDED_PENDING_VIDEO_VERIFICATION'] == 1 and Q['CHEM02_Q01']['eval_track'].startswith('X_EXCLUDED'), 'CHEM02_Q01 still excluded from scoring in reviewed_v1')
chk(tr['A_transcript_qa'] == 139 and tr['A_transcript_qa_abstention'] == 50, 'scored candidate set still 139 + 50', str(dict(tr)))
chk(all(Q[q]['eval_track'] == Q0[q]['eval_track'] for q in Q if q != 'CHEM02_Q01'), 'no other question changed evaluation track (vs v0.1)')
chk('Nothing in this folder changes' in checklist or 'does not change' in readme.lower(), 'documents state that nothing is changed by the review package')
fails = [r for r in res if r[0] == 'FAIL']
lines = [f'# Structural validation of review_v1 artifacts\n', f'Run: {datetime.date.today().isoformat()}; mode: {"filled-log check" if check_log else "blank-template check"}. Result: **{"FAIL" if fails else "PASS"}** ({len(res) - len(fails)} passed, {len(fails)} failed).\n',
         '| Status | Check | Detail |', '|---|---|---|'] + [f'| {s} | {c} | {d} |' for s, c, d in res]
out = '\n'.join(lines) + '\n'
print(out)
if report: open(report, 'w', encoding='utf-8').write(out)
sys.exit(1 if fails else 0)
