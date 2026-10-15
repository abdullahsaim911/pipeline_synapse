#!/usr/bin/env python3
"""ask_prompt_b - controlled test of Change B (one added sentence in rule 3 of the Ask system instruction).

Commands (everything except `run` is offline and never touches .env or the network):

    python -B ask_prompt_b_eval.py verify                      # pins, instruction hashes, identical-evidence gate
    python -B ask_prompt_b_eval.py plan                        # the seeded 60-call schedule
    python -B ask_prompt_b_eval.py template-sheet OUT.csv      # blank blinded scoring sheet
    python -B ask_prompt_b_eval.py report RAW.jsonl [--sheet S.csv --key K.json]
    python -B ask_prompt_b_eval.py make-sheet RAW.jsonl SHEET.csv KEY.json
    python -B ask_prompt_b_eval.py run --out DIR --approve-live-requests 60   # MODEL CALLS; needs explicit approval flag

The two arms differ ONLY in the system instruction. The user prompt (which carries the evidence) is identical; the harness checks its
SHA-256 against the registration before every request. Production code is imported, never edited; the revised instruction exists only in memory.
"""
import collections, contextlib, csv, hashlib, json, os, random, re, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REG_PATH = os.path.join(HERE, 'config_b.json')
REG_SHA256 = '50a6398d2b33de4131e2aaa4c8aae6ff181bcdfbaae55d228694c6e454bb08f5'          # pinned after `register` has written config_b.json

ARMS = ('current', 'revised')
REPEATS = 3
SEED = 20261014
MIN_INTERVAL_SECONDS = 6.0                         # 10 requests per minute
RULE3_ANCHOR = "3. If the excerpts answer only part of the question, answer that part and say plainly what they do not cover."
ADDED_SENTENCE = (' When the instructor describes the steps of a derivation or procedure but never states its result, put only those stated steps together into the result and '
                  'introduce it with "From these steps, it seems that ...". Answer every part of the question that the excerpts support.')
DATASET = 'dataset/synapse_stem25_reviewed_v1'
DEMO_TRANSCRIPT = 'data/3zLQfuMZMF4/transcript.json'
PIN_FILES_DATASET = ('questions.csv', 'transcript_segments.jsonl')


class VerificationError(RuntimeError):
    pass


class LiveRunRefused(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------------------------------------- case definitions (used by `register` only)
DEMO_FORMULA_Q = 'What is the formula for integration by parts, and what do the terms in the formula represent.'
CASE_DEFS = [
    {'id': 'P1', 'role': 'positive-synthesis (motivating case; NOT independent)', 'source': 'demo', 'question': DEMO_FORMULA_Q,
     'expectation_status': 'PROVISIONAL: the author\'s reading of the captions; the project owner has not yet confirmed it against the video (0:37-1:30 and 2:45-3:30)',
     'expected_content': [
         'Integration by parts is described as the reverse of the product rule (stated).',
         'After integrating the product rule, the left side leaves UV (stated).',
         'The right side has an integral term plus "the integral of V du" (the caption garbles the first term as "integral of UV"); the integral of V du is moved to the other side (stated).',
         'U is chosen with the LIATE acronym; du comes from differentiating u; v comes from integrating dv (stated).',
         'A complete answer gives the formula in words, introduced as derived ("From these steps, it seems that ..."), defines u, dv, du and v from the spoken text, and says the final formula is not read aloud in full.',
         'Reading the garbled "integral of UV" as "integral of u dv" is allowed only as a labelled reading.'],
     'acceptable_statuses': ['answered'], 'checks': ['common']},
    {'id': 'P2', 'role': 'positive-synthesis', 'source': 'dataset:MATH04_Q08', 'question': None,
     'expectation_status': 'DATASET NOTE + OWNER REVIEW (log R02): evidence_supported=yes; the video observation is as reported by the project owner and was not independently confirmed',
     'expected_content': [
         'The series is built from four separately spoken terms: constant 1, minus 2x, plus 4x squared, plus 3x cubed (dataset note: MATH04_S0079, S0086, S0098, S0106).',
         'Expected combined series (dataset): 1 - 2x + 4x^2 + 3x^3, written in words.',
         'The lecturer does not state the series as one expression, so a correct answer labels the assembled series as derived.',
         'The dataset\'s second point ("same polynomial as the original with reversed order") is unclear and is NOT scored.'],
     'acceptable_statuses': ['answered'], 'checks': ['common']},
    {'id': 'P3', 'role': 'positive-synthesis', 'source': 'dataset:PHYS04_Q07', 'question': None,
     'expectation_status': 'UNVERIFIED: dataset machine-draft (human review pending)',
     'expected_content': ['Look for a solution of the same form as the pure exponential forcing (I = W e^(i omega t)).',
                          'The derivative brings down i omega; the integral divides by i omega (stated in the gold window).',
                          'Result W = V / (i omega L + R + 1/(i omega C)) (dataset point; whether the lecture states it in the supplied evidence is unverified).'],
     'acceptable_statuses': ['answered'], 'checks': ['common']},
    {'id': 'C1', 'role': 'control-result-stated', 'source': 'dataset:MATH02_Q07', 'question': None,
     'expectation_status': 'UNVERIFIED: dataset machine-draft (human review pending)',
     'expected_content': ['The derivative of (5x^2-1)^(4/3) gives 40x/3 (5x^2-1)^(1/3).', 'The target has 4x, so the constant is off by 10/3.', 'Multiply by 3/10.',
                          'The result is stated by the lecturer; the answer should not need a "From these steps" label (over-labelling is counted, not failed).'],
     'acceptable_statuses': ['answered'], 'checks': ['common']},
    {'id': 'C2', 'role': 'control-result-stated', 'source': 'dataset:MATH03_Q03', 'question': None,
     'expectation_status': 'UNVERIFIED: dataset machine-draft (human review pending)',
     'expected_content': ['If A v = lambda v then A^2 v = lambda^2 v.', 'Same eigenvector, squared eigenvalue.'],
     'acceptable_statuses': ['answered'], 'checks': ['common']},
    {'id': 'N1', 'role': 'negative-absence', 'source': 'dataset:MATH05_Q09', 'question': None,
     'expectation_status': 'ABSENCE: an automatic caption search found no occurrence of "Bayes"; the dataset label is a machine-draft and unverified',
     'expected_content': ['The lecture does not name or give Bayes\' theorem. Evidence is partial (retrieval), so not_in_lecture cannot be returned; the status must be insufficient_context.',
                          'The answer must not state Bayes\' theorem, its name or its terms as lecture content, and must add no outside knowledge.'],
     'acceptable_statuses': ['insufficient_context'], 'checks': ['common', 'n1']},
    {'id': 'N2', 'role': 'negative-absence', 'source': 'dataset:MATH04_Q09', 'question': None,
     'expectation_status': 'ABSENCE: an automatic caption search found no occurrence of "radius" or "converge"; the dataset label is a machine-draft and unverified',
     'expected_content': ['Convergence is not discussed. not_in_lecture or insufficient_context are acceptable.',
                          'The answer must not state or derive a radius of convergence (that would be outside knowledge) and must not use the "From these steps" label to do so.'],
     'acceptable_statuses': ['not_in_lecture', 'insufficient_context'], 'checks': ['common', 'n2']},
    {'id': 'N3', 'role': 'negative-absence', 'source': 'dataset:MATH01_Q10', 'question': None,
     'expectation_status': 'ABSENCE: the dataset note says the derivative of sine is only cited from an earlier lecture; machine-draft, unverified (caption search cannot prove a proof is absent)',
     'expected_content': ['No proof from the limit definition is given. not_in_lecture or insufficient_context are acceptable, or an answered reply that plainly says no proof is given and adds nothing from outside.',
                          'The answer must not supply a limit-definition proof.'],
     'acceptable_statuses': ['not_in_lecture', 'insufficient_context'], 'checks': ['common', 'n3']},
    {'id': 'N4', 'role': 'negative-visual-only', 'source': 'demo', 'question': 'What is written on the slide at the start of the video?',
     'expectation_status': 'CUSTOM question written by the author; whether any text is shown on screen is not verified',
     'expected_content': ['The answer must say it cannot see the screen. It may add what the instructor says at the start.', 'It must not describe or guess slide content.'],
     'acceptable_statuses': ['insufficient_context', 'answered'], 'checks': ['common', 'n4']},
    {'id': 'N5', 'role': 'negative-attribution', 'source': 'demo', 'question': 'State the product rule and use it to derive the formula for integration by parts.',
     'expectation_status': 'CUSTOM question written by the author; premise (the product rule formula is never spoken) confirmed by caption search only',
     'expected_content': ['The captions mention "the formula of the product rule" but never state it. The best answer says the product rule formula is not stated in the lecture.',
                          'It may derive only from the stated steps, labelled "From these steps, it seems that ...".',
                          'STRICT: any product-rule formula in the answer is a violation; attributing it to the instructor is a hard misattribution.'],
     'acceptable_statuses': ['answered', 'insufficient_context'], 'checks': ['common', 'n5']},
]

N5_FORMULA_PATTERNS = [r"u\s*prime\s*v", r"u\s*v\s*prime", r"derivative of (?:the )?(?:product|u times v|u v)\b", r"d\s*of\s*u\s*v", r"u\s*d\s*v\s*plus\s*v\s*d\s*u",
                       r"u\s*times\s*the\s*derivative\s*of\s*v", r"v\s*times\s*the\s*derivative\s*of\s*u", r"derivative of u times v equals", r"d\s*u\s*v\s*equals"]
N5_ATTRIBUTION = r"\b(?:the )?instructor\s+(?:says|states|explains|notes|mentions|writes|shows|gives|presents|uses|derives)\b"
N5_OUTSIDE_MARKERS = r"general knowledge|outside (?:the )?lecture|not (?:stated|given|spoken|said) in the (?:lecture|captions|excerpts)|not from the lecture"
N5_NEGATION = r"(?:does not|doesn't|did not|didn't|do not|don't|never|isn't|is not|was not|wasn't|not|without|nowhere|only (?:refers|mentions|points))"
N5_VERB = (r"(?:stat(?:e|es|ed|ing)|giv(?:e|es|en|ing)|say|says|said|spell(?:s|ed|ing)?|read(?:s)?|spoken|speak|writ(?:e|es|ten|ing)|show(?:s|n|ing)?|provid(?:e|es|ed|ing)|"
           r"includ(?:e|es|ed|ing)|present(?:s|ed|ing)?|specif(?:y|ies|ied|ying)|defin(?:e|es|ed|ing))")


def acknowledges_not_spoken(text):
    """True if some sentence that mentions the product rule also says it is not stated / given / read in the lecture (heuristic; humans decide)."""
    for s in _sentences(text):
        low = s.lower()
        if 'product rule' in low and re.search(N5_NEGATION + r"[^.]{0,30}?(?<![a-z])" + N5_VERB + r"(?![a-z])", low):
            return True
    return False


DERIVED_LABELS = r"from these steps, it seems that|from this, it seems that|it seems that"


# ------------------------------------------------------------------------------------------------- production access (import only)
def _qa():
    """Import production modules. Never reads .env. Refuses to run if any ASK_* variable is set (production defaults are what is tested)."""
    ask_vars = sorted(k for k in os.environ if k.startswith('ASK_'))
    if ask_vars:
        raise VerificationError(f'ASK_* environment variables are set: {ask_vars}')
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from qa import service as S
    from qa.transcript import normalize_segments
    from llm_interface.providers import GenerationResult
    return S, normalize_segments, GenerationResult


def build_revised(orig):
    if RULE3_ANCHOR not in orig:
        raise VerificationError('rule 3 anchor text not found in the production instruction; the registration no longer applies')
    rev = orig.replace(RULE3_ANCHOR, RULE3_ANCHOR + ADDED_SENTENCE)
    if rev.count(ADDED_SENTENCE) != 1:
        raise VerificationError('revised instruction was not built exactly once')
    return rev


@contextlib.contextmanager
def instruction(S, text):
    original = S.SYSTEM_INSTRUCTION
    S.SYSTEM_INSTRUCTION = text
    try:
        yield
    finally:
        S.SYSTEM_INSTRUCTION = original


def _rd(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_case_inputs(case, root=REPO):
    """(question text, segments) for a registered case, using production normalisation only."""
    _, normalize, _ = _qa()
    src = case['source']
    if src == 'demo':
        with open(os.path.join(root, DEMO_TRANSCRIPT), encoding='utf-8') as f:
            d = json.load(f)
        segs = normalize([{'start': e['start'], 'end': e['end'], 'text': e['text']} for e in d['entries']])
        return case['question'], segs
    qid = src.split(':', 1)[1]
    ds = os.path.join(root, DATASET)
    q = {r['question_id']: r for r in _rd(os.path.join(ds, 'questions.csv'))}[qid]
    cues = []
    with open(os.path.join(ds, 'transcript_segments.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['lecture_id'] == q['lecture_id']:
                cues.append({'start': r['start_seconds'], 'end': r['end_seconds'], 'text': r['text']})
    return q['question_text'], normalize(cues)


class _Recorder:
    name = 'offline-recorder'
    model = 'none'

    def __init__(self, result_cls):
        self.prompts, self.settings, self._cls = [], [], result_cls

    def generate(self, prompt, settings=None):
        self.prompts.append(prompt)
        self.settings.append(settings)
        return self._cls(text='The instructor says this.', provider=self.name, model=self.model, latency_ms=0, usage=None)


def offline_prompt(case, instr_text, root=REPO):
    """Run production AskService.answer with a recording stand-in; return (user prompt, estimated tokens, evidence mode). No model, no network."""
    S, _, Result = _qa()
    question, segs = load_case_inputs(case, root)
    rec = _Recorder(Result)
    with instruction(S, instr_text):
        out = S.AskService(provider=rec, config=S.AskConfig()).answer(question=question, segments=segs)
    if not rec.prompts:
        raise VerificationError(f'case {case["id"]}: the gate stopped before the model; this case cannot be tested')
    return rec.prompts[0], out.estimated_prompt_tokens, out.evidence_mode, rec.settings[0]


# ------------------------------------------------------------------------------------------------- registration
def production_pin_paths(root=REPO):
    paths = [os.path.join('qa', n) for n in sorted(os.listdir(os.path.join(root, 'qa'))) if n.endswith('.py')]
    paths += [os.path.join('llm_interface', n) for n in sorted(os.listdir(os.path.join(root, 'llm_interface'))) if n.endswith('.py')]
    return [p.replace('\\', '/') for p in paths]


def build_registration(root=REPO):
    S, _, _ = _qa()
    orig = S.SYSTEM_INSTRUCTION
    rev = build_revised(orig)
    cases = []
    for c in CASE_DEFS:
        question, _ = load_case_inputs(c, root)
        p_cur, t_cur, mode, _ = offline_prompt(c, orig, root)
        p_rev, t_rev, _, _ = offline_prompt(c, rev, root)
        if p_cur != p_rev:
            raise VerificationError(f'case {c["id"]}: the evidence differs between arms; the test would be confounded')
        from qa.tokens import estimate_tokens
        cases.append({**{k: v for k, v in c.items() if k != 'question'}, 'question': question, 'evidence_mode': mode, 'user_prompt_sha256': sha256_text(p_cur),
                      'estimated_input_tokens': {'current': t_cur, 'revised': estimate_tokens(rev, 3.5) + estimate_tokens(p_cur, 3.5)}})
    n_requests = len(cases) * len(ARMS) * REPEATS
    est = {a: sum(c['estimated_input_tokens'][a] for c in cases) * REPEATS for a in ARMS}
    return {
        'experiment_id': 'ask_prompt_b_v1', 'status': 'REGISTERED - NOT RUN', 'created': '2026-10-14',
        'purpose': 'Compare the current and revised (Change B) Ask system instruction with IDENTICAL evidence, on completeness, grounding, attribution and abstention.',
        'not_in_scope': ['Change A (full-transcript threshold)', 'visual evidence integration', 'any change to production code or the frozen retrieval benchmark'],
        'honesty_notes': ['P1 is the motivating case and is not independent.', 'P1 expected content is PROVISIONAL (author reading; owner confirmation pending).',
                          'P3, C1, C2 expected answers come from machine-drafted dataset labels and are UNVERIFIED.',
                          'N1-N3 absence rests on caption search and machine-drafted labels; N4 and N5 are author-written.',
                          'n = 3 repeats per cell: results are descriptive, not statistical.'],
        'instruction': {'rule_3_anchor': RULE3_ANCHOR, 'added_sentence': ADDED_SENTENCE, 'current_sha256': sha256_text(orig), 'revised_sha256': sha256_text(rev),
                        'revised_is_built_in_memory_only': True},
        'arms': {'current': 'production SYSTEM_INSTRUCTION unchanged', 'revised': 'production SYSTEM_INSTRUCTION with the added sentence appended to rule 3'},
        'settings': {'provider': 'gemini', 'model': 'gemini-3.5-flash-lite', 'temperature': 0.2, 'max_output_tokens': 1024, 'ask_config': 'AskConfig() defaults, ASK_* unset',
                     'repeats': REPEATS, 'schedule_seed': SEED, 'schedule': 'per round the cases are shuffled and, per case, the arm order is randomised (seeded)',
                     'pacing': f'at least {MIN_INTERVAL_SECONDS:g} s between requests (10 per minute)', 'retries': 'none',
                     'stop_rule': 'abort on the first provider error or prompt/instruction hash mismatch; partial results are kept and reported',
                     'requests_total': n_requests},
        'identical_evidence_gate': 'the SHA-256 of the user prompt (which carries the evidence) must equal the registered value before every request',
        'quota_estimate': {'requests': n_requests, 'estimated_input_tokens': est, 'estimated_input_tokens_total': sum(est.values()),
                           'note': 'chars/3.5 estimates; the only measured actual-vs-estimate pairs (n = 2) were 11-16% below the estimate; limits for the model are unverified'},
        'cases': cases,
        'automated_checks': {
            'common': ['status in the registered acceptable set', 'sources non-empty only when answered', 'words (over 150 flagged, over 250 failed)', 'math symbols (rule 8)', 'timestamps in text',
                       'markdown bullets', 'derived-content label present'],
            'n1': 'content about Bayes\' theorem terms (posterior, prior, likelihood, conditional probability) -> review flag',
            'n2': 'a radius value, "ratio test" or "converges for" -> review flag',
            'n3': 'limit-definition proof language (angle addition, "limit as h") -> review flag',
            'n4': 'must contain a "cannot see the screen" statement; slide-content phrases without it -> hard flag',
            'n5_strict': {
                'H1_misattribution': 'a product-rule formula pattern in the same sentence as an "instructor says/states/..." phrase (HARD)',
                'H2_product_rule_present': 'a product-rule formula pattern anywhere in the answer (HARD, or SOFT when an explicit outside-the-lecture marker is in the same sentence)',
                'H3_missing_acknowledgement': 'no statement that the product rule formula is not stated/spoken in the lecture (FAIL)',
                'H4_unlabelled_derivation': 'an integration-by-parts formula or derived result without a "From these steps / From this, it seems" label (FAIL)',
                'rule': 'automated N5 flags are advisory; ALL six N5 answers get human review; a human "ok" against an automated HARD flag must be reconciled in writing before the result counts'}},
        'human_scoring': {'blinded': True, 'dimensions': {'completeness': ['all', 'partial', 'none'], 'grounding': ['ok', 'soft', 'hard'], 'attribution': ['ok', 'unlabelled', 'misattributed'],
                                                         'abstention': ['acceptable', 'unacceptable', 'na'], 'labelled_derived': ['yes', 'no', 'na']},
                          'who_reviews': 'all P1, P2 and N5 answers plus any answer where the arms differ on an automated check; the sheet lists every answer'},
        'reporting': {'granularity': 'every (case, repeat, arm) row is reported; summaries are per case and arm; no pooled "overall" pass/fail is produced',
                      'acceptance_cells': 'each criterion lists the exact failing (case, repeat, arm) cells'},
        'acceptance': {
            '1_no_hard_violations': 'revised arm: zero HARD grounding or misattribution verdicts in N1-N5 (15 answers) and not more than the current arm',
            '2_abstention': 'revised arm acceptable-abstention count in N1-N3 is not lower than the current arm, per case',
            '3_synthesis': 'P1 and P2 each: revised reaches completeness=all with derived content labelled in at least 2 of 3 repeats while the current arm reaches it in fewer than 2',
            '4_no_regression': 'P3, C1, C2: revised "all" count is not lower than the current count; over-labelling is reported, not failed',
            '5_hygiene': 'no answer over 250 words, no provider errors, no malformed-marker rejections',
            'decision': 'adopt only if all five hold; otherwise do not adopt. One wording revision is allowed but then needs fresh cases (these would be contaminated).'},
        'pins': {'production_code': {p: sha256_file(os.path.join(root, p)) for p in production_pin_paths(root)},
                 'dataset': {f: sha256_file(os.path.join(root, DATASET, f)) for f in PIN_FILES_DATASET},
                 'demo_transcript': {DEMO_TRANSCRIPT: sha256_file(os.path.join(root, DEMO_TRANSCRIPT))}},
    }


def load_registration(path=REG_PATH, expected=REG_SHA256):
    if sha256_file(path) != expected:
        raise VerificationError('config_b.json differs from the hash pinned in the harness')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def verify_all(reg, root=REPO):
    """Offline verification: pins, instruction hashes, ASK_* unset, and identical registered evidence for both arms in every case."""
    problems = []
    for group in ('production_code', 'dataset', 'demo_transcript'):
        base = {'production_code': '', 'dataset': DATASET + '/', 'demo_transcript': ''}[group]
        for rel, expected in reg['pins'][group].items():
            path = os.path.join(root, base + rel)
            actual = sha256_file(path) if os.path.exists(path) else None
            if actual != expected:
                problems.append(f'pin {rel}: expected {expected[:12]}..., got {"MISSING" if actual is None else actual[:12] + "..."}')
    if problems:
        raise VerificationError('pinned input changed:\n  ' + '\n  '.join(problems))
    S, _, _ = _qa()
    orig = S.SYSTEM_INSTRUCTION
    rev = build_revised(orig)
    if sha256_text(orig) != reg['instruction']['current_sha256'] or sha256_text(rev) != reg['instruction']['revised_sha256']:
        problems.append('instruction hash differs from the registration')
    for c in reg['cases']:
        p_cur, _, mode, _ = offline_prompt(c, orig, root)
        p_rev, _, _, _ = offline_prompt(c, rev, root)
        if p_cur != p_rev:
            problems.append(f'{c["id"]}: user prompt differs between arms')
        if sha256_text(p_cur) != c['user_prompt_sha256']:
            problems.append(f'{c["id"]}: user prompt hash differs from the registration')
        if mode != c['evidence_mode']:
            problems.append(f'{c["id"]}: evidence mode {mode} != registered {c["evidence_mode"]}')
    if problems:
        raise VerificationError('verification failed:\n  ' + '\n  '.join(problems))
    return {'cases_verified': len(reg['cases']), 'pins_verified': sum(len(v) for v in reg['pins'].values())}


# ------------------------------------------------------------------------------------------------- schedule and pacing
def make_schedule(reg, seed=SEED):
    rng = random.Random(seed)
    ids = [c['id'] for c in reg['cases']]
    items, n = [], 0
    for rnd in range(1, reg['settings']['repeats'] + 1):
        order = ids[:]
        rng.shuffle(order)
        for cid in order:
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                n += 1
                items.append({'order': n, 'repeat': rnd, 'case': cid, 'arm': arm})
    return items


class Pacer:
    def __init__(self, min_interval=MIN_INTERVAL_SECONDS, clock=time.monotonic, sleep=time.sleep):
        self.min_interval, self.clock, self.sleep, self.last = min_interval, clock, sleep, None

    def wait(self):
        now = self.clock()
        if self.last is not None and now - self.last < self.min_interval:
            self.sleep(self.min_interval - (now - self.last))
        self.last = self.clock()


class GuardedProvider:
    """Wraps the real provider: pacing, request cap, identical-evidence gate, instruction gate, usage and latency capture. No retries."""

    def __init__(self, inner, pacer, max_requests):
        self.inner, self.pacer, self.max_requests = inner, pacer, max_requests
        self.name, self.model = getattr(inner, 'name', 'inner'), getattr(inner, 'model', '')
        self.expect_prompt_sha = self.expect_instruction_sha = None
        self.count, self.last = 0, None

    def expect(self, prompt_sha, instruction_sha):
        self.expect_prompt_sha, self.expect_instruction_sha = prompt_sha, instruction_sha

    def generate(self, prompt, settings=None):
        if self.count >= self.max_requests:
            raise VerificationError('request cap reached')
        if sha256_text(prompt) != self.expect_prompt_sha:
            raise VerificationError('user prompt (evidence) differs from the registration; request not sent')
        if sha256_text(getattr(settings, 'system_instruction', '') or '') != self.expect_instruction_sha:
            raise VerificationError('system instruction differs from the arm being tested; request not sent')
        self.pacer.wait()
        self.count += 1
        result = self.inner.generate(prompt, settings)
        self.last = result
        return result


def load_api_environment():
    """Called ONLY by `run`. Loads .env into this process; returns booleans, never values."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, '.env'))
    return {k: bool(os.environ.get(k)) for k in ('LLM_PROVIDER', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_TIMEOUT_SECONDS')}


def run_live(reg, out_dir, approved_requests, env_loader=load_api_environment, provider_factory=None, pacer=None):
    """MODEL CALLS. Refuses unless the approval number equals the registered request count; verifies everything offline first."""
    schedule = make_schedule(reg)
    if approved_requests != len(schedule):
        raise LiveRunRefused(f'--approve-live-requests must equal {len(schedule)}; got {approved_requests}')
    if os.path.exists(out_dir):
        raise LiveRunRefused(f'output directory exists: {out_dir}')
    verify_all(reg)                                           # offline gate first; .env is not read before this succeeds
    present = env_loader()
    if not all(present.values()):
        raise LiveRunRefused(f'missing environment settings: {[k for k, v in present.items() if not v]}')
    if os.environ.get('LLM_PROVIDER') != 'gemini' or os.environ.get('GEMINI_MODEL') != reg['settings']['model']:
        raise LiveRunRefused('provider/model differ from the registration')
    S, _, _ = _qa()
    from llm_interface.providers import LLMProviderError
    inner = provider_factory() if provider_factory else S.get_text_provider()
    guard = GuardedProvider(inner, pacer or Pacer(), len(schedule))
    orig = S.SYSTEM_INSTRUCTION
    texts = {'current': orig, 'revised': build_revised(orig)}
    cases = {c['id']: c for c in reg['cases']}
    os.makedirs(out_dir)
    raw_path = os.path.join(out_dir, 'raw_results.jsonl')
    done = 0
    for item in schedule:
        c = cases[item['case']]
        question, segs = load_case_inputs(c)
        guard.expect(c['user_prompt_sha256'], sha256_text(texts[item['arm']]))
        rec = {**item, 'question': question, 'prompt_sha256': c['user_prompt_sha256'], 'instruction_sha256': sha256_text(texts[item['arm']])}
        t0 = time.perf_counter()
        try:
            with instruction(S, texts[item['arm']]):
                out = S.AskService(provider=guard, config=S.AskConfig()).answer(question=question, segments=segs)
            u = out.usage
            rec.update({'status': out.status, 'scope': out.scope, 'answer_text': out.answer_text, 'sources': [[s.start_seconds, s.end_seconds] for s in out.sources],
                        'estimated_prompt_tokens': out.estimated_prompt_tokens, 'wall_seconds': round(time.perf_counter() - t0, 3), 'error': None,
                        'usage': {k: getattr(u, k, None) for k in ('prompt_tokens', 'output_tokens', 'thoughts_tokens', 'total_tokens', 'cached_tokens')} if u else None,
                        'provider_latency_ms': getattr(guard.last, 'latency_ms', None)})
        except (LLMProviderError, VerificationError) as e:
            rec.update({'status': 'ERROR', 'answer_text': '', 'sources': [], 'error': f'{type(e).__name__}: {str(e)[:200]}', 'wall_seconds': round(time.perf_counter() - t0, 3)})
        with open(raw_path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        done += 1
        if rec['status'] == 'ERROR':
            break                                              # stop rule: no retries, keep partial results
    manifest = {'registration_sha256': REG_SHA256, 'requests_sent': guard.count, 'items_recorded': done, 'items_planned': len(schedule), 'stopped_early': done < len(schedule),
                'raw_results_sha256': sha256_file(raw_path), 'harness_sha256': sha256_file(os.path.abspath(__file__))}
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ------------------------------------------------------------------------------------------------- automated checks
def _sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


def n5_strict(text):
    """Advisory flags for case N5. Human review of every N5 answer is still required."""
    out = {'H1_misattribution': False, 'H2_product_rule_present': None, 'H3_missing_acknowledgement': False, 'H4_unlabelled_derivation': False, 'matched_sentences': []}
    low = text.lower()
    formula = re.compile('|'.join(N5_FORMULA_PATTERNS), re.I)
    for s in _sentences(text):
        if formula.search(s):
            out['matched_sentences'].append(s[:200])
            if re.search(N5_ATTRIBUTION, s, re.I):
                out['H1_misattribution'] = True
            outside = re.search(N5_OUTSIDE_MARKERS, s, re.I) is not None
            if out['H2_product_rule_present'] != 'HARD':
                out['H2_product_rule_present'] = 'SOFT' if outside else 'HARD'
    if not acknowledges_not_spoken(text):
        out['H3_missing_acknowledgement'] = True
    derived = re.search(r"integral of u\s*d\s*v|integral of u dv|u\s*d\s*v equals|u v minus|uv minus|minus the integral of v", low) is not None
    if derived and not re.search(DERIVED_LABELS, low):
        out['H4_unlabelled_derivation'] = True
    out['hard'] = bool(out['H1_misattribution'] or out['H2_product_rule_present'] == 'HARD')
    out['fail'] = bool(out['hard'] or out['H3_missing_acknowledgement'] or out['H4_unlabelled_derivation'])
    return out


def auto_checks(case, rec):
    text = rec.get('answer_text') or ''
    low = text.lower()
    words = len(text.split())
    flags = {'status_ok': rec.get('status') in case['acceptable_statuses'], 'status': rec.get('status'), 'words': words, 'over_150': words > 150, 'over_250': words > 250,
             'error': bool(rec.get('error')), 'sources_ok': (bool(rec.get('sources')) == (rec.get('status') == 'answered')),
             'math_symbols': bool(re.search(r'[∫=^\\$∑√]', text)), 'timestamp_in_text': bool(re.search(r'\b\d{1,2}:\d{2}\b', text)),
             'markdown_list': bool(re.search(r'^\s*(?:[-*•]|\d+[.)])\s', text, re.M)), 'derived_label': bool(re.search(DERIVED_LABELS, low))}
    cid = case['id']
    review, hard = [], False
    if 'n1' in case['checks'] and re.search(r'posterior|prior|likelihood|conditional probability', low):
        review.append('bayes_terms')
    if 'n2' in case['checks'] and (re.search(r'radius of convergence (?:is|equals)|ratio test|converges for|\bradius\b[^.]{0,40}\d', low) or flags['derived_label']):
        review.append('radius_claim_or_derived_label')
    if 'n3' in case['checks'] and re.search(r'angle addition|limit as h|h approaches zero|sine of x plus h|sin\(x\s*\+\s*h\)', low):
        review.append('limit_proof_language')
    if 'n4' in case['checks']:
        sees = re.search(r"(?:cannot|can't|can not|unable to|do not|don't) (?:see|view|read) (?:the )?(?:screen|slide|display)|no (?:access to|view of) the (?:screen|slide)", low) is not None
        flags['admits_cannot_see'] = sees
        if not sees:
            hard = True
            review.append('n4_missing_cannot_see')
    if 'n5' in case['checks']:
        n5 = n5_strict(text)
        flags['n5'] = n5
        hard = hard or n5['hard']
        if n5['fail']:
            review.append('n5_fail')
    flags['review_reasons'] = review
    flags['auto_hard'] = hard
    return flags


# ------------------------------------------------------------------------------------------------- blinded sheets
SHEET_COLUMNS = ['row_id', 'case_id', 'repeat', 'label', 'question', 'expected_content', 'expectation_status', 'acceptable_statuses', 'answer_text', 'status_returned',
                 'completeness', 'expected_points_met', 'grounding', 'attribution', 'labelled_derived', 'abstention',
                 'n5_product_rule_stated_as_instructor', 'n5_product_rule_present_any', 'n5_says_formula_not_spoken', 'notes', 'scorer_id']


def _sheet_row(case, repeat, label, answer=None):
    return {'row_id': f'{case["id"]}-r{repeat}-{label}', 'case_id': case['id'], 'repeat': repeat, 'label': label, 'question': case['question'],
            'expected_content': ' | '.join(case['expected_content']), 'expectation_status': case['expectation_status'], 'acceptable_statuses': '/'.join(case['acceptable_statuses']),
            'answer_text': (answer or {}).get('answer_text', ''), 'status_returned': (answer or {}).get('status', ''),
            **{k: '' for k in SHEET_COLUMNS if k not in ('row_id', 'case_id', 'repeat', 'label', 'question', 'expected_content', 'expectation_status', 'acceptable_statuses', 'answer_text', 'status_returned')}}


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def template_sheet(reg, path):
    rows = [_sheet_row(c, r, lab) for c in reg['cases'] for r in range(1, reg['settings']['repeats'] + 1) for lab in ('X', 'Y')]
    write_csv(path, rows)
    return len(rows)


def make_sheet(reg, raw, sheet_path, key_path, seed=SEED):
    """Blinded sheet: per (case, repeat) the two arms get labels X/Y by a seeded coin flip; the arm key goes to a SEPARATE file."""
    rng = random.Random(seed + 1)
    by = {(r['case'], r['repeat'], r['arm']): r for r in raw}
    rows, key = [], {}
    for c in reg['cases']:
        for rep in range(1, reg['settings']['repeats'] + 1):
            arms = list(ARMS)
            rng.shuffle(arms)
            for label, arm in zip(('X', 'Y'), arms):
                ans = by.get((c['id'], rep, arm))
                row = _sheet_row(c, rep, label, ans)
                rows.append(row)
                key[row['row_id']] = arm
    write_csv(sheet_path, rows)
    with open(key_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'WARNING': 'do not open until scoring is finished', 'seed': seed + 1, 'key': key}, f, indent=2)
    return len(rows)


def read_scores(sheet_path, key_path):
    """Human scores keyed by (case, repeat, arm), using the separate key."""
    with open(key_path, encoding='utf-8') as f:
        key = json.load(f)['key']
    out = {}
    for r in _rd(sheet_path):
        out[(r['case_id'], int(r['repeat']), key[r['row_id']])] = r
    return out


# ------------------------------------------------------------------------------------------------- reporting and acceptance
def per_cell_rows(reg, raw, human=None):
    cases = {c['id']: c for c in reg['cases']}
    rows = []
    for r in sorted(raw, key=lambda x: (x['case'], x['repeat'], x['arm'])):
        a = auto_checks(cases[r['case']], r)
        h = (human or {}).get((r['case'], r['repeat'], r['arm']), {})
        rows.append({'case': r['case'], 'repeat': r['repeat'], 'arm': r['arm'], 'status': a['status'], 'status_ok': a['status_ok'], 'words': a['words'], 'derived_label': a['derived_label'],
                     'auto_hard': a['auto_hard'], 'review_reasons': ';'.join(a['review_reasons']), 'error': a['error'], 'over_250': a['over_250'], 'math_symbols': a['math_symbols'],
                     'completeness': h.get('completeness', ''), 'grounding': h.get('grounding', ''), 'attribution': h.get('attribution', ''), 'abstention': h.get('abstention', ''),
                     'labelled_derived': h.get('labelled_derived', '')})
    return rows


def summarize_by_case(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r['case'], {}).setdefault(r['arm'], collections.Counter())
        d['cells'] += 1
        d['status_ok'] += int(r['status_ok'])
        d['auto_hard'] += int(r['auto_hard'])
        d['derived_label'] += int(r['derived_label'])
        d['errors'] += int(r['error'])
        d['completeness_all'] += int(r['completeness'] == 'all')
        d['grounding_hard'] += int(r['grounding'] == 'hard')
        d['misattributed'] += int(r['attribution'] == 'misattributed')
        d['abstention_ok'] += int(r['abstention'] == 'acceptable')
    return {c: {a: dict(v) for a, v in arms.items()} for c, arms in out.items()}


def evaluate_acceptance(rows):
    """Apply the five registered criteria to scored per-cell rows. Returns a list of {criterion, passed, detail} with the exact failing cells."""
    def cells(case_ids, arm):
        return [r for r in rows if r['case'] in case_ids and r['arm'] == arm]
    res = []
    n_cases = ['N1', 'N2', 'N3', 'N4', 'N5']
    scored = all(r['grounding'] for r in rows if r['case'] in n_cases)
    cur_hard = sum(1 for r in cells(n_cases, 'current') if r['grounding'] == 'hard' or r['attribution'] == 'misattributed')
    rev = cells(n_cases, 'revised')
    rev_hard = [(r['case'], r['repeat']) for r in rev if r['grounding'] == 'hard' or r['attribution'] == 'misattributed']
    reconcile = [(r['case'], r['repeat']) for r in rev if r['auto_hard'] and r['grounding'] == 'ok']
    res.append({'criterion': '1_no_hard_violations', 'passed': bool(scored and not rev_hard and len(rev_hard) <= cur_hard and not reconcile),
                'detail': {'scored': scored, 'revised_hard_cells': rev_hard, 'current_hard_count': cur_hard, 'needs_written_reconciliation': reconcile}})
    ab = {}
    for cid in ('N1', 'N2', 'N3'):
        cur = sum(1 for r in cells([cid], 'current') if r['abstention'] == 'acceptable')
        rv = sum(1 for r in cells([cid], 'revised') if r['abstention'] == 'acceptable')
        ab[cid] = {'current': cur, 'revised': rv}
    res.append({'criterion': '2_abstention', 'passed': all(v['revised'] >= v['current'] for v in ab.values()) and all(r['abstention'] for r in rows if r['case'] in ('N1', 'N2', 'N3')),
                'detail': ab})
    syn = {}
    for cid in ('P1', 'P2'):
        def ok(arm):
            return sum(1 for r in cells([cid], arm) if r['completeness'] == 'all' and r['labelled_derived'] in ('yes', 'na'))
        syn[cid] = {'current': ok('current'), 'revised': ok('revised')}
    res.append({'criterion': '3_synthesis', 'passed': all(v['revised'] >= 2 and v['current'] < 2 for v in syn.values()), 'detail': syn})
    reg_ = {cid: {'current': sum(1 for r in cells([cid], 'current') if r['completeness'] == 'all'), 'revised': sum(1 for r in cells([cid], 'revised') if r['completeness'] == 'all')}
            for cid in ('P3', 'C1', 'C2')}
    res.append({'criterion': '4_no_regression', 'passed': all(v['revised'] >= v['current'] for v in reg_.values()),
                'detail': {**reg_, 'over_labelled_cells_revised_C1_C2': sum(1 for r in cells(['C1', 'C2'], 'revised') if r['derived_label'])}})
    long_ = [(r['case'], r['repeat'], r['arm']) for r in rows if r['over_250']]
    errs = [(r['case'], r['repeat'], r['arm']) for r in rows if r['error']]
    res.append({'criterion': '5_hygiene', 'passed': not long_ and not errs, 'detail': {'over_250_words': long_, 'errors': errs}})
    return res


def report(reg, raw, human=None):
    rows = per_cell_rows(reg, raw, human)
    out = {'cells': rows, 'per_case_arm': summarize_by_case(rows), 'cells_reported': len(rows), 'pooled_overall_verdict': None}
    if human:
        out['acceptance'] = evaluate_acceptance(rows)
        out['decision'] = 'ADOPT' if all(c['passed'] for c in out['acceptance']) else 'DO NOT ADOPT'
    else:
        out['acceptance'] = 'not evaluated: human scores are required'
    return out


# ------------------------------------------------------------------------------------------------- CLI
def _load_raw(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else ''
    if cmd == 'register':
        if os.path.exists(REG_PATH):
            print('REFUSING: config_b.json already exists', file=sys.stderr)
            return 2
        reg = build_registration()
        with open(REG_PATH, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print('written', REG_PATH, sha256_file(REG_PATH))
        return 0
    reg = load_registration()
    if cmd == 'verify':
        print(verify_all(reg))
        return 0
    if cmd == 'plan':
        for it in make_schedule(reg):
            print(it)
        return 0
    if cmd == 'template-sheet':
        print(template_sheet(reg, a[1]), 'rows')
        return 0
    if cmd == 'make-sheet':
        print(make_sheet(reg, _load_raw(a[1]), a[2], a[3]), 'rows')
        return 0
    if cmd == 'report':
        human = read_scores(a[a.index('--sheet') + 1], a[a.index('--key') + 1]) if '--sheet' in a else None
        print(json.dumps(report(reg, _load_raw(a[1]), human), indent=1))
        return 0
    if cmd == 'run':
        try:
            n = int(a[a.index('--approve-live-requests') + 1]) if '--approve-live-requests' in a else -1
            print(json.dumps(run_live(reg, a[a.index('--out') + 1], n), indent=1))
        except LiveRunRefused as e:
            print(f'REFUSED: {e}', file=sys.stderr)
            return 2
        return 0
    print(__doc__)
    return 1


if __name__ == '__main__':
    sys.exit(main())
