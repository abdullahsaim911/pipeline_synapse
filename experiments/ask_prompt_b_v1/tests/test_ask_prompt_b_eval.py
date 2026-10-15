"""Offline tests for ask_prompt_b_eval.py. No model call, no network, no .env access (the env loader is replaced by a fake or a tripwire).

    python -B -m unittest discover -s experiments/ask_prompt_b_v1/tests -p "test_*.py"
"""
import copy, json, os, sys, tempfile, unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, EXP)
sys.path.insert(0, REPO)

import ask_prompt_b_eval as H  # noqa: E402
from llm_interface.providers import GenerationResult, LLMRateLimitError  # noqa: E402

REG = H.load_registration()
CASES = {c['id']: c for c in REG['cases']}


def tripwire(*a, **k):
    raise AssertionError('the .env loader must not be called by this code path')


class FakeInner:
    name, model = 'fake', 'fake-model'

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def generate(self, prompt, settings=None):
        self.calls += 1
        if self.fail_at and self.calls == self.fail_at:
            raise LLMRateLimitError('quota', provider='fake')
        return GenerationResult(text='The instructor says this.', provider='fake', model='fake-model', latency_ms=5, usage=None)


def good_text(cid):
    return {'N4': 'I cannot see the screen. The instructor begins by introducing integration by parts.',
            'N5': 'The captions mention the product rule but the lecture does not state its formula. From these steps, it seems that the result follows from moving the integral across.'}.get(cid, 'The instructor says this.')


def fake_raw(text_fn=good_text, drop=None):
    out = []
    for it in H.make_schedule(REG):
        c = CASES[it['case']]
        status = c['acceptable_statuses'][0]
        out.append({**it, 'status': status, 'answer_text': text_fn(it['case']), 'sources': [[1.0, 2.0]] if status == 'answered' else [], 'error': None})
    return [r for r in out if not (drop and drop(r))]


def good_human(raw, current_partial=('P1', 'P2')):
    h = {}
    for r in raw:
        arm_partial = r['arm'] == 'current' and r['case'] in current_partial
        h[(r['case'], r['repeat'], r['arm'])] = {'completeness': 'partial' if arm_partial else 'all', 'grounding': 'ok', 'attribution': 'ok',
                                                 'abstention': 'acceptable' if r['case'] in ('N1', 'N2', 'N3') else 'na', 'labelled_derived': 'yes'}
    return h


class TestRegistrationAndPins(unittest.TestCase):
    def test_registration_hash_gate(self):
        with self.assertRaises(H.VerificationError):
            H.load_registration(expected='0' * 64)

    def test_shape(self):
        self.assertEqual((len(REG['cases']), REG['settings']['requests_total'], REG['settings']['repeats']), (10, 60, 3))
        self.assertEqual([c['id'] for c in REG['cases']], ['P1', 'P2', 'P3', 'C1', 'C2', 'N1', 'N2', 'N3', 'N4', 'N5'])

    def test_expectation_statuses_are_honest(self):
        self.assertIn('PROVISIONAL', CASES['P1']['expectation_status'])
        for cid in ('P3', 'C1', 'C2'):
            self.assertIn('UNVERIFIED', CASES[cid]['expectation_status'])
        self.assertIn('not independent', CASES['P1']['role'].lower() + ' ' + ' '.join(REG['honesty_notes']).lower())

    def test_verify_all_passes_on_real_inputs(self):
        self.assertEqual(H.verify_all(REG)['cases_verified'], 10)

    def test_changed_pin_is_detected(self):
        bad = copy.deepcopy(REG)
        first = next(iter(bad['pins']['production_code']))
        bad['pins']['production_code'][first] = 'f' * 64
        with self.assertRaises(H.VerificationError) as cm:
            H.verify_all(bad)
        self.assertIn(first, str(cm.exception))

    def test_changed_case_prompt_hash_is_detected(self):
        bad = copy.deepcopy(REG)
        bad['cases'][0]['user_prompt_sha256'] = '0' * 64
        with self.assertRaises(H.VerificationError):
            H.verify_all(bad)

    def test_changed_instruction_hash_is_detected(self):
        bad = copy.deepcopy(REG)
        bad['instruction']['current_sha256'] = '0' * 64
        with self.assertRaises(H.VerificationError):
            H.verify_all(bad)

    def test_ask_environment_variable_aborts(self):
        with mock.patch.dict(os.environ, {'ASK_MAX_PROMPT_TOKENS': '9999'}):
            with self.assertRaises(H.VerificationError):
                H.verify_all(REG)


class TestInstructionConstruction(unittest.TestCase):
    def setUp(self):
        self.S, _, _ = H._qa()
        self.orig = self.S.SYSTEM_INSTRUCTION

    def test_only_rule_three_changes_and_exactly_once(self):
        rev = H.build_revised(self.orig)
        self.assertEqual(rev.count(H.ADDED_SENTENCE), 1)
        diff = [i for i, (a, b) in enumerate(zip(self.orig.splitlines(), rev.splitlines())) if a != b]
        self.assertEqual(len(diff), 1)
        self.assertTrue(rev.splitlines()[diff[0]].startswith('3. '))
        self.assertEqual(rev.replace(H.ADDED_SENTENCE, ''), self.orig)

    def test_missing_anchor_is_refused(self):
        with self.assertRaises(H.VerificationError):
            H.build_revised(self.orig.replace('answer that part', 'answer the part'))

    def test_context_manager_restores_even_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with H.instruction(self.S, 'temporary'):
                self.assertEqual(self.S.SYSTEM_INSTRUCTION, 'temporary')
                raise RuntimeError('boom')
        self.assertEqual(self.S.SYSTEM_INSTRUCTION, self.orig)

    def test_registered_hashes_match(self):
        self.assertEqual(H.sha256_text(self.orig), REG['instruction']['current_sha256'])
        self.assertEqual(H.sha256_text(H.build_revised(self.orig)), REG['instruction']['revised_sha256'])

    def test_evidence_identical_between_arms_for_every_case(self):
        rev = H.build_revised(self.orig)
        for c in REG['cases']:
            a = H.offline_prompt(c, self.orig)[0]
            b = H.offline_prompt(c, rev)[0]
            self.assertEqual(a, b, c['id'])
            self.assertEqual(H.sha256_text(a), c['user_prompt_sha256'], c['id'])


class TestScheduleAndGuards(unittest.TestCase):
    def test_schedule_is_complete_and_deterministic(self):
        s = H.make_schedule(REG)
        self.assertEqual(len(s), 60)
        self.assertEqual(len({(i['case'], i['repeat'], i['arm']) for i in s}), 60)
        self.assertEqual(s, H.make_schedule(REG))
        self.assertNotEqual(s, H.make_schedule(REG, seed=1))
        for rnd in (1, 2, 3):
            self.assertEqual(sorted(i['case'] for i in s if i['repeat'] == rnd and i['arm'] == 'current'), sorted(CASES))
        firsts = {s[k]['arm'] for k in range(0, 60, 2)}
        self.assertEqual(firsts, {'current', 'revised'})                     # arm order is randomised, not fixed

    def test_pacer_enforces_the_minimum_gap(self):
        t = {'now': 0.0}
        slept = []
        p = H.Pacer(6.0, clock=lambda: t['now'], sleep=lambda s: (slept.append(s), t.__setitem__('now', t['now'] + s)))
        p.wait()
        t['now'] += 2.0
        p.wait()
        t['now'] += 10.0
        p.wait()
        self.assertEqual(slept, [4.0])

    def test_guard_blocks_wrong_prompt_without_calling_the_provider(self):
        inner = FakeInner()
        g = H.GuardedProvider(inner, H.Pacer(0), 5)
        S, _, _ = H._qa()
        g.expect(H.sha256_text('expected prompt'), H.sha256_text(S.SYSTEM_INSTRUCTION))
        with self.assertRaises(H.VerificationError):
            g.generate('a different prompt', S.GenerationSettings(system_instruction=S.SYSTEM_INSTRUCTION))
        self.assertEqual(inner.calls, 0)

    def test_guard_blocks_wrong_instruction_and_enforces_the_cap(self):
        S, _, _ = H._qa()
        inner = FakeInner()
        g = H.GuardedProvider(inner, H.Pacer(0), 1)
        g.expect(H.sha256_text('p'), H.sha256_text('instr'))
        with self.assertRaises(H.VerificationError):
            g.generate('p', S.GenerationSettings(system_instruction='other'))
        g.generate('p', S.GenerationSettings(system_instruction='instr'))
        with self.assertRaises(H.VerificationError):
            g.generate('p', S.GenerationSettings(system_instruction='instr'))
        self.assertEqual((inner.calls, g.count), (1, 1))


class TestLiveRunGating(unittest.TestCase):
    ENV = {'LLM_PROVIDER': 'gemini', 'GEMINI_MODEL': 'gemini-3.5-flash-lite'}

    def test_refused_before_the_env_loader_when_approval_is_wrong(self):
        with tempfile.TemporaryDirectory() as d:
            for n in (0, 59, 61, -1):
                with self.assertRaises(H.LiveRunRefused):
                    H.run_live(REG, os.path.join(d, 'out'), n, env_loader=tripwire)

    def test_refused_when_output_exists(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(H.LiveRunRefused):
                H.run_live(REG, d, 60, env_loader=tripwire)

    def test_no_offline_command_touches_the_env_loader(self):
        with mock.patch.object(H, 'load_api_environment', tripwire):
            H.verify_all(REG)
            H.make_schedule(REG)
            with tempfile.TemporaryDirectory() as d:
                H.template_sheet(REG, os.path.join(d, 't.csv'))
                H.make_sheet(REG, fake_raw(), os.path.join(d, 's.csv'), os.path.join(d, 'k.json'))
            H.report(REG, fake_raw())

    def test_refused_when_provider_or_model_differs(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'LLM_PROVIDER': 'gemini', 'GEMINI_MODEL': 'some-other-model'}):
            with self.assertRaises(H.LiveRunRefused):
                H.run_live(REG, os.path.join(d, 'out'), 60, env_loader=lambda: {k: True for k in ('LLM_PROVIDER', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_TIMEOUT_SECONDS')},
                           provider_factory=FakeInner)

    def test_refused_when_a_setting_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(H.LiveRunRefused):
                H.run_live(REG, os.path.join(d, 'out'), 60, env_loader=lambda: {'GEMINI_API_KEY': False}, provider_factory=FakeInner)

    def test_end_to_end_with_a_fake_provider_writes_60_verified_records(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, self.ENV):
            out = os.path.join(d, 'out')
            inner = FakeInner()
            m = H.run_live(REG, out, 60, env_loader=lambda: {k: True for k in ('LLM_PROVIDER', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_TIMEOUT_SECONDS')},
                           provider_factory=lambda: inner, pacer=H.Pacer(0))
            self.assertEqual((m['requests_sent'], m['items_recorded'], m['stopped_early']), (60, 60, False))
            raw = [json.loads(l) for l in open(os.path.join(out, 'raw_results.jsonl'), encoding='utf-8')]
            self.assertEqual(len(raw), 60)
            for r in raw:
                self.assertEqual(r['prompt_sha256'], CASES[r['case']]['user_prompt_sha256'])
                self.assertEqual(r['instruction_sha256'], REG['instruction']['current_sha256' if r['arm'] == 'current' else 'revised_sha256'])
                self.assertIsNone(r['error'])
            self.assertNotIn('GEMINI_API_KEY', json.dumps(raw))

    def test_stops_at_the_first_provider_error_and_keeps_partial_results(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, self.ENV):
            out = os.path.join(d, 'out')
            m = H.run_live(REG, out, 60, env_loader=lambda: {k: True for k in ('LLM_PROVIDER', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_TIMEOUT_SECONDS')},
                           provider_factory=lambda: FakeInner(fail_at=5), pacer=H.Pacer(0))
            self.assertTrue(m['stopped_early'])
            self.assertEqual(m['items_recorded'], 5)
            raw = [json.loads(l) for l in open(os.path.join(out, 'raw_results.jsonl'), encoding='utf-8')]
            self.assertEqual([r['status'] for r in raw][-1], 'ERROR')
            self.assertIn('LLMRateLimitError', raw[-1]['error'])


class TestAutomatedChecks(unittest.TestCase):
    def rec(self, text, status='answered', sources=True):
        return {'status': status, 'answer_text': text, 'sources': [[1, 2]] if sources else [], 'error': None}

    def test_common_checks(self):
        a = H.auto_checks(CASES['C2'], self.rec('word ' * 160 + '3:20 x = y'))
        self.assertTrue(a['over_150'] and not a['over_250'] and a['timestamp_in_text'] and a['math_symbols'])
        self.assertTrue(H.auto_checks(CASES['C2'], self.rec('word ' * 260))['over_250'])
        self.assertTrue(H.auto_checks(CASES['C2'], self.rec('- item one\n- item two'))['markdown_list'])
        self.assertFalse(H.auto_checks(CASES['C2'], self.rec('fine', status='answered', sources=False))['sources_ok'])
        self.assertTrue(H.auto_checks(CASES['N1'], self.rec('nothing found', status='insufficient_context', sources=False))['sources_ok'])
        self.assertFalse(H.auto_checks(CASES['N1'], self.rec('x', status='answered'))['status_ok'])        # N1 allows only insufficient_context

    def test_derived_label_detection(self):
        self.assertTrue(H.auto_checks(CASES['P1'], self.rec('From these steps, it seems that u dv equals u v.'))['derived_label'])
        self.assertTrue(H.auto_checks(CASES['P1'], self.rec('From this, it seems that it works.'))['derived_label'])
        self.assertFalse(H.auto_checks(CASES['P1'], self.rec('The instructor says it works.'))['derived_label'])

    def test_n4_requires_a_cannot_see_statement(self):
        bad = H.auto_checks(CASES['N4'], self.rec('The slide shows the title Integration By Parts.'))
        self.assertTrue(bad['auto_hard'] and not bad['admits_cannot_see'])
        good = H.auto_checks(CASES['N4'], self.rec("I can't see the screen, but the instructor starts by introducing the topic."))
        self.assertFalse(good['auto_hard'])

    def test_negative_absence_review_flags(self):
        self.assertIn('bayes_terms', H.auto_checks(CASES['N1'], self.rec('The posterior is the prior times the likelihood.', 'insufficient_context', False))['review_reasons'])
        self.assertIn('radius_claim_or_derived_label', H.auto_checks(CASES['N2'], self.rec('The radius of convergence is 3.'))['review_reasons'])
        self.assertIn('radius_claim_or_derived_label', H.auto_checks(CASES['N2'], self.rec('From these steps, it seems that it converges.'))['review_reasons'])
        self.assertIn('limit_proof_language', H.auto_checks(CASES['N3'], self.rec('Use the angle addition formula and let h approaches zero.'))['review_reasons'])
        self.assertEqual(H.auto_checks(CASES['N3'], self.rec('The lecture only cites this result without a proof.', 'insufficient_context', False))['review_reasons'], [])


class TestN5Strict(unittest.TestCase):
    GOOD = ('The captions mention the formula of the product rule but the lecture does not state it. '
            'From these steps, it seems that moving the integral of v du to the other side gives the result.')

    def test_good_answer_passes(self):
        r = H.n5_strict(self.GOOD)
        self.assertFalse(r['fail'])
        self.assertFalse(r['hard'])

    def test_common_phrasings_of_the_acknowledgement_are_recognised(self):
        for ok in ('The lecture never states the product rule formula.', 'The product rule formula is not stated in the lecture.',
                   'The instructor only refers to the product rule without giving its formula.', 'The captions do not spell out the product rule.',
                   "The instructor doesn't read the product rule aloud."):
            self.assertTrue(H.acknowledges_not_spoken(ok), ok)
        for bad in ('The product rule is the basis of integration by parts.', 'The instructor explains the product rule clearly.',
                    'The lecture does not state the answer.', 'Integration by parts reverses it.'):
            self.assertFalse(H.acknowledges_not_spoken(bad), bad)

    def test_mentioning_the_product_rule_alone_is_not_a_formula(self):
        self.assertIsNone(H.n5_strict('The instructor refers to the product rule.')['H2_product_rule_present'])

    def test_misattribution_is_hard(self):
        r = H.n5_strict('The instructor states that the derivative of u times v equals u prime v plus u v prime. The lecture does not state the product rule elsewhere.')
        self.assertTrue(r['H1_misattribution'] and r['hard'])

    def test_unattributed_formula_is_hard_but_an_outside_marker_makes_it_soft(self):
        hard = H.n5_strict('The product rule says u prime v plus u v prime. The lecture does not state the product rule.')
        self.assertEqual(hard['H2_product_rule_present'], 'HARD')
        soft = H.n5_strict('From general knowledge, the product rule is u prime v plus u v prime. The lecture does not state the product rule.')
        self.assertEqual(soft['H2_product_rule_present'], 'SOFT')
        self.assertFalse(soft['hard'])

    def test_missing_acknowledgement_fails(self):
        r = H.n5_strict('From these steps, it seems that moving the integral gives the result.')
        self.assertTrue(r['H3_missing_acknowledgement'] and r['fail'])

    def test_unlabelled_derived_result_fails(self):
        r = H.n5_strict('The lecture does not state the product rule formula. The integral of u dv equals u v minus the integral of v du.')
        self.assertTrue(r['H4_unlabelled_derivation'] and r['fail'])

    def test_n5_flags_flow_into_auto_checks(self):
        a = H.auto_checks(CASES['N5'], {'status': 'answered', 'answer_text': 'The instructor says d of u v is u prime v plus u v prime.', 'sources': [[1, 2]], 'error': None})
        self.assertTrue(a['auto_hard'])
        self.assertIn('n5_fail', a['review_reasons'])


class TestBlindedSheets(unittest.TestCase):
    def test_template_has_60_rows_and_no_arm_names(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.csv')
            self.assertEqual(H.template_sheet(REG, p), 60)
            text = open(p, encoding='utf-8').read().lower()
            self.assertNotIn('revised', text.replace('revised arm', ''))
            self.assertNotIn('current', text)
            rows = H._rd(p)
            self.assertEqual(sorted({r['label'] for r in rows}), ['X', 'Y'])
            self.assertEqual(len({r['row_id'] for r in rows}), 60)

    def test_make_sheet_blinds_balances_and_round_trips(self):
        raw = fake_raw(lambda cid: f'answer for {cid}')
        with tempfile.TemporaryDirectory() as d:
            sp, kp = os.path.join(d, 's.csv'), os.path.join(d, 'k.json')
            self.assertEqual(H.make_sheet(REG, raw, sp, kp), 60)
            text = open(sp, encoding='utf-8').read().lower()
            self.assertNotIn('revised', text)
            self.assertNotIn('"current"', text)
            key = json.load(open(kp, encoding='utf-8'))['key']
            for c in CASES:
                for rep in (1, 2, 3):
                    self.assertEqual({key[f'{c}-r{rep}-X'], key[f'{c}-r{rep}-Y']}, {'current', 'revised'})
            self.assertNotEqual(len({key[f'{c}-r1-X'] for c in CASES}), 1)              # X is not always the same arm
            H.make_sheet(REG, raw, os.path.join(d, 's2.csv'), os.path.join(d, 'k2.json'))
            self.assertEqual(H.sha256_file(sp), H.sha256_file(os.path.join(d, 's2.csv')))      # deterministic
            scores = H.read_scores(sp, kp)
            self.assertEqual(len(scores), 60)
            self.assertEqual(scores[('P1', 1, 'current')]['answer_text'], 'answer for P1')


class TestReportingAndAcceptance(unittest.TestCase):
    def test_report_lists_every_cell_and_no_pooled_verdict(self):
        rep = H.report(REG, fake_raw())
        self.assertEqual(rep['cells_reported'], 60)
        self.assertIsNone(rep['pooled_overall_verdict'])
        self.assertEqual(set(rep['per_case_arm']), set(CASES))
        self.assertTrue(all(set(v) == {'current', 'revised'} for v in rep['per_case_arm'].values()))
        self.assertIn('human scores are required', rep['acceptance'])

    def test_partial_runs_are_reported_cell_by_cell(self):
        rep = H.report(REG, fake_raw(drop=lambda r: r['repeat'] == 3))
        self.assertEqual(rep['cells_reported'], 40)

    def test_all_criteria_pass_on_a_clean_synthetic_run(self):
        raw = fake_raw()
        rep = H.report(REG, raw, good_human(raw))
        self.assertEqual(rep['decision'], 'ADOPT', [c for c in rep['acceptance'] if not c['passed']])

    def test_hard_violation_in_a_negative_case_blocks_adoption_and_names_the_cell(self):
        raw = fake_raw()
        h = good_human(raw)
        h[('N2', 2, 'revised')]['grounding'] = 'hard'
        rep = H.report(REG, raw, h)
        c1 = rep['acceptance'][0]
        self.assertFalse(c1['passed'])
        self.assertEqual(c1['detail']['revised_hard_cells'], [('N2', 2)])
        self.assertEqual(rep['decision'], 'DO NOT ADOPT')

    def test_misattribution_counts_as_hard(self):
        raw = fake_raw()
        h = good_human(raw)
        h[('N5', 1, 'revised')]['attribution'] = 'misattributed'
        self.assertFalse(H.report(REG, raw, h)['acceptance'][0]['passed'])

    def test_human_ok_against_an_automated_hard_flag_needs_written_reconciliation(self):
        raw = fake_raw(lambda cid: 'The instructor says d of u v is u prime v plus u v prime.' if cid == 'N5' else good_text(cid))
        h = good_human(raw)
        c1 = H.report(REG, raw, h)['acceptance'][0]
        self.assertFalse(c1['passed'])
        self.assertTrue(c1['detail']['needs_written_reconciliation'])

    def test_abstention_regression_is_caught(self):
        raw = fake_raw()
        h = good_human(raw)
        h[('N1', 1, 'revised')]['abstention'] = 'unacceptable'
        rep = H.report(REG, raw, h)
        self.assertFalse(rep['acceptance'][1]['passed'])
        self.assertEqual(rep['acceptance'][1]['detail']['N1'], {'current': 3, 'revised': 2})

    def test_synthesis_criterion_needs_two_of_three_and_a_weaker_baseline(self):
        raw = fake_raw()
        h = good_human(raw)
        h[('P1', 1, 'revised')]['completeness'] = 'partial'
        h[('P1', 2, 'revised')]['completeness'] = 'partial'
        self.assertFalse(H.report(REG, raw, h)['acceptance'][2]['passed'])
        h2 = good_human(raw, current_partial=())
        self.assertFalse(H.report(REG, raw, h2)['acceptance'][2]['passed'])                     # baseline already as good: no demonstrated gain

    def test_no_regression_and_hygiene(self):
        raw = fake_raw()
        h = good_human(raw)
        h[('C1', 1, 'revised')]['completeness'] = 'partial'
        self.assertFalse(H.report(REG, raw, h)['acceptance'][3]['passed'])
        long_raw = fake_raw(lambda cid: 'word ' * 300 if cid == 'C2' else good_text(cid))
        rep = H.report(REG, long_raw, good_human(long_raw))
        self.assertFalse(rep['acceptance'][4]['passed'])
        err_raw = fake_raw()
        err_raw[3]['error'] = 'LLMRateLimitError: quota'
        self.assertFalse(H.report(REG, err_raw, good_human(err_raw))['acceptance'][4]['passed'])

    def test_unscored_run_cannot_pass_criterion_one(self):
        raw = fake_raw()
        h = {k: {**v, 'grounding': ''} for k, v in good_human(raw).items()}
        self.assertFalse(H.report(REG, raw, h)['acceptance'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
