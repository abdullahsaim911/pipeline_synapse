"""Scorecard validation, review resolution, metrics and the gate verdict (synthetic data only)."""

import unittest

from tests.quiz_eval import fixtures as fx
from quiz_eval import scorecard as sc
from quiz_eval.constants import GATE, JUDGMENTS, SUBJECTS


def errors(data, manifest=None):
    return sc.validate_scorecard(data, manifest).errors


EXCL = {"reason": "transcript_defect", "notes": "caption gap", "excluded_by": "Ann", "confirmed_by": "Bo", "decided_at": "2000-01-01"}


def pair(first):
    """Two agreeing reviews by two people, given the first (a review built with the same judgments)."""
    second = fx.deep(first)
    second["reviewer"] = "reviewer-B"
    return [first, second]


def metrics(items, **kw):
    return sc.compute_metrics(fx.scorecard(items, **kw))


class ValidationTests(unittest.TestCase):
    def test_valid_scorecards(self):
        self.assertEqual(errors(fx.scorecard([fx.item()])), [])
        self.assertEqual(errors(fx.scorecard([fx.item(question_type="tf")])), [])
        self.assertEqual(errors(fx.scorecard([])), [])

    def test_values_read_back_from_json_text_are_accepted(self):
        """Regression: "n/a" read from a file is a new string object, so it must be compared by value."""
        import json
        card = json.loads(json.dumps(fx.scorecard([fx.item(question_type="tf"), fx.item("G2")])))
        self.assertEqual(errors(card), [])
        self.assertEqual(sc.resolve_item(card["items"][0])["state"], "agreed")
        self.assertEqual(sc.compute_metrics(card)["states"]["agreed"], 2)

    def test_top_level_rules(self):
        for key, value in (("scorecard_version", 1), ("is_synthetic", None), ("items", {}), ("run", {"source": "x"})):
            data = fx.scorecard([])
            data[key] = value
            self.assertTrue(errors(data), key)
        self.assertTrue(errors(fx.scorecard([], synthetic=False, source="synthetic")))      # synthetic run must say synthetic
        self.assertTrue(errors(fx.scorecard([], synthetic=True, source="recorded_real_run")))  # a real run cannot be synthetic
        self.assertEqual(errors(fx.scorecard([], synthetic=False, source="pending")), [])
        # a recorded real run must carry its provenance and bind to a manifest
        self.assertTrue(errors(fx.scorecard([], synthetic=False, source="recorded_real_run")))
        _, card = fx.frozen_world(2)
        self.assertEqual(errors(card), [])

    def test_item_fields(self):
        cases = {
            "duplicate": [fx.item("G1"), fx.item("G1")], "no id": [fx.item(item_id="")], "bad subject": [fx.item(subject="History")],
            "bad split": [fx.item(split="train")], "bad type": [fx.item(question_type="short")],
            "no evidence": [fx.item(evidence=[])], "four evidence": [fx.item(evidence=[{"start_seconds": 1, "end_seconds": 2, "quote": "q"}] * 4)],
            "evidence w/o quote": [fx.item(evidence=[{"start_seconds": 1, "end_seconds": 2}])],
            "no question": [fx.item(question=None)], "three reviews": [fx.item(reviews=[fx.review(), fx.review(reviewer="b"), fx.review(reviewer="c")])],
            "same reviewer twice": [fx.item(reviews=[fx.review(reviewer="Ann"), fx.review(reviewer=" ann ")])],
        }
        for name, items in cases.items():
            self.assertTrue(errors(fx.scorecard(items)), name)

    def test_judgment_values(self):
        for key in JUDGMENTS:
            self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review(**{key: "yes"})])])), key)
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review(**{"made_up": True})])])))
        partial = fx.review()
        partial["judgments"]["key_correct"] = None                 # not judged yet: allowed while pending
        self.assertEqual(errors(fx.scorecard([fx.item(reviews=[partial])])), [])

    def test_not_applicable_only_where_it_applies(self):
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review("mcq", single_defensible_option="n/a")])])))
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review("mcq", key_correct="n/a")])])))
        self.assertTrue(errors(fx.scorecard([fx.item(question_type="tf", reviews=[fx.review("tf", misleading_distractors=True)])])))
        self.assertTrue(errors(fx.scorecard([fx.item(question_type="tf", reviews=[fx.review("tf", key_correct="n/a")])])))

    def test_exclusions_cannot_hide_a_failure(self):
        ok = fx.item(reviews=[], exclusion=EXCL)
        self.assertEqual(errors(fx.scorecard([ok])), [])
        for reason in ("requires_visual", "not_answerable"):   # these are JUDGED and counted, never excluded
            self.assertTrue(errors(fx.scorecard([fx.item(reviews=[], exclusion=dict(EXCL, reason=reason))])), reason)
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[], exclusion=dict(EXCL, reason="duplicate", notes=""))])))
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[], exclusion=dict(EXCL, reason="bored"))])))

    def test_adjudication_needs_two_reviews_and_a_note(self):
        adj = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "listened again", "decided_at": "2000-01-02"}
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review()], adjudication=adj)])))
        two = [fx.review(), fx.review(reviewer="b", key_correct=False)]
        self.assertEqual(errors(fx.scorecard([fx.item(reviews=two, adjudication=adj)])), [])
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=two, adjudication=dict(adj, notes=""))])))

    def test_cross_check_against_a_manifest(self):
        m = fx.manifest(lectures=[fx.lecture("SYN01", "Mathematics", "confirmatory")])
        self.assertEqual(errors(fx.scorecard([fx.item()]), m), [])
        self.assertTrue(errors(fx.scorecard([fx.item(lecture_id="NOPE")]), m))
        self.assertTrue(errors(fx.scorecard([fx.item(subject="Physics")]), m))
        self.assertTrue(errors(fx.scorecard([fx.item(split="development")]), m))


class ResolutionTests(unittest.TestCase):
    def state(self, **kw):
        return sc.resolve_item(fx.item(**kw))["state"]

    def test_pending_single_agreed(self):
        self.assertEqual(self.state(reviews=[]), "pending")
        incomplete = fx.review()
        incomplete["judgments"]["key_correct"] = None
        self.assertEqual(self.state(reviews=[incomplete]), "pending")
        self.assertEqual(self.state(reviews=[fx.review()]), "single_review")
        self.assertEqual(self.state(reviews=[fx.review(), fx.review(reviewer="b")]), "agreed")

    def test_a_started_second_review_keeps_the_item_pending(self):
        half = fx.review(reviewer="b")
        half["judgments"]["key_correct"] = None
        self.assertEqual(self.state(reviews=[fx.review(), half]), "pending")

    def test_a_review_without_a_reviewer_or_date_does_not_count(self):
        anonymous = dict(fx.review(), reviewer=None)
        undated = dict(fx.review(), completed_at=None)
        self.assertEqual(self.state(reviews=[anonymous]), "pending")
        self.assertEqual(self.state(reviews=[undated]), "pending")

    def test_disagreement_needs_adjudication_of_exactly_the_disputed_judgments(self):
        two = [fx.review(), fx.review(reviewer="b", key_correct=False, evidence_relevant=False)]
        self.assertEqual(self.state(reviews=two), "unresolved")
        partial = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}
        self.assertEqual(self.state(reviews=two, adjudication=partial), "unresolved")      # one dispute left open
        full = {"adjudicator": "lead", "judgments": {"key_correct": False, "evidence_relevant": True}, "notes": "n", "decided_at": "2000-01-02"}
        res = sc.resolve_item(fx.item(reviews=two, adjudication=full))
        self.assertEqual(res["state"], "adjudicated")
        self.assertEqual(sorted(res["disagreements"]), ["evidence_relevant", "key_correct"])
        self.assertIs(res["final"]["key_correct"], False)
        self.assertIs(res["final"]["evidence_relevant"], True)
        self.assertIs(res["final"]["answerable_from_transcript"], True)   # undisputed judgments keep the agreed value

    def test_exclusion_wins(self):
        self.assertEqual(self.state(exclusion={"reason": "duplicate", "notes": "x"}), "excluded")


class StatisticsTests(unittest.TestCase):
    def test_wilson_matches_the_contracts_example(self):
        low, high = sc.wilson(72, 80)                 # 90% observed on 80 questions
        self.assertGreater(low, 0.80)
        self.assertLess(low, 0.83)                    # the contract says about 81%
        self.assertGreater(high, 0.90)

    def test_wilson_edges(self):
        self.assertIsNone(sc.wilson(0, 0))
        low, high = sc.wilson(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = sc.wilson(10, 10)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 1.0)

    def test_kappa(self):
        self.assertEqual(sc.cohens_kappa([(True, True), (False, False), (True, True), (False, False)]), 1.0)
        self.assertAlmostEqual(sc.cohens_kappa([(True, False), (False, True)]), -1.0)
        self.assertIsNone(sc.cohens_kappa([]))
        self.assertIsNone(sc.cohens_kappa([(True, True), (True, True)]))     # no variation: undefined, not 1


class MetricsTests(unittest.TestCase):
    def test_only_reviewed_items_count(self):
        items = fx.many_items(10)
        pending = [fx.item(f"P{i}", reviews=[]) for i in range(5)]
        m = metrics(items + pending)
        self.assertEqual(m["criteria_confirmatory"]["a_answerable_and_key_correct"]["n"], 10)
        self.assertEqual(m["states"]["pending"], 5)
        self.assertEqual(m["states"]["agreed"], 10)

    def test_pending_items_do_not_change_a_rate(self):
        base = metrics(fx.many_items(10, wrong=2))["criteria_confirmatory"]["a_answerable_and_key_correct"]
        more = metrics(fx.many_items(10, wrong=2) + [fx.item(f"P{i}", reviews=[]) for i in range(20)])["criteria_confirmatory"]["a_answerable_and_key_correct"]
        self.assertEqual(base, more)

    def test_a_missing_judgment_is_not_a_failure(self):
        half = fx.review()
        half["judgments"]["key_correct"] = None
        m = metrics(fx.many_items(5) + [fx.item("H", reviews=[half])])
        self.assertEqual(m["criteria_confirmatory"]["a_answerable_and_key_correct"], sc._rate(5, 5))
        self.assertEqual(m["states"]["pending"], 1)

    def test_rates_and_denominators(self):
        items = fx.many_items(10, wrong=3)
        c = metrics(items)["criteria_confirmatory"]
        self.assertEqual((c["a_answerable_and_key_correct"]["count"], c["a_answerable_and_key_correct"]["n"]), (7, 10))
        self.assertEqual((c["b_wrong_key_or_misleading"]["count"], c["b_wrong_key_or_misleading"]["n"]), (3, 10))
        self.assertEqual(c["c_range_contains_support"]["observed"], 1.0)

    def test_each_criterion_counts_its_own_failures(self):
        items = [
            fx.item("1", reviews=pair(fx.review(answerable_from_transcript=False))),
            fx.item("2", reviews=pair(fx.review(unsupported_claims=True))),
            fx.item("3", reviews=pair(fx.review(misleading_distractors=True))),
            fx.item("4", reviews=pair(fx.review(evidence_range_contains_support=False))),
            fx.item("5", reviews=pair(fx.review(single_defensible_option=False))),
            fx.item("6", reviews=pair(fx.review(requires_visual=True))),
        ]
        c = metrics(items)["criteria_confirmatory"]
        self.assertEqual(c["a_answerable_and_key_correct"]["count"], 5)
        self.assertEqual(c["b_wrong_key_or_misleading"]["count"], 2)       # unsupported claim, misleading distractor
        self.assertEqual(c["c_range_contains_support"]["count"], 5)
        self.assertEqual(c["c_mcq_single_defensible_option"]["count"], 5)
        self.assertEqual(c["requires_visual"]["count"], 1)

    def test_true_false_questions_are_not_in_the_mcq_only_denominators(self):
        items = fx.many_items(4, question_type="mcq") + fx.many_items(6, question_type="tf")
        for i, it in enumerate(items):
            it["item_id"] = f"G{i}"
        m = metrics(items)
        c = m["criteria_confirmatory"]
        self.assertEqual(c["c_mcq_single_defensible_option"]["n"], 4)
        self.assertEqual(c["misleading_distractors"]["n"], 4)
        self.assertEqual(c["a_answerable_and_key_correct"]["n"], 10)
        self.assertEqual(m["question_type_mix_reviewed"], {"mcq": 4, "tf": 6})     # recorded, no ratio demanded

    def test_the_development_split_never_feeds_the_gate(self):
        items = fx.many_items(10) + [dict(i, item_id=f"D{n}") for n, i in enumerate(fx.many_items(5, split="development"))]
        m = metrics(items)
        self.assertEqual(m["splits_counted"], {"development": 5, "confirmatory": 10})
        self.assertEqual(m["criteria_confirmatory"]["a_answerable_and_key_correct"]["n"], 10)
        self.assertEqual(metrics(fx.many_items(10, split="pending"))["criteria_confirmatory"]["a_answerable_and_key_correct"]["n"], 0)

    def test_unresolved_and_excluded_items_are_reported_but_in_no_rate(self):
        two = [fx.review(), fx.review(reviewer="b", key_correct=False)]
        items = fx.many_items(5) + [
            fx.item("U", reviews=two),
            fx.item("X1", reviews=[], exclusion={"reason": "duplicate", "notes": "same as G1"}),
            fx.item("X2", reviews=[], exclusion={"reason": "transcript_defect", "notes": "gap"}),
        ]
        m = metrics(items)
        self.assertEqual(m["states"]["unresolved"], 1)
        self.assertEqual(m["states"]["excluded"], 2)
        self.assertEqual(m["exclusions_by_reason"], {"duplicate": 1, "transcript_defect": 1})
        self.assertEqual(m["criteria_confirmatory"]["a_answerable_and_key_correct"]["n"], 5)

    def test_per_subject_figures_are_descriptive_and_use_the_same_items(self):
        m = metrics(fx.many_items(10, wrong=1))
        self.assertEqual(set(m["by_subject_descriptive"]), set(SUBJECTS))
        self.assertEqual(sum(v["a_answerable_and_key_correct"]["n"] for v in m["by_subject_descriptive"].values()), 10)
        self.assertEqual(m["by_subject_descriptive"]["Mathematics"]["a_answerable_and_key_correct"]["count"], 1)  # its first item had the wrong key

    def test_reviewer_agreement_uses_only_items_with_two_complete_reviews(self):
        items = [
            fx.item("1", reviews=[fx.review(), fx.review(reviewer="b")]),
            fx.item("2", reviews=[fx.review(), fx.review(reviewer="b", key_correct=False)],
                    adjudication={"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}),
            fx.item("3", reviews=[fx.review()]),
        ]
        a = metrics(items)["reviewer_agreement"]
        self.assertEqual(a["key_correct"]["pairs"], 2)
        self.assertEqual(a["key_correct"]["percent_agreement"], 0.5)
        self.assertEqual(a["answerable_from_transcript"]["percent_agreement"], 1.0)

    def test_subject_failures_are_tallied_by_subject_and_category(self):
        r = fx.review()
        r["subject_failure"] = {"category": "units dropped", "notes": "kg became g"}
        m = metrics([fx.item("1", subject="Physics", reviews=[r]), fx.item("2", subject="Physics", reviews=[dict(r)])])
        self.assertEqual(m["subject_failures"], [{"subject": "Physics", "category": "units dropped", "count": 2}])

    def test_the_tool_lists_what_it_cannot_measure(self):
        text = " ".join(metrics([]) ["not_measured_by_this_tool"])
        for phrase in ("Criterion d", "screening rejected", "caption error", "cost"):
            self.assertIn(phrase, text)


def both(card, index, **judgments):
    """Set judgments on BOTH reviews of one item (so the two reviewers agree)."""
    for r in card["items"][index]["reviews"]:
        r["judgments"].update(judgments)


def verdict_of(manifest, card):
    return sc.compute_metrics(card, manifest)["verdict"]


class VerdictTests(unittest.TestCase):
    def test_synthetic_data_never_yields_a_verdict(self):
        v = sc.compute_metrics(fx.scorecard(fx.many_items(80)))["verdict"]
        self.assertEqual(v["overall"], "not_evidence")
        self.assertIsNone(v["per_criterion"])
        m, card = fx.frozen_world(80)
        card["is_synthetic"] = True
        card["run"]["source"] = "synthetic"
        self.assertEqual(verdict_of(m, card)["overall"], "not_evidence")

    def test_a_missing_is_synthetic_field_is_treated_as_synthetic(self):
        m, card = fx.frozen_world(80)
        del card["is_synthetic"]
        self.assertEqual(verdict_of(m, card)["overall"], "not_evidence")

    def test_eighty_good_items_in_a_frozen_recorded_run_meet_a_to_c_but_never_claim_approval(self):
        m, card = fx.frozen_world(80)
        v = verdict_of(m, card)
        self.assertEqual(v["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(v["ineligible_because"], [])
        self.assertEqual(set(v["per_criterion"].values()), {"met"})
        self.assertEqual(v["d_human_accessibility_checks"], "not_measured_here")
        self.assertNotIn("approved", v["overall"])

    def test_seventy_nine_reviewed_slots_is_not_evaluable_and_names_the_gap(self):
        m, card = fx.frozen_world(80)
        card["items"].pop()                       # one registered slot has no item at all
        v = verdict_of(m, card)
        self.assertEqual(v["overall"], "not_evaluable")
        self.assertEqual((v["gate_slots"]["fully_reviewed"], v["gate_slots"]["missing"], v["gate_slots"]["required"]), (79, 1, 80))
        self.assertIsNone(v["per_criterion"])

    def test_the_boundary_of_criterion_a_and_b(self):
        m, card = fx.frozen_world(80, wrong=8)     # 72/80 = 90% on a, but 10% on b
        v = verdict_of(m, card)
        self.assertEqual(v["overall"], "criteria_not_met")
        self.assertEqual((v["per_criterion"]["a"], v["per_criterion"]["b"]), ("met", "not_met"))
        m, card = fx.frozen_world(80, wrong=4)     # 95% and exactly 5%: pooled criteria met ...
        v = verdict_of(m, card)
        self.assertTrue(v["pooled_criteria_met"])
        self.assertEqual(v["overall"], "awaiting_subject_approval")      # ... but four subjects each have one wrong key (1/16 > 5%)
        m, card = fx.frozen_world(80, wrong=5)     # 6.25% on b
        self.assertEqual(verdict_of(m, card)["per_criterion"]["b"], "not_met")
        m, card = fx.frozen_world(80, wrong=9)     # 88.75% on a
        self.assertEqual(verdict_of(m, card)["per_criterion"]["a"], "not_met")

    def test_observed_rate_is_required_even_when_the_interval_clears_the_bound(self):
        # 200 items is not a valid gate sample; this checks the arithmetic of criterion a on its own
        criteria = metrics(fx.many_items(200, wrong=21))["criteria_confirmatory"]
        self.assertGreater(criteria["a_answerable_and_key_correct"]["wilson_low"], GATE["a_wilson_lower_min"])
        full = {"required": 200, "fully_reviewed": 200, "pending": 0, "single_review": 0, "unresolved": 0, "excluded": 0, "missing": 0}
        none = {"confirmed": [], "flagged_not_confirmed": []}
        self.assertEqual(sc._verdict(criteria, full, none, [], False)["per_criterion"]["a"], "not_met")        # 89.5%
        ok = metrics(fx.many_items(200, wrong=20))["criteria_confirmatory"]                                    # exactly 90%
        self.assertEqual(sc._verdict(ok, full, none, [], False)["per_criterion"]["a"], "met")

    def test_wilson_lower_bound_of_the_contract_example(self):
        self.assertGreaterEqual(sc.wilson(72, 80)[0], GATE["a_wilson_lower_min"])

    def test_one_key_false_about_the_subject_blocks_release(self):
        m, card = fx.frozen_world(80)
        both(card, 0, key_false_about_subject=True, key_correct=False)
        v = verdict_of(m, card)
        self.assertEqual(v["overall"], "release_blocked")
        self.assertEqual(v["release_blocking_failures_confirmatory"], 1)
        self.assertEqual(v["per_criterion"]["b"], "not_met")

    def test_criterion_c_thresholds(self):
        for failures, expected in ((8, "met"), (9, "not_met")):       # 90% exactly passes
            m, card = fx.frozen_world(80)
            for i in range(failures):
                both(card, i, evidence_range_contains_support=False)
            self.assertEqual(verdict_of(m, card)["per_criterion"]["c_range"], expected, failures)

    def test_criterion_c_for_mcq_single_option(self):
        for failures, expected in ((12, "met"), (13, "not_met")):     # 68/80 = 85% exactly passes
            m, card = fx.frozen_world(80)
            for i in range(failures):
                both(card, i, single_defensible_option=False)
            self.assertEqual(verdict_of(m, card)["per_criterion"]["c_mcq_single"], expected, failures)

    def test_the_thresholds_and_sample_size_are_the_contracts_own(self):
        self.assertEqual((GATE["min_reviewed_for_a_verdict"], GATE["required_confirmatory_items"]), (80, 80))
        self.assertEqual((GATE["a_observed_min"], GATE["a_wilson_lower_min"]), (0.90, 0.80))
        self.assertEqual((GATE["b_wrong_or_misleading_max"], GATE["c_range_contains_support_min"], GATE["c_mcq_single_defensible_min"]), (0.05, 0.90, 0.85))


class EligibilityTests(unittest.TestCase):
    """Stage 6C: a verdict fails closed. Each case below is a way a pass could be wrongly granted."""

    def check(self, mutate, expect="not_eligible", reason=None):
        m, card = fx.frozen_world(80)
        mutate(m, card)
        v = verdict_of(m, card)
        self.assertEqual(v["overall"], expect, v["ineligible_because"])
        if reason:
            self.assertTrue(any(reason in r for r in v["ineligible_because"]), v["ineligible_because"])

    def test_a_pending_run_cannot_pass(self):
        self.check(lambda m, c: c["run"].update(source="pending"), reason="run.source")

    def test_a_draft_or_template_manifest_cannot_pass(self):
        for status in ("draft", "template"):
            self.check(lambda m, c, s=status: (m.update(status=s), fx.rebind(m, c)), reason="not 'frozen'")

    def test_a_synthetic_manifest_cannot_pass(self):
        self.check(lambda m, c: (m.update(is_synthetic=True), fx.rebind(m, c)), reason="manifest is synthetic")

    def test_missing_manifest_fails_closed(self):
        _, card = fx.frozen_world(80)
        v = sc.compute_metrics(card)["verdict"]
        self.assertEqual(v["overall"], "not_eligible")
        self.assertTrue(any("no manifest" in r for r in v["ineligible_because"]))

    def test_every_provenance_field_is_required(self):
        for key in ("run_id", "prompt_version", "prompt_sha256", "model", "config_sha256", "code_version", "generated_at", "raw_responses_sha256"):
            for bad in (None, "", "   ", "PENDING", "PENDING-not-run"):
                self.check(lambda m, c, k=key, b=bad: c["run"].update({k: b}), reason="provenance")
            self.check(lambda m, c, k=key: c["run"].pop(k), reason="provenance")

    def test_hash_fields_must_be_real_hashes(self):
        for key in ("prompt_sha256", "config_sha256", "raw_responses_sha256"):
            self.check(lambda m, c, k=key: c["run"].update({k: "not-a-hash"}), reason="provenance")

    def test_a_scorecard_bound_to_a_different_manifest_cannot_pass(self):
        self.check(lambda m, c: c["manifest_binding"].update(sha256="0" * 64), reason="hash")
        self.check(lambda m, c: c.pop("manifest_binding"), reason="hash")

    def test_editing_the_manifest_after_the_run_invalidates_it(self):
        self.check(lambda m, c: m.update(selection_rule="changed after the run"), reason="hash")

    def test_the_run_must_match_the_frozen_configuration(self):
        for key in ("prompt_version", "prompt_sha256", "model", "config_sha256"):
            other = "ff" * 32 if key.endswith("sha256") else "a-different-value"
            self.check(lambda m, c, k=key, o=other: c["run"].update({k: o}), reason=f"run.{key}")

    def test_the_contract_sample_size_is_fixed_at_eighty(self):
        for n in (79, 81):
            m, card = fx.frozen_world(n)
            v = verdict_of(m, card)
            self.assertEqual(v["overall"], "not_eligible", n)
            self.assertTrue(any("exactly 80" in r for r in v["ineligible_because"]))

    def test_a_scorecard_with_validation_errors_cannot_pass(self):
        self.check(lambda m, c: c["items"][0].update(question_type="short"), reason="validation error")

    def test_an_unregistered_extra_item_is_rejected(self):
        def extra(m, c):
            c["items"].append(dict(c["items"][0], item_id="EXTRA", sample_id="NOT-A-SLOT"))
        self.check(extra, reason="validation error")

    def test_two_items_cannot_share_one_slot(self):
        def dup(m, c):
            c["items"][1]["sample_id"] = c["items"][0]["sample_id"]
        self.check(dup, reason="validation error")

    def test_an_item_whose_subject_differs_from_the_manifest_is_rejected(self):
        self.check(lambda m, c: c["items"][0].update(subject="Physics"), reason="validation error")


class IncompleteReviewTests(unittest.TestCase):
    """Nothing pending, single-reviewed, unresolved, excluded or missing can be counted as a success."""

    def run_with(self, mutate):
        m, card = fx.frozen_world(80)
        mutate(card)
        return sc.compute_metrics(card, m)

    def test_each_kind_of_incomplete_slot_stops_a_pass_and_is_counted_apart(self):
        cases = {
            "pending": lambda c: c["items"][0].update(reviews=[]),
            "single_review": lambda c: c["items"][0].update(reviews=c["items"][0]["reviews"][:1]),
            "unresolved": lambda c: c["items"][0]["reviews"][1]["judgments"].update(evidence_relevant=False),
            "excluded": lambda c: c["items"][0].update(reviews=[], exclusion=dict(EXCL)),
            "missing": lambda c: c["items"].pop(0),
        }
        for state, mutate in cases.items():
            out = self.run_with(mutate)
            slots = out["gate_slots"]
            self.assertNotEqual(out["verdict"]["overall"], "criteria_a_to_c_met_pending_d", state)
            self.assertEqual(out["verdict"]["overall"], "not_evaluable" if state != "excluded" else out["verdict"]["overall"], state)
            self.assertEqual((slots[state], slots["fully_reviewed"]), (1, 79), state)
            self.assertEqual(sum(slots[k] for k in ("fully_reviewed", "pending", "single_review", "unresolved", "excluded", "missing")), slots["required"], state)

    def test_an_exclusion_does_not_raise_the_pass_rate_or_shrink_the_denominator(self):
        m, card = fx.frozen_world(80, wrong=9)             # 88.75%: criterion a fails
        base = sc.compute_metrics(card, m)["criteria_confirmatory"]["a_answerable_and_key_correct"]
        for i in range(3):                                  # exclude three slots that are NOT recorded failures
            card["items"][70 + i].update(reviews=[], exclusion=dict(EXCL))
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["verdict"]["overall"], "not_evaluable")      # never criteria_a_to_c_met
        self.assertEqual(out["gate_slots"]["excluded"], 3)
        after = out["criteria_confirmatory"]["a_answerable_and_key_correct"]
        self.assertEqual((after["count"], after["n"]), (base["count"] - 3, base["n"] - 3))   # excluded items count as no success

    def test_excluding_a_recorded_failure_is_rejected_and_the_failure_stays_visible(self):
        m, card = fx.frozen_world(80)
        both(card, 0, key_correct=False, key_false_about_subject=True)
        card["items"][0]["exclusion"] = dict(EXCL)
        self.assertTrue(sc.validate_scorecard(card, m).errors)
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["verdict"]["overall"], "not_eligible")
        self.assertEqual(out["release_blockers"]["confirmatory"]["flagged_not_confirmed"], [card["items"][0]["item_id"]])

    def test_unresolved_items_are_not_counted_as_agreement(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"][1]["judgments"]["key_correct"] = False
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["states"]["unresolved"], 1)
        self.assertEqual(out["criteria_confirmatory"]["a_answerable_and_key_correct"]["n"], 79)
        self.assertNotEqual(out["verdict"]["overall"], "criteria_a_to_c_met_pending_d")

    def test_one_reviewer_twice_never_counts_as_two(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"][1]["reviewer"] = card["items"][0]["reviews"][0]["reviewer"].upper()
        self.assertEqual(sc.resolve_item(card["items"][0])["state"], "single_review")
        self.assertNotEqual(verdict_of(m, card)["overall"], "criteria_a_to_c_met_pending_d")

    def test_report_counts_every_state_separately(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"] = []
        card["items"][1]["reviews"] = card["items"][1]["reviews"][:1]
        card["items"][2]["reviews"][1]["judgments"]["key_correct"] = False
        card["items"][3].update(reviews=[], exclusion=dict(EXCL))
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["states"], {"pending": 1, "single_review": 1, "agreed": 76, "adjudicated": 0, "unresolved": 1, "excluded": 1})
        self.assertEqual(out["gate_slots"]["fully_reviewed"], 76)


class ReviewerRuleTests(unittest.TestCase):
    def two(self, **second):
        return [fx.review(), fx.review(reviewer="reviewer-B", **second)]

    def test_contradictory_judgments_are_rejected(self):
        cases = [
            {"answerable_from_transcript": False, "key_correct": True},
            {"key_false_about_subject": True, "key_correct": True},
            {"requires_visual": True, "answerable_from_transcript": True},
        ]
        for case in cases:
            self.assertTrue(errors(fx.scorecard([fx.item(reviews=[fx.review(**case)])])), case)

    def test_a_consistent_failure_is_accepted(self):
        ok = fx.review(answerable_from_transcript=False, key_correct=False, requires_visual=True)
        self.assertEqual(errors(fx.scorecard([fx.item(reviews=[ok])])), [])

    def test_a_reviewer_cannot_adjudicate_their_own_item(self):
        adj = {"adjudicator": " REVIEWER-A ", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}
        item = fx.item(reviews=self.two(key_correct=False), adjudication=adj)
        self.assertTrue(errors(fx.scorecard([item])))
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")

    def test_adjudication_must_resolve_every_disagreement(self):
        two = self.two(key_correct=False, evidence_relevant=False)
        adj = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}
        item = fx.item(reviews=two, adjudication=adj)
        self.assertTrue(any("does not resolve" in e for e in errors(fx.scorecard([item]))))
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")

    def test_adjudication_may_not_supply_values_that_would_be_silently_ignored(self):
        adj = {"adjudicator": "lead", "judgments": {"key_correct": True, "evidence_relevant": False}, "notes": "n", "decided_at": "2000-01-02"}
        item = fx.item(reviews=self.two(key_correct=False), adjudication=adj)
        self.assertTrue(any("not in dispute" in e for e in errors(fx.scorecard([item]))))
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")

    def test_adjudication_with_nothing_to_adjudicate_is_rejected(self):
        adj = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}
        self.assertTrue(errors(fx.scorecard([fx.item(adjudication=adj)])))

    def test_an_adjudication_that_would_create_a_contradiction_is_rejected(self):
        # both reviewers say the question is NOT answerable; they differ on key_correct; adjudicating key_correct=true is contradictory
        reviews = [fx.review(answerable_from_transcript=False, key_correct=False),
                   fx.review(reviewer="reviewer-B", answerable_from_transcript=False, key_correct=False)]
        reviews[1]["judgments"]["evidence_relevant"] = False
        adj = {"adjudicator": "lead", "judgments": {"evidence_relevant": True}, "notes": "n", "decided_at": "2000-01-02"}
        self.assertEqual(sc.resolve_item(fx.item(reviews=reviews, adjudication=adj))["state"], "adjudicated")
        reviews = [fx.review(answerable_from_transcript=False, key_correct=False),
                   fx.review(reviewer="reviewer-B", answerable_from_transcript=False, key_correct=True)]
        self.assertTrue(errors(fx.scorecard([fx.item(reviews=reviews)])))           # the second review itself is contradictory

    def test_adjudication_keeps_both_original_reviews_unchanged(self):
        adj = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "n", "decided_at": "2000-01-02"}
        item = fx.item(reviews=self.two(key_correct=False), adjudication=adj)
        before = fx.deep(item["reviews"])
        res = sc.resolve_item(item)
        self.assertEqual((res["state"], res["disagreements"]), ("adjudicated", ["key_correct"]))
        self.assertEqual(item["reviews"], before)                         # resolution never rewrites a reviewer's own judgment
        self.assertIs(item["reviews"][1]["judgments"]["key_correct"], False)
        self.assertIs(res["final"]["key_correct"], True)

    def test_an_exclusion_needs_two_distinct_people_and_a_date(self):
        for bad in (dict(EXCL, confirmed_by="ann"), dict(EXCL, confirmed_by=None), dict(EXCL, excluded_by=None), dict(EXCL, decided_at=None)):
            self.assertTrue(errors(fx.scorecard([fx.item(reviews=[], exclusion=bad)])), bad)

    def test_exclusion_cannot_cover_any_recorded_failure(self):
        from quiz_eval.constants import FAILING_VALUES
        for key, bad in FAILING_VALUES.items():
            review = fx.review(**{key: bad})
            self.assertTrue(any("never be excluded" in e for e in errors(fx.scorecard([fx.item(reviews=[review], exclusion=dict(EXCL))]))), key)


class ReportingTests(unittest.TestCase):
    def test_zero_mcq_is_not_measurable_not_a_pass_or_a_failure(self):
        m, card = fx.frozen_world(80)
        for it in card["items"]:
            it["question_type"] = "tf"
            for r in it["reviews"]:
                r["judgments"].update(single_defensible_option="n/a", misleading_distractors="n/a")
        out = sc.compute_metrics(card, m)
        self.assertIs(out["criteria_confirmatory"]["c_mcq_single_defensible_option"]["measurable"], False)
        self.assertEqual(out["verdict"]["overall"], "not_evaluable")
        self.assertEqual(out["verdict"]["per_criterion"] if out["verdict"]["per_criterion"] else None, None)

    def test_blockers_are_visible_without_a_valid_sample_size(self):
        blocked = dict(key_false_about_subject=True, key_correct=False)
        card = fx.scorecard([fx.item("G1", reviews=[fx.review(**blocked), fx.review(reviewer="b", **blocked)])])
        out = sc.compute_metrics(card)
        self.assertEqual(out["release_blockers"]["confirmatory"]["confirmed"], ["G1"])
        self.assertEqual(out["criteria_confirmatory"]["b_key_false_about_subject"]["count"], 1)

    def test_a_blocker_in_a_single_review_or_pending_split_is_still_listed(self):
        blocked = dict(key_false_about_subject=True, key_correct=False)
        one = fx.item("A", reviews=[fx.review(**blocked)])
        dev = fx.item("B", split="development", reviews=[fx.review(**blocked), fx.review(reviewer="b", **blocked)])
        pend = fx.item("C", split="pending", reviews=[fx.review(**blocked)])
        out = sc.compute_metrics(fx.scorecard([one, dev, pend]))["release_blockers"]
        self.assertEqual(out["confirmatory"]["flagged_not_confirmed"], ["A"])
        self.assertEqual(out["development"]["confirmed"], ["B"])
        self.assertEqual(out["pending"]["flagged_not_confirmed"], ["C"])

    def test_unconfirmed_blocker_flag_is_reported_in_the_verdict(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"][0]["judgments"].update(key_false_about_subject=True, key_correct=False)   # the other reviewer disagrees
        v = verdict_of(m, card)
        self.assertEqual(v["release_blocking_failures_flagged_unconfirmed"], 1)
        self.assertNotEqual(v["overall"], "criteria_a_to_c_met_pending_d")

    def test_development_is_reported_apart_and_cannot_rescue_or_sink_the_gate(self):
        m, card = fx.frozen_world(80)
        m["samples"].append(fx.sample("DEV001", lecture_id="D0", subject="Mathematics"))
        fx.rebind(m, card)
        card["items"].append(fx.item("DEV1", split="development", subject="Mathematics", lecture_id="D0", sample_id="DEV001", returned_position=1,
                                     reviews=[fx.review(key_correct=False), fx.review(reviewer="b", key_correct=False)]))
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["verdict"]["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(out["splits_counted"], {"development": 1, "confirmatory": 80})
        self.assertEqual(out["criteria_development_descriptive"]["a_answerable_and_key_correct"]["count"], 0)

    def test_per_subject_report_is_complete(self):
        m, card = fx.frozen_world(80, wrong=4)
        out = sc.compute_metrics(card, m)
        for subject, row in out["by_subject_descriptive"].items():
            for key in ("a_answerable_and_key_correct", "b_wrong_key_or_misleading", "b_key_false_about_subject",
                        "c_range_contains_support", "c_mcq_single_defensible_option", "items_by_state", "release_blocking_confirmed"):
                self.assertIn(key, row, subject)
            self.assertEqual(row["a_answerable_and_key_correct"]["n"], 16)
        self.assertEqual(sum(r["b_wrong_key_or_misleading"]["count"] for r in out["by_subject_descriptive"].values()), 4)

    def test_cli_text_lists_slots_blockers_and_reasons(self):
        import io
        import json
        import os
        import tempfile
        from contextlib import redirect_stdout
        from quiz_eval.__main__ import main
        m, card = fx.frozen_world(80)
        card["run"]["source"] = "pending"
        with tempfile.TemporaryDirectory() as tmp:
            mp, cp = os.path.join(tmp, "m.json"), os.path.join(tmp, "c.json")
            for path, data in ((mp, m), (cp, card)):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(data, handle)
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["check", mp, "--scorecard", cp])
        text = buf.getvalue()
        for phrase in ("gate slots", "RELEASE BLOCKERS", "NOT ELIGIBLE", "VERDICT: not_eligible", "per subject"):
            self.assertIn(phrase, text)


class AutomaticCheckTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fx.write_transcript(self.tmp.name)
        self.manifest = fx.manifest(lectures=[fx.lecture()])

    def checks(self, items, base=None):
        return sc.automatic_evidence_checks(fx.scorecard(items), self.manifest, base or self.tmp.name)

    def test_counts_true_false_and_unknown_separately(self):
        wrong_time = fx.item("W", evidence=[{"start_seconds": 30.0, "end_seconds": 40.0, "quote": "gamma follows from beta"}])
        invented = fx.item("I", evidence=[{"start_seconds": 10.0, "end_seconds": 20.0, "quote": "never said"}])
        out = self.checks([fx.item("OK"), wrong_time, invented])["counts"]
        self.assertEqual(out["quote_in_range"], {"true": 1, "false": 2})
        self.assertEqual(out["quote_in_transcript"], {"true": 2, "false": 1})
        self.assertEqual(out["quote_elsewhere"], {"false": 2, "true": 1})

    def test_an_unavailable_transcript_is_unknown_not_a_failure(self):
        out = self.checks([fx.item("OK")], base=self.tmp.name + "-missing")
        self.assertEqual(out["counts"]["quote_in_range"], {"unknown": 1})
        self.assertEqual(out["counts"]["range_well_formed"], {"true": 1})        # needs no transcript
        self.assertEqual(out["review_flags"], [])

    def test_a_mechanical_mismatch_prompts_a_second_look_without_overriding_the_reviewer(self):
        invented = fx.item("I", evidence=[{"start_seconds": 10.0, "end_seconds": 20.0, "quote": "never said"}])
        wrong_time = fx.item("W", evidence=[{"start_seconds": 30.0, "end_seconds": 40.0, "quote": "gamma follows from beta"}])
        flags = self.checks([invented, wrong_time])["review_flags"]
        self.assertEqual({f["item_id"] for f in flags}, {"I", "W"})
        self.assertIs(sc.resolve_item(invented)["final"]["evidence_quote_valid"], True)      # the human judgment stands

    def test_pending_items_get_no_flags(self):
        pending = fx.item("P", reviews=[], evidence=[{"start_seconds": 10.0, "end_seconds": 20.0, "quote": "never said"}])
        self.assertEqual(self.checks([pending])["review_flags"], [])


if __name__ == "__main__":
    unittest.main()
