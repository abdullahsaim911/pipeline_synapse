"""
Stage 6E: the approved governance policies, enforced and tested. Synthetic data only (every id says TEST-FIXTURE or SYN;
no real run, review, approval or approver exists anywhere in this file).

A  two independent reviews, a separate adjudicator, disputed judgments only
B  exactly 80 registered slots; a shortfall makes the evaluation incomplete; no replacement
C  an exclusion cap of zero, for every reason
D  per-subject results and a recorded human approval; no invented per-subject threshold
E  a source-lecture error is a different fact from a model-created error
"""

import unittest

from tests.quiz_eval import fixtures as fx
from quiz_eval import manifest as mf
from quiz_eval import scorecard as sc
from quiz_eval.constants import EXCLUSION_CAP_CONFIRMATORY, GATE, SCORECARD_EXCLUSION_REASONS, SUBJECTS

EXCL = {"reason": "transcript_defect", "notes": "caption gap", "excluded_by": "Ann", "confirmed_by": "Bo", "decided_at": "2000-01-01"}
ADJ = {"adjudicator": "lead", "judgments": {"key_correct": True}, "notes": "listened again", "decided_at": "2000-01-02"}
SOURCE_ERROR = {"quote": "synthetic caption says two plus two is five", "note": "synthetic: four is correct"}


def verdict(m, card):
    return sc.compute_metrics(card, m)["verdict"]


def set_both(card, index, source_error=None, **judgments):
    for r in card["items"][index]["reviews"]:
        r["judgments"].update(judgments)
        if source_error is not None:
            r["source_error"] = dict(source_error)


def approve(m, card, for_subject, decision="granted", **over):
    """Record a subject approval on the figures the report currently shows (a TEST-FIXTURE person, not a real approver)."""
    basis = sc.compute_metrics(card, m)["by_subject_descriptive"][for_subject]["basis_sha256"]
    record = {"subject": for_subject, "decision": decision, "approver": "Approver One", "rationale": "TEST-FIXTURE rationale",
              "decided_at": "2000-01-03", "basis_sha256": basis}
    record.update(over)
    card.setdefault("subject_approvals", []).append(record)
    return record


class IndependentReviewTests(unittest.TestCase):
    def state(self, **kw):
        return sc.resolve_item(fx.item(**kw))["state"]

    def test_the_same_reviewer_twice_is_not_independent(self):
        two = [fx.review(reviewer="Ann"), fx.review(reviewer="  ANN ")]
        self.assertEqual(self.state(reviews=two), "single_review")
        self.assertTrue(sc.validate_scorecard(fx.scorecard([fx.item(reviews=two)])).errors)

    def test_two_distinct_agreeing_reviewers_complete_the_requirement(self):
        self.assertEqual(self.state(reviews=[fx.review(reviewer="Ann"), fx.review(reviewer="Bo")]), "agreed")

    def test_a_single_review_never_fills_a_confirmatory_slot(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"] = card["items"][0]["reviews"][:1]
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["gate_slots"]["single_review"], 1)
        self.assertEqual(out["gate_slots"]["fully_reviewed"], 79)
        self.assertFalse(out["verdict"]["evaluation_complete"])

    def test_a_disagreement_needs_a_separate_adjudicator(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False)]
        self.assertEqual(self.state(reviews=two), "unresolved")
        self.assertEqual(self.state(reviews=two, adjudication=ADJ), "adjudicated")

    def test_an_adjudicator_who_reviewed_the_item_is_rejected(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False)]
        for name in ("reviewer-A", " Reviewer-B "):
            item = fx.item(reviews=two, adjudication=dict(ADJ, adjudicator=name))
            self.assertTrue(sc.validate_scorecard(fx.scorecard([item])).errors, name)
            self.assertEqual(sc.resolve_item(item)["state"], "unresolved", name)

    def test_adjudicating_an_undisputed_judgment_is_rejected(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False)]
        item = fx.item(reviews=two, adjudication=dict(ADJ, judgments={"key_correct": True, "evidence_relevant": False}))
        self.assertTrue(any("not in dispute" in e for e in sc.validate_scorecard(fx.scorecard([item])).errors))
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")

    def test_incomplete_adjudication_leaves_the_slot_unresolved(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False, evidence_relevant=False)]
        item = fx.item(reviews=two, adjudication=dict(ADJ))           # resolves key_correct only
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")
        no_date = fx.item(reviews=two[:1] + [fx.review(reviewer="reviewer-B", key_correct=False)], adjudication=dict(ADJ, decided_at=None))
        self.assertEqual(sc.resolve_item(no_date)["state"], "unresolved")

    def test_original_reviews_survive_adjudication_untouched(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False)]
        item = fx.item(reviews=two, adjudication=dict(ADJ))
        before = fx.deep(item["reviews"])
        self.assertEqual(sc.resolve_item(item)["state"], "adjudicated")
        self.assertEqual(item["reviews"], before)


class ShortfallTests(unittest.TestCase):
    def test_a_missing_slot_prevents_a_pass_and_is_named(self):
        m, card = fx.frozen_world(80)
        card["items"].pop(0)
        v = verdict(m, card)
        self.assertNotEqual(v["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(v["gate_slots"]["missing"], 1)
        self.assertTrue(any("missing" in r for r in v["blocking_reasons"]))

    def test_a_lecture_with_too_few_usable_questions_makes_the_evaluation_incomplete(self):
        m, card = fx.frozen_world(80)
        card["items"] = [i for i in card["items"] if not (i["lecture_id"] == "L0" and i["returned_position"] > 14)]   # L0 gave 14 of its 16
        v = verdict(m, card)
        self.assertFalse(v["evaluation_complete"])
        self.assertEqual(v["overall"], "not_evaluable")
        self.assertEqual(set(v["shortfall_by_lecture"]), {"L0"})
        self.assertEqual((v["shortfall_by_lecture"]["L0"]["required"], v["shortfall_by_lecture"]["L0"]["missing"]), (16, 2))
        self.assertEqual(v["gate_slots"]["required"], 80)          # the denominator is not lowered

    def test_a_different_lecture_cannot_fill_a_missing_slot(self):
        m, card = fx.frozen_world(80)
        donor = next(i for i in card["items"] if i["lecture_id"] == "L1")
        victim = next(i for i in card["items"] if i["lecture_id"] == "L0")
        card["items"].remove(victim)
        donor_copy = fx.deep(donor)
        donor_copy.update(item_id="BORROWED", sample_id=victim["sample_id"], returned_position=victim["returned_position"])   # L1's question in L0's slot
        card["items"].append(donor_copy)
        self.assertTrue(any("different lecture" in e for e in sc.validate_scorecard(card, m).errors))
        v = verdict(m, card)
        self.assertEqual(v["overall"], "not_eligible")

    def test_extra_items_cannot_pad_the_sample(self):
        m, card = fx.frozen_world(80)
        card["items"].append(dict(fx.deep(card["items"][0]), item_id="EXTRA", sample_id="SYN-EXTRA"))
        self.assertTrue(sc.validate_scorecard(card, m).errors)
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")

    def test_slots_are_filled_in_the_models_returned_order_not_by_choice(self):
        m, card = fx.frozen_world(80)
        a, b = [i for i in card["items"] if i["lecture_id"] == "L0"][:2]
        a["returned_position"], b["returned_position"] = b["returned_position"], a["returned_position"]      # a hand-picked order
        self.assertTrue(any("returned_position" in e for e in sc.validate_scorecard(card, m).errors))
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")
        m, card = fx.frozen_world(80)
        del card["items"][0]["returned_position"]
        self.assertTrue(any("returned_position" in e for e in sc.validate_scorecard(card, m).errors))
        m, card = fx.frozen_world(80)
        card["items"][0]["returned_position"] = True            # a boolean is not a position
        self.assertTrue(sc.validate_scorecard(card, m).errors)

    def test_every_non_complete_state_is_reported_separately(self):
        m, card = fx.frozen_world(80)
        card["items"][0]["reviews"] = []
        card["items"][1]["reviews"] = card["items"][1]["reviews"][:1]
        card["items"][2]["reviews"][1]["judgments"]["key_correct"] = False
        card["items"][3].update(reviews=[], exclusion=dict(EXCL))
        card["items"].pop()
        slots = sc.compute_metrics(card, m)["gate_slots"]
        self.assertEqual({k: slots[k] for k in ("pending", "single_review", "unresolved", "excluded", "missing", "fully_reviewed")},
                         {"pending": 1, "single_review": 1, "unresolved": 1, "excluded": 1, "missing": 1, "fully_reviewed": 75})


class ExclusionCapTests(unittest.TestCase):
    def test_the_cap_is_zero(self):
        self.assertEqual(EXCLUSION_CAP_CONFIRMATORY, 0)

    def test_any_exclusion_blocks_a_pass_whatever_its_reason(self):
        for reason in SCORECARD_EXCLUSION_REASONS:
            m, card = fx.frozen_world(80)
            card["items"][0].update(reviews=[], exclusion=dict(EXCL, reason=reason))
            self.assertEqual(sc.validate_scorecard(card, m).errors, [], reason)
            v = verdict(m, card)
            self.assertNotEqual(v["overall"], "criteria_a_to_c_met_pending_d", reason)
            self.assertFalse(v["evaluation_complete"], reason)
            self.assertTrue(any("exclusion cap is 0" in r for r in v["blocking_reasons"]), reason)

    def test_an_exclusion_stays_in_the_denominator_and_is_never_replaced(self):
        m, card = fx.frozen_world(80)
        card["items"][0].update(reviews=[], exclusion=dict(EXCL))
        out = sc.compute_metrics(card, m)
        self.assertEqual((out["gate_slots"]["required"], out["gate_slots"]["excluded"], out["gate_slots"]["fully_reviewed"]), (80, 1, 79))
        self.assertEqual(out["verdict"]["needed_for_a_verdict"], 80)
        self.assertEqual(out["exclusions_by_reason"], {"transcript_defect": 1})
        card["items"].append(dict(fx.deep(card["items"][1]), item_id="REPLACEMENT", sample_id="SYN-REPL", returned_position=17))
        self.assertTrue(sc.validate_scorecard(card, m).errors)

    def test_the_audit_trail_of_an_exclusion_is_preserved_and_validated(self):
        item = fx.item(reviews=[], exclusion=dict(EXCL))
        self.assertEqual(sc.resolve_item(item)["state"], "excluded")
        self.assertEqual(item["exclusion"], EXCL)
        for bad in (dict(EXCL, confirmed_by="ann"), dict(EXCL, decided_at=None), dict(EXCL, notes=""), dict(EXCL, reason="bored")):
            self.assertTrue(sc.validate_scorecard(fx.scorecard([fx.item(reviews=[], exclusion=bad)])).errors, bad)

    def test_an_item_cannot_be_both_excluded_and_adjudicated(self):
        two = [fx.review(), fx.review(reviewer="reviewer-B", key_correct=False)]
        item = fx.item(reviews=two, adjudication=dict(ADJ), exclusion=dict(EXCL))
        self.assertTrue(any("excluded and adjudicated" in e for e in sc.validate_scorecard(fx.scorecard([item])).errors))

    def test_a_recorded_source_error_cannot_be_excluded_away(self):
        review = fx.review(lecture_factually_incorrect=True)
        review["source_error"] = dict(SOURCE_ERROR)
        item = fx.item(reviews=[review], exclusion=dict(EXCL))
        self.assertTrue(any("never be excluded" in e for e in sc.validate_scorecard(fx.scorecard([item])).errors))


class SubjectApprovalTests(unittest.TestCase):
    def test_a_pooled_pass_with_a_weak_subject_awaits_approval(self):
        m, card = fx.frozen_world(80, wrong=4)             # pooled 95% and 5%; four subjects have one wrong key each
        v = verdict(m, card)
        self.assertTrue(v["pooled_criteria_met"])
        self.assertEqual(v["overall"], "awaiting_subject_approval")
        self.assertEqual(sorted(v["subjects_awaiting_approval"]), sorted(SUBJECTS[:4]))
        self.assertEqual(v["subject_criteria"]["Computer science"], "met")
        self.assertEqual(v["subject_approvals"]["Computer science"], "not_required")
        self.assertEqual(v["subject_approvals"]["Mathematics"], "pending")

    def test_granted_approvals_for_every_weak_subject_allow_the_pending_d_state(self):
        m, card = fx.frozen_world(80, wrong=4)
        for subject in SUBJECTS[:4]:
            approve(m, card, subject)
        self.assertEqual(sc.validate_scorecard(card, m).errors, [])
        v = verdict(m, card)
        self.assertEqual(v["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(v["subject_approvals"]["Physics"], "granted")
        self.assertEqual(v["d_human_accessibility_checks"], "not_measured_here")      # still never a release approval

    def test_one_missing_approval_keeps_the_gate_closed(self):
        m, card = fx.frozen_world(80, wrong=4)
        for subject in SUBJECTS[:3]:
            approve(m, card, subject)
        self.assertEqual(verdict(m, card)["overall"], "awaiting_subject_approval")

    def test_a_denied_subject_blocks_release(self):
        m, card = fx.frozen_world(80, wrong=4)
        for subject in SUBJECTS[:3]:
            approve(m, card, subject)
        approve(m, card, SUBJECTS[3], decision="denied")
        v = verdict(m, card)
        self.assertEqual(v["overall"], "release_blocked")
        self.assertIn(f"subject approval denied: {SUBJECTS[3]}", v["blocking_reasons"])

    def test_an_approval_cannot_rescue_failed_pooled_criteria_or_a_blocker(self):
        m, card = fx.frozen_world(80, wrong=8)             # pooled b is 10%
        for subject in SUBJECTS:
            approve(m, card, subject)
        self.assertEqual(verdict(m, card)["overall"], "criteria_not_met")
        m, card = fx.frozen_world(80)
        for r in card["items"][0]["reviews"]:
            r["judgments"].update(key_false_about_subject=True, key_correct=False)
        approve(m, card, SUBJECTS[0])
        self.assertEqual(verdict(m, card)["overall"], "release_blocked")

    def test_a_partly_missing_subject_is_incomplete_not_met(self):
        m, card = fx.frozen_world(80)
        victim = next(i for i in card["items"] if i["subject"] == "Biology")
        card["items"].remove(victim)
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["verdict"]["subject_criteria"]["Biology"], "incomplete")
        self.assertEqual(out["verdict"]["subject_approvals"]["Biology"], "pending")
        self.assertIn("Biology", out["verdict"]["subjects_awaiting_approval"])

    def test_missing_subject_evidence_cannot_pass(self):
        m, card = fx.frozen_world(80)
        card["items"] = [i for i in card["items"] if i["subject"] != "Chemistry"]
        out = sc.compute_metrics(card, m)
        self.assertEqual(out["verdict"]["subject_criteria"]["Chemistry"], "no_evidence")
        self.assertNotEqual(out["verdict"]["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(out["by_subject_descriptive"]["Chemistry"]["slots"]["missing"], 16)
        approve(m, card, "Chemistry")             # even with an approval, the missing slots keep the evaluation incomplete
        self.assertEqual(verdict(m, card)["overall"], "not_evaluable")

    def test_approval_is_traceable_to_an_authorized_approver_with_a_rationale(self):
        m, card = fx.frozen_world(80, wrong=4)
        approve(m, card, "Mathematics")
        row = sc.compute_metrics(card, m)["by_subject_descriptive"]["Mathematics"]["approval"]
        self.assertEqual(row["state"], "granted")
        self.assertEqual({k: row["record"][k] for k in ("approver", "rationale", "decided_at")},
                         {"approver": "Approver One", "rationale": "TEST-FIXTURE rationale", "decided_at": "2000-01-03"})

    def test_an_unauthorized_or_conflicted_or_stale_approval_is_not_accepted(self):
        m, card = fx.frozen_world(80, wrong=4)
        approve(m, card, "Mathematics", approver="Someone Else")
        approve(m, card, "Physics", approver="reviewer-A")               # authorized only if listed; also reviewed these items
        approve(m, card, "Chemistry", basis_sha256="0" * 64)
        out = sc.compute_metrics(card, m)["by_subject_descriptive"]
        for subject, phrase in (("Mathematics", "not in the frozen manifest"), ("Physics", "not in the frozen manifest"), ("Chemistry", "basis hash differs")):
            self.assertEqual(out[subject]["approval"]["state"], "pending", subject)
            self.assertTrue(any(phrase in p for p in out[subject]["approval"]["problems"]), subject)
        self.assertTrue(sc.validate_scorecard(card, m).errors)

    def test_a_reviewer_cannot_approve_their_own_subject_even_if_listed(self):
        m, card = fx.frozen_world(80, wrong=4)
        m["authorized_approvers"].append("reviewer-A")
        fx.rebind(m, card)
        approve(m, card, "Mathematics", approver="reviewer-A")
        approval = sc.compute_metrics(card, m)["by_subject_descriptive"]["Mathematics"]["approval"]
        self.assertEqual(approval["state"], "pending")
        self.assertTrue(any("reviewed or adjudicated" in p for p in approval["problems"]))

    def test_an_approval_goes_stale_when_the_evidence_changes(self):
        m, card = fx.frozen_world(80, wrong=4)
        for subject in SUBJECTS[:4]:
            approve(m, card, subject)
        set_both(card, 70, key_correct=False)                         # a later change to Mathematics' figures? slot 70 is Mathematics
        v = verdict(m, card)
        self.assertNotEqual(v["overall"], "criteria_a_to_c_met_pending_d")

    def test_approval_records_are_validated(self):
        m, card = fx.frozen_world(80, wrong=4)
        for bad in ({"decision": "maybe"}, {"approver": ""}, {"rationale": "PENDING"}, {"decided_at": None}, {"subject": "History"}, {"basis_sha256": "x"}):
            m2, card2 = fx.frozen_world(80, wrong=4)
            approve(m2, card2, "Mathematics", **bad)
            self.assertTrue(sc.validate_scorecard(card2, m2).errors, bad)
        approve(m, card, "Mathematics")
        approve(m, card, "Mathematics", decision="denied")
        self.assertTrue(any("more than one approval" in e for e in sc.validate_scorecard(card, m).errors))

    def test_no_numeric_subject_threshold_exists(self):
        self.assertFalse([k for k in GATE if "subject" in k])

    def test_authorized_approvers_are_validated_in_the_manifest(self):
        for bad in ("Ann", ["Ann", "ann"], ["PENDING"], [""], [3]):
            self.assertTrue(mf.validate_manifest(fx.manifest(authorized_approvers=bad)).errors, bad)
        self.assertEqual(mf.validate_manifest(fx.manifest(authorized_approvers=["Ann", "Bo"])).errors, [])
        m, card = fx.frozen_world(80)
        self.assertIn("authorized_approvers", m)
        m["authorized_approvers"] = ["Mallory"]                       # editing the list after the run changes the manifest hash
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")


class SourceErrorTests(unittest.TestCase):
    """A lecture that is wrong (source) and a question the model got wrong (model) are different facts."""

    def rows(self, **judgments):
        m, card = fx.frozen_world(80)
        set_both(card, 0, source_error=SOURCE_ERROR if judgments.get("lecture_factually_incorrect") else None, **judgments)
        return m, card

    def attribution(self, **judgments):
        m, card = self.rows(**judgments)
        return m, card, sc.compute_metrics(card, m)

    def test_case_1_lecture_correct_and_model_correct(self):
        _, _, out = self.attribution()
        self.assertEqual(out["error_attribution_confirmatory"], {"clean": 80, "model_error_only": 0, "source_error_reproduced": 0, "source_error_and_model_error": 0})

    def test_case_2_lecture_correct_and_the_model_introduces_a_false_claim(self):
        m, card, out = self.attribution(unsupported_claims=True, key_false_about_subject=True, key_correct=False)
        self.assertEqual(out["error_attribution_confirmatory"]["model_error_only"], 1)
        self.assertEqual(out["criteria_confirmatory"]["lecture_factually_incorrect"]["count"], 0)
        self.assertEqual(out["release_blockers"]["confirmatory"]["confirmed_by_cause"], {"model_error": ["G1"]})
        self.assertEqual(out["verdict"]["overall"], "release_blocked")

    def test_case_3_wrong_lecture_faithfully_reproduced_is_not_a_model_error_but_stays_visible(self):
        m, card, out = self.attribution(lecture_factually_incorrect=True, key_false_about_subject=True, key_correct=True)
        self.assertEqual(sc.validate_scorecard(card, m).errors, [])
        self.assertEqual(out["error_attribution_confirmatory"]["source_error_reproduced"], 1)
        self.assertEqual(out["error_attribution_confirmatory"]["model_error_only"], 0)
        self.assertEqual(out["criteria_confirmatory"]["b_wrong_key_or_misleading"]["count"], 0)       # not attributed to the model
        self.assertEqual(out["criteria_confirmatory"]["lecture_factually_incorrect"]["count"], 1)
        self.assertEqual(out["release_blockers"]["confirmatory"]["confirmed_by_cause"], {"source_error_reproduced": ["G1"]})
        # a key that is false about the subject still blocks release (contract 11.1 b); the cause is reported, not excused
        self.assertEqual(out["verdict"]["overall"], "release_blocked")

    def test_case_4_wrong_lecture_and_a_separate_model_error_are_both_recorded(self):
        m, card, out = self.attribution(lecture_factually_incorrect=True, unsupported_claims=True, key_false_about_subject=True, key_correct=True)
        self.assertEqual(sc.validate_scorecard(card, m).errors, [])
        self.assertEqual(out["error_attribution_confirmatory"]["source_error_and_model_error"], 1)
        self.assertEqual(out["criteria_confirmatory"]["b_wrong_key_or_misleading"]["count"], 1)       # the model's own error still counts
        self.assertEqual(out["release_blockers"]["confirmatory"]["confirmed_by_cause"], {"model_error": ["G1"]})

    def test_a_source_error_never_excuses_a_model_error(self):
        _, _, out = self.attribution(lecture_factually_incorrect=True, key_correct=False)
        self.assertEqual(out["criteria_confirmatory"]["a_answerable_and_key_correct"]["count"], 79)
        self.assertEqual(out["criteria_confirmatory"]["b_wrong_key_or_misleading"]["count"], 1)

    def test_a_source_error_alone_does_not_count_against_the_model(self):
        _, _, out = self.attribution(lecture_factually_incorrect=True)
        self.assertEqual(out["criteria_confirmatory"]["b_wrong_key_or_misleading"]["count"], 0)
        self.assertEqual(out["criteria_confirmatory"]["a_answerable_and_key_correct"]["count"], 80)
        self.assertEqual(out["error_attribution_confirmatory"]["source_error_reproduced"], 1)

    def test_case_5_reviewers_who_disagree_about_the_lecture_stay_unresolved_until_adjudicated(self):
        m, card = fx.frozen_world(80)
        item = card["items"][0]
        item["reviews"][1]["judgments"]["lecture_factually_incorrect"] = True
        item["reviews"][1]["source_error"] = dict(SOURCE_ERROR)
        out = sc.compute_metrics(card, m)
        self.assertEqual(sc.resolve_item(item)["state"], "unresolved")
        self.assertEqual(out["gate_slots"]["unresolved"], 1)
        self.assertNotEqual(out["verdict"]["overall"], "criteria_a_to_c_met_pending_d")
        item["adjudication"] = {"adjudicator": "lead", "judgments": {"lecture_factually_incorrect": False}, "notes": "checked the textbook", "decided_at": "2000-01-02"}
        self.assertEqual(sc.validate_scorecard(card, m).errors, [])
        self.assertEqual(sc.resolve_item(item)["state"], "adjudicated")
        self.assertIs(sc.resolve_item(item)["final"]["lecture_factually_incorrect"], False)

    def test_an_adjudication_that_finds_a_source_error_needs_evidence(self):
        m, card = fx.frozen_world(80)
        item = card["items"][0]
        item["reviews"][1]["judgments"]["lecture_factually_incorrect"] = True       # reviewer B says so, without evidence
        item["adjudication"] = {"adjudicator": "lead", "judgments": {"lecture_factually_incorrect": True}, "notes": "agree", "decided_at": "2000-01-02"}
        errors = sc.validate_scorecard(card, m).errors
        self.assertTrue(any("source_error" in e for e in errors))
        item["reviews"][1]["source_error"] = dict(SOURCE_ERROR)        # evidence always comes from the reviewer who claimed the error
        self.assertEqual(sc.validate_scorecard(card, m).errors, [])

    def test_claiming_a_lecture_error_requires_evidence(self):
        review = fx.review(lecture_factually_incorrect=True)
        self.assertTrue(any("source_error" in e for e in sc.validate_scorecard(fx.scorecard([fx.item(reviews=[review])])).errors))
        review["source_error"] = {"quote": "", "note": "x"}
        self.assertTrue(sc.validate_scorecard(fx.scorecard([fx.item(reviews=[review])])).errors)
        review["source_error"] = dict(SOURCE_ERROR)
        self.assertEqual(sc.validate_scorecard(fx.scorecard([fx.item(reviews=[review])])).errors, [])

    def test_the_guide_contradiction_is_resolved_without_loosening_the_rule(self):
        # a key false about the subject cannot be "correct" while the lecture is correct ...
        bad = fx.review(key_false_about_subject=True, key_correct=True)
        self.assertTrue(sc.validate_scorecard(fx.scorecard([fx.item(reviews=[bad])])).errors)
        # ... nor while the lecture error has not been judged at all (the review is then incomplete, not valid)
        undecided = fx.review(key_false_about_subject=True, key_correct=True, lecture_factually_incorrect=None)
        self.assertEqual(sc.resolve_item(fx.item(reviews=[undecided]))["state"], "pending")
        self.assertEqual(sc.contradictions(undecided["judgments"]), [])        # not an error while the review is still being filled in
        # ... but it is consistent when the lecture itself is wrong
        ok = fx.review(key_false_about_subject=True, key_correct=True, lecture_factually_incorrect=True)
        ok["source_error"] = dict(SOURCE_ERROR)
        self.assertEqual(sc.validate_scorecard(fx.scorecard([fx.item(reviews=[ok])])).errors, [])

    def test_an_unjudged_source_question_means_the_review_is_not_complete(self):
        review = fx.review()
        review["judgments"]["lecture_factually_incorrect"] = None
        self.assertEqual(sc.resolve_item(fx.item(reviews=[review, fx.review(reviewer="b")]))["state"], "pending")

    def test_per_subject_reports_carry_the_attribution(self):
        m, card = fx.frozen_world(80)
        set_both(card, 0, source_error=SOURCE_ERROR, lecture_factually_incorrect=True)
        row = sc.compute_metrics(card, m)["by_subject_descriptive"]["Mathematics"]
        self.assertEqual(row["error_attribution"]["source_error_reproduced"], 1)
        self.assertEqual(row["lecture_factually_incorrect"]["count"], 1)


class PreservedSafeguardTests(unittest.TestCase):
    """Stage 6C protections must still hold (the full 6C tests also run unchanged)."""

    def test_synthetic_and_pending_runs_still_cannot_pass(self):
        m, card = fx.frozen_world(80)
        card["run"]["source"] = "pending"
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")
        m, card = fx.frozen_world(80)
        card["is_synthetic"] = True
        self.assertEqual(verdict(m, card)["overall"], "not_evidence")

    def test_hash_binding_and_frozen_config_still_fail_closed(self):
        m, card = fx.frozen_world(80)
        card["manifest_binding"]["sha256"] = "0" * 64
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")
        m, card = fx.frozen_world(80)
        card["run"]["prompt_sha256"] = "ff" * 32
        self.assertEqual(verdict(m, card)["overall"], "not_eligible")

    def test_an_all_good_run_with_no_weak_subject_still_reaches_only_pending_d(self):
        m, card = fx.frozen_world(80)
        v = verdict(m, card)
        self.assertEqual(v["overall"], "criteria_a_to_c_met_pending_d")
        self.assertEqual(set(v["subject_approvals"].values()), {"not_required"})
        self.assertEqual(v["blocking_reasons"], [])

    def test_reports_keep_per_subject_breakdowns_blockers_and_slot_accounting(self):
        m, card = fx.frozen_world(80, wrong=4)
        out = sc.compute_metrics(card, m)
        self.assertEqual(set(out["by_subject_descriptive"]), set(SUBJECTS))
        row = out["by_subject_descriptive"]["Mathematics"]
        for key in ("a_answerable_and_key_correct", "b_wrong_key_or_misleading", "b_key_false_about_subject", "c_range_contains_support",
                    "c_mcq_single_defensible_option", "slots", "items_by_state", "release_blocking_confirmed", "approval", "status", "basis_sha256"):
            self.assertIn(key, row)
        self.assertEqual(sum(r["slots"]["required"] for r in out["by_subject_descriptive"].values()), 80)
        self.assertEqual(set(out["release_blockers"]), {"confirmatory", "development", "pending"})

    def test_thresholds_are_the_contracts_own_numbers(self):
        self.assertEqual((GATE["min_reviewed_for_a_verdict"], GATE["required_confirmatory_items"], GATE["z"]), (80, 80, 1.96))
        self.assertEqual((GATE["a_observed_min"], GATE["a_wilson_lower_min"], GATE["b_wrong_or_misleading_max"]), (0.90, 0.80, 0.05))
        self.assertEqual((GATE["c_range_contains_support_min"], GATE["c_mcq_single_defensible_min"]), (0.90, 0.85))


class ReportOutputTests(unittest.TestCase):
    def test_cli_prints_blockers_origin_of_errors_approvals_and_the_basis_hash(self):
        import io
        import json
        import os
        import tempfile
        from contextlib import redirect_stdout
        from quiz_eval.__main__ import main
        m, card = fx.frozen_world(80, wrong=4)
        card["items"][5].update(reviews=[], exclusion=dict(EXCL))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, data in (("m.json", m), ("c.json", card)):
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(data, handle)
                paths.append(path)
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["check", paths[0], "--scorecard", paths[1]])
        text = buf.getvalue()
        for phrase in ("BLOCKING: 1 excluded confirmatory slot", "errors by origin", "approval: pending", "basis for an approval", "evaluation complete: False"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
