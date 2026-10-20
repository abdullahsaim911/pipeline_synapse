"""Shared vocabulary, and the contract's provisional release-gate thresholds."""

import re

MANIFEST_VERSION = 1
# Version 2 (Stage 6E): adds the required judgment lecture_factually_incorrect, the per-item returned_position,
# and top-level subject_approvals. No version-1 scorecard of real data exists.
SCORECARD_VERSION = 2

SUBJECTS = ("Mathematics", "Physics", "Chemistry", "Biology", "Computer science")
SPLITS = ("development", "confirmatory", "pending")
MANIFEST_STATUSES = ("template", "draft", "frozen")
QUESTION_TYPES = ("mcq", "tf")
SAMPLE_REVIEW_STATUSES = ("pending", "approved", "rejected", "excluded")
ITEM_STATUSES = ("pending", "reviewed", "excluded")
WORD_COUNT_STATUSES = ("pending", "from_dataset_manifest", "measured")
RUN_SOURCES = ("pending", "recorded_real_run", "synthetic")

# Why an item leaves the evaluation. An exclusion is a recorded decision with notes, never a silent drop.
EXCLUSION_REASONS = (
    "requires_visual",       # the question needs a diagram/chart/slide the transcript does not carry
    "not_answerable",        # the cited transcript does not contain the answer (reviewer's choice to exclude from a reference set)
    "ambiguous_source",      # the transcript itself is too unclear to judge
    "transcript_defect",     # a caption error or gap makes the item unjudgeable
    "duplicate",
    "out_of_scope_type",     # not MCQ or true/false
    "other",
)

# In a SCORECARD an exclusion means "this item can not be judged at all". A question that is wrong,
# unanswerable or visual-dependent IS judged (and counts as a failure); it must never be "excluded" away.
SCORECARD_EXCLUSION_REASONS = tuple(r for r in EXCLUSION_REASONS if r not in ("requires_visual", "not_answerable"))

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,60}$")
LECTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,40}$")

# The human judgments recorded for each generated question. Values: True, False, "n/a" (only where
# the criterion does not apply) or None (not judged yet).
JUDGMENTS = (
    "answerable_from_transcript",       # the answer follows from what is SAID in the lecture
    "key_correct",                      # the marked answer is right ACCORDING TO THE LECTURE (REVIEWER_GUIDE.md)
    "single_defensible_option",         # MCQ: exactly one option can be defended; true/false: n/a
    "evidence_quote_valid",             # the quote is real, accurate transcript text
    "evidence_relevant",                # the quote actually supports the question and key
    "evidence_range_contains_support",  # the cited time range contains the supporting statement
    "unsupported_claims",               # question/explanation asserts something the lecture does not
    "misleading_distractors",           # MCQ: a distractor is defensible or misleading; true/false: n/a
    "requires_visual",                  # needs a figure/slide/animation unavailable in the transcript
    "key_false_about_subject",          # the key is a FALSE statement about the subject itself
    "lecture_factually_incorrect",      # SOURCE error: the lecture statement(s) this question rests on are factually wrong
)
NA_ONLY_FOR_TF = ("single_defensible_option", "misleading_distractors")

# Contract docs/QUIZ_CONTRACT.md section 11.1. PROVISIONAL: these are the contract's own numbers,
# copied here, not tuned. Changing one means changing the contract first.
GATE = {
    "min_reviewed_for_a_verdict": 80,     # 80 questions from the 20 confirmatory lectures
    "required_confirmatory_items": 80,    # the pre-registered sample: exactly this many frozen confirmatory slots
    "a_observed_min": 0.90,               # answerable AND key correct, observed
    "a_wilson_lower_min": 0.80,           # 95% Wilson lower bound
    "b_wrong_or_misleading_max": 0.05,
    "c_range_contains_support_min": 0.90,
    "c_mcq_single_defensible_min": 0.85,
    "z": 1.96,
}

# POLICY (not in the contract text): every confirmatory item needs this many INDEPENDENT reviews before it
# can count toward the gate. docs/QUIZ_CONTRACT.md 11.2 only estimates one reviewer's time; the two-review
# rule comes from quiz_eval/REVIEWER_GUIDE.md and needs explicit human confirmation (see README, open policy issues).
REVIEWS_REQUIRED_FOR_GATE = 2

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# What a recorded real run must state about itself. Missing, blank or placeholder values fail CLOSED.
RUN_REQUIRED_FIELDS = ("run_id", "prompt_version", "prompt_sha256", "model", "config_sha256", "code_version",
                       "generated_at", "raw_responses_sha256")
RUN_HASH_FIELDS = ("prompt_sha256", "config_sha256", "raw_responses_sha256")
# What a FROZEN manifest pins, so the run can be checked against what was registered before it.
FROZEN_CONFIG_FIELDS = ("prompt_version", "prompt_sha256", "model", "config_sha256")
FROZEN_CONFIG_HASH_FIELDS = ("prompt_sha256", "config_sha256")

BLOCKING_JUDGMENT = "key_false_about_subject"

# Judgment values that record a failure. An item with any of these can never be EXCLUDED.
FAILING_VALUES = {
    "answerable_from_transcript": False, "key_correct": False, "single_defensible_option": False,
    "evidence_quote_valid": False, "evidence_relevant": False, "evidence_range_contains_support": False,
    "unsupported_claims": True, "misleading_distractors": True, "requires_visual": True, "key_false_about_subject": True,
    "lecture_factually_incorrect": True,
}

# Judgments that are errors CREATED BY THE MODEL (it asserted or marked something the lecture does not support).
# A source-lecture error is the separate judgment lecture_factually_incorrect and is never counted here.
SOURCE_ERROR_JUDGMENT = "lecture_factually_incorrect"

# Approved policy (Stage 6E): the number of excluded confirmatory slots the gate tolerates. Zero, for every reason.
EXCLUSION_CAP_CONFIRMATORY = 0

# A subject-level approval is a recorded human decision; it never infers itself from scores.
SUBJECT_APPROVAL_DECISIONS = ("granted", "denied")

# Pairs of judgments that cannot both be recorded: (key, value, key, value, why).
# Also checked in scorecard.contradictions(): key_false_about_subject true with key_correct true is a contradiction
# ONLY while lecture_factually_incorrect is false. If the lecture itself is wrong, a key that faithfully follows it
# is correct according to the lecture and false about the subject (REVIEWER_GUIDE.md, Stage 6E).
CONTRADICTIONS = (
    ("answerable_from_transcript", False, "key_correct", True, "a question that is not answerable from the transcript cannot have a key judged correct"),
    ("requires_visual", True, "answerable_from_transcript", True, "a question that needs a visual cannot be answerable from the transcript alone"),
)

def is_placeholder(value) -> bool:
    return not isinstance(value, str) or value.strip() == "" or value.strip().upper().startswith("PENDING")

# Contract section 11.2 protocol shape (checked as warnings on a frozen manifest, not enforced).
PROTOCOL = {"development_per_subject": 1, "confirmatory_per_subject": 4, "questions_per_confirmatory_lecture": 4}
