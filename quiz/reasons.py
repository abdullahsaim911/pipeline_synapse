"""
Fixed rejection reason codes.

Reasons are constants, never built from candidate text, so a rejection record can
be logged or returned without leaking untrusted model output.
"""

from enum import Enum


class Reason(str, Enum):
    # candidate structure
    NOT_AN_OBJECT = "not_an_object"
    UNKNOWN_FIELD = "unknown_field"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    UNSUPPORTED_QUESTION_TYPE = "unsupported_question_type"
    OVER_CANDIDATE_LIMIT = "over_candidate_limit"
    # model-supplied timestamps (the server alone owns times)
    MODEL_SUPPLIED_TIMESTAMP = "model_supplied_timestamp"
    TIME_IN_TEXT = "time_in_text"
    # evidence
    NO_EVIDENCE = "no_evidence"
    INVALID_EVIDENCE_ID = "invalid_evidence_id"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    DUPLICATE_EVIDENCE_ID = "duplicate_evidence_id"
    TOO_MANY_EVIDENCE_IDS = "too_many_evidence_ids"
    UNSUPPORTED_MOMENT_EVIDENCE = "unsupported_moment_evidence"
    EXACT_FIGURE_NEEDS_TWO_PASSAGES = "exact_figure_needs_two_passages"
    # text rules
    TEXT_LENGTH = "text_length"
    FORMATTING_NOT_ALLOWED = "formatting_not_allowed"
    CONTROL_CHARACTER = "control_character"
    INVALID_CONCEPT = "invalid_concept"
    # choices and key
    INVALID_CHOICE_COUNT = "invalid_choice_count"
    INVALID_CHOICES = "invalid_choices"
    DUPLICATE_CHOICES = "duplicate_choices"
    BANNED_CHOICE_PATTERN = "banned_choice_pattern"
    INVALID_ANSWER_KEY = "invalid_answer_key"
    # hints
    TOO_MANY_HINTS = "too_many_hints"
    HINT_REVEALS_ANSWER = "hint_reveals_answer"       # the hint is dropped, not the question
    HINT_INVALID = "hint_invalid"                     # the hint is dropped, not the question
    # visual dependence
    DEPENDS_ON_VISUAL = "depends_on_visual"
    VISUAL_REFERENCE = "visual_reference"
    # evidence support (lexical / numeric screening, not proof)
    NO_SUPPORTING_SENTENCE = "no_supporting_sentence"
    CITED_PASSAGE_NOT_RELEVANT = "cited_passage_not_relevant"
    KEY_CONFLICTS_WITH_EVIDENCE = "key_conflicts_with_evidence"
    ANSWER_MARGIN_TOO_SMALL = "answer_margin_too_small"
    NUMBER_NOT_IN_EVIDENCE = "number_not_in_evidence"
    NUMBER_CONFLICTS_WITH_EVIDENCE = "number_conflicts_with_evidence"
    OPERATOR_CONFLICTS_WITH_EVIDENCE = "operator_conflicts_with_evidence"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    # defensive: a question built from validated parts still failed the output schema
    SCHEMA_REJECTED = "schema_rejected"
    # likely automatic-caption error carried into the question (see screening._suspect_caption_token)
    SUSPECT_CAPTION_TOKEN = "suspect_caption_token"
    # batch level
    DUPLICATE_QUESTION = "duplicate_question"
    PASSAGE_OVERUSED = "passage_overused"
