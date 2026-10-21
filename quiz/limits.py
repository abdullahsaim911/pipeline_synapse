"""
Limits for quiz generation.

Every value is PROVISIONAL unless the contract says otherwise: none has been
measured against real model output. Values marked "contract" come from
docs/QUIZ_CONTRACT.md; they are still not validated in practice.
"""

from dataclasses import dataclass

MIN_QUESTION_COUNT = 5      # contract
MAX_QUESTION_COUNT = 10     # contract
DEFAULT_QUESTION_COUNT = 6  # contract
MIN_USABLE_QUESTIONS = 3    # contract: fewer is "insufficient_material"
MAX_HINTS = 2               # contract: 0 to 2 hints per question
MAX_EVIDENCE_IDS = 3        # contract: 1 to 3 evidence items


@dataclass(frozen=True)
class QuizLimits:
    # Candidates asked of the model = requested + extra, never above the cap.
    candidate_extra: int = 2           # provisional
    max_candidates: int = 12           # provisional

    # Evidence passages (estimated tokens, qa.tokens.estimate_tokens).
    passage_min_tokens: int = 300      # provisional
    passage_max_tokens: int = 400      # provisional
    max_passages: int = 40             # provisional; longer transcripts are flagged as truncated

    max_evidence_ids: int = MAX_EVIDENCE_IDS
    max_hints: int = MAX_HINTS

    # Text bounds (characters, after stripping); contract section 4.3.
    prompt_max: int = 400
    choice_max: int = 200
    explanation_max: int = 600
    hint_max: int = 200
    concept_label_max: int = 60
    quote_max: int = 240

    mcq_min_choices: int = 3
    mcq_max_choices: int = 4

    # Lexical support screening (PROVISIONAL, unmeasured). Windows are sentence-sized;
    # captions without punctuation are cut into overlapping word windows.
    window_words: int = 24
    window_step: int = 12
    stem_min_coverage: float = 0.2        # share of the question's content words in one window (>= 1 word)
    answer_min_coverage: float = 0.5      # share of the key's distinctive words in that same window
    statement_min_coverage: float = 0.5   # true/false: share of the statement's content words
    answer_margin: float = 0.2            # key must beat each distractor's coverage by this much

    # Generation (PROVISIONAL, unmeasured; no real model has been called).
    min_transcript_words: int = 300       # below this the lecture is "too_short"
    deadline_seconds: float = 100.0       # overall limit across both model calls
    second_call_min_remaining: float = 45.0   # a second call is not started with less time left
    temperature: float = 0.4
    output_tokens_base: int = 1000
    output_tokens_per_candidate: int = 700
    output_tokens_cap: int = 16384

    # No passage may back more than ceil(requested / divisor) questions.
    passage_share_divisor: int = 3     # provisional
    # Token-set Jaccard at or above which two prompts count as duplicates.
    duplicate_overlap: float = 0.7     # provisional
    # Two questions whose server-chosen supporting text in the same passage overlaps at least this much test the same
    # fact. Supporting windows are 24 words stepping by 12, so an identical window is 1.0 and neighbours are about 0.33.
    duplicate_support_overlap: float = 0.6   # provisional


DEFAULT_LIMITS = QuizLimits()


def max_output_tokens(candidates: int, limits: "QuizLimits" = None) -> int:
    """Output budget for `candidates` questions (PROVISIONAL; thinking tokens, if any, are unmeasured)."""
    limits = limits or DEFAULT_LIMITS
    return min(limits.output_tokens_cap, limits.output_tokens_base + limits.output_tokens_per_candidate * candidates)


def candidate_target(requested_count: int, limits: QuizLimits = DEFAULT_LIMITS) -> int:
    """How many candidates to ask for: requested + extra, capped."""
    return min(requested_count + limits.candidate_extra, limits.max_candidates)
