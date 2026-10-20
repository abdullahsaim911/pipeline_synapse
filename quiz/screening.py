"""
Deterministic, reject-only screening of untrusted quiz candidates.

A "candidate" is one item of the model's (already JSON-parsed) reply. This module
never calls a model. Its guarantees:

  * REJECT-ONLY. Passing screening does not mean a question is true, unambiguous,
    or has a correct key. It means none of the rules below fired. Accepted
    questions carry grounding="screened", never "verified".
  * The server owns times. A candidate that supplies any timestamp is rejected;
    evidence is cited by passage id only and mapped by quiz.passages.
  * Accepted questions are REBUILT from whitelisted, validated parts. Rejections
    record fixed reason codes only, never candidate text.
  * Nothing is relaxed to reach a requested count; shortfall is the caller's to
    report.

Candidate shape (an internal PROPOSAL; the model-facing schema is a later stage):

  common:  type ("mcq" | "tf"), prompt, explanation, hints (optional list, 0..2),
           concept {id, label}, evidence_ids [str], relies_on_exact_figure (bool),
           depends_on_visual (bool)
  mcq:     choices [str] (3..4), correct_index (int, 0-based)
  tf:      answer (bool)

Support screening (quiz.support) is lexical and numeric. It can reject a question
whose cited text shares nothing with it, whose MCQ key is less supported than a
distractor, or whose checkable numbers/units/signs/operators conflict with a
comparable value in the evidence. It cannot detect negation or antonym flips, a
true/false statement that is false for other reasons, or caption errors; it
rejects paraphrases that share no content word with the lecture.
"""

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from quiz.limits import DEFAULT_LIMITS, MAX_QUESTION_COUNT, MIN_QUESTION_COUNT, QuizLimits
from quiz.passages import PassageSet, map_evidence_ids, truncate_quote, verify_evidence_ranges
from quiz.reasons import Reason
from quiz.schemas import CHOICE_IDS, CONCEPT_ID_PATTERN, TF_CHOICES, QuizQuestion
from quiz.support import evaluate_support

_COMMON_FIELDS = {
    "type", "prompt", "explanation", "hints", "concept", "evidence_ids",
    "relies_on_exact_figure", "depends_on_visual",
}
_TYPE_FIELDS = {"mcq": {"choices", "correct_index"}, "tf": {"answer"}}
_REQUIRED_COMMON = _COMMON_FIELDS - {"hints"}

# Keys that would carry a model-written time or evidence object.
_TIMESTAMP_KEYS = {
    "start", "end", "start_seconds", "end_seconds", "start_time", "end_time", "time",
    "times", "timestamp", "timestamps", "timestamp_formatted", "seconds", "evidence",
    "evidence_times", "quote", "quotes", "range", "ranges", "moment", "intervention_id",
}

_TIME_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
_FORMAT_RE = re.compile(
    r"[`$\\^]|_\{|<\s*[A-Za-z/!][^>]*>|\*\*|__|~~|\]\(|^\s{0,3}#{1,6}\s|^\s*[-*+•]\s+"
)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_CONCEPT_ID_RE = re.compile(CONCEPT_ID_PATTERN)

# Provisional phrase list: things shown but not necessarily said.
_VISUAL_PHRASES = (
    "as shown", "shown here", "shown above", "shown below", "this graph", "this diagram",
    "this figure", "this chart", "this picture", "this image", "this slide", "this table",
    "on the slide", "on the screen", "in the picture", "pictured", "as you can see",
)
_BANNED_CHOICES = (
    "all of the above", "none of the above", "both of the above", "all of these",
    "none of these", "both a and b",
)
_HINT_NAMES_OPTION = re.compile(
    r"\b(?:option|choice|answer|letter)\s+(?:is\s+)?[a-d]\b"
    r"|\b(?:option|choice)\s+[1-4]\b"
    r"|\b(?:first|second|third|fourth|last)\s+(?:option|choice|answer)\b"
    r"|\b(?:correct|right)\s+(?:answer|option|choice)\b"
    r"|\banswer\s+is\b"
)
_HINT_TF_WORDS = re.compile(r"\b(?:true|false|correct|incorrect)\b")


_ELEMENTS = frozenset("""
h he li be b c n o f ne na mg al si p s cl ar k ca sc ti v cr mn fe co ni cu zn ga ge as se br kr rb sr y zr nb mo tc
ru rh pd ag cd in sn sb te i xe cs ba la ce pr nd pm sm eu gd tb dy ho er tm yb lu hf ta w re os ir pt au hg tl pb bi
po at rn fr ra ac th pa u np pu am cm bk cf es fm md no lr rf db sg bh hs mt ds rg cn nh fl mc lv ts og
""".split())
# Elements that really occur as a molecule with a digit, plus ozone.
_ELEMENTAL_MOLECULES = frozenset("h2 n2 o2 f2 cl2 br2 i2 o3 p4 s8".split())
# Common computing tokens that mix letters and digits (a lecture on encodings or networks may use them legitimately).
_KNOWN_TECH_TOKENS = frozenset("""
utf8 utf16 utf32 base64 base32 ipv4 ipv6 python2 python3 sha1 sha256 sha512 md5 mp3 mp4 html5 css3 es6 java8 cp1252
iso8859 win32 win64 x86 x64 h264 h265 http2 tls12 tls13 sql92 sql99 ieee754 rs232 rs485 ethernet802 wifi6 bluetooth5
""".split())
_MIXED_TOKEN_RE = re.compile(r"\b[a-z]{2,}[a-z0-9]*\d[a-z0-9]*\b")


def _segments(token: str) -> Optional[List[str]]:
    """
    Split a lowercase token into element symbols with optional counts (a chemical-formula shape), using as many symbols
    as possible ("co2" is carbon and oxygen, not cobalt), or None when it cannot be read that way.
    """
    memo: Dict[int, Optional[List[str]]] = {len(token): []}

    def solve(i: int) -> Optional[List[str]]:
        if i in memo:
            return memo[i]
        best: Optional[List[str]] = None
        for size in (1, 2):
            symbol = token[i:i + size]
            if len(symbol) != size or symbol not in _ELEMENTS:
                continue
            j = i + size
            while j < len(token) and token[j].isdigit():
                j += 1
            rest = solve(j)
            if rest is not None and (best is None or len(rest) + 1 > len(best)):
                best = [token[i:j]] + rest
        memo[i] = best
        return best

    return solve(0)


def _suspect_caption_token(text: str) -> bool:
    """
    A word that mixes letters and digits and is not a recognisable chemical formula is very likely an automatic-caption
    error (a real lecture produced "li8" for the mnemonic LIATE, and the model wrote a question whose key was "li8").
    Accepted: a formula made of two or more element symbols (h2o, co2, h2so4, c6h12o6) and the elemental molecules
    (o2, n2, ...). Single letters with a digit (x2, v0) are variables and are not examined. This catches only this one
    kind of caption error; it cannot catch a wrong word that looks normal, so it is a screen, not a guarantee.
    """
    for token in _MIXED_TOKEN_RE.findall(text.casefold()):
        if token in _KNOWN_TECH_TOKENS:
            continue
        parts = _segments(token)
        if parts is None:
            return True
        if len(parts) == 1 and token not in _ELEMENTAL_MOLECULES:
            return True
    return False


def fold(text: str) -> str:
    """Case-fold, drop punctuation, collapse spaces (comparison form only)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).replace("_", " ").split())


def _contains_phrase(folded: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", folded) is not None


@dataclass(frozen=True)
class Rejection:
    index: int                      # position in the candidate list
    reasons: Tuple[Reason, ...]     # fixed codes only, no candidate text


@dataclass(frozen=True)
class HintDrop:
    candidate_index: int
    reason: Reason


@dataclass(frozen=True)
class ScreeningResult:
    accepted: Tuple[QuizQuestion, ...]
    rejected: Tuple[Rejection, ...]
    dropped_hints: Tuple[HintDrop, ...]
    candidates_seen: int

    @property
    def type_mix(self) -> Dict[str, int]:
        """Recorded only. No MCQ / true-false ratio is enforced."""
        mix = {"mcq": 0, "tf": 0}
        for question in self.accepted:
            mix[question.type] += 1
        return mix

    @property
    def rejection_counts(self) -> Dict[str, int]:
        """Count of rejected candidates by their first reason."""
        counts: Dict[str, int] = {}
        for rejection in self.rejected:
            key = rejection.reasons[0].value
            counts[key] = counts.get(key, 0) + 1
        return counts


# --------------------------------------------------------------------------- text


def _text_reasons(value: Any, max_len: int) -> Tuple[Optional[str], List[Reason]]:
    """Common plain-text rules. Returns (stripped text or None, reasons)."""
    if not isinstance(value, str):
        return None, [Reason.WRONG_TYPE]
    text = value.strip()
    reasons: List[Reason] = []
    if not text or len(text) > max_len or not fold(text):
        reasons.append(Reason.TEXT_LENGTH)
    if any(unicodedata.category(ch) in ("Cc", "Cf") for ch in text):
        reasons.append(Reason.CONTROL_CHARACTER)
    if _FORMAT_RE.search(text) or _EMOJI_RE.search(text):
        reasons.append(Reason.FORMATTING_NOT_ALLOWED)
    if _TIME_RE.search(text):
        reasons.append(Reason.TIME_IN_TEXT)
    return (None if reasons else text), reasons


def _visual_reference(text: str) -> bool:
    folded = fold(text)
    return any(_contains_phrase(folded, phrase) for phrase in _VISUAL_PHRASES)


# --------------------------------------------------------------------------- hints


def _hint_reveals(hint: str, qtype: str, correct_text: Optional[str]) -> bool:
    folded = fold(hint)
    if _HINT_NAMES_OPTION.search(folded):
        return True
    if qtype == "tf":
        return _HINT_TF_WORDS.search(folded) is not None
    if correct_text:
        correct = fold(correct_text)
        if correct and _contains_phrase(folded, correct):
            return True
    return False


# --------------------------------------------------------------------------- one candidate


@dataclass
class _Parsed:
    qtype: str
    prompt: str
    explanation: str
    hints: List[str]
    concept_id: str
    concept_label: str
    evidence_ids: List[str]
    evidence_ranges: list
    choices: List[str]         # mcq only
    correct_index: int         # mcq only
    answer: bool               # tf only
    primary_id: str            # passage holding the best supporting window
    support_text: str = ""     # that best supporting window (server-chosen), used to spot two questions on one fact


def _validate_candidate(
    raw: Any, passages: PassageSet, limits: QuizLimits, index: int
) -> Tuple[Optional[_Parsed], List[Reason], List[HintDrop]]:
    if not isinstance(raw, dict):
        return None, [Reason.NOT_AN_OBJECT], []

    reasons: List[Reason] = []
    drops: List[HintDrop] = []

    def add(*items: Reason) -> None:
        for item in items:
            if item not in reasons:
                reasons.append(item)

    qtype = raw.get("type")
    if "type" not in raw:
        add(Reason.MISSING_FIELD)
    elif not isinstance(qtype, str) or qtype not in _TYPE_FIELDS:
        add(Reason.UNSUPPORTED_QUESTION_TYPE)
    if reasons:
        # without a supported type, field-level checks are meaningless
        _check_keys(raw, _COMMON_FIELDS | _TYPE_FIELDS["mcq"] | _TYPE_FIELDS["tf"], add)
        return None, reasons, drops

    _check_keys(raw, _COMMON_FIELDS | _TYPE_FIELDS[qtype], add)
    for field in _REQUIRED_COMMON | _TYPE_FIELDS[qtype]:
        if field not in raw:
            add(Reason.MISSING_FIELD)

    prompt, found = _text_reasons(raw.get("prompt"), limits.prompt_max) if "prompt" in raw else (None, [])
    add(*found)
    if prompt is not None and _visual_reference(prompt):
        add(Reason.VISUAL_REFERENCE)
    explanation, found = (
        _text_reasons(raw.get("explanation"), limits.explanation_max) if "explanation" in raw else (None, [])
    )
    add(*found)

    # flags
    if "depends_on_visual" in raw:
        if not isinstance(raw["depends_on_visual"], bool):
            add(Reason.WRONG_TYPE)
        elif raw["depends_on_visual"]:
            add(Reason.DEPENDS_ON_VISUAL)
    exact_figure = raw.get("relies_on_exact_figure")
    if "relies_on_exact_figure" in raw and not isinstance(exact_figure, bool):
        add(Reason.WRONG_TYPE)

    # concept
    concept_id = concept_label = ""
    concept = raw.get("concept")
    if "concept" in raw:
        if not isinstance(concept, dict) or set(concept) != {"id", "label"}:
            add(Reason.INVALID_CONCEPT)
        else:
            label, _ = _text_reasons(concept["label"], limits.concept_label_max)
            if (
                label is None
                or not isinstance(concept["id"], str)
                or not _CONCEPT_ID_RE.fullmatch(concept["id"])
            ):
                add(Reason.INVALID_CONCEPT)
            else:
                concept_id, concept_label = concept["id"], label

    # evidence
    ranges: list = []
    ids = raw.get("evidence_ids")
    if "evidence_ids" in raw:
        ranges, found = map_evidence_ids(ids, passages, limits)
        add(*found)
        if exact_figure is True and isinstance(ids, (list, tuple)) and len(set(map(str, ids))) < 2:
            add(Reason.EXACT_FIGURE_NEEDS_TWO_PASSAGES)

    # choices and key
    choices: List[str] = []
    correct_index = -1
    answer = False
    if qtype == "mcq":
        choices, correct_index = _check_mcq(raw, limits, add)
    elif "answer" in raw:
        if isinstance(raw["answer"], bool):
            answer = raw["answer"]
        else:
            add(Reason.INVALID_ANSWER_KEY)

    # hints (0..max_hints). Too many rejects the question; a bad hint is dropped.
    hints: List[str] = []
    raw_hints = raw.get("hints", [])
    if not isinstance(raw_hints, list):
        add(Reason.WRONG_TYPE)
    elif len(raw_hints) > limits.max_hints:
        add(Reason.TOO_MANY_HINTS)
    elif any(not isinstance(h, str) for h in raw_hints):
        add(Reason.WRONG_TYPE)
    else:
        correct_text = choices[correct_index] if qtype == "mcq" and 0 <= correct_index < len(choices) else None
        for hint in raw_hints:
            text, found = _text_reasons(hint, limits.hint_max)
            if text is None or _visual_reference(text):
                drops.append(HintDrop(index, Reason.HINT_INVALID))
            elif _hint_reveals(text, qtype, correct_text):
                drops.append(HintDrop(index, Reason.HINT_REVEALS_ANSWER))
            else:
                hints.append(text)

    if prompt is not None and any(_suspect_caption_token(t) for t in [prompt, *choices]):
        add(Reason.SUSPECT_CAPTION_TOKEN)

    primary_id = ""
    support_text = ""
    if not reasons:
        # Lexical/numeric support screening against the cited text (not proof).
        cited = [passages.passages[i - 1] for r in ranges for i in r.passage_indexes]
        evaluation = evaluate_support(
            qtype=qtype, prompt=prompt, options=choices, correct_index=correct_index,
            answer=answer, passages=cited, limits=limits,
        )
        add(*evaluation.reasons)
        if not reasons:
            ranges = _requote(ranges, evaluation.best_by_passage, limits)
            if not verify_evidence_ranges(ranges, passages, limits):
                add(Reason.EVIDENCE_INCONSISTENT)
            primary_id = f"S{evaluation.primary_index}"
            support_text = evaluation.best_by_passage.get(evaluation.primary_index, (0.0, ""))[1]

    if reasons:
        return None, reasons, []
    return _Parsed(
        qtype, prompt, explanation, hints, concept_id, concept_label,
        list(ids), ranges, choices, correct_index, answer, primary_id, support_text,
    ), [], drops


def _requote(ranges: list, best_by_passage: dict, limits: QuizLimits) -> list:
    """Quote = the best-supporting window among the range's own passages (real transcript text)."""
    out = []
    for r in ranges:
        scored = [best_by_passage[i] for i in r.passage_indexes if i in best_by_passage]
        if scored:
            best = max(scored, key=lambda item: item[0])  # first wins ties
            r = replace(r, quote=truncate_quote(best[1], limits.quote_max))
        out.append(r)
    return out


def _check_keys(raw: dict, allowed: set, add) -> None:
    for key in raw:
        if not isinstance(key, str):
            add(Reason.UNKNOWN_FIELD)
        elif key in allowed:
            continue
        elif key.strip().casefold() in _TIMESTAMP_KEYS:
            add(Reason.MODEL_SUPPLIED_TIMESTAMP)
        else:
            add(Reason.UNKNOWN_FIELD)


def _check_mcq(raw: dict, limits: QuizLimits, add) -> Tuple[List[str], int]:
    choices: List[str] = []
    raw_choices = raw.get("choices")
    if "choices" in raw:
        if not isinstance(raw_choices, list):
            add(Reason.WRONG_TYPE)
        elif not limits.mcq_min_choices <= len(raw_choices) <= limits.mcq_max_choices:
            add(Reason.INVALID_CHOICE_COUNT)
        else:
            for choice in raw_choices:
                text, found = _text_reasons(choice, limits.choice_max)
                if text is not None and _visual_reference(text):
                    found = found + [Reason.VISUAL_REFERENCE]
                    text = None
                add(*found)
                if text is not None:
                    choices.append(text)
            if len(choices) == len(raw_choices):
                folded = [fold(c) for c in choices]
                if len(set(folded)) != len(folded):
                    add(Reason.DUPLICATE_CHOICES)
                if any(_contains_phrase(f, b) for f in folded for b in _BANNED_CHOICES):
                    add(Reason.BANNED_CHOICE_PATTERN)
            else:
                choices = []

    key = raw.get("correct_index")
    correct_index = -1
    if "correct_index" in raw:
        if isinstance(key, bool) or not isinstance(key, int) or not (
            isinstance(raw_choices, list) and 0 <= key < len(raw_choices)
        ):
            add(Reason.INVALID_ANSWER_KEY)
        else:
            correct_index = key
    return choices, correct_index


# --------------------------------------------------------------------------- build


def _order(quiz_id: str, candidate_index: int, count: int) -> List[int]:
    """Deterministic shuffle independent of Python's RNG implementation."""
    def sort_key(position: int) -> str:
        return hashlib.sha256(f"{quiz_id}:{candidate_index}:{position}".encode()).hexdigest()
    return sorted(range(count), key=sort_key)


def _build(parsed: _Parsed, number: int, candidate_index: int, quiz_id: str) -> QuizQuestion:
    if parsed.qtype == "tf":
        choices = [{"id": cid, "text": text} for cid, text in TF_CHOICES]
        correct = "true" if parsed.answer else "false"
    else:
        order = _order(quiz_id, candidate_index, len(parsed.choices))
        choices = [{"id": CHOICE_IDS[slot], "text": parsed.choices[src]} for slot, src in enumerate(order)]
        correct = CHOICE_IDS[order.index(parsed.correct_index)]
    return QuizQuestion(
        id=f"q{number}",
        type=parsed.qtype,
        prompt=parsed.prompt,
        choices=choices,
        correct_choice_id=correct,
        explanation=parsed.explanation,
        hints=parsed.hints,
        concept={"id": parsed.concept_id, "label": parsed.concept_label},
        evidence=[
            {"kind": "transcript", "start_seconds": r.start, "end_seconds": r.end, "quote": r.quote}
            for r in parsed.evidence_ranges
        ],
        grounding="screened",
    )


# --------------------------------------------------------------------------- batch


def screen_candidates(
    candidates: Sequence[Any],
    passages: PassageSet,
    *,
    quiz_id: str,
    requested_count: int,
    limits: QuizLimits = DEFAULT_LIMITS,
) -> ScreeningResult:
    """
    Screen parsed candidates in order. The caller must already have parsed the
    model reply as strict JSON and obtained a list; anything else is a caller bug.

    Batch rules, applied in order against previously ACCEPTED questions only:
      * duplicate prompts (token-set overlap >= limits.duplicate_overlap)
      * no passage is the PRIMARY support (the passage holding the best supporting
        window, decided by the server, not by citation order) of more than
        max(ceil(requested / divisor), ceil(requested / passage_count)) questions.
        Secondary citations do not count toward the cap.
    Candidates beyond limits.max_candidates are rejected, not silently dropped.
    """
    if not isinstance(candidates, (list, tuple)):
        raise TypeError("candidates must be a list")
    if not MIN_QUESTION_COUNT <= requested_count <= MAX_QUESTION_COUNT:
        raise ValueError("requested_count out of range")
    if not isinstance(quiz_id, str) or not quiz_id:
        raise ValueError("quiz_id is required")

    # Per-passage cap is a DIVERSITY policy, not a correctness check. The floor keeps it
    # from making the requested count impossible on a lecture with few passages.
    cap = max(
        math.ceil(requested_count / limits.passage_share_divisor),
        math.ceil(requested_count / max(1, len(passages.passages))),
    )
    accepted: List[QuizQuestion] = []
    accepted_prompts: List[set] = []
    accepted_folded: List[str] = []
    accepted_concepts: List[Tuple[str, str]] = []
    accepted_support: List[Tuple[str, set]] = []
    usage: Dict[str, int] = {}
    rejected: List[Rejection] = []
    dropped: List[HintDrop] = []

    for index, raw in enumerate(candidates):
        if index >= limits.max_candidates:
            rejected.append(Rejection(index, (Reason.OVER_CANDIDATE_LIMIT,)))
            continue
        parsed, reasons, drops = _validate_candidate(raw, passages, limits, index)
        if parsed is None:
            rejected.append(Rejection(index, tuple(reasons)))
            continue

        folded = fold(parsed.prompt)
        tokens = set(folded.split())
        if any(_similar(folded, tokens, f, t, limits.duplicate_overlap)
               for f, t in zip(accepted_folded, accepted_prompts)):
            rejected.append(Rejection(index, (Reason.DUPLICATE_QUESTION,)))
            continue
        concept = (parsed.concept_id, fold(parsed.concept_label))
        support_tokens = set(fold(parsed.support_text).split())
        if any(concept[0] == c[0] or (concept[1] and concept[1] == c[1]) for c in accepted_concepts) or any(
                parsed.primary_id == pid and bool(support_tokens | toks)
                and len(support_tokens & toks) / len(support_tokens | toks) >= limits.duplicate_support_overlap
                for pid, toks in accepted_support):
            # the same concept label, or the same supporting text: two questions on one fact
            rejected.append(Rejection(index, (Reason.DUPLICATE_QUESTION,)))
            continue
        if usage.get(parsed.primary_id, 0) >= cap:
            rejected.append(Rejection(index, (Reason.PASSAGE_OVERUSED,)))
            continue

        try:
            question = _build(parsed, len(accepted) + 1, index, quiz_id)
        except ValidationError:
            rejected.append(Rejection(index, (Reason.SCHEMA_REJECTED,)))
            continue

        accepted.append(question)
        accepted_folded.append(folded)
        accepted_prompts.append(tokens)
        accepted_concepts.append(concept)
        accepted_support.append((parsed.primary_id, support_tokens))
        usage[parsed.primary_id] = usage.get(parsed.primary_id, 0) + 1
        dropped.extend(drops)

    return ScreeningResult(tuple(accepted), tuple(rejected), tuple(dropped), len(candidates))


def _similar(folded_a: str, tokens_a: set, folded_b: str, tokens_b: set, threshold: float) -> bool:
    if folded_a == folded_b:
        return True
    union = tokens_a | tokens_b
    return bool(union) and len(tokens_a & tokens_b) / len(union) >= threshold
