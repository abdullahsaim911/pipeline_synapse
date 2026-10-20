"""
Deterministic lexical and numeric SCREENING of a question against its cited evidence.

Nothing here verifies factual truth. Every check compares surface forms (stemmed
content words, parsed numbers and units, arithmetic words) between the candidate
and the transcript text the server cited. A candidate can pass all of them and
still be wrong, ambiguous, or based on a caption error; a good candidate can fail
them. All thresholds are PROVISIONAL (quiz.limits) and unmeasured.

What the checks can and cannot catch (see also quiz/README in the contract):

  CAN reject: a cited passage that shares no content with the question or key;
    an MCQ key that is less supported by the evidence than a distractor; a
    checkable number, decimal, sign or unit that differs from a comparable value
    in the evidence; an arithmetic word whose opposite is the only one cited.
  CANNOT detect: negation or antonym flips ("increases" vs "decreases" are two
    unrelated stems), a true/false statement that is false for a reason other
    than a number/sign/unit/operator, a wrong key whose words still appear, a
    paraphrase that shares no content word with the lecture (it is rejected), or
    errors in the captions themselves. Distractors drawn from the same sentence as
    the key tie with it and are rejected as ambiguous (a known false rejection).
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from quiz.limits import QuizLimits
from quiz.reasons import Reason

_EPS = 1e-9

STOPWORDS = frozenset("""
a an the and or but nor so yet of in on at to for from by with without within into onto over under
about above below between among through during before after as than then that this these those it its
is are was were be been being am do does did done doing have has had having can could will would shall
should may might must not no yes if else when where which who whom whose what why how there here
i you he she we they me him her us them my your his our their one ones some any each every all both
either neither such very just also only more most much many few other another same own too up down out
off again further once now lecture lecturer speaker instructor teacher professor video explanation
according say says said saying state states stated mention mentions mentioned describe describes
described explain explains explained following given
""".split())

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_DIGIT_WORDS = {w: v for w, v in _ONES.items() if v <= 9}

_UNITS = {
    "percent": "percent", "%": "percent", "degree": "degree", "degrees": "degree", "°": "degree",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "m": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "km": "km",
    "centimeter": "cm", "centimeters": "cm", "cm": "cm", "millimeter": "mm", "millimeters": "mm", "mm": "mm",
    "second": "s", "seconds": "s", "sec": "s", "s": "s", "minute": "min", "minutes": "min", "min": "min",
    "hour": "h", "hours": "h", "hr": "h",
    "kilogram": "kg", "kilograms": "kg", "kg": "kg", "gram": "g", "grams": "g",
    "newton": "newton", "newtons": "newton", "joule": "joule", "joules": "joule",
    "volt": "volt", "volts": "volt", "ampere": "ampere", "amperes": "ampere", "amp": "ampere", "amps": "ampere",
    "ohm": "ohm", "ohms": "ohm", "watt": "watt", "watts": "watt", "hertz": "hertz", "hz": "hertz",
    "mole": "mole", "moles": "mole", "mol": "mole", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "kelvin": "kelvin", "celsius": "celsius",
}

_TOKEN_RE = re.compile(r"(?<![\w])-?\d+(?:,\d{3})*(?:\.\d+)?|[^\W\d_]+|[%°/]")
_NUMERIC_RE = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MINUS_SIGN_CONTEXT = {"is", "equals", "equal", "be", "of", "to", "at", "by", "than", "from", "was", "are"}

# Opposing arithmetic / sign words. A conflict needs the opposite group present in
# the evidence and the claimed group absent from it.
_GROUPS = {
    "add": {"plus", "add", "adds", "added", "adding", "addition"},
    "subtract": {"minus", "subtract", "subtracts", "subtracted", "subtracting", "subtraction"},
    "multiply": {"times", "multiply", "multiplies", "multiplied", "multiplying", "multiplication"},
    "divide": {"divide", "divides", "divided", "dividing", "division"},
    "positive": {"positive"},
    "negative": {"negative"},
}
_OPPOSITES = (("add", "subtract"), ("multiply", "divide"), ("positive", "negative"))


# --------------------------------------------------------------------------- stems


def stem(word: str) -> str:
    """Crude suffix stripping. Consistent, not linguistically correct."""
    w = word
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ations"):
        w = w[:-6]
    elif w.endswith("ation"):
        w = w[:-5]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4 and (w[-3] in "sxz" or w.endswith(("ches", "shes"))):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("at"):
        w = w[:-2]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
        w = w[:-1]
    return w


# --------------------------------------------------------------------------- numbers


@dataclass(frozen=True)
class Quantity:
    value: Decimal               # signed
    unit: Optional[str]          # canonical unit, or None
    is_decimal: bool
    labels: FrozenSet[str]       # nearby content stems, used to judge comparability

    @property
    def checkable(self) -> bool:
        """Contract: 10 or more and decimals; extended to signed values and values with a unit.
        Unitless integers 1 to 9 stay unchecked."""
        return abs(self.value) >= 10 or self.is_decimal or self.value < 0 or self.unit is not None


@dataclass(frozen=True)
class Analysis:
    tokens: FrozenSet[str]               # stemmed content words and "#<number>" tokens
    quantities: Tuple[Quantity, ...]
    words: FrozenSet[str]                # raw alphabetic words (for arithmetic-word checks)


def _tokenize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    text = text.replace("'", "").replace("’", "")
    return _TOKEN_RE.findall(text)


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _below_100(toks: List[str], i: int) -> Optional[Tuple[int, int]]:
    if i >= len(toks):
        return None
    t = toks[i]
    if t in _TENS:
        if i + 1 < len(toks) and toks[i + 1] in _DIGIT_WORDS and toks[i + 1] != "zero":
            return _TENS[t] + _DIGIT_WORDS[toks[i + 1]], i + 2
        return _TENS[t], i + 1
    if t in _ONES and t != "one":
        return _ONES[t], i + 1
    return None


def _below_1000(toks: List[str], i: int) -> Optional[Tuple[int, int]]:
    if i + 1 < len(toks) and toks[i] in _DIGIT_WORDS and toks[i] != "zero" and toks[i + 1] == "hundred":
        base, j = _DIGIT_WORDS[toks[i]] * 100, i + 2
        if j < len(toks) and toks[j] == "and":
            j += 1
        rest = _below_100(toks, j)
        return (base + rest[0], rest[1]) if rest else (base, i + 2)
    return _below_100(toks, i)


def _word_number(toks: List[str], i: int) -> Optional[Tuple[int, int]]:
    first = _below_1000(toks, i)
    if first is None and i + 1 < len(toks) and toks[i] in _DIGIT_WORDS and toks[i + 1] == "thousand":
        first = (_DIGIT_WORDS[toks[i]], i + 1)
    if first is None:
        return None
    value, j = first
    if j < len(toks) and toks[j] == "thousand":
        value, j = value * 1000, j + 1
        rest = _below_1000(toks, j)
        if rest:
            value, j = value + rest[0], rest[1]
    return value, j


def _read_number(toks: List[str], i: int) -> Optional[Tuple[Decimal, bool, int]]:
    """(value, is_decimal, next index) for a number starting at toks[i], or None."""
    if i >= len(toks):
        return None
    t = toks[i]
    if _NUMERIC_RE.match(t):
        try:
            return Decimal(t.replace(",", "")), "." in t, i + 1
        except InvalidOperation:
            return None
    parsed = _word_number(toks, i)
    if parsed is None:
        return None
    value, j = parsed
    if j + 1 < len(toks) and toks[j] == "point" and (toks[j + 1] in _DIGIT_WORDS or toks[j + 1].isdigit()):
        digits, k = "", j + 1
        while k < len(toks) and (toks[k] in _DIGIT_WORDS or (toks[k].isdigit() and len(toks[k]) == 1)):
            digits += str(_DIGIT_WORDS[toks[k]]) if toks[k] in _DIGIT_WORDS else toks[k]
            k += 1
        return Decimal(f"{value}.{digits}"), True, k
    return Decimal(value), False, j


def _read_leading_decimal(toks: List[str], i: int) -> Optional[Tuple[Decimal, bool, int]]:
    """'point five' -> 0.5 (a decimal written without its integer part)."""
    k, digits = i + 1, ""
    while k < len(toks) and (toks[k] in _DIGIT_WORDS or (toks[k].isdigit() and len(toks[k]) == 1)):
        digits += str(_DIGIT_WORDS[toks[k]]) if toks[k] in _DIGIT_WORDS else toks[k]
        k += 1
    return (Decimal("0." + digits), True, k) if digits else None


def _read_unit(toks: List[str], j: int) -> Tuple[Optional[str], int]:
    if j >= len(toks) or toks[j] not in _UNITS:
        return None, j
    unit, k = _UNITS[toks[j]], j + 1
    if unit == "percent":
        return unit, k
    if k + 1 < len(toks) and toks[k] in ("per", "/") and toks[k + 1] in _UNITS:
        unit, k = unit + "/" + _UNITS[toks[k + 1]], k + 2
    if k < len(toks) and toks[k] in ("squared", "cubed"):
        unit, k = unit + ("^2" if toks[k] == "squared" else "^3"), k + 1
    return unit, k


def _is_content_word(t: str) -> bool:
    return len(t) > 1 and t.isalpha() and t not in STOPWORDS and t not in _UNITS and t not in _ONES and t not in _TENS


def _labels(toks: List[str], start: int, end: int) -> FrozenSet[str]:
    found = set()
    for k in range(start - 1, max(-1, start - 4), -1):
        if _is_content_word(toks[k]):
            found.add(stem(toks[k]))
            break
    for k in range(end, min(len(toks), end + 3)):
        if _is_content_word(toks[k]):
            found.add(stem(toks[k]))
            break
    return frozenset(found)


def analyze(text: str) -> Analysis:
    toks = _tokenize(text)
    words = frozenset(t for t in toks if t.isalpha())
    stems: Set[str] = set()
    quantities: List[Quantity] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        start, negative = i, False
        parsed = None
        if t == "negative" or (t == "minus" and (i == 0 or toks[i - 1] in _MINUS_SIGN_CONTEXT)):
            parsed = _read_number(toks, i + 1)
            negative = parsed is not None
        if parsed is None and t == "point":
            parsed = _read_leading_decimal(toks, i)
        if parsed is None:
            parsed = _read_number(toks, i)
        if parsed is not None:
            value, is_decimal, j = parsed
            if negative:
                value = -value
            unit, after = _read_unit(toks, j)
            quantities.append(Quantity(value, unit, is_decimal, _labels(toks, start, after)))
            stems.add("#" + _fmt(value))
            i = j
            continue
        if t == "%":
            stems.add("percent")
        elif len(t) > 1 and t.isalpha() and t not in STOPWORDS:
            stems.add(stem(t))
        i += 1
    return Analysis(frozenset(stems), tuple(quantities), words)


def compare_quantity(q: Quantity, evidence: Sequence[Quantity]) -> str:
    """'supported', 'conflict' (evidence has a comparable but different value) or 'absent'."""
    same = [e for e in evidence if e.value == q.value]
    if q.unit is not None and same:
        if any(e.unit == q.unit for e in same):
            return "supported"
        if any(e.unit is not None for e in same):
            return "conflict"                               # same number, but only with other units
        return "supported"                                  # evidence repeats the number without a unit
    if same:
        return "supported"
    if q.value != 0 and any(e.value == -q.value for e in evidence):
        return "conflict"                                   # same magnitude, opposite sign
    for e in evidence:
        if (q.unit is not None and q.unit == e.unit) or (q.labels & e.labels):
            return "conflict"                               # comparable quantity, different value
    return "absent"


def operator_conflict(claim_words: FrozenSet[str], evidence_words: FrozenSet[str]) -> bool:
    for left, right in _OPPOSITES:
        for a, b in ((left, right), (right, left)):
            if (claim_words & _GROUPS[a]) and not (claim_words & _GROUPS[b]):
                if (evidence_words & _GROUPS[b]) and not (evidence_words & _GROUPS[a]):
                    return True
    return False


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    passage_index: int
    text: str
    tokens: FrozenSet[str]


def window_texts(text: str, limits: QuizLimits) -> List[str]:
    """
    Sentence-sized units. Captions often have no punctuation, so a unit longer than
    window_words is cut into overlapping word windows (window_words, step window_step).
    """
    out: List[str] = []
    for unit in _SENTENCE_SPLIT_RE.split(text):
        words = unit.split()
        if not words:
            continue
        if len(words) <= limits.window_words:
            out.append(" ".join(words))
            continue
        start = 0
        while True:
            out.append(" ".join(words[start:start + limits.window_words]))
            if start + limits.window_words >= len(words):
                break
            start += limits.window_step
    return out


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class Evaluation:
    reasons: Tuple[Reason, ...]
    primary_index: Optional[int]                       # passage holding the best supporting window
    best_by_passage: Dict[int, Tuple[float, str]]      # passage index -> (score, best window text)


def _coverage(needed: FrozenSet[str], tokens: FrozenSet[str]) -> float:
    return len(needed & tokens) / len(needed) if needed else 0.0


def _distinctive(all_tokens: List[FrozenSet[str]], stem_tokens: FrozenSet[str]) -> List[FrozenSet[str]]:
    """Per option: words unique to it, minus the question's own words; widen if that is empty."""
    out = []
    for i, tokens in enumerate(all_tokens):
        others = frozenset().union(*[t for j, t in enumerate(all_tokens) if j != i])
        chosen = tokens - others - stem_tokens
        out.append(chosen or (tokens - stem_tokens) or tokens)
    return out


def evaluate_support(
    *,
    qtype: str,
    prompt: str,
    options: Sequence[str],
    correct_index: int,
    answer: bool,
    passages: Sequence,
    limits: QuizLimits,
) -> Evaluation:
    """
    Screen one question against its cited passages. `passages` are the quiz.passages
    Passage objects the model cited (already validated and ordered by the caller).
    """
    reasons: List[Reason] = []

    def add(reason: Reason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    prompt_analysis = analyze(prompt)
    stem_tokens = prompt_analysis.tokens
    mcq = qtype == "mcq"
    option_analyses = [analyze(o) for o in options] if mcq else []
    uniques = _distinctive([a.tokens for a in option_analyses], stem_tokens) if mcq else []
    key_tokens = uniques[correct_index] if mcq else frozenset()

    windows: List[Window] = []
    for passage in passages:
        for text in window_texts(passage.text, limits):
            windows.append(Window(passage.index, text, analyze(text).tokens))

    best_by_passage: Dict[int, Tuple[float, str]] = {}
    relevant_passages: Set[int] = set()
    stem_windows: List[Window] = []
    supporting: List[Tuple[float, Window]] = []

    if not stem_tokens or (mcq and not key_tokens):
        add(Reason.NO_SUPPORTING_SENTENCE)  # nothing checkable in the question or the key
    else:
        if mcq:
            need = 1  # the key must also be present in the same window (answer_min_coverage)
        else:
            need = min(3, len(stem_tokens))
        for w in windows:
            hits = len(stem_tokens & w.tokens)
            stem_cov = hits / len(stem_tokens)
            key_cov = _coverage(key_tokens, w.tokens) if mcq else 0.0
            score = stem_cov + key_cov
            if score > best_by_passage.get(w.passage_index, (-1.0, ""))[0]:
                best_by_passage[w.passage_index] = (score, w.text)
            if hits >= 1 or (mcq and key_cov > 0):
                relevant_passages.add(w.passage_index)
            if mcq:
                stem_ok = hits >= need and stem_cov >= limits.stem_min_coverage - _EPS
                if stem_ok:
                    stem_windows.append(w)
                    if key_cov >= limits.answer_min_coverage - _EPS:
                        supporting.append((score, w))
            elif hits >= need and stem_cov >= limits.statement_min_coverage - _EPS:
                stem_windows.append(w)
                supporting.append((score, w))

        if not supporting:
            add(Reason.NO_SUPPORTING_SENTENCE)
        if any(p.index not in relevant_passages for p in passages):
            add(Reason.CITED_PASSAGE_NOT_RELEVANT)

        if mcq and stem_windows:
            def best_cov(tokens: FrozenSet[str]) -> float:
                return max((_coverage(tokens, w.tokens) for w in stem_windows), default=0.0)
            key_cov_best = best_cov(key_tokens)
            for i, tokens in enumerate(uniques):
                if i == correct_index:
                    continue
                d = best_cov(tokens)
                if d > key_cov_best + _EPS:
                    add(Reason.KEY_CONFLICTS_WITH_EVIDENCE)
                elif key_cov_best - d < limits.answer_margin - _EPS:
                    add(Reason.ANSWER_MARGIN_TOO_SMALL)

    # numbers, decimals, units, signs (against ALL cited text, not one window)
    evidence_quantities: List[Quantity] = []
    evidence_words: Set[str] = set()
    for passage in passages:
        a = analyze(passage.text)
        evidence_quantities.extend(a.quantities)
        evidence_words |= a.words
    claim_quantities = list(prompt_analysis.quantities)
    claim_words = set(prompt_analysis.words)
    if mcq:
        claim_quantities += list(option_analyses[correct_index].quantities)
        claim_words |= option_analyses[correct_index].words
    for q in claim_quantities:
        if not q.checkable:
            continue
        status = compare_quantity(q, evidence_quantities)
        if status == "conflict" and (mcq or answer):
            add(Reason.NUMBER_CONFLICTS_WITH_EVIDENCE)
        elif status == "absent":
            add(Reason.NUMBER_NOT_IN_EVIDENCE)
        # a FALSE true/false statement may deliberately carry a different comparable value

    if (mcq or answer) and operator_conflict(frozenset(claim_words), frozenset(evidence_words)):
        add(Reason.OPERATOR_CONFLICTS_WITH_EVIDENCE)

    primary = max(supporting, key=lambda s: (s[0], -s[1].passage_index))[1].passage_index if supporting else None
    return Evaluation(tuple(reasons), primary, best_by_passage)
