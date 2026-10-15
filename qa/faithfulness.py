"""
Programmatic grounding guardrail for Ask a Question (OFF by default).

After the model has answered, each sentence of the answer is checked against the passages that
were actually supplied to it (lecture excerpts, plus the moment's caption / explanation / visual
analysis when one was attached). The share of supported sentences is the faithfulness score. If
it is below a threshold the response is replaced by `insufficient_context`, instead of relying
only on the model's own obedience to the prompt.

Modes (ASK_FAITHFULNESS_MODE): off (default, nothing runs) | shadow (score and log, never change
the response; use this first to see real scores) | enforce (replace low-scoring answers).

Methods (ASK_FAITHFULNESS_METHOD):
  lexical (default): no model. A sentence is supported when enough of its content words, after
    stemming and the retrieval module's synonym groups, occur in the supplied passages; digits
    must occur too (spoken numbers count). Statements about what the excerpts do NOT say, and
    very short sentences, are not checked.
  nli: an entailment cross-encoder (ASK_NLI_MODEL) scores each sentence against the two most
    overlapping passages. Needs sentence-transformers and a locally cached model.

What it can and cannot do:
  * It catches answers that drift away from the supplied text. It does NOT catch an answer that
    faithfully repeats a wrong caption (a mis-heard "minus" for "plus" passes), and it cannot
    judge whether the evidence was complete.
  * The threshold is NOT calibrated: no labelled set of faithful and unfaithful answers exists
    yet. Run `shadow` on real traffic and review before using `enforce`. A wrongly blocked answer
    costs a student a useful reply.
  * A blocked answer is always `insufficient_context`. `not_in_lecture` is never used here: a
    low faithfulness score proves nothing about what the lecture covers.
  * Fail-open: if the checker raises, the original response is returned and a warning is logged.
"""

import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from .retrieval import content_terms, expand_term, tokenize
from .service import STATUS_ANSWERED, STATUS_INSUFFICIENT, AskOutcome

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "enforce")
METHODS = ("lexical", "nli")
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
GATE_MESSAGE = ("I couldn't confirm that answer against the lecture text, so I'm not going to give it. "
                "Try rephrasing your question, or ask about a specific time such as 3:20.")

_ATTRIBUTION_PREFIX = re.compile(
    r"^\s*(?:the\s+(?:instructor|lecturer|speaker|professor|teacher|captions?)\s+(?:also\s+)?"
    r"(?:says|said|states|stated|explains|notes|mentions|describes|uses|points out|adds|appears? to say)\s+(?:that\s+)?"
    r"|from\s+(?:this|these steps|these|the steps described),?\s*it\s+seems\s+(?:that\s+)?)",
    re.IGNORECASE)
_ABSENCE = re.compile(
    r"\b(?:do(?:es)? not|doesn't|don't|did not|didn't|never|no mention|not (?:stated|mentioned|defined|given|said|covered|"
    r"discussed|explained|included|specified)|cannot|can't|unable to|isn't|aren't|are not)\b", re.IGNORECASE)
_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FaithfulnessConfig:
    mode: str = "off"
    method: str = "lexical"
    threshold: float = 0.6               # minimum share of supported sentences (uncalibrated)
    term_support_ratio: float = 0.6      # lexical: share of a sentence's content words found in the passages
    min_terms: int = 2                   # sentences with fewer content words are not checked
    nli_model: str = DEFAULT_NLI_MODEL
    entail_threshold: float = 0.5
    allow_download: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("threshold", "term_support_ratio", "entail_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_terms < 1:
            raise ValueError("min_terms must be at least 1")

    @classmethod
    def from_env(cls) -> "FaithfulnessConfig":
        def pick(name, allowed, default):
            raw = (os.environ.get(name) or "").strip().lower()
            if raw and raw not in allowed:
                logger.warning("Ignoring invalid %s", name)
            return raw if raw in allowed else default

        threshold = 0.6
        raw = os.environ.get("ASK_FAITHFULNESS_THRESHOLD")
        if raw:
            try:
                value = float(raw)
                if 0.0 <= value <= 1.0:
                    threshold = value
                else:
                    raise ValueError
            except ValueError:
                logger.warning("Ignoring invalid ASK_FAITHFULNESS_THRESHOLD")
        return cls(mode=pick("ASK_FAITHFULNESS_MODE", MODES, "off"), method=pick("ASK_FAITHFULNESS_METHOD", METHODS, "lexical"),
                   threshold=threshold, nli_model=os.environ.get("ASK_NLI_MODEL") or DEFAULT_NLI_MODEL,
                   allow_download=(os.environ.get("ASK_MODELS_ALLOW_DOWNLOAD", "").strip().lower() in _TRUE))


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float                         # supported sentences / checked sentences (1.0 when nothing needed checking)
    method: str
    n_sentences: int
    n_checked: int
    n_supported: int
    unsupported: Tuple[str, ...] = ()    # first characters of the unsupported sentences (for tests; never logged)


# ---------------------------------------------------------------------------
# Sentence preparation
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def checkable_sentences(answer: str, min_terms: int = 2) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """(all sentences, [(sentence, content terms)] for the sentences that must be checked)."""
    sentences = split_sentences(answer)
    checked = []
    for sentence in sentences:
        if _ABSENCE.search(sentence):                    # "the excerpts do not state..." is a claim about the evidence
            continue
        body = _ATTRIBUTION_PREFIX.sub("", sentence)
        terms = content_terms(body)
        if len(terms) >= min_terms:
            checked.append((sentence, terms))
    return sentences, checked


_UNITS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * n for n, w in enumerate("x x twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "x"}


def spoken_numbers(tokens: Sequence[str]) -> set:
    """Integers (0-999) that occur as spoken words in a token list: 'seven', 'forty six', 'two hundred'."""
    values, i = set(), 0
    while i < len(tokens):
        token, nxt = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
        value = None
        if token in _TENS:
            value = _TENS[token]
            if _UNITS.get(nxt, 0) in range(1, 10) and nxt in _UNITS:
                value, i = value + _UNITS[nxt], i + 1
        elif token in _UNITS:
            value = _UNITS[token]
        if value is not None:
            if i + 1 < len(tokens) and tokens[i + 1] == "hundred":
                value, i = value * 100, i + 1
            values.add(value)
        i += 1
    return values


def _term_supported(term: str, stems: set, words: set, numbers: set) -> bool:
    if term.isdigit():
        return term in words or (len(term) <= 3 and int(term) in numbers)
    return any(variant in stems for variant in expand_term(term))


class LexicalSupportChecker:
    """No-model check: enough of each sentence's content words must occur in the supplied passages."""

    method = "lexical"

    def __init__(self, config: Optional[FaithfulnessConfig] = None):
        self.config = config or FaithfulnessConfig(mode="shadow")

    def check(self, answer: str, passages: Sequence[str]) -> FaithfulnessResult:
        sentences, checked = checkable_sentences(answer, self.config.min_terms)
        if not checked:
            return FaithfulnessResult(1.0, self.method, len(sentences), 0, 0)
        text = " ".join(passages)
        stems = set(tokenize(text))
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        words, numbers = set(tokens), spoken_numbers(tokens)
        supported, unsupported = 0, []
        for sentence, terms in checked:
            flags = [_term_supported(t, stems, words, numbers) for t in terms]
            digits_ok = all(ok for t, ok in zip(terms, flags) if t.isdigit())      # a number that is not in the text fails the sentence
            if digits_ok and sum(flags) / len(terms) >= self.config.term_support_ratio:
                supported += 1
            else:
                unsupported.append(sentence[:160])
        return FaithfulnessResult(supported / len(checked), self.method, len(sentences), len(checked), supported, tuple(unsupported))


class NLIChecker:
    """Entailment check with a cross-encoder NLI model; each sentence is judged against its two most overlapping passages."""

    method = "nli"

    def __init__(self, config: Optional[FaithfulnessConfig] = None, model=None, entail_index: Optional[int] = None):
        self.config = config or FaithfulnessConfig(mode="shadow", method="nli")
        self._model = model
        self._entail_index = entail_index

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.config.nli_model, device="cpu", local_files_only=not self.config.allow_download)
        if self._entail_index is None:
            id2label = getattr(getattr(self._model, "config", None), "id2label", None) \
                or getattr(getattr(getattr(self._model, "model", None), "config", None), "id2label", None) or {}
            self._entail_index = next((int(i) for i, name in id2label.items() if str(name).lower().startswith("entail")), 1)
        return self._model

    @staticmethod
    def _probabilities(row: Sequence[float]) -> List[float]:
        row = [float(x) for x in row]
        if all(0.0 <= x <= 1.0 for x in row) and abs(sum(row) - 1.0) < 1e-3:
            return row
        top = max(row)
        exp = [pow(2.718281828459045, x - top) for x in row]
        total = sum(exp)
        return [x / total for x in exp]

    def check(self, answer: str, passages: Sequence[str]) -> FaithfulnessResult:
        sentences, checked = checkable_sentences(answer, self.config.min_terms)
        if not checked:
            return FaithfulnessResult(1.0, self.method, len(sentences), 0, 0)
        model = self._load()
        passage_stems = [set(tokenize(p)) for p in passages]
        pairs = []
        for sentence, terms in checked:
            overlap = sorted(range(len(passages)), key=lambda i: (-sum(1 for t in terms if t in passage_stems[i]), i))[:2]
            premise = " ".join(passages[i][:900] for i in sorted(overlap))
            pairs.append((premise, _ATTRIBUTION_PREFIX.sub("", sentence)))
        supported, unsupported = 0, []
        for (sentence, _), row in zip(checked, model.predict(pairs)):
            if self._probabilities(row)[self._entail_index] >= self.config.entail_threshold:
                supported += 1
            else:
                unsupported.append(sentence[:160])
        return FaithfulnessResult(supported / len(checked), self.method, len(sentences), len(checked), supported, tuple(unsupported))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def apply_faithfulness_gate(outcome: AskOutcome, checker, config: FaithfulnessConfig) -> AskOutcome:
    """
    Return `outcome` unchanged unless mode is `enforce`, the answer was `answered`, and its score
    is below the threshold; then return an `insufficient_context` outcome without sources.
    """
    if config.mode == "off" or checker is None or outcome.status != STATUS_ANSWERED:
        return outcome
    if not outcome.evidence_passages:
        return outcome                                   # nothing to compare against (older callers)
    try:
        result = checker.check(outcome.answer_text, outcome.evidence_passages)
    except Exception as exc:                             # fail open: a broken checker must not silence answers
        logger.warning("Faithfulness check failed (%s); response left unchanged", type(exc).__name__)
        return outcome
    blocked = result.score < config.threshold
    logger.info("Faithfulness: mode=%s method=%s score=%.3f threshold=%.2f checked=%d/%d decision=%s",
                config.mode, result.method, result.score, config.threshold, result.n_checked, result.n_sentences,
                "block" if (blocked and config.mode == "enforce") else ("would_block" if blocked else "pass"))
    if blocked and config.mode == "enforce":
        return replace(outcome, status=STATUS_INSUFFICIENT, answer_text=GATE_MESSAGE, sources=[])
    return outcome


def build_gate_from_env() -> Optional[Callable[[AskOutcome], AskOutcome]]:
    """None when the guardrail is off; otherwise a function outcome -> outcome."""
    config = FaithfulnessConfig.from_env()
    if config.mode == "off":
        return None
    checker = NLIChecker(config) if config.method == "nli" else LexicalSupportChecker(config)
    return lambda outcome: apply_faithfulness_gate(outcome, checker, config)
