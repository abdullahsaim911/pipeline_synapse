"""
Lightweight evidence selection for Ask a Question.

Pure Python, no embeddings. Evidence is chosen from ONE lecture's segments:

  0. SHORT TRANSCRIPTS: if the whole transcript fits `full_transcript_chars`
     (set from a token budget by the service), all of it is sent. Retrieval
     cannot save meaningful tokens there, and a retrieval miss would be a
     false "not found". Only in this mode is the evidence "complete".
  1. explicit time references ("at 3:20", "between 2:00 and 4:00", "the last
     five minutes", "in the first two minutes");
  2. whole-lecture requests ("summarize the lecture"), sampled evenly if the
     transcript exceeds the budget;
  3. the stretch just before and after the playback position, when the question
     points at the present ("this", "here") or a moment is attached;
  4. BM25 ranking over ~40 second chunks, with
       * query expansion (small synonym/morphology groups, irregular stems and
         the ASR sound alias "8" -> "ate", so "li8" matches "LIATE"),
       * an IDF-weighted coverage gate (ubiquitous words such as "function" or
         "u" cannot pass it alone),
       * a relative-score cutoff and a few segments of context padding so an
         explanation split across a chunk boundary is not cut in half.

Selected segments are returned as labelled passages (S1, S2, ...) so the model
can cite them and the backend can verify the citations.

LIMITATIONS
  * Matching is lexical. The synonym groups are a short hand-written list, not
    a thesaurus: "speed" will not match "velocity", and spoken numbers ("two")
    will not match digits ("2"). Only the ASR alias 8 -> "ate" is modelled; no
    fuzzy matching is used because it matches common words.
  * When retrieval fails on a long transcript the result is "no match", which
    callers must NOT present as proof the lecture lacks the topic.
  * Relationship questions work only if both concepts share words with the
    question; nothing models how concepts relate.
  * Whole-lecture questions on transcripts above the budget see an even sample
    of the lecture, not all of it.
  * Time phrases are English-only and heuristic; "minute 5" means the moment
    around 5:00, not 4:00-5:00.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .transcript import TranscriptSegment

# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

MIN_EVIDENCE_CHARS = 500


@dataclass(frozen=True)
class EvidenceConfig:
    max_chars: int = 12000          # evidence budget (the service derives it from a token budget)
    chunk_seconds: float = 40.0
    chunk_max_chars: int = 900
    top_k: int = 6                  # keyword-ranked chunks
    local_before_seconds: float = 45.0
    local_after_seconds: float = 15.0
    full_transcript_chars: int = 0  # send everything when the transcript fits (0 = never)
    min_coverage: float = 0.25      # IDF-weighted share of question terms that must match
    min_relative_score: float = 0.35  # drop chunks scoring below this fraction of the best
    context_padding_seconds: float = 25.0  # context added either side of a ranked chunk

    def __post_init__(self):
        if self.max_chars < MIN_EVIDENCE_CHARS:
            object.__setattr__(self, "max_chars", MIN_EVIDENCE_CHARS)


@dataclass(frozen=True)
class EvidenceBlock:
    """One labelled passage of transcript text (a contiguous run of segments)."""
    start: float
    end: float
    text: str
    first_index: int
    last_index: int
    id: str = ""                    # "S1", "S2", ... in chronological order


@dataclass
class EvidencePlan:
    # "ok", "empty_transcript", "needs_current_time", "time_out_of_range",
    # "no_question_terms", or "no_evidence"
    decision: str
    blocks: List[EvidenceBlock] = field(default_factory=list)
    complete: bool = False          # True only when the WHOLE transcript was selected
    mode: str = ""                  # "full", "time", "summary", "playback" or "retrieval"
    coverage: Optional[float] = None  # IDF-weighted question-term coverage, when computed

    @property
    def ok(self) -> bool:
        return self.decision == "ok" and bool(self.blocks)


@dataclass(frozen=True)
class TimeRef:
    kind: str                       # "range" or "needs_current"
    start: float = 0.0
    end: float = 0.0


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset("""
a an the and or but if then than so as of in on at to from by for with without into onto over under
up down out off again further once here there is are was were be been being am do does did doing done
have has had having can could would should will shall may might must i me my mine we us our you your
yours he him his she her it its they them their this that these those what which who whom whose when
where why how all any both each few more most other some such no nor not only own same too very just
also about between through during before after above below against while because until
explain explained tell describe describes mention mentioned mentions say said says talk talked talks
discuss discussed discusses show shown shows instructor teacher professor lecturer speaker lecture
lectures video earlier previously previous please give another example examples let know
relate related relates relationship relationships difference differences compare compared concept
concepts idea ideas thing things mean means meaning understand simply simple
""".split())

_DEICTIC_WORDS = frozenset("""
this that these those here now current currently moment screen slide shown showing displayed picture
image figure diagram graph chart equation visual drawing curve plot
""".split())


# Contraction suffixes are dropped ("what's" -> "what") so "s", "t" never become terms.
_CONTRACTION_RE = re.compile(r"(?<=[a-z])['’](?:s|t|d|m|re|ve|ll)\b")
_STOPWORDS = _STOPWORDS | frozenset(
    "don doesn didn isn aren wasn weren won wouldn couldn shouldn haven hasn hadn "
    "minute minutes min mins second seconds sec secs last past ago".split())

# Single letters that are usually mathematical symbols and worth matching (u, x, ...).
# a, i, o, s, t and d are excluded: they are English words or contraction/derivative debris.
_KEEP_SINGLE = frozenset("bcefghjklmnpqruvwxyz")

# Past/irregular forms the suffix stripper cannot reach ("chosen" -> "choose").
_IRREGULAR = {
    "chosen": "choose", "chose": "choose", "taken": "take", "took": "take", "given": "give",
    "gave": "give", "written": "write", "wrote": "write", "shown": "show", "found": "find",
    "made": "make", "seen": "see", "saw": "see", "known": "know", "knew": "know",
    "gone": "go", "went": "go", "begun": "begin", "began": "begin",
}

# Digits that automatic captioning writes for a spoken syllable inside a word
# ("li8" for "liate", "gr8" for "great"). Only 8 -> "ate" is modelled because it is
# the case observed in the sample data; each alias is added next to the original token.
_ASR_DIGIT_SOUNDS = {"8": "ate"}

# Query expansion: stems in one group are treated as the same search term. Kept short
# and specific on purpose; broad groups make unrelated chunks match.
_SYNONYM_GROUP_WORDS = (
    "choose select selection pick decide decision determine prioritize prioritise prefer choice",
    "order ranking rank priority prioritize prioritise sequence hierarchy precedence",
    "substitute substitution replace replacement plug swap",
    "type kind sort category variety",
    "acronym abbreviation stand",
    "integrate integral integration",
    "differentiate differentiation derivative",
)


def stem(token: str) -> str:
    """Crude suffix stripping so 'parts'/'part' and 'derived'/'derive' match."""
    token = _IRREGULAR.get(token, token)
    if token.isdigit() or len(token) <= 3:
        return token
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and not (suffix == "s" and token.endswith("ss")):
            candidate = token[: -len(suffix)] + replacement
            if len(candidate) >= 3:
                token = candidate
                break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _asr_alias(raw: str) -> Optional[str]:
    """The stem a caption token probably stands for, e.g. 'li8' -> 'liat'."""
    if raw.isdigit() or sum(c.isalpha() for c in raw) < 2:
        return None
    spoken = raw
    for digit, sound in _ASR_DIGIT_SOUNDS.items():
        spoken = spoken.replace(digit, sound)
    return stem(spoken) if spoken != raw else None


def _words(text: str) -> List[str]:
    return _TOKEN_RE.findall(_CONTRACTION_RE.sub("", text.lower()))


def tokenize(text: str) -> List[str]:
    """Stemmed tokens of caption text, each ASR-garbled token followed by its alias."""
    tokens: List[str] = []
    for raw in _words(text):
        tokens.append(stem(raw))
        alias = _asr_alias(raw)
        if alias:
            tokens.append(alias)
    return tokens


def content_terms(question: str) -> List[str]:
    """Unique stemmed non-stopword terms of the question, in order."""
    seen: Set[str] = set()
    terms: List[str] = []
    for raw in _words(question):
        if raw in _STOPWORDS or (len(raw) < 2 and raw not in _KEEP_SINGLE and not raw.isdigit()):
            continue
        term = stem(raw)
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _build_group_index() -> Dict[str, FrozenSet[str]]:
    index: Dict[str, Set[str]] = {}
    for line in _SYNONYM_GROUP_WORDS:
        members = {stem(w) for w in line.split()}
        for member in members:
            index.setdefault(member, set()).update(members)
    return {k: frozenset(v) for k, v in index.items()}


_GROUP_INDEX = _build_group_index()


def expand_term(term: str) -> FrozenSet[str]:
    """The term plus any stems the synonym groups treat as equivalent."""
    return _GROUP_INDEX.get(term, frozenset()) | {term}


def is_deictic(question: str) -> bool:
    return any(t in _DEICTIC_WORDS for t in _TOKEN_RE.findall(question.lower()))


# ---------------------------------------------------------------------------
# Time references
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}
_COUNT = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_STOPWORDS = _STOPWORDS | frozenset(_NUMBER_WORDS)     # "five" in "the last five minutes"
# Time expressions are handled by parse_time_reference; their digits are not content terms.
_TIME_TEXT_RE = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?|(?:minutes?|mins?|seconds?|secs?)\s+\d+|\d+\s*(?:minutes?|mins?|seconds?|secs?)\b")
_TS = r"(\d{1,2}):(\d{2})(?::(\d{2}))?"
_SEP = r"\s*(?:to|until|through|and|-|–|—)\s*"

_TS_RANGE_RE = re.compile(_TS + _SEP + _TS)
_TS_ONE_RE = re.compile(r"(?<![\d:])" + _TS + r"(?![\d:])")
_MIN_RANGE_RE = re.compile(r"minutes?\s+(\d{1,3})" + _SEP + r"(?:minutes?\s+)?(\d{1,3})\b")
_BETWEEN_MIN_RE = re.compile(r"between\s+(\d{1,3})\s+and\s+(\d{1,3})\s+minutes?")
_MIN_POINT_RES = (
    re.compile(r"\bminute\s+(\d{1,3})\b"),
    re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?[\s-]minute\s+mark\b"),
    re.compile(r"\b(\d{1,3})\s+minutes?\s+(?:in|into)\b"),
)
_SEC_POINT_RES = (
    re.compile(r"\bsecond\s+(\d{1,4})\b"),
    re.compile(r"\b(\d{1,4})\s+seconds?\s+(?:in|into)\b"),
)
_LAST_RE = re.compile(r"\b(?:last|past|previous)\s+(?:" + _COUNT + r"\s+)?(minutes?|mins?|seconds?|secs?)\b")
_OF_LECTURE_RE = re.compile(r"^\s+of\s+(?:the\s+)?(?:lecture|video|talk|recording)\b")
_JUST_NOW_RE = re.compile(r"\b(?:just now|just said|just explained|just mentioned|a moment ago|a few seconds ago)\b")
_FIRST_RE = re.compile(r"\bfirst\s+(?:" + _COUNT + r"\s+)?(minutes?|mins?|seconds?|secs?)\b")
_BEGINNING_RE = re.compile(r"\b(?:at|in|from)\s+the\s+(?:very\s+)?(?:beginning|start|opening)\b")
_END_RE = re.compile(
    r"\b(?:at|towards?|near)\s+the\s+(?:very\s+)?end(?:\s+of\s+(?:the\s+)?(?:lecture|video|talk))?\s*[?.!]*\s*$"
)
_GLOBAL_RE = re.compile(
    r"\b(?:summar(?:y|ize|ise)|overview|recap|whole lecture|entire (?:lecture|video)|"
    r"main (?:points?|ideas?)|key (?:points?|takeaways?|ideas?)|"
    r"what (?:is|was) (?:this|the) (?:lecture|video) about|what does (?:this|the) (?:lecture|video) cover)\b"
)

_POINT_BEFORE = 15.0
_POINT_AFTER = 45.0
_JUST_NOW_SECONDS = 60.0
_EDGE_SECONDS = 120.0


def is_global_question(question: str) -> bool:
    return bool(_GLOBAL_RE.search(question.lower()))


def _count(token: Optional[str]) -> int:
    if token is None:
        return 1
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)


def _clock(a: str, b: str, c: Optional[str]) -> float:
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def _point(t: float) -> TimeRef:
    return TimeRef("range", max(0.0, t - _POINT_BEFORE), t + _POINT_AFTER)


def _unit_seconds(unit: str) -> int:
    return 60 if unit.startswith("min") else 1


def parse_time_reference(
    question: str,
    current_time: Optional[float],
    lecture_end: float,
) -> Optional[TimeRef]:
    """
    Resolve a time phrase in the question to a range of seconds.

    Returns a TimeRef(kind="needs_current") when the phrase is relative to the
    playback position but none was supplied, or None if there is no time phrase.
    """
    q = question.lower()

    m = _TS_RANGE_RE.search(q)
    if m:
        a, b = _clock(*m.group(1, 2, 3)), _clock(*m.group(4, 5, 6))
        return TimeRef("range", min(a, b), max(a, b))

    for regex in (_MIN_RANGE_RE, _BETWEEN_MIN_RE):
        m = regex.search(q)
        if m:
            a, b = int(m.group(1)) * 60, int(m.group(2)) * 60
            return TimeRef("range", min(a, b), max(a, b))

    m = _TS_ONE_RE.search(q)
    if m:
        return _point(_clock(*m.group(1, 2, 3)))

    for regex in _MIN_POINT_RES:
        m = regex.search(q)
        if m:
            return _point(int(m.group(1)) * 60)
    for regex in _SEC_POINT_RES:
        m = regex.search(q)
        if m:
            return _point(int(m.group(1)))

    m = _LAST_RE.search(q)
    if m:
        span = _count(m.group(1)) * _unit_seconds(m.group(2))
        if _OF_LECTURE_RE.match(q[m.end():]):
            return TimeRef("range", max(0.0, lecture_end - span), lecture_end)
        if current_time is None:
            return TimeRef("needs_current")
        return TimeRef("range", max(0.0, current_time - span), current_time)

    if _JUST_NOW_RE.search(q) and current_time is not None:
        return TimeRef("range", max(0.0, current_time - _JUST_NOW_SECONDS), current_time)

    m = _FIRST_RE.search(q)
    if m:
        return TimeRef("range", 0.0, _count(m.group(1)) * _unit_seconds(m.group(2)))
    if _BEGINNING_RE.search(q):
        return TimeRef("range", 0.0, _EDGE_SECONDS)
    if _END_RE.search(q):
        return TimeRef("range", max(0.0, lecture_end - _EDGE_SECONDS), lecture_end)

    return None


# ---------------------------------------------------------------------------
# Chunking and ranking
# ---------------------------------------------------------------------------

Range = Tuple[int, int]  # inclusive segment index range


def _group(indices: Sequence[int], segments: Sequence[TranscriptSegment],
           config: EvidenceConfig) -> List[Range]:
    """Group consecutive segment indices into chunks of bounded duration and size."""
    ranges: List[Range] = []
    first = last = -1
    chars = 0
    for i in indices:
        seg = segments[i]
        if first >= 0 and i == last + 1 \
                and seg.end - segments[first].start <= config.chunk_seconds \
                and chars + len(seg.text) + 1 <= config.chunk_max_chars:
            last = i
            chars += len(seg.text) + 1
        else:
            if first >= 0:
                ranges.append((first, last))
            first = last = i
            chars = len(seg.text) + 1
    if first >= 0:
        ranges.append((first, last))
    return ranges


@dataclass(frozen=True)
class Ranking:
    scores: List[float]             # BM25 score per chunk
    matched: Set[int]               # indices of question-term groups found anywhere in the lecture
    coverage: float                 # IDF-weighted share of the question that matched, 0..1


def _rank_groups(groups: Sequence[FrozenSet[str]], chunks: Sequence[Range],
                 segments: Sequence[TranscriptSegment]) -> Ranking:
    """
    BM25 over chunks for expanded question terms.

    Each group is one question term plus its equivalents; its frequency in a chunk
    is the sum over the group. `coverage` weights every term by IDF, counting a term
    absent from the lecture at the maximum weight, so a question made of rare
    unmatched words scores low even if one ubiquitous word ("function", "u") matched.
    """
    k1, b = 1.5, 0.75
    docs: List[Counter] = []
    for first, last in chunks:
        counter: Counter = Counter()
        for i in range(first, last + 1):
            counter.update(tokenize(segments[i].text))
        docs.append(counter)

    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
    n_docs = len(docs)

    matched: Set[int] = set()
    idf: List[float] = []
    for g, group in enumerate(groups):
        df = sum(1 for d in docs if any(d[t] > 0 for t in group))
        if df:
            matched.add(g)
        idf.append(math.log(1 + (n_docs - df + 0.5) / (df + 0.5)))

    scores: List[float] = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for g, group in enumerate(groups):
            tf = sum(doc[t] for t in group)
            if tf:
                score += idf[g] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)

    total = sum(idf)
    coverage = (sum(idf[g] for g in matched) / total) if total else 0.0
    return Ranking(scores, matched, coverage)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class _Selection:
    """Selected segment indices under a character budget."""

    def __init__(self, segments: Sequence[TranscriptSegment], config: "EvidenceConfig"):
        self.segments = segments
        self.config = config
        self.max_chars = config.max_chars
        self.indices: Set[int] = set()
        self.chars = 0

    def _cost(self, rng: Range) -> int:
        return sum(len(self.segments[i].text) + 1
                   for i in range(rng[0], rng[1] + 1) if i not in self.indices)

    def add(self, rng: Range) -> bool:
        """Add a range if it fits (the very first range may be cut down to fit)."""
        cost = self._cost(rng)
        if self.chars + cost <= self.max_chars:
            self._commit(rng)
            return True
        if not self.indices:
            first, last = rng
            while last > first and self._cost((first, last)) > self.max_chars:
                last -= 1
            self._commit((first, last))
            return True
        return False

    def add_evenly(self, ranges: Sequence[Range]) -> None:
        """Add all ranges, or an even chronological sample if they do not fit."""
        if not ranges:
            return
        total = sum(self._cost(r) for r in ranges)
        remaining = self.max_chars - self.chars
        if total <= remaining:
            for r in ranges:
                self._commit(r)
            return
        keep = max(1, int(len(ranges) * remaining / total))
        if keep >= len(ranges):
            picks = list(ranges)
        elif keep == 1:
            picks = [ranges[0]]
        else:
            step = (len(ranges) - 1) / (keep - 1)
            picks = [ranges[round(i * step)] for i in range(keep)]
        for r in picks:
            self.add(r)

    def _commit(self, rng: Range) -> None:
        for i in range(rng[0], rng[1] + 1):
            if i not in self.indices:
                self.indices.add(i)
                self.chars += len(self.segments[i].text) + 1

    def blocks(self) -> List[EvidenceBlock]:
        """Labelled passages (S1, S2, ...): each run is split to chunk size so citations stay narrow."""
        runs: List[List[int]] = []
        for i in sorted(self.indices):
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        blocks: List[EvidenceBlock] = []
        for run in runs:
            for first, last in _group(run, self.segments, self.config):
                segs = self.segments[first:last + 1]
                blocks.append(EvidenceBlock(
                    start=segs[0].start, end=segs[-1].end,
                    text=" ".join(s.text for s in segs),
                    first_index=first, last_index=last,
                    id=f"S{len(blocks) + 1}"))
        return blocks


def _pad(rng: Range, segments: Sequence[TranscriptSegment], seconds: float) -> Range:
    """Extend a chunk by `seconds` each side so an explanation split across a boundary stays whole."""
    lo, hi = rng
    start, end = segments[lo].start - seconds, segments[hi].end + seconds
    while lo > 0 and segments[lo - 1].end > start:
        lo -= 1
    while hi < len(segments) - 1 and segments[hi + 1].start < end:
        hi += 1
    return lo, hi


def _overlapping(segments: Sequence[TranscriptSegment], start: float, end: float) -> List[int]:
    return [s.index for s in segments if s.end > start and s.start < end]


def select_evidence(
    question: str,
    segments: Sequence[TranscriptSegment],
    current_time: Optional[float] = None,
    has_moment: bool = False,
    config: Optional[EvidenceConfig] = None,
) -> EvidencePlan:
    """
    Choose transcript evidence for a question about a single lecture.

    `segments` must already belong to the requested video; nothing here looks
    anywhere else. `current_time` is the playback position (the attached
    intervention's timestamp if none was supplied). The plan's `complete` flag is
    True only when the whole transcript was selected, which is the only case in
    which "the lecture does not cover this" is a justified conclusion.
    """
    config = config or EvidenceConfig()
    if not segments:
        return EvidencePlan("empty_transcript")

    lecture_end = segments[-1].end
    ref = parse_time_reference(question, current_time, lecture_end)
    if ref and ref.kind == "needs_current":
        return EvidencePlan("needs_current_time")

    selection = _Selection(segments, config)
    global_question = is_global_question(question)
    terms = content_terms(_TIME_TEXT_RE.sub(" ", question))

    # Nothing to search for and nothing to anchor to: answerable by no model, so do not ask one.
    anchored = has_moment or (current_time is not None and is_deictic(question))
    if not terms and ref is None and not global_question and not anchored:
        return EvidencePlan("no_question_terms")

    # 0. Short transcript: send all of it; retrieval could not save tokens here.
    total_chars = sum(len(s.text) + 1 for s in segments)
    if ref is None and total_chars <= min(config.full_transcript_chars, config.max_chars):
        selection.add_evenly(_group(range(len(segments)), segments, config))
        return EvidencePlan("ok", selection.blocks(), complete=True, mode="full")

    mode = "retrieval"

    # 1. Explicit time reference.
    if ref:
        indices = _overlapping(segments, ref.start, ref.end)
        if not indices:
            return EvidencePlan("time_out_of_range")
        selection.add_evenly(_group(indices, segments, config))
        mode = "time"

    # 2. Whole-lecture request: an even sample, no keyword filtering.
    elif global_question:
        selection.add_evenly(_group(range(len(segments)), segments, config))
        mode = "summary"

    # 3. Around the playback position.
    if current_time is not None and (has_moment or is_deictic(question)) and not global_question:
        window = _overlapping(
            segments, current_time - config.local_before_seconds,
            current_time + config.local_after_seconds)
        if window:
            selection.add((window[0], window[-1]))
            if not selection.indices - set(range(window[0], window[-1] + 1)):
                mode = "playback"

    # 4. Ranking over the whole lecture, with expanded question terms.
    coverage: Optional[float] = None
    if terms and not (global_question and not ref):
        chunks = _group(range(len(segments)), segments, config)
        ranking = _rank_groups([expand_term(t) for t in terms], chunks, segments)
        coverage = ranking.coverage
        if ranking.matched and ranking.coverage >= config.min_coverage:
            cutoff = max(ranking.scores) * config.min_relative_score
            ranked = sorted((i for i, s in enumerate(ranking.scores) if s > 0 and s >= cutoff),
                            key=lambda i: ranking.scores[i], reverse=True)
            top = ranked[: config.top_k]
            for i in top:                       # matched text first, so padding never displaces it
                selection.add(chunks[i])
            for i in top:                       # then context, only where it still fits
                selection.add(_pad(chunks[i], segments, config.context_padding_seconds))

    if not selection.indices:
        return EvidencePlan("no_question_terms" if not terms else "no_evidence", coverage=coverage)
    return EvidencePlan("ok", selection.blocks(), mode=mode, coverage=coverage)
