"""
Optional hybrid retrieval and reranking for Ask a Question (OFF by default).

Pipeline, applied only when a question goes through ordinary BM25 retrieval (mode "retrieval"):

    1. qa.retrieval.select_evidence runs unchanged. If it returns no evidence (the IDF coverage
       gate or the relative-score cutoff said "no match"), that answer is final: the dense model
       is never consulted, so "couldn't find it" decisions are identical to today.
    2. Chunks are ranked three ways: BM25 (positive scores only), dense cosine similarity over a
       sentence-embedding model (all chunks), then merged with Reciprocal Rank Fusion
       (score = sum of 1/(k + rank), one-based ranks, a chunk missing from a list adds nothing).
    3. The top `candidates` fused chunks are scored by a cross-encoder against the question and the
       best `rerank_top_k` chunks are kept (default 4, allowed 1-8).
    4. Those chunks get the same context padding and character budget as before, and become the
       labelled passages S1, S2, ... that the model cites.

Whole-transcript, time-range, summary and playback evidence is never touched.

Fail-open: if the embedding model, the reranker or an import is unavailable, or anything raises,
the BM25 evidence is returned unchanged and a warning is logged. Models are loaded offline from the
local Hugging Face cache unless ASK_MODELS_ALLOW_DOWNLOAD=1.

STATUS: not evaluated for benefit. A preregistered comparison found no demonstrated gain from
dense/hybrid retrieval over BM25 on the project benchmark, and the cross-encoder rerank step has
not been evaluated at all. Enable it only after a new evaluation on questions that were not used
for tuning. It adds a heavy dependency (sentence-transformers, torch) and CPU latency per question.

Environment: ASK_HYBRID_RETRIEVAL=1, ASK_EMBED_MODEL, ASK_RERANK_MODEL, ASK_RRF_K (60),
ASK_HYBRID_CANDIDATES (12), ASK_RERANK_TOP_K (4), ASK_MODELS_ALLOW_DOWNLOAD.
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import retrieval as R
from .retrieval import EvidenceConfig, EvidencePlan
from .transcript import TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name)
    if raw:
        try:
            value = int(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        logger.warning("Ignoring invalid %s", name)
    return default


@dataclass(frozen=True)
class HybridConfig:
    enabled: bool = False
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL
    rrf_k: int = 60
    candidates: int = 12
    rerank_top_k: int = 4
    allow_download: bool = False

    def __post_init__(self):
        if not 1 <= self.rerank_top_k <= 8:
            raise ValueError("rerank_top_k must be between 1 and 8")
        if not self.rerank_top_k <= self.candidates <= 50:
            raise ValueError("candidates must be between rerank_top_k and 50")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")

    @classmethod
    def from_env(cls) -> "HybridConfig":
        top_k = _int("ASK_RERANK_TOP_K", 4, 1, 8)
        return cls(
            enabled=_flag("ASK_HYBRID_RETRIEVAL"),
            embed_model=os.environ.get("ASK_EMBED_MODEL") or DEFAULT_EMBED_MODEL,
            rerank_model=os.environ.get("ASK_RERANK_MODEL") or DEFAULT_RERANK_MODEL,
            rrf_k=_int("ASK_RRF_K", 60, 1, 1000),
            candidates=_int("ASK_HYBRID_CANDIDATES", max(12, top_k), top_k, 50),
            rerank_top_k=top_k,
            allow_download=_flag("ASK_MODELS_ALLOW_DOWNLOAD"),
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60,
             weights: Optional[Sequence[float]] = None) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion. Each ranking lists item ids best first (one-based ranks). An item
    missing from a ranking contributes nothing from it (no penalty rank). Result: every item in
    any ranking as (id, score), by score descending, ties by the smaller id.
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: Dict[int, float] = {}
    for weight, ranking in zip(weights, rankings):
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                raise ValueError("an item appears twice in one ranking")
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def chunk_text(segments: Sequence[TranscriptSegment], rng: Tuple[int, int]) -> str:
    return " ".join(segments[i].text for i in range(rng[0], rng[1] + 1))


# ---------------------------------------------------------------------------
# Models (lazy, offline by default). Both classes satisfy tiny protocols so tests use fakes.
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """encode(texts) -> list of unit vectors (lists of floats)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, allow_download: bool = False):
        self.model_name, self.allow_download = model_name, allow_download
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer     # heavy import, only when enabled
                self._model = SentenceTransformer(self.model_name, device="cpu",
                                                  local_files_only=not self.allow_download)
            return self._model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._load().encode(list(texts), batch_size=32, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]


class CrossEncoderReranker:
    """score(query, passages) -> one relevance score per passage (higher is better)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, allow_download: bool = False):
        self.model_name, self.allow_download = model_name, allow_download
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device="cpu",
                                           local_files_only=not self.allow_download)
            return self._model

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        scores = self._load().predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
        return [float(s) for s in scores]


class _LRU:
    def __init__(self, capacity: int):
        self.capacity, self._data, self._lock = capacity, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkRanking:
    """All ranking stages for one question; chunk numbers index `chunks`."""
    chunks: List[Tuple[int, int]]
    texts: List[str]
    bm25_rank: List[int]                  # positive BM25 scores only, best first
    dense_rank: List[int]                 # every chunk, best first
    fused: List[Tuple[int, float]]        # RRF order over the union, with scores
    candidates: List[int]                 # the first `candidates` fused chunks
    rerank_scores: List[float]            # aligned with `candidates`
    rerank_order: List[int]               # `candidates` re-ordered by the cross-encoder, best first

    def reranked_full(self) -> List[int]:
        """Cross-encoder order for the candidates, followed by the remaining fused chunks (a full ranking)."""
        taken = set(self.rerank_order)
        return self.rerank_order + [i for i, _ in self.fused if i not in taken]

class HybridSelector:
    """
    Drop-in replacement for qa.retrieval.select_evidence (same call signature). Returns the BM25
    plan itself whenever hybrid retrieval does not apply or fails.
    """

    def __init__(self, config: Optional[HybridConfig] = None, embedder=None, reranker=None, cache_size: int = 64):
        self.config = config or HybridConfig.from_env()
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.embed_model, self.config.allow_download)
        self.reranker = reranker or CrossEncoderReranker(self.config.rerank_model, self.config.allow_download)
        self._chunk_vectors = _LRU(cache_size)

    def __call__(self, question: str, segments: Sequence[TranscriptSegment], current_time: Optional[float] = None,
                 has_moment: bool = False, config: Optional[EvidenceConfig] = None) -> EvidencePlan:
        config = config or EvidenceConfig()
        plan = R.select_evidence(question, segments, current_time=current_time, has_moment=has_moment, config=config)
        if not (self.config.enabled and plan.ok and not plan.complete and plan.mode == "retrieval"):
            return plan                                          # BM25 gate said no, or another evidence mode
        try:
            refined = self._refine(question, segments, config, plan)
        except Exception as exc:                                  # fail open: never break /ask over an optional step
            logger.warning("Hybrid retrieval failed (%s); using BM25 evidence", type(exc).__name__)
            return plan
        return refined or plan

    def _vectors_for(self, texts: List[str]) -> List[List[float]]:
        key = hashlib.sha256(("\x1f".join([self.config.embed_model] + texts)).encode("utf-8")).hexdigest()
        cached = self._chunk_vectors.get(key)
        if cached is not None:
            return cached
        vectors = self.embedder.encode(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned the wrong number of vectors")
        self._chunk_vectors.put(key, vectors)
        return vectors

    def rank_chunks(self, question: str, segments: Sequence[TranscriptSegment],
                    config: Optional[EvidenceConfig] = None) -> Optional["ChunkRanking"]:
        """
        Every ranking stage for one question (BM25, dense, fused, reranked). Used by the selector itself and by
        evaluation code, so what is measured is exactly what runs. Returns None when there is nothing to rank.
        Raises if a model fails.
        """
        config = config or EvidenceConfig()
        terms = R.content_terms(R._TIME_TEXT_RE.sub(" ", question))
        groups = [R.expand_term(t) for t in terms]
        chunks = R._group(range(len(segments)), segments, config)
        if not chunks or not groups:
            return None
        scores = R._rank_groups(groups, chunks, segments).scores
        bm25_rank = [i for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)) if scores[i] > 0]

        texts = [chunk_text(segments, rng) for rng in chunks]
        query_vec = self.embedder.encode([question])[0]
        dense = [cosine(query_vec, v) for v in self._vectors_for(texts)]
        dense_rank = sorted(range(len(chunks)), key=lambda i: (-dense[i], i))

        fused = rrf_fuse([bm25_rank, dense_rank], k=self.config.rrf_k)
        candidates = [i for i, _ in fused[: self.config.candidates]]
        rerank_scores = self.reranker.score(question, [texts[i] for i in candidates])
        if len(rerank_scores) != len(candidates):
            raise ValueError("reranker returned the wrong number of scores")
        order = sorted(zip(candidates, rerank_scores), key=lambda pair: (-pair[1], pair[0]))
        return ChunkRanking(chunks=chunks, texts=texts, bm25_rank=bm25_rank, dense_rank=dense_rank, fused=fused,
                            candidates=candidates, rerank_scores=list(rerank_scores), rerank_order=[i for i, _ in order])

    def _refine(self, question: str, segments: Sequence[TranscriptSegment], config: EvidenceConfig,
                bm25_plan: EvidencePlan) -> Optional[EvidencePlan]:
        ranking = self.rank_chunks(question, segments, config)
        if ranking is None:
            return None
        chunks, candidates = ranking.chunks, ranking.candidates
        bm25_rank = ranking.bm25_rank
        chosen = ranking.rerank_order[: self.config.rerank_top_k]

        selection = R._Selection(segments, config)
        for i in chosen:                                          # matched text first, so padding never displaces it
            selection.add(chunks[i])
        for i in chosen:
            selection.add(R._pad(chunks[i], segments, config.context_padding_seconds))
        if not selection.indices:
            return None
        logger.info("Hybrid retrieval: chunks=%d bm25=%d candidates=%d chosen=%d", len(chunks), len(bm25_rank),
                    len(candidates), len(chosen))
        return EvidencePlan("ok", selection.blocks(), complete=False, mode="retrieval", coverage=bm25_plan.coverage)


def build_selector_from_env() -> Optional[Callable]:
    """None when hybrid retrieval is disabled, so nothing heavy is imported or loaded."""
    config = HybridConfig.from_env()
    return HybridSelector(config) if config.enabled else None
