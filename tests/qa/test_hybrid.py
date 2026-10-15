"""qa.hybrid: RRF, gating, reranking, caching and fail-open behaviour (fake models), plus real-model smoke tests."""

import os
import unittest
from unittest import mock

from tests.qa.helpers import covered_indices, long_lecture_segments, make_segments
from qa import retrieval as R
from qa.hybrid import (
    CrossEncoderReranker,
    HybridConfig,
    HybridSelector,
    SentenceTransformerEmbedder,
    build_selector_from_env,
    cosine,
    rrf_fuse,
)

QUESTION = "What is an eigenvalue?"


def lecture_lines():
    """54 captions, 6 per chunk: chunk 2 (lines 12-17) mentions eigenvalues, chunk 6 (lines 36-41) the determinant."""
    lines = []
    for i in range(54):
        if 12 <= i < 15:
            lines.append(f"an eigenvalue is a scalar that scales its eigenvector line {i}")
        elif 36 <= i < 39:
            lines.append(f"set the determinant of A minus lambda I to zero line {i}")
        else:
            lines.append(f"the ribosome moves along the membrane protein number {i}")
    return lines


EIGEN_LINES = set(range(12, 18))
DET_LINES = set(range(36, 42))


class FakeEmbedder:
    """Deterministic 3-d vectors: eigenvalue text -> x, determinant text -> near x, everything else -> z."""

    def __init__(self, fail=False, short=False):
        self.calls, self.fail, self.short = [], fail, short

    @staticmethod
    def vec(text):
        t = text.lower()
        if "determinant" in t:
            return [0.9, 0.436, 0.0]
        if "eigenvalue" in t:
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedder down")
        vectors = [self.vec(t) for t in texts]
        return vectors[:-1] if self.short else vectors


class FakeReranker:
    def __init__(self, fail=False, scorer=None):
        self.calls, self.fail = [], fail
        self.scorer = scorer or (lambda p: 0.9 if "determinant" in p else (0.5 if "eigenvalue" in p else 0.0))

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.fail:
            raise RuntimeError("reranker down")
        return [self.scorer(p) for p in passages]


def blocks_of(plan):
    return [(b.first_index, b.last_index) for b in plan.blocks]


class RRFTests(unittest.TestCase):
    def test_hand_calculated_example(self):
        fused = rrf_fuse([[2, 0], [0, 1, 2]], k=60)
        scores = dict(fused)
        self.assertEqual(scores[0], 1 / 62 + 1 / 61)           # bm25 rank 2, dense rank 1
        self.assertEqual(scores[1], 1 / 62)                    # absent from the first list: no penalty rank
        self.assertEqual(scores[2], 1 / 61 + 1 / 63)           # bm25 rank 1, dense rank 3
        self.assertEqual([i for i, _ in fused], [0, 2, 1])

    def test_ranks_are_one_based(self):
        self.assertEqual(rrf_fuse([[7]], k=60), [(7, 1 / 61)])

    def test_equal_scores_break_by_smaller_id(self):
        fused = rrf_fuse([[1, 0], [0, 1]])
        self.assertEqual(fused[0][1], fused[1][1])
        self.assertEqual([i for i, _ in fused], [0, 1])

    def test_empty_first_list_falls_back_to_the_second(self):
        self.assertEqual([i for i, _ in rrf_fuse([[], [2, 0, 1]])], [2, 0, 1])

    def test_weights_and_invalid_input(self):
        self.assertEqual([i for i, _ in rrf_fuse([[0], [1]], weights=[1.0, 2.0])], [1, 0])
        with self.assertRaises(ValueError):
            rrf_fuse([[0, 0]])
        with self.assertRaises(ValueError):
            rrf_fuse([[0], [1]], weights=[1.0])

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"rerank_top_k": 0}, {"rerank_top_k": 9}, {"candidates": 2, "rerank_top_k": 4}, {"candidates": 51}, {"rrf_k": 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                HybridConfig(**kwargs)

    def test_defaults_are_off(self):
        self.assertFalse(HybridConfig().enabled)

    def test_from_env(self):
        env = {"ASK_HYBRID_RETRIEVAL": "1", "ASK_RERANK_TOP_K": "3", "ASK_HYBRID_CANDIDATES": "9", "ASK_RRF_K": "30"}
        with mock.patch.dict(os.environ, env):
            cfg = HybridConfig.from_env()
        self.assertEqual((cfg.enabled, cfg.rerank_top_k, cfg.candidates, cfg.rrf_k), (True, 3, 9, 30))

    def test_invalid_values_fall_back_to_defaults(self):
        env = {"ASK_RERANK_TOP_K": "99", "ASK_HYBRID_CANDIDATES": "2", "ASK_RRF_K": "abc"}
        with mock.patch.dict(os.environ, env):
            cfg = HybridConfig.from_env()
        self.assertEqual((cfg.rerank_top_k, cfg.candidates, cfg.rrf_k), (4, 12, 60))

    def test_build_selector_from_env_is_none_when_disabled_and_loads_nothing_when_enabled(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("ASK_HYBRID_RETRIEVAL", None)
            self.assertIsNone(build_selector_from_env())
            os.environ["ASK_HYBRID_RETRIEVAL"] = "1"
            selector = build_selector_from_env()
        self.assertIsInstance(selector, HybridSelector)
        self.assertIsNone(selector.embedder._model)             # models are loaded lazily, on first use
        self.assertIsNone(selector.reranker._model)


class SelectorBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.segs = make_segments(lecture_lines())
        self.embedder, self.reranker = FakeEmbedder(), FakeReranker()

    def selector(self, **cfg):
        cfg.setdefault("enabled", True)
        cfg.setdefault("rerank_top_k", 1)
        cfg.setdefault("candidates", 6)
        return HybridSelector(HybridConfig(**cfg), embedder=self.embedder, reranker=self.reranker)

    def test_bm25_baseline_covers_only_the_eigenvalue_region(self):
        plan = R.select_evidence(QUESTION, self.segs)
        self.assertTrue(plan.ok)
        self.assertEqual(plan.mode, "retrieval")
        covered = covered_indices(plan.blocks)
        self.assertTrue(EIGEN_LINES <= covered)
        self.assertFalse(DET_LINES & covered)

    def test_disabled_returns_the_bm25_plan_without_touching_the_models(self):
        sel = self.selector(enabled=False)
        plan = sel(QUESTION, self.segs)
        self.assertEqual(blocks_of(plan), blocks_of(R.select_evidence(QUESTION, self.segs)))
        self.assertEqual((self.embedder.calls, self.reranker.calls), ([], []))

    def test_dense_candidate_that_bm25_scored_zero_can_win_the_rerank(self):
        plan = self.selector(rerank_top_k=1)(QUESTION, self.segs)
        covered = covered_indices(plan.blocks)
        self.assertTrue(DET_LINES <= covered)                   # the reranker preferred the determinant chunk
        self.assertFalse(EIGEN_LINES & covered)

    def test_rerank_top_k_two_keeps_both_regions(self):
        covered = covered_indices(self.selector(rerank_top_k=2)(QUESTION, self.segs).blocks)
        self.assertTrue(EIGEN_LINES <= covered and DET_LINES <= covered)

    def test_plan_shape_is_unchanged(self):
        base = R.select_evidence(QUESTION, self.segs)
        plan = self.selector(rerank_top_k=2)(QUESTION, self.segs)
        self.assertEqual((plan.decision, plan.mode, plan.complete), ("ok", "retrieval", False))
        self.assertEqual(plan.coverage, base.coverage)
        self.assertEqual([b.id for b in plan.blocks], [f"S{i}" for i in range(1, len(plan.blocks) + 1)])
        self.assertEqual(blocks_of(plan), sorted(blocks_of(plan)))      # chronological, like the BM25 path
        self.assertLessEqual(sum(len(b.text) for b in plan.blocks), R.EvidenceConfig().max_chars)

    def test_bm25_no_match_stays_no_match_and_the_models_are_never_called(self):
        plan = self.selector()("Who won the football world cup in Brazil?", self.segs)
        self.assertEqual(plan.decision, "no_evidence")
        self.assertEqual((self.embedder.calls, self.reranker.calls), ([], []))

    def test_whole_transcript_mode_is_untouched(self):
        cfg = R.EvidenceConfig(full_transcript_chars=100000, max_chars=100000)
        plan = self.selector()(QUESTION, self.segs, config=cfg)
        self.assertEqual((plan.mode, plan.complete), ("full", True))
        self.assertEqual(self.embedder.calls, [])

    def test_time_reference_mode_is_untouched(self):
        plan = self.selector()("What happens at 1:00 about eigenvalue?", self.segs)
        self.assertEqual(plan.mode, "time")
        self.assertEqual(self.embedder.calls, [])

    def test_candidate_cap_limits_what_the_reranker_sees(self):
        self.selector(candidates=2, rerank_top_k=1)(QUESTION, self.segs)
        self.assertEqual(len(self.reranker.calls[0][1]), 2)

    def test_reranker_receives_the_raw_question_and_chunk_texts(self):
        self.selector()(QUESTION, self.segs)
        query, passages = self.reranker.calls[0]
        self.assertEqual(query, QUESTION)
        self.assertTrue(all(isinstance(p, str) and p for p in passages))

    def test_result_is_deterministic(self):
        sel = self.selector(rerank_top_k=2)
        self.assertEqual(blocks_of(sel(QUESTION, self.segs)), blocks_of(sel(QUESTION, self.segs)))

    def test_chunk_vectors_are_cached_between_questions(self):
        sel = self.selector()
        sel(QUESTION, self.segs)
        sel("What is an eigenvalue exactly?", self.segs)
        bulk_encodes = [c for c in self.embedder.calls if len(c) > 1]
        self.assertEqual(len(bulk_encodes), 1)

    def test_rank_chunks_exposes_every_stage_and_matches_what_the_selector_chooses(self):
        sel = self.selector(rerank_top_k=1)
        ranking = sel.rank_chunks(QUESTION, self.segs)
        self.assertEqual(ranking.bm25_rank, [2])                                  # only the eigenvalue chunk has a BM25 score
        self.assertEqual(sorted(ranking.dense_rank), list(range(len(ranking.chunks))))
        self.assertEqual(ranking.candidates, [i for i, _ in ranking.fused[:6]])
        self.assertEqual(ranking.rerank_order[0], 6)                              # the reranker prefers the determinant chunk
        full = ranking.reranked_full()
        self.assertEqual(sorted(full), sorted(i for i, _ in ranking.fused))       # a full ranking, no duplicates
        self.assertEqual(full[: len(ranking.candidates)], ranking.rerank_order)
        plan = sel(QUESTION, self.segs)
        first, last = ranking.chunks[ranking.rerank_order[0]]
        self.assertTrue(set(range(first, last + 1)) <= covered_indices(plan.blocks))

    def test_rank_chunks_returns_none_without_content_terms(self):
        self.assertIsNone(self.selector().rank_chunks("what is it?", self.segs))

    def test_blocks_stay_inside_the_character_budget(self):
        cfg = R.EvidenceConfig(max_chars=600)
        plan = self.selector(rerank_top_k=3)(QUESTION, self.segs, config=cfg)
        self.assertTrue(plan.ok)
        self.assertLessEqual(sum(len(b.text) + 1 for b in plan.blocks), 600 + 200)       # one oversize first chunk may be cut to fit


class FailOpenTests(unittest.TestCase):
    def setUp(self):
        self.segs = make_segments(lecture_lines())
        self.baseline = blocks_of(R.select_evidence(QUESTION, self.segs))

    def run_with(self, embedder, reranker):
        sel = HybridSelector(HybridConfig(enabled=True, rerank_top_k=1, candidates=6), embedder=embedder, reranker=reranker)
        with self.assertLogs("qa.hybrid", level="WARNING") as logs:
            plan = sel(QUESTION, self.segs)
        self.assertIn("using BM25 evidence", logs.output[0])
        return plan

    def test_embedder_failure_returns_the_bm25_evidence(self):
        self.assertEqual(blocks_of(self.run_with(FakeEmbedder(fail=True), FakeReranker())), self.baseline)

    def test_reranker_failure_returns_the_bm25_evidence(self):
        self.assertEqual(blocks_of(self.run_with(FakeEmbedder(), FakeReranker(fail=True))), self.baseline)

    def test_wrong_number_of_vectors_returns_the_bm25_evidence(self):
        self.assertEqual(blocks_of(self.run_with(FakeEmbedder(short=True), FakeReranker())), self.baseline)

    def test_wrong_number_of_rerank_scores_returns_the_bm25_evidence(self):
        class Short(FakeReranker):
            def score(self, query, passages):
                return [0.1]
        self.assertEqual(blocks_of(self.run_with(FakeEmbedder(), Short())), self.baseline)

    def test_missing_sentence_transformers_is_handled(self):
        sel = HybridSelector(HybridConfig(enabled=True, rerank_top_k=1, candidates=6))   # real, lazy classes
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertLogs("qa.hybrid", level="WARNING"):
                plan = sel(QUESTION, self.segs)
        self.assertEqual(blocks_of(plan), self.baseline)


def _real_models_available():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_real_models_available(), "sentence-transformers not installed")
class RealModelSmokeTests(unittest.TestCase):
    """Loads the CACHED Hugging Face models offline (no download). Skips if they are not cached."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        cls.embedder = SentenceTransformerEmbedder()
        cls.reranker = CrossEncoderReranker()
        try:
            cls.embedder.encode(["warm up"])
            cls.reranker.score("warm up", ["warm up"])
        except Exception as exc:                                 # not cached: nothing to test
            raise unittest.SkipTest(f"models are not cached locally: {type(exc).__name__}")

    def test_embeddings_are_384_dimensional_finite_unit_vectors(self):
        vectors = self.embedder.encode(["The derivative measures change.", "Mitochondria make ATP."])
        self.assertEqual((len(vectors), len(vectors[0])), (2, 384))
        for v in vectors:
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=4)
            self.assertTrue(all(x == x and abs(x) < 10 for x in v))

    def test_embeddings_separate_related_from_unrelated(self):
        q, a, b = self.embedder.encode(["What does a derivative measure?", "The derivative measures the rate of change of a function.",
                                        "The French revolution began in 1789."])
        self.assertGreater(cosine(q, a), cosine(q, b))

    def test_reranker_prefers_the_relevant_passage(self):
        s = self.reranker.score("What does a derivative measure?", ["The French revolution began in 1789.",
                                                                    "The derivative measures the rate of change of a function."])
        self.assertEqual(len(s), 2)
        self.assertGreater(s[1], s[0])

    def test_end_to_end_selector_on_a_real_transcript_keeps_the_relevant_region(self):
        segs, region = long_lecture_segments()
        sel = HybridSelector(HybridConfig(enabled=True, rerank_top_k=4, candidates=12))
        plan = sel("What order does the lecturer give for choosing u?", segs)
        self.assertTrue(plan.ok)
        self.assertEqual(plan.mode, "retrieval")
        self.assertTrue(set(region) & covered_indices(plan.blocks))


if __name__ == "__main__":
    unittest.main()
