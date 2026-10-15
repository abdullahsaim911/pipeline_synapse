"""
Regression tests for the LIATE failures: retrieval misses on paraphrases, a keyword miss reported as
"not in lecture", over-broad sources, and unbounded prompt size.

All data is the real transcript of video 3zLQfuMZMF4 (tests/qa/liate_fixture.py), including its
speech-recognition errors ("li8" for LIATE). Nothing here needs the data/ directory, an API key or
the network; the model is always a fake.
"""

import json
import os
import re
import tempfile
import unittest
from unittest import mock

from tests.qa.helpers import (
    FakeProvider,
    LIATE_REGION,
    covered_indices,
    forbidden_provider_factory,
    liate_segments,
    long_lecture_segments,
)
from tests.qa.liate_fixture import ROWS, VIDEO_ID
from llm_interface.providers import TokenUsage
from qa import retrieval as R
from qa.moment import MomentContext
from qa.service import (
    PARTIAL_EVIDENCE_MESSAGE,
    SYSTEM_INSTRUCTION,
    AskConfig,
    AskService,
    extract_citations,
)
from qa.tokens import estimate_tokens
from qa.transcript import normalize_segments

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api.database import Base, get_db
    from api.routers import video as video_module
    HAVE_STACK = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_STACK = False

Q1 = "What is the order used to choose u in integration by parts?"
Q2 = "Which type of function should I prioritize when deciding what to substitute for u?"

# Every one of these is answered by captions 45-56 of the lecture.
PARAPHRASES = [
    Q1,
    Q2,
    "What does LIATE stand for?",
    "How do I pick which part is u?",
    "How do I decide what u and dv should be?",
    "Which function goes first, algebraic or trigonometric?",
    "Which has priority when choosing u, an algebraic or a trigonometric function?",
    "What ranking of function types tells me what to differentiate?",
]
UNRELATED = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "What is the speed of light?",
    "What is a function in Python programming?",
    "How do I bake an integration cake with parts?",
]

SEGMENTS = liate_segments()
RETRIEVAL = R.EvidenceConfig(full_transcript_chars=0)          # force retrieval mode
FULL = R.EvidenceConfig(full_transcript_chars=7000)             # the lecture fits whole
TOTAL_CHARS = sum(len(s.text) + 1 for s in SEGMENTS)
LECTURE_END = SEGMENTS[-1].end
FORCE_RETRIEVAL = AskConfig(full_transcript_max_tokens=1)       # service-level equivalent


def region_hits(plan, region=LIATE_REGION):
    return len(set(region) & covered_indices(plan.blocks))


class FixtureTests(unittest.TestCase):
    def test_fixture_is_the_real_asr_text(self):
        self.assertEqual(len(SEGMENTS), 109)
        self.assertGreater(TOTAL_CHARS, 3500)
        text = " ".join(s.text for s in SEGMENTS[LIATE_REGION.start:LIATE_REGION.stop])
        self.assertIn("li8", text)
        self.assertNotIn("liate", text.lower())
        self.assertIn("L for logarithmic", text)
        self.assertIn("e for exponential", text)


class QueryNormalisationTests(unittest.TestCase):
    def test_asr_digit_alias_links_li8_to_liate(self):
        self.assertIn(R.stem("liate"), R.tokenize("just remember the acronym li8 where L for logarithmic"))
        self.assertIn(R.stem("liate"), R.content_terms("What does LIATE stand for?"))

    def test_alias_is_narrow(self):
        # Only a word with two or more letters and an 8 gets an alias; maths and plain numbers do not.
        for text in ("2x", "x8", "3d", "e8", "8", "2022"):
            self.assertEqual(len(R.tokenize(text)), 1, text)
        self.assertEqual(len(R.tokenize("li8")), 2)
        self.assertEqual(len(R.tokenize("gr8")), 2)

    def test_synonym_groups_connect_the_question_to_the_lecture_wording(self):
        for word in ("prioritize", "decide", "pick", "select"):
            self.assertIn(R.stem("choose"), R.expand_term(R.stem(word)), word)
        self.assertIn(R.stem("plug"), R.expand_term(R.stem("substitute")))
        self.assertIn(R.stem("acronym"), R.expand_term(R.stem("stand")))

    def test_groups_are_tight(self):
        for word in ("function", "part", "integral", "derive", "first", "the", "u"):
            self.assertLessEqual(len(R.expand_term(R.stem(word))), 3, word)
        self.assertEqual(R.expand_term("function"), frozenset({"function"}))

    def test_irregular_forms(self):
        self.assertEqual(R.stem("chosen"), R.stem("choose"))
        self.assertEqual(R.stem("chose"), R.stem("choosing"))

    def test_single_letter_symbols_are_kept_but_not_english_debris(self):
        self.assertIn("u", R.content_terms("How do I choose u?"))
        self.assertIn("x", R.content_terms("What is x here"))
        for debris in ("a", "i", "s", "t", "o"):
            self.assertNotIn(debris, R.content_terms(f"what is {debris} in the lecture"))

    def test_contractions_do_not_leave_fragments(self):
        terms = R.content_terms("What's the order? Don't skip it, isn't it key?")
        for fragment in ("s", "t", "don", "isn"):
            self.assertNotIn(fragment, terms)

    def test_time_expressions_are_not_content_terms(self):
        cleaned = R._TIME_TEXT_RE.sub(" ", "explain 1:45 to 2:20 and minute 2")
        self.assertEqual(R.content_terms(cleaned), [])
        self.assertEqual(R.content_terms(R._TIME_TEXT_RE.sub(" ", "what is LIATE at 2:00")), [R.stem("liate")])


class RetrievalModeTests(unittest.TestCase):
    """Retrieval alone (whole-transcript mode disabled): this is the long-lecture path."""

    def test_every_paraphrase_retrieves_the_whole_ordering_explanation(self):
        for question in PARAPHRASES:
            with self.subTest(question=question):
                plan = R.select_evidence(question, SEGMENTS, config=RETRIEVAL)
                self.assertEqual(plan.decision, "ok", plan.coverage)
                self.assertEqual(plan.mode, "retrieval")
                self.assertFalse(plan.complete)
                self.assertEqual(region_hits(plan), len(LIATE_REGION))

    def test_a_specific_question_does_not_pull_in_the_whole_lecture(self):
        plan = R.select_evidence("What does LIATE stand for?", SEGMENTS, config=RETRIEVAL)
        chars = sum(len(b.text) for b in plan.blocks)
        self.assertLess(chars, TOTAL_CHARS * 0.5)
        joined = " ".join(b.text for b in plan.blocks)
        self.assertNotIn("understand integration by parts in just", joined)    # the introduction
        self.assertNotIn("thank you", joined)

    def test_explanation_split_across_a_chunk_boundary_is_kept_whole(self):
        # The ordering straddles two chunks (the boundary is at 116.3 s, mid-list).
        chunks = R._group(range(len(SEGMENTS)), SEGMENTS, RETRIEVAL)
        straddling = [c for c in chunks if c[0] <= LIATE_REGION.start <= c[1] and c[1] < LIATE_REGION.stop - 1]
        self.assertTrue(straddling, "fixture should straddle a chunk boundary")
        plan = R.select_evidence("What does LIATE stand for?", SEGMENTS, config=RETRIEVAL)
        self.assertEqual(region_hits(plan), len(LIATE_REGION))

    def test_unrelated_questions_match_nothing(self):
        for question in UNRELATED:
            with self.subTest(question=question):
                plan = R.select_evidence(question, SEGMENTS, config=RETRIEVAL)
                self.assertEqual((plan.decision, plan.blocks), ("no_evidence", []))
                self.assertLess(plan.coverage, RETRIEVAL.min_coverage)

    def test_threshold_separates_relevant_from_unrelated_with_margin(self):
        relevant = [R.select_evidence(q, SEGMENTS, config=RETRIEVAL).coverage for q in PARAPHRASES]
        unrelated = [R.select_evidence(q, SEGMENTS, config=RETRIEVAL).coverage for q in UNRELATED]
        self.assertGreater(min(relevant), RETRIEVAL.min_coverage + 0.1)
        self.assertLess(max(unrelated), RETRIEVAL.min_coverage - 0.1)

    def test_one_common_word_alone_cannot_pass_the_gate(self):
        # "function" and "part" occur in most chunks; with rare unmatched words they must not carry the question.
        for question in ("What is the best function for baking bread?",
                         "Which engine part fails when the car is first started?"):
            plan = R.select_evidence(question, SEGMENTS, config=RETRIEVAL)
            self.assertEqual(plan.decision, "no_evidence", question)

    def test_gate_threshold_is_what_rejects(self):
        loose = R.EvidenceConfig(full_transcript_chars=0, min_coverage=0.0)
        self.assertEqual(R.select_evidence(UNRELATED[3], SEGMENTS, config=loose).decision, "ok")

    def test_question_with_nothing_to_search_for(self):
        for question in ("what is it?", "why?", "how does this"):
            plan = R.select_evidence(question, SEGMENTS, config=RETRIEVAL)
            self.assertEqual(plan.decision, "no_question_terms", question)

    def test_explicit_time_range_returns_only_that_stretch(self):
        plan = R.select_evidence("what is said between 1:45 and 2:20", SEGMENTS, config=RETRIEVAL)
        self.assertEqual(plan.mode, "time")
        self.assertFalse(plan.complete)
        self.assertEqual(region_hits(plan), len(LIATE_REGION))
        self.assertTrue(all(b.start >= 100 and b.end <= 145 for b in plan.blocks), [(b.start, b.end) for b in plan.blocks])
        self.assertLess(sum(len(b.text) for b in plan.blocks), 700)      # no keyword noise from "1", "45", "2"

    def test_single_timestamp_is_a_window(self):
        plan = R.select_evidence("what is said at 2:00", SEGMENTS, config=RETRIEVAL)
        self.assertTrue(any(b.start <= 120 <= b.end for b in plan.blocks))
        self.assertTrue(all(b.start >= 100 for b in plan.blocks))

    def test_relative_time_uses_playback_position(self):
        plan = R.select_evidence("what happened in the last minute", SEGMENTS, current_time=150.0, config=RETRIEVAL)
        self.assertTrue(all(b.end <= 151 and b.start >= 88 for b in plan.blocks))
        self.assertEqual(R.select_evidence("what happened in the last minute", SEGMENTS, config=RETRIEVAL).decision,
                         "needs_current_time")

    def test_time_beyond_the_lecture(self):
        self.assertEqual(R.select_evidence("what is said at 59:00", SEGMENTS, config=RETRIEVAL).decision,
                         "time_out_of_range")

    def test_deictic_question_uses_the_playback_position(self):
        plan = R.select_evidence("What does this graph show?", SEGMENTS, current_time=126.0, config=RETRIEVAL)
        self.assertTrue(plan.ok)
        self.assertTrue(any(b.start <= 126.0 <= b.end for b in plan.blocks))
        self.assertEqual(R.select_evidence("What does this graph show?", SEGMENTS, config=RETRIEVAL).decision,
                         "no_evidence")  # no position, "graph" is not in the lecture

    def test_empty_and_malformed_transcripts(self):
        self.assertEqual(R.select_evidence("What is LIATE?", [], config=RETRIEVAL).decision, "empty_transcript")
        broken = normalize_segments([
            {"start": float("nan"), "text": "bad"}, {"start": None, "text": "worse"},
            {"start": -4, "text": "negative"}, {"start": "x", "text": "text"}, {"start": 3, "text": "  "}])
        self.assertEqual(broken, [])
        self.assertEqual(R.select_evidence("What is LIATE?", broken, config=RETRIEVAL).decision, "empty_transcript")
        partly = normalize_segments([{"start": 0, "text": "the acronym li8 is useful"},
                                     {"start": float("nan"), "text": "ignored"},
                                     {"start": "oops", "end": 4, "text": "ignored too"}])
        self.assertEqual(len(partly), 1)
        self.assertTrue(R.select_evidence("What does LIATE stand for?", partly, config=RETRIEVAL).ok)


class WholeTranscriptModeTests(unittest.TestCase):
    def test_short_transcript_is_sent_whole_and_marked_complete(self):
        plan = R.select_evidence(Q2, SEGMENTS, config=FULL)
        self.assertEqual((plan.decision, plan.mode, plan.complete), ("ok", "full", True))
        self.assertEqual(covered_indices(plan.blocks), set(range(len(SEGMENTS))))

    def test_whole_transcript_is_split_into_narrow_citable_passages(self):
        plan = R.select_evidence(Q1, SEGMENTS, config=FULL)
        self.assertEqual([b.id for b in plan.blocks], [f"S{i}" for i in range(1, len(plan.blocks) + 1)])
        self.assertGreaterEqual(len(plan.blocks), 6)
        for block in plan.blocks:
            self.assertLessEqual(block.end - block.start, 45)
        self.assertEqual([b.first_index for b in plan.blocks][0], 0)
        self.assertEqual([b.last_index for b in plan.blocks][-1], len(SEGMENTS) - 1)

    def test_a_transcript_over_the_cap_falls_back_to_retrieval(self):
        small = R.EvidenceConfig(full_transcript_chars=TOTAL_CHARS - 1)
        plan = R.select_evidence(Q1, SEGMENTS, config=small)
        self.assertEqual((plan.mode, plan.complete), ("retrieval", False))
        exact = R.EvidenceConfig(full_transcript_chars=TOTAL_CHARS)
        self.assertTrue(R.select_evidence(Q1, SEGMENTS, config=exact).complete)

    def test_the_evidence_budget_caps_full_mode_too(self):
        tight = R.EvidenceConfig(max_chars=1500, full_transcript_chars=100000)
        plan = R.select_evidence(Q1, SEGMENTS, config=tight)
        self.assertFalse(plan.complete)

    def test_explicit_time_beats_full_mode(self):
        plan = R.select_evidence("what is said at 2:00", SEGMENTS, config=FULL)
        self.assertEqual((plan.mode, plan.complete), ("time", False))

    def test_empty_question_is_not_sent_even_when_everything_fits(self):
        self.assertEqual(R.select_evidence("what is it?", SEGMENTS, config=FULL).decision, "no_question_terms")
        self.assertTrue(R.select_evidence("what is it?", SEGMENTS, has_moment=True, config=FULL).ok)
        self.assertTrue(R.select_evidence("summarize the lecture", SEGMENTS, config=FULL).complete)


class CitationParsingTests(unittest.TestCase):
    def test_sources_line_and_inline_markers_are_removed(self):
        self.assertEqual(extract_citations("The order is L, I, A, T, E.\nSOURCES: S3, S4"),
                         ("The order is L, I, A, T, E.", ["S3", "S4"]))
        self.assertEqual(extract_citations("Answer [S2] here [S3, S4].")[1], ["S2", "S3", "S4"])
        self.assertEqual(extract_citations("Answer.\nsources: s2 and s5"), ("Answer.", ["S2", "S5"]))

    def test_invented_times_on_a_sources_line_are_dropped_not_read_aloud(self):
        body, cited = extract_citations("Answer.\n**Sources:** 01:56-02:35")
        self.assertEqual((body, cited), ("Answer.", []))

    def test_ordinary_sentences_survive(self):
        text = "Sources of error include noise. It matters."
        self.assertEqual(extract_citations(text), (text, []))
        self.assertEqual(extract_citations("No citation."), ("No citation.", []))


class ServiceShortLectureTests(unittest.TestCase):
    """Default configuration on the real 3.9k-character lecture: it is sent whole."""

    def ask(self, question, reply, **kwargs):
        provider = FakeProvider(reply)
        outcome = AskService(provider=provider, config=kwargs.pop("config", AskConfig())).answer(
            question, SEGMENTS, **kwargs)
        return outcome, provider

    ANSWER = ("The instructor lists logarithmic, inverse trigonometric, algebraic, trigonometric and "
              "exponential, in that order. The captions appear to say LIATE as li8.")

    def test_both_original_questions_reach_the_model_with_the_full_explanation(self):
        for question in (Q1, Q2):
            with self.subTest(question=question):
                out, provider = self.ask(question, self.ANSWER + "\nSOURCES: S3, S4")
                prompt = provider.calls[0]["prompt"]
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual((out.status, out.evidence_mode), ("answered", "full"))
                for caption in SEGMENTS[LIATE_REGION.start:LIATE_REGION.stop]:
                    self.assertIn(caption.text, prompt)
                self.assertIn("COMPLETE transcript", prompt)

    def test_cited_passages_narrow_the_sources(self):
        out, _ = self.ask(Q1, self.ANSWER + "\nSOURCES: S3, S4")
        self.assertEqual(len(out.sources), 1)                       # adjacent passages merge
        source = out.sources[0]
        self.assertAlmostEqual(source.start_seconds, 77.7, delta=1.0)
        self.assertAlmostEqual(source.end_seconds, 155.0, delta=1.0)
        self.assertLess(source.end_seconds - source.start_seconds, LECTURE_END / 2)
        self.assertTrue(source.start_seconds <= SEGMENTS[LIATE_REGION.start].start
                        and source.end_seconds >= SEGMENTS[LIATE_REGION.stop - 1].end)   # covers the explanation

    def test_non_adjacent_citations_stay_separate(self):
        out, _ = self.ask(Q1, self.ANSWER + "\nSOURCES: S1, S4")
        self.assertEqual(len(out.sources), 2)
        self.assertLess(out.sources[0].end_seconds, out.sources[1].start_seconds)

    def test_citation_text_never_reaches_the_user(self):
        out, _ = self.ask(Q1, "The order is given [S3]. It continues [S4].\nSOURCES: S3, S4")
        for token in ("SOURCES", "S3", "S4", "["):
            self.assertNotIn(token, out.answer_text)

    def test_fabricated_citation_ids_are_ignored(self):
        out, _ = self.ask(Q1, self.ANSWER + "\nSOURCES: S99, S3")
        self.assertEqual(len(out.sources), 1)
        self.assertLessEqual(out.sources[0].end_seconds, 116.5)     # only S3 survived
        self.assertGreaterEqual(out.sources[0].start_seconds, 77.0)

    def test_without_valid_citations_every_supplied_passage_is_reported(self):
        for reply in (self.ANSWER, self.ANSWER + "\nSOURCES: S99", self.ANSWER + "\nSOURCES: 01:56-02:35"):
            out, _ = self.ask(Q1, reply)
            self.assertEqual(out.status, "answered")
            self.assertEqual(len(out.sources), 1)
            self.assertAlmostEqual(out.sources[0].start_seconds, SEGMENTS[0].start, delta=0.5)
            self.assertAlmostEqual(out.sources[0].end_seconds, LECTURE_END, delta=0.5)
            self.assertNotIn("SOURCES", out.answer_text)

    def test_sources_are_always_valid_ranges_inside_the_lecture(self):
        for cited in ("S1", "S2, S5", "S7, S8", "S1, S2, S3, S4, S5, S6, S7, S8", "S0", ""):
            out, _ = self.ask(Q1, self.ANSWER + (f"\nSOURCES: {cited}" if cited else ""))
            for s in out.sources:
                self.assertTrue(0 <= s.start_seconds < s.end_seconds <= LECTURE_END + 0.01, (cited, s))

    def test_unrelated_questions_are_decided_by_the_model_which_saw_everything(self):
        for question in ("What is the capital of France?", "How do I bake sourdough bread?"):
            out, provider = self.ask(question, "[[NOT_IN_LECTURE]] This lecture is about integration by parts.")
            self.assertEqual((out.status, out.sources), ("not_in_lecture", []))
            self.assertIn("COMPLETE transcript", provider.calls[0]["prompt"])

    def test_nothing_to_search_for_is_not_sent(self):
        service = AskService(provider_factory=forbidden_provider_factory)
        out = service.answer("what is it?", SEGMENTS)
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))

    def test_explicit_timestamp_sends_only_that_stretch(self):
        out, provider = self.ask("what is said between 1:45 and 2:20", "They list the order. SOURCES: S1")
        prompt = provider.calls[0]["prompt"]
        self.assertEqual(out.evidence_mode, "time")
        self.assertNotIn("COMPLETE transcript", prompt)
        self.assertIn("li8", prompt)
        self.assertNotIn("understand integration by parts in just", prompt)
        self.assertTrue(all(100 <= s.start_seconds and s.end_seconds <= 145 for s in out.sources))

    def test_playback_position_question_without_a_position_is_not_sent(self):
        service = AskService(provider_factory=forbidden_provider_factory)
        self.assertEqual(service.answer("explain the last five minutes", SEGMENTS).status, "insufficient_context")
        out, _ = self.ask("explain the last minute", "They derive u and dv. SOURCES: S1", current_time=150.0)
        self.assertTrue(all(s.end_seconds <= 151 for s in out.sources))

    def test_empty_transcript_is_never_sent(self):
        service = AskService(provider_factory=forbidden_provider_factory)
        out = service.answer(Q1, [])
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))

    def test_intervention_source_is_added_only_when_generated_content_was_used(self):
        moment = MomentContext(intervention_id="int-9", timestamp=126.0, content_type="graph",
                               caption="and lastly e for exponential function", explanation="A graph of the ordering.",
                               visual=None)
        out, _ = self.ask("What does this show?", self.ANSWER + "\nSOURCES: S4", moment=moment)
        self.assertEqual([s.intervention_id for s in out.sources if s.intervention_id], ["int-9"])
        self.assertEqual(out.scope, "lecture_with_moment")


class ServiceLongLectureTests(unittest.TestCase):
    """A transcript far beyond the prompt budget: retrieval path."""

    SEGMENTS_LONG, REGION = long_lecture_segments()

    def ask(self, question, reply="The instructor lists the order.", config=None, **kwargs):
        provider = FakeProvider(reply)
        out = AskService(provider=provider, config=config or AskConfig()).answer(
            question, self.SEGMENTS_LONG, **kwargs)
        return out, provider

    def prompt_tokens(self, prompt):
        return estimate_tokens(SYSTEM_INSTRUCTION) + estimate_tokens(prompt)

    def test_the_fixture_is_really_over_budget(self):
        chars = sum(len(s.text) + 1 for s in self.SEGMENTS_LONG)
        self.assertGreater(estimate_tokens("x" * chars), AskConfig().max_prompt_tokens)

    def test_paraphrased_questions_find_the_explanation_in_the_haystack(self):
        for question in (Q1, Q2, "What does LIATE stand for?", "How do I pick which part is u?"):
            with self.subTest(question=question):
                out, provider = self.ask(question)
                prompt = provider.calls[0]["prompt"]
                self.assertEqual(out.evidence_mode, "retrieval")
                for caption in self.SEGMENTS_LONG[self.REGION.start:self.REGION.stop]:
                    self.assertIn(caption.text, prompt)

    def test_irrelevant_filler_is_kept_out_of_the_prompt(self):
        _, provider = self.ask("What does LIATE stand for?")
        prompt = provider.calls[0]["prompt"]
        filler_in_prompt = prompt.count("cell membrane and ribosome number")
        filler_total = sum("ribosome number" in s.text for s in self.SEGMENTS_LONG)
        self.assertLess(filler_in_prompt, filler_total * 0.15)
        self.assertIn("only selected parts", prompt)
        self.assertNotIn("COMPLETE transcript", prompt)

    def test_prompt_never_exceeds_the_token_budget(self):
        for budget in (1500, 2500, 4000, 8000):
            for question in (Q1, "summarize the whole lecture", "explain 0:00 to 14:00", "cell membrane ribosome transport"):
                with self.subTest(budget=budget, question=question):
                    _, provider = self.ask(question, config=AskConfig(max_prompt_tokens=budget))
                    self.assertLessEqual(self.prompt_tokens(provider.calls[0]["prompt"]), budget)

    def test_a_tiny_budget_degrades_to_a_floor_not_to_the_whole_lecture(self):
        with self.assertLogs("qa.service", level="WARNING"):
            _, provider = self.ask(Q1, config=AskConfig(max_prompt_tokens=100))
        prompt = provider.calls[0]["prompt"]
        self.assertLess(len(prompt), 4000)

    def test_legacy_evidence_cap_still_bounds_the_text(self):
        _, provider = self.ask("summarize the whole lecture", config=AskConfig(max_evidence_chars=800))
        evidence = sum(len(m) for m in re.findall(r"\[S\d+ [^\]]*\] ([^\n]*)", provider.calls[0]["prompt"]))
        self.assertLessEqual(evidence, 800)

    def test_unmatched_question_is_insufficient_and_costs_no_request(self):
        service = AskService(provider_factory=forbidden_provider_factory)
        for question in ("What is the capital of France?", "How do I bake sourdough bread?"):
            out = service.answer(question, self.SEGMENTS_LONG)
            self.assertEqual((out.status, out.sources), ("insufficient_context", []))
            self.assertNotIn("not covered", out.answer_text.lower())
            self.assertNotIn("isn't in", out.answer_text.lower())

    def test_not_in_lecture_from_partial_evidence_is_downgraded(self):
        out, provider = self.ask(Q1, reply="[[NOT_IN_LECTURE]] This lecture never mentions that.")
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))
        self.assertEqual(out.answer_text, PARTIAL_EVIDENCE_MESSAGE)
        self.assertNotIn("never mentions", out.answer_text)

    def test_insufficient_context_from_the_model_is_passed_through(self):
        out, _ = self.ask(Q1, reply="[[INSUFFICIENT_CONTEXT]] The excerpts do not list the full order.")
        self.assertEqual((out.status, out.answer_text), ("insufficient_context", "The excerpts do not list the full order."))

    def test_citation_of_the_passage_that_holds_the_explanation_narrows_sources(self):
        provider = FakeProvider("placeholder")
        service = AskService(provider=provider)
        service.answer("What does LIATE stand for?", self.SEGMENTS_LONG)          # first call to read the passage ids
        prompt = provider.calls[0]["prompt"]
        ids = [pid for pid, text in re.findall(r"\[(S\d+) [^\]]*\] ([^\n]*)", prompt) if "li8" in text]
        self.assertEqual(len(ids), 1)
        provider.reply = f"The captions appear to say LIATE.\nSOURCES: {ids[0]}"
        out = service.answer("What does LIATE stand for?", self.SEGMENTS_LONG)
        self.assertEqual(out.status, "answered")
        li8_caption = next(s for s in self.SEGMENTS_LONG if "li8" in s.text)
        self.assertTrue(any(s.start_seconds <= li8_caption.start <= s.end_seconds for s in out.sources))
        self.assertLess(sum(s.end_seconds - s.start_seconds for s in out.sources), 60)

    def test_explicit_time_in_a_long_lecture(self):
        start = self.SEGMENTS_LONG[self.REGION.start].start
        question = f"what is said at {int(start // 60)}:{int(start % 60):02d}"
        out, provider = self.ask(question, "They list the order. SOURCES: S1")
        self.assertEqual(out.evidence_mode, "time")
        self.assertTrue(all(abs(s.start_seconds - start) < 60 for s in out.sources))


class TokenAccountingTests(unittest.TestCase):
    def test_estimate_is_conservative_for_the_real_lecture(self):
        words = sum(len(s.text.split()) for s in SEGMENTS)
        tokens = estimate_tokens(" ".join(s.text for s in SEGMENTS))
        self.assertGreaterEqual(tokens, words * 1.2)       # English averages about 1.3 tokens per word
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd", 4), 1)
        self.assertEqual(estimate_tokens("abcde", 4), 2)

    def test_provider_reported_usage_is_kept_with_the_outcome_and_logged(self):
        usage = TokenUsage(prompt_tokens=1700, output_tokens=60, thoughts_tokens=12, total_tokens=1772)
        provider = FakeProvider("Answer text. SOURCES: S3", usage=usage)
        with self.assertLogs("qa.service", level="INFO") as logs:
            out = AskService(provider=provider).answer(Q1, SEGMENTS)
        self.assertEqual(out.usage, usage)
        self.assertGreater(out.estimated_prompt_tokens, 0)
        line = next(l for l in logs.output if "Ask tokens" in l)
        self.assertIn("actual_prompt=1700", line)
        self.assertIn(f"estimated_prompt={out.estimated_prompt_tokens}", line)
        self.assertNotIn("Answer text", "\n".join(logs.output))      # no content in logs

    def test_missing_usage_is_reported_as_such(self):
        with self.assertLogs("qa.service", level="INFO") as logs:
            AskService(provider=FakeProvider("Answer. SOURCES: S1")).answer(Q1, SEGMENTS)
        self.assertTrue(any("provider reported no usage" in l for l in logs.output))

    def test_config_from_environment(self):
        env = {"ASK_MAX_PROMPT_TOKENS": "2500", "ASK_FULL_TRANSCRIPT_MAX_TOKENS": "900",
               "ASK_CHARS_PER_TOKEN": "3.0", "ASK_MAX_EVIDENCE_CHARS": "5000"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = AskConfig.from_env()
        self.assertEqual((config.max_prompt_tokens, config.full_transcript_max_tokens,
                          config.chars_per_token, config.max_evidence_chars), (2500, 900, 3.0, 5000))

    def test_invalid_environment_values_fall_back_to_defaults(self):
        env = {"ASK_MAX_PROMPT_TOKENS": "lots", "ASK_FULL_TRANSCRIPT_MAX_TOKENS": "-5",
               "ASK_CHARS_PER_TOKEN": "0", "ASK_MAX_EVIDENCE_CHARS": "x"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertLogs(level="WARNING"):
                config = AskConfig.from_env()
        self.assertEqual(config, AskConfig())

    def test_evidence_budget_shrinks_as_the_fixed_parts_grow(self):
        config = AskConfig(max_prompt_tokens=4000)
        self.assertGreater(config.evidence_chars(500), config.evidence_chars(1500))
        self.assertGreaterEqual(config.evidence_chars(10 ** 6), 500)       # never below the floor
        self.assertLessEqual(config.full_transcript_chars(config.evidence_chars(800)),
                             config.evidence_chars(800))

    def test_default_budget_keeps_the_sample_lecture_whole(self):
        config = AskConfig()
        overhead = estimate_tokens(SYSTEM_INSTRUCTION) + 250
        self.assertGreaterEqual(config.full_transcript_chars(config.evidence_chars(overhead)), TOTAL_CHARS)


@unittest.skipUnless(HAVE_STACK, "fastapi, httpx or sqlalchemy not installed")
class EndpointLiateTests(unittest.TestCase):
    """The two original questions through POST /video/ask, with the real transcript on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.Session = sessionmaker(bind=engine)
        self.provider = FakeProvider()
        self.service = AskService(provider=self.provider)

        app = FastAPI()
        app.include_router(video_module.router)

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[video_module.get_ask_service] = lambda: self.service
        app.dependency_overrides[video_module.get_data_dir] = lambda: self.tmp.name
        self.client = TestClient(app, raise_server_exceptions=False)

        self.write(VIDEO_ID, [{"start": a, "end": b, "duration": b - a, "text": t} for a, b, t in ROWS])

    def write(self, video_id, entries):
        folder = os.path.join(self.tmp.name, video_id)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "transcript.json"), "w", encoding="utf-8") as handle:
            json.dump({"video_id": video_id, "duration_seconds": 4.8, "entries": entries}, handle)

    def ask(self, question, video_id=VIDEO_ID, **extra):
        return self.client.post("/video/ask", json={"video_id": video_id, "question": question, **extra})

    def test_original_questions_are_answered_with_narrow_valid_sources(self):
        self.provider.reply = "The instructor lists the order, which the captions render as li8.\nSOURCES: S3, S4"
        for question in (Q1, Q2):
            with self.subTest(question=question):
                response = self.ask(question)
                body = response.json()
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(set(body), {"status", "scope", "answer_text", "sources"})
                self.assertEqual(body["status"], "answered")
                self.assertNotIn("SOURCES", body["answer_text"])
                self.assertEqual(len(body["sources"]), 1)
                source = body["sources"][0]
                self.assertTrue(77 <= source["start_seconds"] < source["end_seconds"] <= 156)
                self.assertTrue(source["start_seconds"] <= 103.96 and source["end_seconds"] >= 139.48)
        self.assertEqual(len(self.provider.calls), 2)

    def test_short_lecture_unrelated_question_is_the_models_call(self):
        self.provider.reply = "[[NOT_IN_LECTURE]] This lecture is about integration by parts."
        body = self.ask("How do I bake sourdough bread?").json()
        self.assertEqual((body["status"], body["sources"]), ("not_in_lecture", []))

    def test_long_lecture_unmatched_question_is_insufficient_and_free(self):
        segments, _ = long_lecture_segments()
        self.write("LLLLLLLLLLL", [{"start": s.start, "end": s.end, "text": s.text} for s in segments])
        self.service = AskService(provider_factory=forbidden_provider_factory)
        body = self.ask("How do I bake sourdough bread?", video_id="LLLLLLLLLLL").json()
        self.assertEqual((body["status"], body["sources"]), ("insufficient_context", []))
        self.assertTrue(body["answer_text"])

    def test_long_lecture_relevant_question_reaches_the_model_within_budget(self):
        segments, region = long_lecture_segments()
        self.write("LLLLLLLLLLL", [{"start": s.start, "end": s.end, "text": s.text} for s in segments])
        self.provider.reply = "The instructor lists the order."
        body = self.ask(Q2, video_id="LLLLLLLLLLL").json()
        self.assertEqual(body["status"], "answered")
        prompt = self.provider.calls[-1]["prompt"]
        self.assertIn("li8", prompt)
        self.assertLessEqual(estimate_tokens(SYSTEM_INSTRUCTION) + estimate_tokens(prompt), AskConfig().max_prompt_tokens)
        start, end = segments[region.start].start, segments[region.stop - 1].end
        self.assertTrue(any(s["start_seconds"] <= start and s["end_seconds"] >= end for s in body["sources"]))

    def test_malformed_timestamps_in_the_transcript_file_do_not_break_the_request(self):
        entries = [{"start": a, "end": b, "text": t} for a, b, t in ROWS]
        entries += [{"start": "oops", "text": "bad"}, {"start": None, "text": "worse"}, {"text": "no start"},
                    "not an object", {"start": float(10 ** 9), "text": "far away"}]
        self.write("MMMMMMMMMMM", entries)
        self.provider.reply = "The order is listed. SOURCES: S3"
        response = self.ask(Q1, video_id="MMMMMMMMMMM")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "answered")
        for source in response.json()["sources"]:
            self.assertTrue(0 <= source["start_seconds"] <= source["end_seconds"])


if __name__ == "__main__":
    unittest.main()
