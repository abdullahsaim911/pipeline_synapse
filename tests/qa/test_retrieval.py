"""Retrieval tests: time references, keyword ranking, budgets, isolation of decisions."""

import unittest

from tests.qa.helpers import LECTURE_LINES, make_segments  # noqa: F401  (sets sys.path)
from qa.retrieval import (
    EvidenceConfig,
    content_terms,
    is_deictic,
    parse_time_reference,
    select_evidence,
    stem,
)

SEGMENTS = make_segments()          # 40 captions, 6 s apart => 0:00 - 4:00
END = SEGMENTS[-1].end


def plan_text(plan):
    return " ".join(b.text for b in plan.blocks)


class TimeReferenceTests(unittest.TestCase):
    def check(self, question, expected, current=None):
        ref = parse_time_reference(question, current, END)
        if expected is None:
            self.assertIsNone(ref, question)
        else:
            self.assertIsNotNone(ref, question)
            self.assertEqual((ref.kind, ref.start, ref.end), expected, question)

    def test_clock_range(self):
        self.check("What happened between 2:00 and 4:00?", ("range", 120, 240))
        self.check("explain 1:30 to 2:10", ("range", 90, 130))

    def test_single_timestamp_is_a_window(self):
        self.check("what is said at 3:20", ("range", 185, 245))
        self.check("what is said at 1:02:03", ("range", 3723 - 15, 3723 + 45))

    def test_minute_phrases(self):
        self.check("what about minute 2", ("range", 105, 165))
        self.check("at the 3 minute mark", ("range", 165, 225))
        self.check("from minute 1 to minute 2", ("range", 60, 120))
        self.check("between 1 and 2 minutes", ("range", 60, 120))

    def test_relative_to_playback_position(self):
        self.check("explain the last five minutes", ("range", 0, 200), current=200)
        self.check("what happened in the past 30 seconds", ("range", 170, 200), current=200)
        self.check("in the last minute", ("range", 140, 200), current=200)
        self.check("explain the last five minutes", ("needs_current", 0.0, 0.0))

    def test_end_of_lecture_does_not_need_playback_position(self):
        self.check("summarize the last 2 minutes of the lecture", ("range", END - 120, END))

    def test_beginning_and_end(self):
        self.check("what did he say in the first two minutes", ("range", 0, 120))
        self.check("what was said at the beginning?", ("range", 0, 120))
        self.check("what happens at the end?", ("range", END - 120, END))

    def test_just_now_phrases_need_a_playback_position(self):
        self.check("what did he say just now", ("range", 140, 200), current=200)
        self.check("what did he just explain", None, current=200)   # not one of the supported phrases
        self.check("what did he say just now", None)                 # no position: keyword path

    def test_no_time_phrase(self):
        self.check("what is the formula for integration by parts", None)
        self.check("explain chapter 3", None)


class TermTests(unittest.TestCase):
    def test_stemming_matches_inflections(self):
        self.assertEqual(stem("parts"), stem("part"))
        self.assertEqual(stem("derived"), stem("derive"))
        self.assertEqual(stem("functions"), stem("function"))
        self.assertEqual(stem("class"), "class")

    def test_stemming_is_crude(self):
        # Documented limitation: derivational forms do not match.
        self.assertNotEqual(stem("integration"), stem("integrate"))

    def test_content_terms_drop_question_words(self):
        self.assertEqual(content_terms("What did the instructor say about LIATE?"), [stem("liate")])

    def test_deictic(self):
        self.assertTrue(is_deictic("What does this diagram represent?"))
        self.assertFalse(is_deictic("What is the formula?"))


class SelectionTests(unittest.TestCase):
    def test_empty_transcript(self):
        self.assertEqual(select_evidence("anything", []).decision, "empty_transcript")

    def test_explicit_timestamp_returns_that_stretch(self):
        plan = select_evidence("What is explained at 2:30?", SEGMENTS)
        self.assertTrue(plan.ok)
        text = plan_text(plan)
        self.assertIn("plug into the formula", text)          # ~2:12
        self.assertNotIn("welcome to this lecture", text)
        self.assertTrue(any(b.start <= 150 <= b.end for b in plan.blocks))

    def test_time_range_limits_evidence(self):
        plan = select_evidence("Explain 1:00 to 1:30", SEGMENTS)
        self.assertTrue(plan.ok)
        self.assertTrue(all(b.start >= 54 and b.end <= 96 for b in plan.blocks), plan.blocks)
        self.assertIn("LIATE", plan_text(plan))

    def test_time_beyond_lecture(self):
        self.assertEqual(select_evidence("what happens at 59:00", SEGMENTS).decision,
                         "time_out_of_range")

    def test_relative_time_requires_playback_position(self):
        self.assertEqual(select_evidence("explain the last five minutes", SEGMENTS).decision,
                         "needs_current_time")

    def test_last_minutes_uses_playback_position(self):
        plan = select_evidence("what happened in the last minute", SEGMENTS, current_time=150)
        self.assertTrue(plan.ok)
        self.assertTrue(all(b.end <= 156 for b in plan.blocks))
        self.assertIn("plug into the formula", plan_text(plan))

    def test_keyword_retrieval_finds_earlier_definition(self):
        plan = select_evidence("What did the instructor say about LIATE?", SEGMENTS, current_time=230)
        self.assertTrue(plan.ok)
        self.assertIn("acronym LIATE", plan_text(plan))
        self.assertLessEqual(len(plan.blocks), 6)

    def test_keyword_retrieval_works_without_playback_position(self):
        plan = select_evidence("how does the tabular method work", SEGMENTS)
        self.assertTrue(plan.ok)
        self.assertIn("tabular method", plan_text(plan))

    def test_relationship_question_pulls_both_concepts(self):
        plan = select_evidence("How is the product rule related to the tabular method?", SEGMENTS)
        self.assertTrue(plan.ok)
        text = plan_text(plan)
        self.assertIn("product rule", text)
        self.assertIn("tabular method", text)
        self.assertEqual(plan.blocks, sorted(plan.blocks, key=lambda b: b.start))

    def test_blocks_are_chronological_and_disjoint(self):
        plan = select_evidence("product rule and exponentials and LIATE", SEGMENTS)
        for a, b in zip(plan.blocks, plan.blocks[1:]):
            self.assertLess(a.last_index + 1, b.first_index + 1)
            self.assertLess(a.start, b.start)

    def test_off_topic_question_is_rejected_without_evidence(self):
        plan = select_evidence("Who won the football world cup in Brazil?", SEGMENTS)
        self.assertEqual(plan.decision, "no_evidence")
        self.assertEqual(plan.blocks, [])

    def test_majority_of_terms_must_appear(self):
        # only 'formula' appears in the lecture; three unrelated terms do not.
        plan = select_evidence("formula for baking sourdough bread", SEGMENTS)
        self.assertEqual(plan.decision, "no_evidence")

    def test_question_with_no_content_terms(self):
        self.assertEqual(select_evidence("what is it?", SEGMENTS).decision, "no_question_terms")

    def test_deictic_question_uses_playback_window(self):
        plan = select_evidence("What does this graph show?", SEGMENTS, current_time=174)
        self.assertTrue(plan.ok)
        self.assertIn("graph of this function", plan_text(plan))
        self.assertTrue(any(b.start <= 174 <= b.end for b in plan.blocks))

    def test_attached_moment_adds_playback_window_for_any_question(self):
        plan = select_evidence("Why is that harder?", SEGMENTS, current_time=60, has_moment=True)
        self.assertTrue(plan.ok)
        self.assertTrue(any(b.start <= 60 <= b.end for b in plan.blocks))

    def test_deictic_without_position_falls_back_to_keywords(self):
        plan = select_evidence("What does this graph show?", SEGMENTS)
        self.assertTrue(plan.ok)
        self.assertIn("graph", plan_text(plan))

    def test_summary_includes_whole_short_lecture(self):
        plan = select_evidence("Can you summarize the whole lecture?", SEGMENTS)
        self.assertTrue(plan.ok)
        text = plan_text(plan)
        self.assertIn("welcome to this lecture", text)
        self.assertIn("thank you for watching", text)


class BudgetTests(unittest.TestCase):
    def total_chars(self, plan):
        return sum(len(b.text) for b in plan.blocks)

    def test_budget_is_enforced_for_keyword_hits(self):
        config = EvidenceConfig(max_chars=500, top_k=6)
        plan = select_evidence("integration parts formula product rule", SEGMENTS, config=config)
        self.assertTrue(plan.ok)
        self.assertLessEqual(self.total_chars(plan), 500)

    def test_summary_is_sampled_evenly_under_small_budget(self):
        config = EvidenceConfig(max_chars=700)
        plan = select_evidence("summarize the lecture", SEGMENTS, config=config)
        full = sum(len(s.text) + 1 for s in SEGMENTS)
        self.assertGreater(full, 700)
        self.assertTrue(plan.ok)
        self.assertLessEqual(self.total_chars(plan), 700)
        self.assertGreater(len(plan.blocks), 1)                 # several parts of the lecture
        self.assertLess(plan.blocks[0].start, 30)               # reaches the start
        self.assertGreater(plan.blocks[-1].start, 150)          # and the end

    def test_time_range_over_budget_is_sampled(self):
        config = EvidenceConfig(max_chars=600)
        plan = select_evidence("explain 0:00 to 4:00", SEGMENTS, config=config)
        self.assertTrue(plan.ok)
        self.assertLessEqual(self.total_chars(plan), 600)

    def test_minimum_budget_floor(self):
        self.assertEqual(EvidenceConfig(max_chars=10).max_chars, 500)

    def test_ranking_prefers_the_matching_chunk_under_tight_budget(self):
        config = EvidenceConfig(max_chars=500, top_k=1)
        plan = select_evidence("What is the acronym LIATE?", SEGMENTS, config=config)
        self.assertTrue(plan.ok)
        self.assertIn("LIATE", " ".join(b.text for b in plan.blocks))
        self.assertLessEqual(sum(len(b.text) for b in plan.blocks), 500)
        self.assertNotIn("welcome to this lecture", " ".join(b.text for b in plan.blocks))


if __name__ == "__main__":
    unittest.main()
