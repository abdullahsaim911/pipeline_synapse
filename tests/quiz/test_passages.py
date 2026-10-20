"""Evidence passages and server-side evidence-id -> time mapping."""

import math
import unittest

from tests.quiz.helpers import make_passages, make_segments
from qa.transcript import TranscriptSegment
from quiz.limits import QuizLimits
from quiz.passages import build_passages, map_evidence_ids
from quiz.reasons import Reason


class BuildPassagesTests(unittest.TestCase):
    def test_ids_are_sequential_and_stable(self):
        first = make_passages()
        second = make_passages()
        self.assertGreaterEqual(len(first.passages), 6)
        self.assertEqual([p.id for p in first.passages], [f"S{i}" for i in range(1, len(first.passages) + 1)])
        self.assertEqual(first, second)

    def test_times_come_from_segments_and_cover_the_transcript(self):
        passages = make_passages()
        self.assertEqual(passages.passages[0].start, 0.0)
        self.assertEqual(passages.lecture_end, 120 * 6.0)
        for before, after in zip(passages.passages, passages.passages[1:]):
            self.assertEqual(before.end, after.start)

    def test_passages_respect_the_token_ceiling(self):
        limits = QuizLimits()
        for passage in make_passages().passages:
            self.assertLessEqual(passage.tokens, limits.passage_max_tokens)

    def test_empty_inputs_yield_no_passages(self):
        for empty in ([], None, ()):
            result = build_passages(empty)
            self.assertEqual(result.passages, ())
            self.assertEqual(result.lecture_end, 0.0)
            self.assertFalse(result.truncated)

    def test_malformed_segments_are_skipped_and_counted_not_repaired(self):
        good = TranscriptSegment(0, 0.0, 5.0, "a real caption")
        bad = [
            TranscriptSegment(1, -1.0, 2.0, "negative start"),
            TranscriptSegment(2, 3.0, 2.0, "end before start"),
            TranscriptSegment(3, float("nan"), 4.0, "nan start"),
            TranscriptSegment(4, 5.0, math.inf, "infinite end"),
            TranscriptSegment(5, 6.0, 7.0, "   "),
            {"start": 1, "end": 2, "text": "a dict, not a segment"},
            "not a segment",
            None,
        ]
        result = build_passages([good] + bad)
        self.assertEqual(result.skipped_segments, len(bad))
        self.assertEqual(len(result.passages), 1)
        self.assertEqual(result.passages[0].text, "a real caption")
        self.assertEqual(result.lecture_end, 5.0)

    def test_all_segments_malformed_gives_empty_set(self):
        result = build_passages([TranscriptSegment(0, 1.0, 0.5, "x"), None])
        self.assertEqual(result.passages, ())
        self.assertEqual(result.skipped_segments, 2)

    def test_oversized_single_segment_is_its_own_passage(self):
        big = TranscriptSegment(0, 0.0, 10.0, "word " * 600)
        small = TranscriptSegment(1, 10.0, 12.0, "tail")
        result = build_passages([big, small])
        self.assertEqual(len(result.passages), 2)
        self.assertGreater(result.passages[0].tokens, QuizLimits().passage_max_tokens)

    def test_small_tail_merges_into_previous_when_it_fits(self):
        limits = QuizLimits(passage_min_tokens=300, passage_max_tokens=400)
        fill = TranscriptSegment(0, 0.0, 6.0, "x" * 1000)       # ~286 tokens
        tail = TranscriptSegment(1, 6.0, 8.0, "tiny tail")
        result = build_passages([fill, tail], limits)
        self.assertEqual(len(result.passages), 1)
        self.assertEqual(result.passages[0].end, 8.0)

    def test_max_passages_truncates_and_says_so(self):
        limits = QuizLimits(max_passages=3)
        result = build_passages(make_segments(), limits)
        self.assertEqual(len(result.passages), 3)
        self.assertTrue(result.truncated)
        self.assertLess(result.lecture_end, 120 * 6.0)

    def test_unsorted_input_is_ordered_by_start(self):
        segments = list(reversed(make_segments(10)))
        result = build_passages(segments)
        self.assertEqual(result.passages[0].start, 0.0)


class MapEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.passages = make_passages()

    def test_single_id_maps_to_server_times(self):
        ranges, reasons = map_evidence_ids(["S2"], self.passages)
        self.assertEqual(reasons, [])
        passage = self.passages.passages[1]
        self.assertEqual((ranges[0].start, ranges[0].end), (round(passage.start, 2), round(passage.end, 2)))

    def test_adjacent_ids_merge_and_separate_ids_stay_sorted(self):
        ranges, reasons = map_evidence_ids(["S5", "S2", "S3"], self.passages)
        self.assertEqual(reasons, [])
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0].start, self.passages.passages[1].start)
        self.assertEqual(ranges[0].end, self.passages.passages[2].end)
        self.assertEqual(ranges[1].start, self.passages.passages[4].start)

    def test_quote_is_real_bounded_transcript_text(self):
        ranges, _ = map_evidence_ids(["S1"], self.passages, QuizLimits(quote_max=40))
        self.assertLessEqual(len(ranges[0].quote), 40)
        self.assertIn(ranges[0].quote, self.passages.passages[0].text)

    def test_unknown_ids_are_rejected_without_partial_results(self):
        total = len(self.passages.passages)
        for ids in (["S%d" % (total + 1)], ["S1", "S%d" % (total + 1)], ["S99999"]):
            ranges, reasons = map_evidence_ids(ids, self.passages)
            self.assertEqual(ranges, [], ids)
            self.assertTrue(reasons, ids)
        self.assertEqual(map_evidence_ids(["S%d" % (total + 1)], self.passages)[1], [Reason.UNKNOWN_EVIDENCE_ID])

    def test_malformed_ids_are_rejected(self):
        for bad in ("S0", "s1", "S01", " S1", "S1 ", "S1\n", "S-1", "S1.5", "1", "", "T1", "S", 1, None, 1.0, ["S1"]):
            ranges, reasons = map_evidence_ids([bad], self.passages)
            self.assertEqual((ranges, reasons), ([], [Reason.INVALID_EVIDENCE_ID]), repr(bad))

    def test_missing_or_non_list_evidence_is_rejected(self):
        for bad in (None, [], (), "S1", 5, {"S1": 1}):
            self.assertEqual(map_evidence_ids(bad, self.passages), ([], [Reason.NO_EVIDENCE]), repr(bad))

    def test_moment_evidence_is_unsupported(self):
        for ids in (["M1"], ["m2"], ["S1", "M1"]):
            ranges, reasons = map_evidence_ids(ids, self.passages)
            self.assertEqual(ranges, [])
            self.assertIn(Reason.UNSUPPORTED_MOMENT_EVIDENCE, reasons)

    def test_duplicate_ids_and_too_many_ids(self):
        self.assertEqual(map_evidence_ids(["S1", "S1"], self.passages)[1], [Reason.DUPLICATE_EVIDENCE_ID])
        self.assertEqual(map_evidence_ids(["S1", "S2", "S3", "S4"], self.passages)[1], [Reason.TOO_MANY_EVIDENCE_IDS])
        ranges, reasons = map_evidence_ids(["S1", "S3", "S5"], self.passages)  # exactly the limit
        self.assertEqual((len(ranges), reasons), (3, []))

    def test_empty_transcript_has_no_valid_ids(self):
        empty = build_passages([])
        self.assertEqual(map_evidence_ids(["S1"], empty), ([], [Reason.UNKNOWN_EVIDENCE_ID]))

    def test_ranges_are_within_the_lecture(self):
        ranges, _ = map_evidence_ids([f"S{len(self.passages.passages)}"], self.passages)
        self.assertLessEqual(ranges[0].end, self.passages.lecture_end)
        self.assertGreaterEqual(ranges[0].start, 0)


if __name__ == "__main__":
    unittest.main()
