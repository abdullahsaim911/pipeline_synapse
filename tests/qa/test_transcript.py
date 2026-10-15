"""Transcript loading: normalization, database/file sources, malformed data, isolation."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.qa.helpers import LECTURE_LINES, OTHER_LECTURE_LINES, PROJECT_ROOT, raw_rows
from qa.transcript import (
    is_valid_video_id,
    load_transcript,
    normalize_segments,
    transcript_file_path,
    video_folder,
)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api.database import Base
    from api import models
    HAVE_SQLALCHEMY = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_SQLALCHEMY = False

VIDEO_A = "AAAAAAAAAAA"
VIDEO_B = "BBBBBBBBBBB"


def memory_session(test_case):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    test_case.addCleanup(engine.dispose)
    return sessionmaker(bind=engine)()


def write_transcript(data_dir, video_id, lines, step=6.0):
    folder = os.path.join(data_dir, video_id)
    os.makedirs(folder, exist_ok=True)
    entries = [{"time": "00:00", "start": r["start"], "end": r["end"],
                "duration": r["end"] - r["start"], "text": r["text"]}
               for r in raw_rows(lines, step)]
    path = os.path.join(folder, "transcript.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"video_id": video_id, "duration_seconds": 4.8, "entries": entries}, handle)
    return path


class NormalizeTests(unittest.TestCase):
    def test_overlapping_end_is_capped_at_next_start(self):
        segs = normalize_segments([
            {"start": 0.3, "end": 6.3, "text": "one"},
            {"start": 3.1, "end": 9.4, "text": "two"},
            {"start": 6.3, "end": 12.2, "text": "three"},
        ])
        self.assertEqual([(s.start, s.end) for s in segs], [(0.3, 3.1), (3.1, 6.3), (6.3, 12.2)])

    def test_gap_is_preserved_when_end_precedes_next_start(self):
        segs = normalize_segments([{"start": 0, "end": 2, "text": "a"},
                                   {"start": 10, "end": 12, "text": "b"}])
        self.assertEqual(segs[0].end, 2)

    def test_missing_or_bad_end_uses_next_start(self):
        for bad in (None, "x", -1, 0, float("nan")):
            segs = normalize_segments([{"start": 5, "end": bad, "text": "a"},
                                       {"start": 9, "text": "b"}])
            self.assertEqual(segs[0].end, 9, bad)

    def test_last_segment_without_end_gets_a_default(self):
        segs = normalize_segments([{"start": 5, "text": "a"}])
        self.assertGreater(segs[0].end, 5)

    def test_invalid_rows_are_dropped_and_order_fixed(self):
        segs = normalize_segments([
            {"start": 10, "text": "late"},
            {"start": "abc", "text": "bad start"},
            {"start": -3, "text": "negative"},
            {"start": 4, "text": "   "},
            {"start": None, "text": "none"},
            {"start": 2, "text": "early"},
        ])
        self.assertEqual([s.text for s in segs], ["early", "late"])
        self.assertEqual([s.index for s in segs], [0, 1])

    def test_text_is_cleaned(self):
        segs = normalize_segments([{"start": 0, "text": "  &gt;&gt; it&#39;s [Music] \n here "}])
        self.assertEqual(segs[0].text, "it's here")


class VideoIdTests(unittest.TestCase):
    def test_valid_and_invalid_ids(self):
        self.assertTrue(is_valid_video_id("3zLQfuMZMF4"))
        self.assertTrue(is_valid_video_id("3x6-3_yF6q4"))
        for bad in ("short", "../../etc/pw", "3zLQfuMZMF4/", "3zLQfuMZMFé", "", None, 12345678901):
            self.assertFalse(is_valid_video_id(bad), bad)

    def test_unsafe_id_never_builds_a_path(self):
        self.assertIsNone(transcript_file_path("../../secrets"))


class FileLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = self.tmp.name

    def test_loads_file_and_normalizes(self):
        write_transcript(self.data, VIDEO_A, LECTURE_LINES)
        result = load_transcript(VIDEO_A, data_dir=self.data)
        self.assertEqual(result.source, "file")
        self.assertTrue(result.file_exists)
        self.assertEqual(len(result.segments), len(LECTURE_LINES))
        self.assertEqual(result.segments[0].end, 6.0)  # capped at next start, not 9.0

    def test_missing_file(self):
        result = load_transcript(VIDEO_A, data_dir=self.data)
        self.assertEqual((result.segments, result.source, result.file_exists), ([], "none", False))

    def test_failed_transcription_file_has_no_entries(self):
        folder = os.path.join(self.data, VIDEO_A)
        os.makedirs(folder)
        with open(os.path.join(folder, "transcript.json"), "w") as handle:
            json.dump({"video_id": VIDEO_A, "method": "failed", "entry_count": 0,
                       "error": "No audio data received from stream", "entries": []}, handle)
        result = load_transcript(VIDEO_A, data_dir=self.data)
        self.assertEqual((result.segments, result.source, result.file_exists), ([], "none", True))

    def test_malformed_files_degrade_to_empty(self):
        folder = os.path.join(self.data, VIDEO_A)
        os.makedirs(folder)
        path = os.path.join(folder, "transcript.json")
        for content in ("{not json", "[]", '{"entries": "oops"}', '{"entries": [1, "x", null]}',
                        '{"entries": [{"start": "a"}]}', ""):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            result = load_transcript(VIDEO_A, data_dir=self.data)
            self.assertEqual(result.segments, [], content)
            self.assertEqual(result.source, "none", content)

    def test_path_traversal_id_is_rejected(self):
        write_transcript(self.data, VIDEO_A, LECTURE_LINES)
        result = load_transcript("../" + VIDEO_A, data_dir=os.path.join(self.data, VIDEO_A))
        self.assertEqual(result.segments, [])

    def test_id_differing_only_by_case_never_loads_the_other_transcript(self):
        # On case-insensitive filesystems (Windows, macOS) the OS would resolve the
        # alias to the same folder; the loader must still refuse it.
        write_transcript(self.data, VIDEO_A, LECTURE_LINES)
        alias = VIDEO_A.lower()
        self.assertNotEqual(alias, VIDEO_A)
        self.assertIsNone(video_folder(alias, self.data))
        self.assertIsNone(transcript_file_path(alias, self.data))
        result = load_transcript(alias, data_dir=self.data)
        self.assertEqual((result.segments, result.source, result.file_exists), ([], "none", False))
        # the exact ID still works
        self.assertEqual(load_transcript(VIDEO_A, data_dir=self.data).source, "file")

    def test_mixed_case_ids_with_their_own_folders_stay_separate(self):
        probe = os.path.join(self.data, "probe_CASE")
        os.makedirs(probe)
        insensitive = os.path.isdir(os.path.join(self.data, "PROBE_case"))
        if insensitive:
            self.skipTest("filesystem is case-insensitive: two IDs differing only by case cannot coexist")
        upper, lower = "AbCdEfGhIjK", "aBcDeFgHiJk"
        write_transcript(self.data, upper, ["upper case video"])
        write_transcript(self.data, lower, ["lower case video"])
        self.assertEqual(load_transcript(upper, data_dir=self.data).segments[0].text, "upper case video")
        self.assertEqual(load_transcript(lower, data_dir=self.data).segments[0].text, "lower case video")

    def test_transcript_file_name_must_match_exactly(self):
        folder = os.path.join(self.data, VIDEO_A)
        os.makedirs(folder)
        with open(os.path.join(folder, "Transcript.JSON"), "w", encoding="utf-8") as handle:
            json.dump({"entries": [{"start": 0, "text": "wrong name"}]}, handle)
        self.assertIsNone(transcript_file_path(VIDEO_A, self.data))
        self.assertEqual(load_transcript(VIDEO_A, data_dir=self.data).segments, [])

    def test_symlinked_transcript_pointing_elsewhere_is_rejected(self):
        outside = os.path.join(self.data, "outside.json")
        with open(outside, "w", encoding="utf-8") as handle:
            json.dump({"entries": [{"start": 0, "text": "outside"}]}, handle)
        folder = os.path.join(self.data, VIDEO_A)
        os.makedirs(folder)
        try:
            os.symlink(outside, os.path.join(folder, "transcript.json"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        self.assertIsNone(transcript_file_path(VIDEO_A, self.data))

    def test_io_errors_are_not_swallowed(self):
        write_transcript(self.data, VIDEO_A, LECTURE_LINES)
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                load_transcript(VIDEO_A, data_dir=self.data)

    def test_invalid_ids_do_not_touch_the_database_or_filesystem(self):
        class ForbiddenSession:
            def query(self, *args, **kwargs):
                raise AssertionError("database must not be queried for an invalid video ID")

        with mock.patch("os.listdir", side_effect=AssertionError("filesystem must not be listed")):
            for bad in ("short", "../" + VIDEO_A, VIDEO_A + "\n", "AAAAAAAAAA\u00e9", "", None):
                with self.subTest(bad=bad):
                    result = load_transcript(bad, ForbiddenSession(), self.data)
                    self.assertEqual((result.segments, result.source, result.file_exists), ([], "none", False))

    def test_videos_are_isolated(self):
        write_transcript(self.data, VIDEO_A, LECTURE_LINES)
        write_transcript(self.data, VIDEO_B, OTHER_LECTURE_LINES)
        a = load_transcript(VIDEO_A, data_dir=self.data)
        b = load_transcript(VIDEO_B, data_dir=self.data)
        self.assertNotIn("mitochondria", " ".join(s.text for s in a.segments))
        self.assertNotIn("integration", " ".join(s.text for s in b.segments))

    @unittest.skipUnless(os.path.isfile(os.path.join(PROJECT_ROOT, "data", "3zLQfuMZMF4", "transcript.json")),
                         "sample lecture data not present")
    def test_real_sample_lecture(self):
        result = load_transcript("3zLQfuMZMF4")
        self.assertEqual(result.source, "file")
        self.assertGreater(len(result.segments), 50)
        self.assertTrue(all(s.end > s.start for s in result.segments))
        self.assertTrue(all(a.end <= b.start + 1e-9 for a, b in zip(result.segments, result.segments[1:])))


@unittest.skipUnless(HAVE_SQLALCHEMY, "SQLAlchemy not installed")
class DatabaseLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = self.tmp.name
        self.db = memory_session(self)
        self.addCleanup(self.db.close)
        for vid in (VIDEO_A, VIDEO_B):
            self.db.add(models.Video(id=vid, title=vid, source_url="http://x"))
        self.db.commit()

    def add_rows(self, video_id, lines):
        for r in raw_rows(lines):
            self.db.add(models.Transcript(video_id=video_id, start_time=r["start"],
                                          end_time=r["end"], text=r["text"]))
        self.db.commit()

    def test_database_rows_are_preferred_over_file(self):
        self.add_rows(VIDEO_A, ["from the database"])
        write_transcript(self.data, VIDEO_A, ["from the file"])
        result = load_transcript(VIDEO_A, self.db, self.data)
        self.assertEqual(result.source, "database")
        self.assertEqual(result.segments[0].text, "from the database")
        self.assertTrue(result.file_exists)

    def test_falls_back_to_file_when_database_has_no_rows(self):
        write_transcript(self.data, VIDEO_A, ["from the file"])
        result = load_transcript(VIDEO_A, self.db, self.data)
        self.assertEqual((result.source, result.segments[0].text), ("file", "from the file"))

    def test_falls_back_to_file_when_database_rows_are_unusable(self):
        self.add_rows(VIDEO_A, ["   ", ""])
        write_transcript(self.data, VIDEO_A, ["from the file"])
        self.assertEqual(load_transcript(VIDEO_A, self.db, self.data).source, "file")

    def test_database_errors_propagate_instead_of_falling_back(self):
        class BrokenSession:
            def query(self, *a, **k):
                raise RuntimeError("programming error")
        write_transcript(self.data, VIDEO_A, ["from the file"])
        with self.assertRaises(RuntimeError):
            load_transcript(VIDEO_A, BrokenSession(), self.data)

    def test_missing_table_error_propagates(self):
        from sqlalchemy.exc import OperationalError
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        self.addCleanup(engine.dispose)
        no_tables = sessionmaker(bind=engine)()   # schema never created
        self.addCleanup(no_tables.close)
        write_transcript(self.data, VIDEO_A, ["from the file"])
        with self.assertRaises(OperationalError):
            load_transcript(VIDEO_A, no_tables, self.data)

    def test_import_errors_propagate(self):
        write_transcript(self.data, VIDEO_A, ["from the file"])
        with mock.patch.dict(sys.modules, {"api.models": None}):
            with self.assertRaises(ImportError):
                load_transcript(VIDEO_A, self.db, self.data)

    def test_empty_database_result_is_the_only_trigger_for_file_fallback(self):
        # the database answered, it simply has nothing for this video
        write_transcript(self.data, VIDEO_A, ["from the file"])
        with mock.patch("qa.transcript._load_from_database", return_value=[]) as query:
            result = load_transcript(VIDEO_A, self.db, self.data)
        query.assert_called_once()
        self.assertEqual(result.source, "file")

    def test_database_overlapping_ends_are_normalized(self):
        self.add_rows(VIDEO_A, LECTURE_LINES[:3])
        segs = load_transcript(VIDEO_A, self.db, self.data).segments
        self.assertEqual([(s.start, s.end) for s in segs[:2]], [(0.0, 6.0), (6.0, 12.0)])

    def test_database_videos_are_isolated(self):
        self.add_rows(VIDEO_A, LECTURE_LINES)
        self.add_rows(VIDEO_B, OTHER_LECTURE_LINES)
        a = load_transcript(VIDEO_A, self.db, self.data).segments
        b = load_transcript(VIDEO_B, self.db, self.data).segments
        self.assertEqual(len(a), len(LECTURE_LINES))
        self.assertEqual(len(b), len(OTHER_LECTURE_LINES))
        self.assertFalse({s.text for s in a} & {s.text for s in b})

    def test_unknown_video_has_nothing(self):
        result = load_transcript("CCCCCCCCCCC", self.db, self.data)
        self.assertEqual((result.segments, result.source, result.file_exists), ([], "none", False))


if __name__ == "__main__":
    unittest.main()
