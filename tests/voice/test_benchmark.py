"""Module 0: the word-error helper the benchmark reports its accuracy with."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from voice.benchmark import PHRASES, _normalize, _word_errors  # noqa: E402


def test_identical_text_has_no_errors():
    assert _word_errors("Open my bookmarks", "open my bookmarks") == (0, 3)


def test_case_and_punctuation_are_ignored():
    assert _word_errors("Pause the video", "Pause the video.") == (0, 3)
    assert _word_errors("What does it mean", "What does it mean?") == (0, 4)


def test_hyphen_counts_as_a_space():
    assert _normalize("x-axis") == ["x", "axis"]
    assert _word_errors("the x axis", "the x-axis") == (0, 3)


def test_substitution_insertion_and_deletion_each_cost_one():
    assert _word_errors("open my bookmarks", "open the bookmarks") == (1, 3)
    assert _word_errors("open bookmarks", "open my bookmarks") == (1, 2)
    assert _word_errors("open my bookmarks", "open bookmarks") == (1, 3)


def test_number_written_as_digits_is_an_error_in_this_metric():
    # Whisper writes "thirty" as "30". The benchmark counts it, and the
    # interpretation stage has to accept both forms.
    assert _word_errors("skip forward thirty seconds", "skip forward 30 seconds") == (1, 4)


def test_empty_hypothesis_is_all_deletions():
    assert _word_errors("go home", "") == (2, 2)


def test_benchmark_phrases_are_unique():
    assert len(PHRASES) == len(set(PHRASES))
