"""qa.asr_cleanup: conservative disfluency removal and STEM term repair."""

import json
import os
import tempfile
import unittest
from unittest import mock

from qa.asr_cleanup import (
    TERMS_FILE_ENV,
    clean_caption_text,
    load_extra_terms,
    sanitize_segments,
    sanitize_stats,
)
from qa.transcript import normalize_segments


class DisfluencyTests(unittest.TestCase):
    def test_filler_sounds_are_removed(self):
        self.assertEqual(clean_caption_text("so um we take the derivative"), "so we take the derivative")
        self.assertEqual(clean_caption_text("Uh, the integral of x"), "the integral of x")
        self.assertEqual(clean_caption_text("er, well that works"), "well that works")
        self.assertEqual(clean_caption_text("hmm that is odd"), "that is odd")
        self.assertEqual(clean_caption_text("mmm okay"), "okay")

    def test_commas_around_fillers_are_tidied(self):
        self.assertEqual(clean_caption_text("so, um, we start"), "so, we start")

    def test_words_that_merely_contain_a_filler_are_untouched(self):
        for text in ("an umbrella and a human", "the number of errors", "to err is human", "hummus", "5 mm long", "mm and cm"):
            self.assertEqual(clean_caption_text(text), text, text)

    def test_cut_off_word_fragments_are_removed(self):
        self.assertEqual(clean_caption_text("th- the derivative"), "the derivative")
        self.assertEqual(clean_caption_text("wh- what is v"), "what is v")
        self.assertEqual(clean_caption_text("I- I think so"), "I think so")

    def test_a_hyphen_that_is_not_a_false_start_is_kept(self):
        self.assertEqual(clean_caption_text("pre- and post-test"), "pre- and post-test")

    def test_repeated_words_collapse_but_variables_and_numbers_do_not(self):
        self.assertEqual(clean_caption_text("the the integral"), "the integral")
        self.assertEqual(clean_caption_text("no no no that is wrong"), "no that is wrong")
        self.assertEqual(clean_caption_text("so x x equals 2 2"), "so x x equals 2 2")
        self.assertEqual(clean_caption_text("it is 3 3 3"), "it is 3 3 3")

    def test_disfluency_removal_can_be_switched_off(self):
        self.assertEqual(clean_caption_text("um the the thing", remove_disfluencies=False), "um the the thing")


class TermRepairTests(unittest.TestCase):
    def test_split_terms_are_joined(self):
        cases = {
            "the anti derivative of x": "the antiderivative of x",
            "find the anti-derivatives": "find the antiderivatives",
            "the eigen values of A": "the eigenvalues of A",
            "each eigen vector": "each eigenvector",
            "a tan gent line": "a tangent line",
            "the co efficient of x": "the coefficient of x",
            "take the log arithm": "take the logarithm",
            "inside the mito chondria": "inside the mitochondria",
        }
        for heard, expected in cases.items():
            self.assertEqual(clean_caption_text(heard), expected, heard)

    def test_capitalisation_of_the_first_letter_is_kept(self):
        self.assertEqual(clean_caption_text("Eigen vector basis"), "Eigenvector basis")

    def test_liate_alias_is_spelled_out(self):
        self.assertEqual(clean_caption_text("remember the acronym li8 for picking u"), "remember the acronym LIATE for picking u")

    def test_cosign_is_repaired_only_in_a_trigonometric_context(self):
        self.assertEqual(clean_caption_text("the cosign of x is even"), "the cosine of x is even")
        self.assertEqual(clean_caption_text("you need someone to cosign the loan"), "you need someone to cosign the loan")

    def test_term_repair_can_be_switched_off(self):
        self.assertEqual(clean_caption_text("an eigen value", normalize_terms=False), "an eigen value")


class GeneralBehaviourTests(unittest.TestCase):
    TEXTS = [
        "so um we take th- the anti derivative", "the the eigen values of A", "hmm li8 is the rule", "plain text with nothing to fix",
        "x x plus 2 2 equals y", "uh, er, um", "5 mm and 3 cm", "Co efficient of the tan gent line",
    ]

    def test_idempotent(self):
        for text in self.TEXTS:
            once = clean_caption_text(text)
            self.assertEqual(clean_caption_text(once), once, text)

    def test_clean_text_is_unchanged(self):
        text = "The derivative of x squared is two x."
        self.assertEqual(clean_caption_text(text), text)

    def test_text_with_nothing_left_becomes_empty(self):
        for text in ("uh", "um, er", "hmm.", "", "  "):
            self.assertEqual(clean_caption_text(text), "", repr(text))

    def test_digits_and_variables_survive(self):
        text = "evaluate at x equals 0 then 1 over 2"
        self.assertEqual(clean_caption_text(text), text)


class ExtraTermsTests(unittest.TestCase):
    def write(self, content):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, "terms.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_whole_word_case_insensitive_literal_replacements(self):
        pairs = load_extra_terms(self.write(json.dumps({"heart rate": "heart-rate", "foo": "bar\\1"})))
        self.assertEqual(clean_caption_text("Heart Rate and foo and food", extra_terms=pairs), "Heart-rate and bar\\1 and food")

    def test_invalid_files_are_ignored(self):
        self.assertEqual(load_extra_terms(self.write("not json")), [])
        self.assertEqual(load_extra_terms(self.write("[1, 2]")), [])
        self.assertEqual(load_extra_terms("/no/such/file.json"), [])
        self.assertEqual(load_extra_terms(None), [])

    def test_environment_variable_is_used_by_sanitize_segments(self):
        path = self.write(json.dumps({"gauss law": "Gauss's law"}))
        segs = normalize_segments([{"start": 0.0, "end": 5.0, "text": "now gauss law applies"}])
        with mock.patch.dict(os.environ, {TERMS_FILE_ENV: path}):
            self.assertEqual(sanitize_segments(segs)[0].text, "now Gauss's law applies")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TERMS_FILE_ENV, None)
            self.assertEqual(sanitize_segments(segs)[0].text, "now gauss law applies")


class SanitizeSegmentsTests(unittest.TestCase):
    RAW = [
        {"start": 0.0, "end": 6.0, "text": "welcome um to the lecture"},
        {"start": 6.0, "end": 12.0, "text": "uh"},
        {"start": 12.0, "end": 18.0, "text": "the the anti derivative of x"},
        {"start": 18.0, "end": 24.0, "text": "li8 helps us choose u"},
    ]

    def setUp(self):
        self.segs = normalize_segments(self.RAW)
        self.out = sanitize_segments(self.segs, extra_terms=[])

    def test_cleaned_text_and_dropped_empty_caption(self):
        self.assertEqual([s.text for s in self.out],
                         ["welcome to the lecture", "the antiderivative of x", "LIATE helps us choose u"])

    def test_start_times_are_preserved_and_indexes_renumbered(self):
        self.assertEqual([s.start for s in self.out], [0.0, 12.0, 18.0])
        self.assertEqual([s.index for s in self.out], [0, 1, 2])

    def test_effective_end_times_stay_valid(self):
        for a, b in zip(self.out, self.out[1:]):
            self.assertLessEqual(a.end, b.start)
            self.assertGreater(a.end, a.start)

    def test_input_is_not_mutated_and_result_is_idempotent(self):
        self.assertEqual(len(self.segs), 4)
        self.assertEqual(self.segs[1].text, "uh")
        again = sanitize_segments(self.out, extra_terms=[])
        self.assertEqual([(s.start, s.text) for s in again], [(s.start, s.text) for s in self.out])

    def test_stats_count_without_exposing_text(self):
        stats = sanitize_stats(self.segs, self.out)
        self.assertEqual((stats["segments_before"], stats["segments_after"], stats["dropped"]), (4, 3, 1))
        self.assertGreater(stats["chars_removed"], 0)
        self.assertTrue(all(isinstance(v, int) for v in stats.values()))


if __name__ == "__main__":
    unittest.main()
