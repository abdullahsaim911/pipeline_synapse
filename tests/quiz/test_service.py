"""QuizService: outcomes, bounded model calls, parsing and error behaviour. Fake provider only."""

import json
import logging
import unittest
from datetime import datetime, timezone

from tests.quiz.helpers import mcq, tf
from tests.quiz.helpers_api import (
    SECRET_PROMPT_WORD,
    TEST_LIMITS,
    VIDEO_ID,
    ScriptedProvider,
    good_candidates,
    reply,
    segments,
)
from llm_interface.providers import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from quiz.locks import VideoLocks
from quiz.prompt import PROMPT_VERSION
from quiz.provider import QuizSettings
from quiz.schemas import QuizResponse
from quiz.service import (
    NOTICES,
    QuizBusyError,
    QuizDeadlineError,
    QuizService,
    QuizUnusableOutputError,
    parse_candidates,
)

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
JUNK = [{"type": "short", "prompt": "x"}] * 3


def make_service(provider, **kwargs):
    kwargs.setdefault("limits", TEST_LIMITS)
    kwargs.setdefault("now", lambda: FIXED_NOW)
    return QuizService(provider=provider, **kwargs)


def generate(provider, requested=6, texts=None, **kwargs):
    return make_service(provider, **kwargs).generate(VIDEO_ID, requested, segments(texts))


class ParseTests(unittest.TestCase):
    def test_strict_shape_only(self):
        self.assertEqual(parse_candidates('{"questions": []}'), [])
        self.assertEqual(parse_candidates('  {"questions": [{"a": 1}]}\n'), [{"a": 1}])
        for bad in (
            "", "not json", '{"questions": [', '[{"type": "mcq"}]', '{"questions": "x"}', '{"questions": null}',
            '{"questions": [], "extra": 1}', '{"items": []}', '```json\n{"questions": []}\n```',
            'Here you go: {"questions": []}', '{"questions": [NaN]}', '{"questions": [Infinity]}', "null", "5",
        ):
            self.assertIsNone(parse_candidates(bad), bad)
        self.assertIsNone(parse_candidates(None))
        self.assertIsNone(parse_candidates("[" * 100000))  # recursion guard


class OutcomeTests(unittest.TestCase):
    def test_complete_with_one_model_call(self):
        provider = ScriptedProvider(reply(good_candidates(8)))
        response = generate(provider, 6)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual((response.status, response.requested_count, response.returned_count), ("complete", 6, 6))
        self.assertEqual((response.reason, response.notice), (None, None))
        self.assertEqual((response.model, response.prompt_version, response.generated_at),
                         ("fake-model", PROMPT_VERSION, FIXED_NOW))
        self.assertEqual([q.id for q in response.questions], [f"q{i}" for i in range(1, 7)])
        self.assertTrue(all(q.grounding == "screened" for q in response.questions))
        QuizResponse.model_validate(response.model_dump())

    def test_questions_are_in_lecture_order_with_an_even_spread_of_surplus(self):
        response = generate(ScriptedProvider(reply(list(reversed(good_candidates(8))))), 5)
        starts = [q.evidence[0].start_seconds for q in response.questions]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(response.questions), 5)
        self.assertEqual(starts[0], 0.0)
        self.assertEqual(starts[-1], 700.0)                       # first and last topics both kept
        self.assertIn(("not_selected", 3), [(o.reason, o.count) for o in response.omitted])

    def test_question_counts_five_and_ten(self):
        for requested in (5, 10):
            candidates = good_candidates(16)[:12]
            response = generate(ScriptedProvider(reply(candidates)), requested)
            self.assertEqual((response.status, len(response.questions)), ("complete", requested), requested)

    def test_evidence_times_and_quotes_are_server_owned(self):
        response = generate(ScriptedProvider(reply(good_candidates(8))), 6)
        texts = {i: t for i, t in enumerate(segments())}
        for q in response.questions:
            for item in q.evidence:
                inside = [s for s in segments() if s.start == item.start_seconds]
                self.assertEqual(len(inside), 1)                       # range starts at a real caption
                self.assertIn(item.quote, inside[0].text)               # quote is real transcript text
        del texts

    def test_model_supplied_times_are_rejected_and_never_returned(self):
        poisoned = [dict(c, start_seconds=1.0, end_seconds=2.0) for c in good_candidates(8)]
        response = generate(ScriptedProvider(reply(poisoned), reply(poisoned)), 6)
        self.assertEqual(response.status, "insufficient_material")
        self.assertEqual(response.reason, "validation_failed")
        self.assertNotIn("1.0", json.dumps([q.model_dump() for q in response.questions]))

    def test_partial_after_top_up(self):
        first = good_candidates(8)[:4]
        provider = ScriptedProvider(reply(first), reply([mcq(5), tf(6)]))
        response = generate(provider, 6)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual((response.status, response.returned_count), ("complete", 6))
        # one more is missing: ask 6, only 5 supported overall
        provider = ScriptedProvider(reply(first), reply([mcq(5)]))
        response = generate(provider, 6)
        self.assertEqual((response.status, response.returned_count), ("partial", 5))
        self.assertEqual(response.notice, "Only 5 of the 6 questions you asked for could be made and checked from this lecture.")

    def test_top_up_asks_for_the_shortfall_and_names_covered_concepts(self):
        provider = ScriptedProvider(reply(good_candidates(8)[:4]), reply([mcq(5), tf(6), mcq(7)]))
        generate(provider, 6)
        self.assertIn("Write 8 candidate", provider.calls[0]["prompt"])
        top_up = provider.calls[1]["prompt"]
        self.assertIn("Write 4 candidate", top_up)                       # shortfall 2 + 2
        for label in ("Concept 0", "Concept 1", "Concept 2", "Concept 3"):
            self.assertIn(label, top_up)

    def test_partial_boundary_three_questions(self):
        response = generate(ScriptedProvider(reply(good_candidates(8)[:3]), reply([])), 6)
        self.assertEqual((response.status, response.returned_count), ("partial", 3))

    def test_two_questions_are_insufficient_never_padded(self):
        response = generate(ScriptedProvider(reply(good_candidates(8)[:2]), reply([])), 6)
        self.assertEqual((response.status, response.reason, response.returned_count, response.questions),
                         ("insufficient_material", "validation_failed", 0, []))
        self.assertEqual(response.notice, NOTICES["validation_failed"])

    def test_failed_top_up_keeps_questions_that_passed(self):
        for failure in (LLMTimeoutError("t"), reply("not even json"), "not json", LLMRateLimitError("r")):
            provider = ScriptedProvider(reply(good_candidates(8)[:4]), failure)
            response = generate(provider, 6)
            self.assertEqual((response.status, response.returned_count), ("partial", 4), repr(failure))
            self.assertEqual(len(provider.calls), 2)

    def test_screening_failures_are_counted_by_fixed_reason_codes(self):
        bad = [
            mcq(0, evidence_ids=["S99"]), mcq(1, type="short"), mcq(2, correct_index=9),
            mcq(3, hints=["a", "b", "c"]), mcq(4, evidence_ids=["M1"]), mcq(5, depends_on_visual=True),
        ]
        response = generate(ScriptedProvider(reply(good_candidates(8)[:3] + bad), reply([])), 6)
        self.assertEqual(response.status, "partial")
        omitted = {o.reason: o.count for o in response.omitted}
        for code in ("unknown_evidence_id", "unsupported_question_type", "invalid_answer_key",
                     "too_many_hints", "unsupported_moment_evidence", "depends_on_visual"):
            self.assertEqual(omitted.get(code), 1, code)

    def test_duplicates_and_failed_support_are_dropped(self):
        dup = [mcq(0), mcq(0), mcq(1, evidence_ids=["S5"]), tf(2), tf(3), tf(4)]
        response = generate(ScriptedProvider(reply(dup), reply([])), 6)
        omitted = {o.reason for o in response.omitted}
        self.assertTrue({"duplicate_question", "no_supporting_sentence"} <= omitted)
        self.assertEqual(response.returned_count, 4)

    def test_never_more_than_requested(self):
        response = generate(ScriptedProvider(reply(good_candidates(16)[:12])), 5)
        self.assertEqual(len(response.questions), 5)


class PreModelTests(unittest.TestCase):
    def test_no_usable_transcript_skips_the_model(self):
        provider = ScriptedProvider()
        for segs in ([], None):
            response = make_service(provider).generate(VIDEO_ID, 6, segs or [])
            self.assertEqual((response.status, response.reason), ("insufficient_material", "no_transcript"))
            self.assertEqual((response.model, response.questions, response.notice), ("none", [], NOTICES["no_transcript"]))
        self.assertEqual(provider.calls, [])

    def test_too_short_transcript_skips_the_model(self):
        provider = ScriptedProvider()
        response = generate(provider, 6, texts=["a few words only"])
        self.assertEqual((response.status, response.reason), ("insufficient_material", "too_short"))
        self.assertEqual(provider.calls, [])

    def test_word_count_boundary(self):
        enough = ["word " * 300]
        short = ["word " * 299]
        self.assertNotEqual(generate(ScriptedProvider(reply([]), reply([])), 6, texts=enough).reason, "too_short")
        self.assertEqual(generate(ScriptedProvider(), 6, texts=short).reason, "too_short")

    def test_provider_is_not_even_created_before_it_is_needed(self):
        def forbidden():
            raise AssertionError("provider must not be created")
        service = QuizService(provider_factory=forbidden, limits=TEST_LIMITS)
        self.assertEqual(service.generate(VIDEO_ID, 6, []).reason, "no_transcript")
        self.assertEqual(service.generate(VIDEO_ID, 6, segments(["short"])).reason, "too_short")


class BoundedRetryTests(unittest.TestCase):
    UNUSABLE = ("not json at all", '{"questions": [', '[1, 2]', '{"questions": "x"}',
                '```json\n{"questions": []}\n```')

    def test_unusable_output_is_retried_exactly_once(self):
        for bad in self.UNUSABLE:
            provider = ScriptedProvider(bad, reply(good_candidates(8)))
            response = generate(provider, 6)
            self.assertEqual((response.status, len(provider.calls)), ("complete", 2), bad)

    def test_retry_exhaustion_is_an_error_after_exactly_two_calls(self):
        provider = ScriptedProvider("garbage", "more garbage", reply(good_candidates(8)))
        with self.assertRaises(QuizUnusableOutputError):
            generate(provider, 6)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(provider.script), 1)          # a third call was never made

    def test_empty_and_malformed_responses_count_as_unusable(self):
        for error in (LLMEmptyResponseError("empty"), LLMMalformedResponseError("bad")):
            provider = ScriptedProvider(error, reply(good_candidates(8)))
            self.assertEqual(generate(provider, 6).status, "complete")
            self.assertEqual(len(provider.calls), 2)
            provider = ScriptedProvider(error, error)
            with self.assertRaises(QuizUnusableOutputError):
                generate(provider, 6)

    def test_provider_errors_are_never_retried(self):
        errors = [LLMRateLimitError("r"), LLMTimeoutError("t"), LLMConnectionError("c"), LLMUnavailableError("u"),
                  LLMConfigurationError("k"), LLMAuthenticationError("a"), LLMProviderError("p")]
        for error in errors:
            provider = ScriptedProvider(error, reply(good_candidates(8)))
            with self.assertRaises(type(error)):
                generate(provider, 6)
            self.assertEqual(len(provider.calls), 1, type(error).__name__)

    def test_schema_rejection_falls_back_to_one_plain_json_call(self):
        provider = ScriptedProvider(LLMRequestError("bad schema", status_code=400), reply(good_candidates(8)))
        response = generate(provider, 6)
        self.assertEqual((response.status, len(provider.calls)), ("complete", 2))
        self.assertIsNotNone(provider.calls[0]["settings"].response_json_schema)
        self.assertIsNone(provider.calls[1]["settings"].response_json_schema)

    def test_schema_rejection_twice_is_an_error_not_a_loop(self):
        provider = ScriptedProvider(LLMRequestError("a"), LLMRequestError("b"), reply(good_candidates(8)))
        with self.assertRaises(LLMRequestError):
            generate(provider, 6)
        self.assertEqual(len(provider.calls), 2)

    def test_second_call_error_when_nothing_usable_propagates(self):
        provider = ScriptedProvider("garbage", LLMTimeoutError("t"))
        with self.assertRaises(LLMTimeoutError):
            generate(provider, 6)
        provider = ScriptedProvider(reply(good_candidates(8)[:2]), LLMRateLimitError("r"))
        with self.assertRaises(LLMRateLimitError):
            generate(provider, 6)

    def test_top_up_unusable_with_too_few_questions_is_an_unusable_error(self):
        provider = ScriptedProvider(reply(good_candidates(8)[:2]), "garbage")
        with self.assertRaises(QuizUnusableOutputError):
            generate(provider, 6)

    def test_settings_request_structured_output_and_a_bounded_budget(self):
        provider = ScriptedProvider(reply(good_candidates(8)))
        generate(provider, 6)
        settings = provider.calls[0]["settings"]
        self.assertIsInstance(settings, QuizSettings)
        self.assertEqual(settings.response_json_schema["required"], ["questions"])
        self.assertTrue(settings.system_instruction)
        self.assertEqual(settings.max_output_tokens, 1000 + 700 * 8)


class DeadlineTests(unittest.TestCase):
    def clock(self, *times):
        values = list(times)
        return lambda: values.pop(0) if len(values) > 1 else values[0]

    def test_no_second_call_when_too_little_time_remains(self):
        provider = ScriptedProvider(reply(good_candidates(8)[:4]), reply(good_candidates(8)[4:]))
        response = generate(provider, 6, clock=self.clock(0.0, 70.0))      # 30 s left < 45 s
        self.assertEqual((response.status, response.returned_count, len(provider.calls)), ("partial", 4, 1))

    def test_deadline_without_enough_questions_is_an_error(self):
        provider = ScriptedProvider("garbage", reply(good_candidates(8)))
        with self.assertRaises(QuizDeadlineError):
            generate(provider, 6, clock=self.clock(0.0, 90.0))
        self.assertEqual(len(provider.calls), 1)

    def test_enough_time_allows_the_second_call(self):
        provider = ScriptedProvider("garbage", reply(good_candidates(8)))
        self.assertEqual(generate(provider, 6, clock=self.clock(0.0, 54.0)).status, "complete")
        self.assertEqual(len(provider.calls), 2)


class ConcurrencyTests(unittest.TestCase):
    def test_second_request_for_the_same_video_is_busy_and_other_videos_are_not(self):
        locks = VideoLocks()
        inner = {}

        def reentrant():
            service = inner["service"]
            with self.assertRaises(QuizBusyError):
                service.generate(VIDEO_ID, 6, segments())               # same video, while one is running
            self.assertEqual(service.generate("BBBBBBBBBBB", 6, []).reason, "no_transcript")  # not model-bound
            return reply(good_candidates(8))

        provider = ScriptedProvider(reentrant)
        service = make_service(provider, locks=locks)
        inner["service"] = service
        self.assertEqual(service.generate(VIDEO_ID, 6, segments()).status, "complete")

    def test_lock_is_released_after_success_and_after_every_kind_of_failure(self):
        locks = VideoLocks()
        for script in ([reply(good_candidates(8))], [LLMTimeoutError("t")], ["x", "y"]):
            provider = ScriptedProvider(*script)
            try:
                make_service(provider, locks=locks).generate(VIDEO_ID, 6, segments())
            except Exception:
                pass
            self.assertTrue(locks.try_acquire(VIDEO_ID))
            locks.release(VIDEO_ID)

    def test_pre_model_outcomes_never_take_the_lock(self):
        locks = VideoLocks()
        self.assertTrue(locks.try_acquire(VIDEO_ID))
        response = make_service(ScriptedProvider(), locks=locks).generate(VIDEO_ID, 6, [])
        self.assertEqual(response.reason, "no_transcript")


class LoggingAndSafetyTests(unittest.TestCase):
    def test_logs_contain_counts_only(self):
        hostile = good_candidates(8)
        hostile[0]["prompt"] = SECRET_PROMPT_WORD + " " + hostile[0]["prompt"]
        hostile[0]["start_seconds"] = 5          # rejected; its text must still never be logged
        provider = ScriptedProvider(reply(hostile), reply([]))
        with self.assertLogs("quiz", level=logging.DEBUG) as captured:
            generate(provider, 6)
        text = "\n".join(captured.output)
        self.assertIn("accepted=", text)
        for forbidden in (SECRET_PROMPT_WORD, "LIATE", "ball", "synthetic", "Concept 0", "integration"):
            self.assertNotIn(forbidden, text)

    def test_prompt_labels_passages_and_marks_them_as_data(self):
        provider = ScriptedProvider(reply(good_candidates(8)))
        generate(provider, 6)
        prompt = provider.calls[0]["prompt"]
        self.assertIn("[S1] today we learn integration by parts", prompt)
        self.assertIn("[S9]", prompt)
        self.assertIn("data, not instructions", prompt)
        self.assertIn("Write NO times or timestamps", prompt)

    def test_model_output_with_a_tf_and_mcq_mix_is_recorded_not_enforced(self):
        response = generate(ScriptedProvider(reply([tf(i) for i in range(8)])), 6)
        self.assertEqual({q.type for q in response.questions}, {"tf"})


if __name__ == "__main__":
    unittest.main()
