"""AskService: deterministic decisions, prompt contents, statuses, sources, provider errors."""

import unittest

from tests.qa.helpers import (
    FakeProvider,
    LECTURE_LINES,
    OTHER_LECTURE_LINES,
    forbidden_provider_factory,
    make_segments,
)
from llm_interface.providers import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from qa.moment import MomentContext
from qa.service import (
    SYSTEM_INSTRUCTION,
    AskConfig,
    AskOutcome,
    AskService,
    AskSourceInfo,
    build_prompt,
    parse_model_reply,
    to_plain_text,
)

SEGMENTS = make_segments()


def moment(explanation=None, visual=None, caption="u equals x so du equals dx"):
    return MomentContext(intervention_id="int-1", timestamp=126.0, content_type="graph",
                         caption=caption, explanation=explanation, visual=visual)


class DeterministicDecisionTests(unittest.TestCase):
    def service(self):
        return AskService(provider_factory=forbidden_provider_factory)

    def test_empty_transcript_never_calls_the_llm(self):
        out = self.service().answer("What is integration by parts?", [])
        self.assertEqual((out.status, out.scope, out.sources), ("insufficient_context", "lecture", []))
        self.assertIn("transcript", out.answer_text)

    def test_unmatched_question_on_a_long_transcript_never_calls_the_llm(self):
        # Forced retrieval mode: the transcript is treated as too long to send whole.
        service = AskService(provider_factory=forbidden_provider_factory,
                             config=AskConfig(full_transcript_max_tokens=1))
        out = service.answer("Who won the football world cup in Brazil?", SEGMENTS)
        # A keyword miss on partial evidence cannot prove absence, so it is not "not_in_lecture".
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))
        self.assertTrue(out.answer_text.strip())

    def test_unmatched_question_message_names_both_possibilities_without_claiming_absence(self):
        service = AskService(provider_factory=forbidden_provider_factory,
                             config=AskConfig(full_transcript_max_tokens=1))
        text = service.answer("Who won the football world cup in Brazil?", SEGMENTS).answer_text
        self.assertIn("may not be covered", text)          # possibility 1: the lecture does not cover it
        self.assertIn("different words", text)             # possibility 2: the wording differs
        self.assertIn("3:20", text)                        # a way forward that works without keywords
        for flat_claim in ("does not cover", "doesn't cover", "not in this lecture", "never covers"):
            self.assertNotIn(flat_claim, text.lower())     # absence is never asserted from a keyword miss
        self.assertLessEqual(len(text.split()), 45)        # short enough to listen to

    def test_relative_time_without_position(self):
        out = self.service().answer("Explain the last five minutes", SEGMENTS)
        self.assertEqual(out.status, "insufficient_context")
        self.assertIn("where you are", out.answer_text)

    def test_time_beyond_lecture_mentions_covered_range(self):
        out = self.service().answer("What happens at 59:00?", SEGMENTS)
        self.assertEqual(out.status, "insufficient_context")
        self.assertIn("00:00 to 04:03", out.answer_text)

    def test_question_without_content_terms(self):
        out = self.service().answer("what is it?", SEGMENTS)
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))

    def test_scope_reflects_attached_moment_even_without_llm(self):
        out = self.service().answer("what is it?", [], moment=moment())
        self.assertEqual(out.scope, "lecture_with_moment")


class VisualInstructionTests(unittest.TestCase):
    """The model is told how to behave when the screen cannot be seen (checked on the prompt text only)."""

    def test_instruction_tells_the_model_to_say_it_cannot_see_the_screen_in_plain_words(self):
        self.assertIn("cannot see the screen", SYSTEM_INSTRUCTION)
        self.assertIn('never mention a "visual analysis" section', SYSTEM_INSTRUCTION)

    def test_screen_only_questions_use_the_insufficient_marker(self):
        rule = [line for line in SYSTEM_INSTRUCTION.splitlines() if line.startswith("6.")][0]
        self.assertIn("[[INSUFFICIENT_CONTEXT]]", rule)
        self.assertIn("only what is shown on the screen", rule)

    def test_instruction_still_forbids_guessing_visuals_and_timestamps(self):
        self.assertIn("Never guess what a visual shows", SYSTEM_INSTRUCTION)
        self.assertIn("Never state, guess or cite timestamps", SYSTEM_INSTRUCTION)

    def test_missing_visual_analysis_is_stated_in_the_prompt_for_a_moment_without_visuals(self):
        provider = FakeProvider()
        AskService(provider=provider).answer("What does the graph show?", SEGMENTS, moment=moment())
        self.assertIn("Visual analysis: unavailable", provider.calls[-1]["prompt"])


class AnswerTests(unittest.TestCase):
    def ask(self, question, provider=None, **kwargs):
        provider = provider or FakeProvider()
        service = AskService(provider=provider)
        return service.answer(question, kwargs.pop("segments", SEGMENTS), **kwargs), provider

    def test_answered_with_backend_computed_sources(self):
        out, provider = self.ask("What did the instructor say about LIATE?")
        self.assertEqual((out.status, out.scope), ("answered", "lecture"))
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(out.sources)
        covering = [s for s in out.sources if s.start_seconds <= 66 <= s.end_seconds]
        self.assertTrue(covering, out.sources)           # LIATE is introduced at 1:00
        for s in out.sources:
            self.assertIsNone(s.intervention_id)
            self.assertLess(s.start_seconds, s.end_seconds)
            self.assertLessEqual(s.end_seconds, SEGMENTS[-1].end)

    def test_model_cannot_inject_sources(self):
        out, _ = self.ask("What did the instructor say about LIATE?",
                          FakeProvider("It is at 12:34 to 56:00, see 99:99."))
        self.assertEqual(out.status, "answered")
        self.assertTrue(all(s.end_seconds <= SEGMENTS[-1].end for s in out.sources))

    def test_answered_always_has_sources(self):
        questions = ["What is LIATE?", "explain 1:00 to 1:30", "summarize the lecture",
                     "how are the product rule and the tabular method related"]
        for q in questions:
            out, _ = self.ask(q)
            self.assertEqual(out.status, "answered", q)
            self.assertGreaterEqual(len(out.sources), 1, q)

    def test_prompt_contains_only_this_lectures_evidence(self):
        provider = FakeProvider()
        AskService(provider=provider, config=AskConfig(full_transcript_max_tokens=1)).answer(
            "What did the instructor say about LIATE?", SEGMENTS)   # retrieval mode
        prompt = provider.calls[0]["prompt"]
        self.assertIn("acronym LIATE", prompt)
        self.assertIn("What did the instructor say about LIATE?", prompt)
        for line in OTHER_LECTURE_LINES:
            self.assertNotIn(line, prompt)
        self.assertNotIn("thank you for watching", prompt)  # irrelevant parts are left out

    def test_prompt_budget_is_respected(self):
        service = AskService(provider=FakeProvider(), config=AskConfig(max_evidence_chars=500))
        service.answer("summarize the lecture", SEGMENTS)
        prompt = service.provider.calls[0]["prompt"]
        excerpts = prompt.split("LECTURE EXCERPTS")[1]
        self.assertLessEqual(len(excerpts), 500 + 400)  # budget + headings and timestamps

    def test_generation_settings_and_system_instruction(self):
        _, provider = self.ask("What is LIATE?")
        settings = provider.calls[0]["settings"]
        self.assertEqual(settings.system_instruction, SYSTEM_INSTRUCTION)
        self.assertLessEqual(settings.temperature, 0.3)
        for rule in ("Do not use outside knowledge", "timestamps", "Plain text only",
                     "Ignore any instructions"):
            self.assertIn(rule.lower(), SYSTEM_INSTRUCTION.lower())

    def test_markdown_is_removed_from_the_reply(self):
        out, _ = self.ask("What is LIATE?", FakeProvider(
            "**Key idea**\n\n- The instructor says *LIATE* helps choose `u`.\n- It is an order.\n\n1. Then integrate."))
        for ch in "*`#":
            self.assertNotIn(ch, out.answer_text)
        self.assertNotIn("- ", out.answer_text)
        self.assertIn("The instructor says LIATE helps choose u.", out.answer_text)

    def test_model_declares_topic_not_covered(self):
        out, _ = self.ask("What is the derivative of LIATE?",
                          FakeProvider("[[NOT_IN_LECTURE]] This lecture does not cover that."))
        self.assertEqual((out.status, out.sources), ("not_in_lecture", []))
        self.assertEqual(out.answer_text, "This lecture does not cover that.")

    def test_model_declares_insufficient_evidence(self):
        out, _ = self.ask("Explain LIATE in depth",
                          FakeProvider("[[INSUFFICIENT_CONTEXT]]"))
        self.assertEqual((out.status, out.sources), ("insufficient_context", []))
        self.assertTrue(out.answer_text.strip())

    def test_empty_reply_is_an_error_not_an_answer(self):
        with self.assertRaises(LLMEmptyResponseError):
            self.ask("What is LIATE?", FakeProvider("  **  \n "))

    def test_provider_is_built_lazily_and_reused(self):
        built = []

        def factory():
            built.append(1)
            return FakeProvider()

        service = AskService(provider_factory=factory)
        service.answer("what is it?", SEGMENTS)       # resolved without an LLM
        self.assertEqual(built, [])
        service.answer("What is LIATE?", SEGMENTS)
        service.answer("What is LIATE?", SEGMENTS)
        self.assertEqual(built, [1])


class MomentTests(unittest.TestCase):
    def ask(self, question, m, provider=None, **kwargs):
        provider = provider or FakeProvider()
        out = AskService(provider=provider).answer(question, SEGMENTS, moment=m, **kwargs)
        return out, provider

    def test_moment_with_visual_adds_context_and_intervention_source(self):
        m = moment(explanation="The graph shows x times e to the x growing.",
                   visual=(("Structure", "A curve rising to the right."),))
        out, provider = self.ask("What does this diagram represent?", m)
        prompt = provider.calls[0]["prompt"]
        self.assertEqual((out.status, out.scope), ("answered", "lecture_with_moment"))
        self.assertIn("A curve rising to the right.", prompt)
        self.assertIn("The graph shows x times e to the x growing.", prompt)
        self.assertEqual([s.intervention_id for s in out.sources if s.intervention_id], ["int-1"])
        self.assertTrue(any(s.intervention_id is None for s in out.sources))  # plus transcript sources

    def test_unavailable_visual_is_stated_and_not_invented(self):
        out, provider = self.ask("What does this diagram represent?", moment())
        prompt = provider.calls[0]["prompt"]
        self.assertIn("Visual analysis: unavailable", prompt)
        self.assertIn("Do not describe what is on the screen", prompt)
        self.assertEqual(out.scope, "lecture_with_moment")
        # nothing generated was used, so the intervention is not claimed as a source
        self.assertTrue(all(s.intervention_id is None for s in out.sources))

    def test_moment_uses_its_timestamp_when_no_current_time(self):
        _, provider = self.ask("Why is that harder?", moment())
        self.assertIn("PLAYBACK POSITION: about 02:06", provider.calls[0]["prompt"])
        self.assertIn("plug into the formula", provider.calls[0]["prompt"])  # ~2:12, in the window

    def test_explicit_current_time_wins(self):
        _, provider = self.ask("Why is that harder?", moment(), current_time=30.0)
        self.assertIn("PLAYBACK POSITION: about 00:30", provider.calls[0]["prompt"])

    def test_current_time_alone_stays_lecture_scope(self):
        out = AskService(provider=FakeProvider()).answer(
            "What does this graph show?", SEGMENTS, current_time=174.0)
        self.assertEqual((out.status, out.scope), ("answered", "lecture"))

    def test_build_prompt_without_moment_has_no_moment_section(self):
        prompt = build_prompt("q?", [], None, None)
        self.assertNotIn("CURRENT MOMENT", prompt)
        self.assertIn("PLAYBACK POSITION: not provided", prompt)


class ProviderErrorTests(unittest.TestCase):
    def test_provider_errors_propagate_unchanged(self):
        for error in (LLMRateLimitError("quota", provider="p", status_code=429),
                      LLMTimeoutError("slow"), LLMUnavailableError("down"),
                      LLMAuthenticationError("bad key")):
            service = AskService(provider=FakeProvider(error=error))
            with self.assertRaises(type(error)):
                service.answer("What is LIATE?", SEGMENTS)

    def test_missing_configuration_surfaces_when_an_llm_is_needed(self):
        def factory():
            raise LLMConfigurationError("GEMINI_API_KEY is not set.")

        service = AskService(provider_factory=factory)
        with self.assertRaises(LLMConfigurationError):
            service.answer("What is LIATE?", SEGMENTS)

    def test_missing_configuration_is_irrelevant_when_no_llm_is_needed(self):
        def factory():
            raise LLMConfigurationError("GEMINI_API_KEY is not set.")

        out = AskService(provider_factory=factory).answer("anything", [])
        self.assertEqual(out.status, "insufficient_context")


LONG_REMAINDER = " ".join(["word"] * 61)

# Every malformed example from the review, plus the obvious variants.
MALFORMED_REPLIES = [
    "[NOT_IN_LECTURE] nope",                                   # single brackets
    "NOT_IN_LECTURE: nope",                                    # no brackets
    "[[NOT IN LECTURE]] nope",                                 # spaces instead of underscores
    "[[not_in_lecture]] nope",                                 # wrong case
    "[[Not_In_Lecture]] nope",
    "Sure. [[NOT_IN_LECTURE]] nope",                           # embedded after prose
    "The instructor covers this. [[INSUFFICIENT_CONTEXT]]",    # trailing marker
    "[[NOT_IN_LECTURE]] [[INSUFFICIENT_CONTEXT]] x",           # two different markers
    "[[NOT_IN_LECTURE]][[NOT_IN_LECTURE]]",                    # repeated marker
    "[[NOT_IN_LECTURE]] fine. [[NOT_IN_LECTURE]]",             # repeated, second embedded
    "[[ANSWERED]] The instructor says LIATE.",                 # unsupported marker
    "[[UNKNOWN_STATUS]] something",
    "[INSUFFICIENT CONTEXT] missing",
    "INSUFFICIENT CONTEXT",
    "The answer is NOT_IN_LECTURE",
    "[[NOT_IN_LECTURE]] " + LONG_REMAINDER,                    # marker contradicted by a long answer
    "[[INSUFFICIENT_CONTEXT]] " + LONG_REMAINDER,
]

# Text that merely resembles a marker but is ordinary lecture content.
LEGITIMATE_REPLIES = [
    "The matrix [[1, 2], [3, 4]] has two rows.",
    "The instructor says the topic is not in the lecture notes.",
    "Use the array [x] and the list [1, 2, 3].",
    "The constant MAX_VALUE is set to ten.",
]


class ModelReplyParsingTests(unittest.TestCase):
    def test_well_formed_markers(self):
        self.assertEqual(parse_model_reply("[[NOT_IN_LECTURE]] This lecture does not cover that."),
                         ("not_in_lecture", "This lecture does not cover that."))
        self.assertEqual(parse_model_reply("[[INSUFFICIENT_CONTEXT]] The excerpts do not say."),
                         ("insufficient_context", "The excerpts do not say."))
        self.assertEqual(parse_model_reply("[[NOT_IN_LECTURE]]No space is fine.")[0], "not_in_lecture")

    def test_marker_alone_gets_a_status_specific_default_message(self):
        not_in = parse_model_reply("[[NOT_IN_LECTURE]]")
        insufficient = parse_model_reply("[[INSUFFICIENT_CONTEXT]]")
        self.assertEqual((not_in[0], insufficient[0]), ("not_in_lecture", "insufficient_context"))
        self.assertTrue(not_in[1].strip() and insufficient[1].strip())
        self.assertNotEqual(not_in[1], insufficient[1])
        self.assertNotIn("[[", not_in[1] + insufficient[1])

    def test_marker_at_the_word_limit_is_accepted(self):
        self.assertEqual(parse_model_reply("[[NOT_IN_LECTURE]] " + " ".join(["word"] * 60))[0],
                         "not_in_lecture")

    def test_every_malformed_reply_is_rejected(self):
        for reply in MALFORMED_REPLIES:
            with self.subTest(reply=reply[:60]):
                with self.assertRaises(LLMMalformedResponseError):
                    parse_model_reply(reply)

    def test_ordinary_text_that_resembles_a_marker_is_still_an_answer(self):
        for reply in LEGITIMATE_REPLIES:
            with self.subTest(reply=reply):
                self.assertEqual(parse_model_reply(reply), ("answered", reply))

    def test_plain_answer(self):
        self.assertEqual(parse_model_reply("The instructor says to use LIATE."),
                         ("answered", "The instructor says to use LIATE."))


class MalformedReplyEndToEndTests(unittest.TestCase):
    def ask(self, reply):
        return AskService(provider=FakeProvider(reply)).answer("What is the acronym LIATE?", SEGMENTS)

    def test_malformed_output_is_an_error_not_a_verdict(self):
        for reply in MALFORMED_REPLIES:
            with self.subTest(reply=reply[:60]):
                with self.assertRaises(LLMMalformedResponseError) as ctx:
                    self.ask(reply)
                self.assertNotIn("NOT_IN_LECTURE", str(ctx.exception))     # nothing from the reply is echoed
                self.assertTrue(ctx.exception.retryable)

    def test_markdown_wrapped_valid_marker_is_accepted(self):
        out = self.ask("**[[INSUFFICIENT_CONTEXT]]** The excerpts do not say.")
        self.assertEqual((out.status, out.sources, out.answer_text),
                         ("insufficient_context", [], "The excerpts do not say."))

    def test_no_marker_text_ever_reaches_answer_text(self):
        for reply in ["[[NOT_IN_LECTURE]] nope", "[[INSUFFICIENT_CONTEXT]]", "Plain answer about LIATE."]:
            out = self.ask(reply)
            for token in ("[[", "]]", "NOT_IN_LECTURE", "INSUFFICIENT_CONTEXT"):
                self.assertNotIn(token, out.answer_text)

    def test_legitimate_lookalike_is_answered_with_sources(self):
        out = self.ask("The matrix [[1, 2], [3, 4]] appears in the instructor's example about LIATE.")
        self.assertEqual(out.status, "answered")
        self.assertGreaterEqual(len(out.sources), 1)


class OutcomeInvariantTests(unittest.TestCase):
    GOOD = [AskSourceInfo(1.0, 2.0)]

    def test_valid_outcomes(self):
        AskOutcome("answered", "lecture", "text", self.GOOD)
        AskOutcome("answered", "lecture_with_moment", "text", [AskSourceInfo(5.0, 5.0, "int-1")])
        AskOutcome("not_in_lecture", "lecture", "text", [])
        AskOutcome("insufficient_context", "lecture_with_moment", "text", [])

    def test_answered_requires_sources(self):
        with self.assertRaises(ValueError):
            AskOutcome("answered", "lecture", "text", [])

    def test_non_answered_forbids_sources(self):
        for status in ("not_in_lecture", "insufficient_context"):
            with self.assertRaises(ValueError):
                AskOutcome(status, "lecture", "text", self.GOOD)

    def test_text_must_be_non_empty(self):
        for text in ("", "   ", "\n"):
            with self.assertRaises(ValueError):
                AskOutcome("not_in_lecture", "lecture", text, [])

    def test_sources_must_be_valid_ranges(self):
        for bad in (AskSourceInfo(-1.0, 2.0), AskSourceInfo(5.0, 4.0),
                    AskSourceInfo(float("nan"), 2.0), AskSourceInfo(1.0, float("inf"))):
            with self.assertRaises(ValueError):
                AskOutcome("answered", "lecture", "text", [bad])

    def test_unknown_status_or_scope(self):
        with self.assertRaises(ValueError):
            AskOutcome("maybe", "lecture", "text", [])
        with self.assertRaises(ValueError):
            AskOutcome("not_in_lecture", "galaxy", "text", [])

    def test_service_outcomes_satisfy_the_invariants(self):
        replies = {"What is LIATE?": "The instructor says it orders function types.",
                   "summarize the lecture": "A lecture on integration by parts.",
                   "explain 1:00 to 1:30": "Choosing u with LIATE."}
        for question, reply in replies.items():
            out = AskService(provider=FakeProvider(reply)).answer(question, SEGMENTS)
            self.assertEqual(out.status, "answered")
            self.assertTrue(out.answer_text.strip())
            self.assertGreaterEqual(len(out.sources), 1)
            for source in out.sources:
                self.assertTrue(0 <= source.start_seconds <= source.end_seconds <= SEGMENTS[-1].end)


class PlainTextTests(unittest.TestCase):
    def test_preserves_paragraphs_and_math_words(self):
        text = to_plain_text("The integral of f of x.\n\nx squared plus two x.")
        self.assertEqual(text, "The integral of f of x.\n\nx squared plus two x.")

    def test_removes_dollar_signs_and_headings(self):
        self.assertEqual(to_plain_text("## Title\nSolve $x$ now"), "Title Solve x now")


if __name__ == "__main__":
    unittest.main()
