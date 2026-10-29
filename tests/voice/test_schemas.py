"""Module 1: the request/response contract shared with the frontend."""

import os
import sys

import pytest
from pydantic import ValidationError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from voice.schemas import (  # noqa: E402
    ActionCandidate,
    ContextItem,
    InterpretRequest,
    InterpretResponse,
    TranscribeResponse,
    VoiceContext,
)


def context(**overrides):
    base = {"screen": "explanation"}
    base.update(overrides)
    return VoiceContext(**base)


def item(ref="m1", position=1, **overrides):
    base = {"ref": ref, "kind": "moment", "label": "Fourier integral", "position": position}
    base.update(overrides)
    return ContextItem(**base)


# ── VoiceContext ─────────────────────────────────────────────────────────────

def test_minimal_context_uses_safe_defaults():
    ctx = context()
    assert ctx.playing == "unknown"
    assert ctx.items == [] and ctx.available_actions == []
    assert ctx.pending_confirmation is None


def test_full_context_is_accepted():
    ctx = context(
        video_id="aircAruvnKk", current_moment_ref="m1", playing="lecture",
        presentation_mode="study", lecture_seconds=187.5,
        items=[item("m1", 1), item("m2", 2)],
        available_actions=["pause_lecture", "next_moment"],
        pending_confirmation="remove_item", last_action="next_moment", language_hint="en",
    )
    assert [i.ref for i in ctx.items] == ["m1", "m2"]


@pytest.mark.parametrize("bad", [
    {"screen": "settings"},
    {"video_id": "short"},
    {"playing": "music"},
    {"lecture_seconds": -1},
    {"lecture_seconds": float("inf")},
    {"available_actions": ["Pause Lecture"]},
    {"pending_confirmation": "not an id"},
    {"language_hint": "english"},
    {"presentation_mode": "fancy"},
])
def test_invalid_context_is_rejected(bad):
    payload = {"screen": "home"}
    payload.update(bad)
    with pytest.raises(ValidationError):
        VoiceContext(**payload)


def test_duplicate_item_refs_are_rejected():
    with pytest.raises(ValidationError):
        context(items=[item("m1", 1), item("m1", 2)])


def test_too_many_items_are_rejected():
    with pytest.raises(ValidationError):
        context(items=[item(f"m{i}", i) for i in range(1, 62)])


def test_item_label_and_position_limits():
    with pytest.raises(ValidationError):
        item(label="x" * 161)
    with pytest.raises(ValidationError):
        item(position=0)
    with pytest.raises(ValidationError):
        item(label="")


# ── InterpretRequest ─────────────────────────────────────────────────────────

def test_request_text_is_stripped():
    req = InterpretRequest(text="  open bookmarks \n", context=context())
    assert req.text == "open bookmarks"


@pytest.mark.parametrize("text", ["", "   ", "x" * 301])
def test_request_text_length_is_enforced(text):
    with pytest.raises(ValidationError):
        InterpretRequest(text=text, context=context())


def test_request_language_must_look_like_a_code():
    assert InterpretRequest(text="hi", language="pt-BR", context=context()).language == "pt-BR"
    with pytest.raises(ValidationError):
        InterpretRequest(text="hi", language="not a language", context=context())


# ── InterpretResponse invariants ─────────────────────────────────────────────

def response(**fields):
    base = {"decision": "unsupported", "message_key": "not_understood"}
    base.update(fields)
    return InterpretResponse(**base)


def test_execute_requires_an_action():
    with pytest.raises(ValidationError):
        response(decision="execute")
    assert response(decision="execute", action="next_moment", confidence=0.9).action == "next_moment"


def test_confirm_requires_an_action():
    with pytest.raises(ValidationError):
        response(decision="confirm")
    assert response(decision="confirm", action="remove_item", args={"index": 2}).args == {"index": 2}


@pytest.mark.parametrize("decision", ["unsupported", "no_speech"])
def test_decisions_without_an_action_reject_one(decision):
    assert response(decision=decision).action is None
    with pytest.raises(ValidationError):
        response(decision=decision, action="next_moment")


def test_clarify_needs_two_candidates():
    one = [ActionCandidate(action="pause_lecture", score=0.5)]
    two = one + [ActionCandidate(action="pause_explanation", score=0.48)]
    with pytest.raises(ValidationError):
        response(decision="clarify", candidates=one)
    assert len(response(decision="clarify", candidates=two).candidates) == 2


def test_ask_question_needs_the_question_text():
    ok = response(decision="ask_question", action="ask_question", args={"question": "why is it negative"})
    assert ok.args["question"] == "why is it negative"
    with pytest.raises(ValidationError):
        response(decision="ask_question", action="ask_question", args={})
    with pytest.raises(ValidationError):
        response(decision="ask_question", action="next_moment", args={"question": "x"})
    with pytest.raises(ValidationError):
        response(decision="ask_question", action="ask_question", args={"question": "   "})


def test_at_most_three_candidates():
    many = [ActionCandidate(action="next_moment", score=0.5) for _ in range(4)]
    with pytest.raises(ValidationError):
        response(decision="clarify", candidates=many)


def test_scores_and_confidence_are_bounded():
    with pytest.raises(ValidationError):
        ActionCandidate(action="next_moment", score=1.2)
    with pytest.raises(ValidationError):
        response(confidence=-0.1)


def test_invalid_action_id_is_rejected():
    with pytest.raises(ValidationError):
        response(decision="execute", action="Next Moment")


# ── TranscribeResponse ───────────────────────────────────────────────────────

def test_transcribe_response_defaults_and_bounds():
    ok = TranscribeResponse(text="open bookmarks", language="en", language_probability=0.98, duration_seconds=1.6)
    assert ok.no_speech is False
    with pytest.raises(ValidationError):
        TranscribeResponse(text="", language_probability=1.5, duration_seconds=1)
    with pytest.raises(ValidationError):
        TranscribeResponse(text="", duration_seconds=-1)
