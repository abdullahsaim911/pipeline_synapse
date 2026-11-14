import numpy as np
import pytest
from voice.explanation_answers import exact_answer
from voice.interpret import Interpreter
from voice.schemas import InterpretRequest


@pytest.mark.parametrize("text,choices,expected", [
    ("Go ahead", ["listen", "skip"], "listen"),
    ("Not now", ["listen", "skip"], "skip"),
    ("I understand", ["yes", "no", "replay"], "yes"),
    ("I don't understand", ["yes", "no", "replay"], "no"),
    ("I understand but play it again", ["yes", "no", "replay"], "replay"),
    ("No, actually explain more", ["brief", "detailed", "resume"], "detailed"),
    ("Just the main point", ["brief", "detailed", "resume"], "brief"),
    ("Don't resume", ["wait", "resume"], "wait"),
    ("Keep going while it prepares", ["wait", "resume"], "resume"),
    ("Yes", ["brief", "detailed", "resume"], None),
    ("Maybe", ["listen", "skip"], None),
    ("Brief or detailed", ["brief", "detailed", "resume"], None),
    ("Do not play it", ["listen", "skip"], None),
])
def test_contextual_answer(text, choices, expected):
    interpreter = Interpreter(embedder=lambda texts: np.zeros((len(texts), 2), dtype=np.float32))
    result = interpreter.interpret(InterpretRequest(text=text, language="en", context={
        "screen": "explanation", "available_actions": ["answer_explanation"],
        "explanation_question": {"id": "question-1", "choices": choices},
    }))
    if expected:
        assert result.action == "answer_explanation"
        assert result.args == {"choice": expected}
    else:
        assert result.decision == "unsupported"


def test_answer_is_unavailable_without_question():
    interpreter = Interpreter(embedder=lambda texts: np.zeros((len(texts), 2), dtype=np.float32))
    result = interpreter.interpret(InterpretRequest(text="yes", language="en", context={"screen": "explanation"}))
    assert result.action != "answer_explanation"


def test_non_english_is_not_executed():
    interpreter = Interpreter(embedder=lambda texts: np.zeros((len(texts), 2), dtype=np.float32))
    result = interpreter.interpret(InterpretRequest(text="oui", language="fr", context={"screen": "explanation"}))
    assert result.message_key == "english_only"


@pytest.mark.parametrize("text,expected", [
    ("I do not understand this, explain it", {}),
    ("give me the standard version", {"depth": "standard"}),
    ("give me the detailed version", {"depth": "detailed"}),
    ("No, actually give me the brief version", {"depth": "brief"}),
])
def test_standalone_requests_are_available_during_lecture_playback(text, expected):
    interpreter = Interpreter(embedder=lambda texts: np.zeros((len(texts), 2), dtype=np.float32))
    result = interpreter.interpret(InterpretRequest(text=text, language="en", context={
        "screen": "explanation", "playing": "lecture", "available_actions": ["request_explanation"]}))
    assert result.action == "request_explanation"
    assert result.args == expected


@pytest.mark.parametrize("text,choices,expected", [
    ("listen later", ["wait", "resume"], "resume"),
    ("listen later", ["listen", "later"], "later"),
    ("no", ["listen", "later"], "later"),
    ("process", ["process", "later"], "process"),
])
def test_completion_and_saved_view_answers(text, choices, expected):
    assert exact_answer(text, choices)[0] == expected


@pytest.mark.parametrize("text", ["do not explain this", "brief or detailed"])
def test_uncertain_standalone_requests_do_not_execute(text):
    interpreter = Interpreter(embedder=lambda texts: np.zeros((len(texts), 2), dtype=np.float32))
    result = interpreter.interpret(InterpretRequest(text=text, language="en", context={
        "screen": "explanation", "available_actions": ["request_explanation"]}))
    assert result.decision == "unsupported"
