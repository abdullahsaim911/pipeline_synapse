"""Modules 3, 4, 6: speech recognition, intent interpretation and the voice routes (fakes; no models needed)."""

import io
import os
import sys
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from voice import routes  # noqa: E402
from voice.catalog import ActionCatalog  # noqa: E402
from voice.interpret import Interpreter, extract_args  # noqa: E402
from voice.schemas import InterpretRequest, VoiceContext  # noqa: E402
from voice.stt import MAX_BYTES, AudioError, Recognizer, decode_wav  # noqa: E402

CATALOG = ActionCatalog.load()


def wav_bytes(samples, rate=16000, channels=1, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


TONE = np.sin(np.linspace(0, 400, 16000)) * 0.4


class FakeModel:
    def __init__(self, segments, language="en"):
        self.segments, self.language, self.calls = segments, language, 0

    def transcribe(self, samples, **kwargs):
        self.calls += 1
        info = type("Info", (), {"language": self.language, "language_probability": 0.97})()
        return iter(self.segments), info


def seg(text, p=0.1):
    return type("Seg", (), {"text": f" {text} ", "no_speech_prob": p})()


# ── Speech recognition ───────────────────────────────────────────────────────

def test_decode_rejects_bad_audio():
    with pytest.raises(AudioError):
        decode_wav(b"not a wav")
    with pytest.raises(AudioError):
        decode_wav(wav_bytes(TONE, rate=8000))
    with pytest.raises(AudioError):
        decode_wav(b"x" * (MAX_BYTES + 1))
    with pytest.raises(AudioError):
        decode_wav(wav_bytes(np.zeros(16000 * 31)))


def test_silence_never_reaches_the_model():
    model = FakeModel([seg("Thank you.")])
    rec = Recognizer(loader=lambda: model)
    out = rec.transcribe(wav_bytes(np.zeros(16000)))
    assert out.no_speech and out.text == "" and model.calls == 0 and not rec.loaded


def test_transcribes_and_drops_non_speech_segments():
    rec = Recognizer(loader=lambda: FakeModel([seg("open my bookmarks"), seg("Thank you.", p=0.9)]))
    out = rec.transcribe(wav_bytes(TONE))
    assert out.text == "open my bookmarks" and out.language == "en" and not out.no_speech


def test_model_loads_lazily_once_and_unloads_when_idle():
    now = [0.0]
    loads = []
    rec = Recognizer(loader=lambda: loads.append(1) or FakeModel([seg("hello")]), clock=lambda: now[0])
    rec.transcribe(wav_bytes(TONE))
    rec.transcribe(wav_bytes(TONE))
    assert len(loads) == 1 and rec.loaded
    now[0] = 10_000
    assert rec.unload_if_idle() and not rec.loaded


def test_bad_language_code_is_rejected():
    with pytest.raises(AudioError):
        Recognizer(loader=lambda: FakeModel([])).transcribe(wav_bytes(TONE), "english!")


# ── Interpretation ───────────────────────────────────────────────────────────

class FakeEmbedder:
    """Maps each action's examples to one axis, and a test sentence to the axis named by 'axis:<action>'."""

    def __init__(self, sentence_axis=None):
        self.axes = {a: i for i, a in enumerate(CATALOG.ids())}
        self.lookup = {s: a for lang in CATALOG.languages for a, items in CATALOG.examples(lang).items() for s in items}
        self.sentence_axis = sentence_axis or {}

    def __call__(self, texts):
        out = np.zeros((len(texts), len(self.axes)), dtype=np.float32)
        for i, t in enumerate(texts):
            action = self.lookup.get(t) or self.sentence_axis.get(t)
            if action:
                out[i, self.axes[action]] = 1.0
            else:
                out[i, :] = 0.01
        return out


def ask(text, action=None, **ctx):
    interp = Interpreter(CATALOG, embedder=FakeEmbedder({text: action} if action else {}))
    ctx.setdefault("screen", "explanation")
    ctx.setdefault("video_id", "aircAruvnKk")
    return interp.interpret(InterpretRequest(text=text, context=VoiceContext(**ctx)))


def test_clear_match_executes_with_arguments():
    res = ask("skip ahead a minute please", "seek_lecture")
    assert (res.decision, res.action, res.args) == ("execute", "seek_lecture", {"direction": "forward", "seconds": 60})


def test_unrelated_sentence_is_refused_not_guessed():
    res = ask("blah blah")
    assert res.decision == "unsupported" and res.action is None


def test_risky_action_always_asks_first():
    res = ask("delete the second one", "remove_item", screen="bookmarks", video_id=None)
    assert (res.decision, res.action, res.args) == ("confirm", "remove_item", {"index": 2})


def test_missing_required_value_is_reported_not_run():
    res = ask("open it", "open_item", screen="bookmarks", video_id=None)
    assert res.decision == "unsupported" and res.message_key == "missing_index"


def test_question_is_routed_to_qa():
    res = ask("why is the curve flat", "ask_question")
    assert (res.decision, res.action, res.args) == ("ask_question", "ask_question", {"question": "why is the curve flat"})


def test_questions_are_recognised_on_any_screen():
    assert ask("why is it flat", "ask_question", video_id=None).decision == "ask_question"


def test_how_do_i_question_is_answered_from_the_catalog():
    res = ask("how do I bookmark this", "app_question", screen="home")
    assert (res.decision, res.action) == ("execute", "app_question")


def test_actions_not_on_this_screen_are_never_chosen():
    assert ask("play the video", "play_lecture", screen="library", video_id=None).decision == "unsupported"


def test_what_is_playing_decides_between_lecture_and_explanation():
    assert ask("pause", "pause_explanation", playing="lecture").decision == "unsupported"
    assert ask("pause", "pause_explanation", playing="explanation").action == "pause_explanation"


def test_yes_only_counts_while_a_confirmation_is_pending():
    assert ask("yes", "confirm_yes").decision == "unsupported"
    assert ask("yes", "confirm_yes", pending_confirmation="remove_item").action == "confirm_yes"


@pytest.mark.parametrize("action,text,expected", [
    ("open_item", "open the second one", {"index": 2}),
    ("open_item", "number 12", {"index": 12}),
    ("seek_lecture", "go back thirty five seconds", {"direction": "backward", "seconds": 35}),
    ("seek_lecture", "rewind two minutes", {"direction": "backward", "seconds": 120}),
    ("change_text_size", "the letters are too big, make them smaller", {"change": "smaller"}),
    ("change_listening_speed", "the voice is too quick, slow it down", {"direction": "slower"}),
    ("set_theme", "turn on dark mode", {"theme": "night"}),
    ("search_lectures", "find lectures on quantum mechanics", {"query": "quantum mechanics"}),
    ("explore_subject", "open the biology lectures", {"subject": "Biology"}),
    ("set_presentation_mode", "hide the video", {"mode": "accessibility"}),
])
def test_argument_values(action, text, expected):
    assert extract_args(action, text, CATALOG) == expected


# ── Routes ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    rec = Recognizer(loader=lambda: FakeModel([seg("open my bookmarks")]))
    interp = Interpreter(CATALOG, embedder=FakeEmbedder({"open my bookmarks": "navigate_bookmarks"}))
    spoken = []

    class Tts:
        def generate(self, text, path):
            spoken.append(text)
            open(path, "wb").write(b"ID3fake")

    app.dependency_overrides[routes.get_recognizer] = lambda: rec
    app.dependency_overrides[routes.get_interpreter] = lambda: interp
    app.dependency_overrides[routes.get_tts] = lambda: Tts()
    c = TestClient(app)
    c.spoken = spoken
    return c


def test_transcribe_route(client):
    r = client.post("/voice/transcribe", files={"file": ("a.wav", wav_bytes(TONE), "audio/wav")})
    assert r.status_code == 200 and r.json()["text"] == "open my bookmarks"
    assert client.post("/voice/transcribe", files={"file": ("a.wav", b"junk", "audio/wav")}).status_code == 422
    big = client.post("/voice/transcribe", files={"file": ("a.wav", b"x" * (MAX_BYTES + 10), "audio/wav")})
    assert big.status_code == 413


def test_interpret_route(client):
    body = {"text": "open my bookmarks", "context": {"screen": "home"}}
    r = client.post("/voice/interpret", json=body)
    assert r.status_code == 200 and r.json()["action"] == "navigate_bookmarks" and r.json()["decision"] == "execute"
    body["context"]["available_actions"] = ["fly_away"]
    assert client.post("/voice/interpret", json=body).status_code == 422
    assert client.post("/voice/interpret", json={"text": "", "context": {"screen": "home"}}).status_code == 422


def test_speak_route_caches_repeated_phrases(client):
    first = client.post("/voice/speak", json={"text": "Opening your bookmarks."})
    again = client.post("/voice/speak", json={"text": "Opening   your bookmarks."})
    assert first.status_code == again.status_code == 200 and first.headers["content-type"] == "audio/mpeg"
    assert client.spoken.count("Opening your bookmarks.") <= 1
    assert client.post("/voice/speak", json={"text": "x" * 401}).status_code == 422


def test_foreign_browser_origin_is_refused(client):
    r = client.post("/voice/interpret", json={"text": "hi", "context": {"screen": "home"}}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_quiz_answer_letters_and_numbers():
    from voice.catalog import ActionCatalog
    from voice.interpret import extract_args
    cat = ActionCatalog.load()
    for text, want in [("option two", 2), ("answer B", 2), ("I choose answer c", 3), ("pick the first one", 1), ("go with option 4", 4)]:
        assert extract_args("quiz_answer", text, cat) == {"index": want}, text
    assert extract_args("quiz_answer", "I have a question", cat) == {}
