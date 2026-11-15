"""
Intent interpretation (module 4): what did the learner mean, in the context of the screen they are on?

English commands are compared by meaning with catalog examples using the existing embedding model.
An active explanation question first checks bounded answers and negation; uncertain answers are clarified.
Other deterministic parsing extracts argument values after choosing an action.
"""

import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from voice.catalog import ActionCatalog, CatalogError
from voice.explanation_answers import EXAMPLES, exact_answer, explanation_request
from voice.schemas import ActionCandidate, InterpretRequest, InterpretResponse, VoiceContext

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Calibrated on the held-out set (python -m voice.evaluate). Below CONFIRM_MIN nothing is guessed.
EXECUTE_MIN = 0.62
CONFIRM_MIN = 0.45
MARGIN = 0.05
MAX_SCORE_CACHE = 256

# A "how do I ...?" question about the app has this shape; a plain command ("pause the video", "export to PDF")
# does not. Embeddings alone confuse the two because both talk about the same topic, so this only GATES the
# app_question intent; it never picks an action.
_APP_QUESTION_FORM = re.compile(
    r"\b(how (do|can|would|should|could) i|how does|is there a way|is it possible|"
    r"what (do|should|can) i (say|ask)|what is the command|tell me how|does (this|the) app|do you support)\b")
_VOICE_LAYER = {"stop_speaking", "repeat_last_response", "cancel", "help", "app_question", "where_am_i"}
_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_NUMBERS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90})
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10}
# Embeddings put "next" and "previous" almost on top of each other, so for these pairs the direction WORD in the
# sentence settles which one was meant (the same way a number settles "the second one").
_PAIRS = {("next_moment", "previous_moment"), ("quiz_next", "quiz_previous"), ("next_saved_page", "previous_saved_page")}
_BACK_WORDS = {"previous", "back", "before", "earlier", "prior", "preceding", "last", "backward", "backwards", "behind"}
_FORWARD_WORDS = {"next", "following", "forward", "after", "ahead", "later", "onward", "upcoming"}
_LETTERS = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
_LETTER_LEADS = {"answer", "option", "choice", "letter", "choose", "pick", "select", "is"}
_ENUM_WORDS = {
    ("request_explanation", "depth"): {"brief": ["brief", "shorter", "short"],
        "standard": ["standard", "normal"], "detailed": ["detailed", "detail", "fuller"]},
    ("set_saved_view", "view"): {"recent": ["recent", "recently", "viewed"], "lectures": ["lecture", "lectures"]},
    ("set_theme", "theme"): {"night": ["night", "knight", "nite", "dark", "black"], "sepia": ["sepia", "warm"], "paper": ["paper", "light"]},
    ("set_presentation_mode", "mode"): {"accessibility": ["accessibility", "hide"], "study": ["study", "show"]},
    ("quiz_start", "mode"): {"assessment": ["assessment", "assess", "test", "exam", "graded"],
                             "practice": ["practice", "practise", "practising", "practicing"]},
    ("change_text_size", "change"): {"larger": ["bigger", "larger", "increase", "enlarge", "big", "large"],
                                     "smaller": ["smaller", "reduce", "shrink", "decrease", "small"]},
    ("change_listening_speed", "direction"): {"faster": ["faster", "quicker", "quick", "speed", "fast", "increase"],
                                              "slower": ["slower", "slow", "decrease"]},
    ("filter_moments", "category"): {
        "graph": ["graph", "graphs", "plot", "plots", "curve", "curves"],
        "equation": ["equation", "equations", "formula", "formulas", "formulae", "maths", "math"],
        "chart": ["chart", "charts"], "diagram": ["diagram", "diagrams", "slide", "slides", "figure", "figures"],
        "code": ["code", "program", "programs", "snippet", "snippets"], "table": ["table", "tables"],
        "all": ["all", "everything", "every"]},
    ("set_visual_tab", "tab"): {"visual": ["visual", "picture", "image", "diagram"],
                                "structured": ["structured", "text", "description", "list"]},
    ("explore_subject", "subject"): {"Mathematics": ["mathematics", "math", "maths"], "Physics": ["physics"],
                                     "Computer Science": ["computer", "computing"], "Chemistry": ["chemistry"],
                                     "Biology": ["biology"]},
}
_QUERY_LEAD = re.compile(
    r"^(please\s+|can you\s+|could you\s+)?(search for|search|find me|find|look up|look for|i'?m looking for|"
    r"i want to learn about|do you have)\s+(a |an |the |some |any )?(lectures?\s+|videos?\s+)?(about |on |for |of )?",
    re.I)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


def _numbers(tokens: List[str]) -> List[int]:
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            out.append(int(t))
        elif t in _NUMBERS:
            value = _NUMBERS[t]
            if value >= 20 and i + 1 < len(tokens) and 0 < _NUMBERS.get(tokens[i + 1], -1) < 10:
                value += _NUMBERS[tokens[i + 1]]
                i += 1
            out.append(value)
        i += 1
    return out


def _last_enum(text_tokens: List[str], options: Dict[str, List[str]]) -> Optional[str]:
    """The option whose word appears LAST ("too big, make them smaller" means smaller)."""
    best, best_at = None, -1
    for value, words in options.items():
        for word in words:
            for at, token in enumerate(text_tokens):
                if token == word and at > best_at:
                    best, best_at = value, at
    return best


_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")  # the account service's rule (api/auth.py)
_USERNAME_LEAD = re.compile(
    r"^.*?\b(?:user\s*name|login(?:\s+name)?|my\s+name)\b(?:\s*,)?\s*(?:is|it\s+is|it's|should\s+be|will\s+be|to|as)?\s*", re.I)
_USERNAME_VERB_LEAD = re.compile(r"^(?:please\s+)?(?:type|enter|use|put|set|sign\s+in\s+as|log\s+in\s+as|it's|it\s+is)\s+", re.I)
_USERNAME_SYMBOLS = {"underscore": "_", "dot": ".", "period": ".", "point": ".", "dash": "-", "hyphen": "-", "minus": "-"}


def _spoken_username(text: str) -> Optional[str]:
    """
    The username in "my username is m a h a 2 2" -> "maha22". Spelled letters and digits are joined, spoken
    symbols become characters ("learner underscore one" -> "learner_one"), number words become digits, and the
    whole is joined without spaces. Returns None unless the result is a valid username, so a misheard name is
    asked for again rather than typed.
    """
    rest = _USERNAME_LEAD.sub("", text.strip(), count=1) if _USERNAME_LEAD.search(text) else _USERNAME_VERB_LEAD.sub("", text.strip())
    rest = rest.strip(" .?!,'\"")
    parts = []
    for token in re.findall(r"[A-Za-z]+|\d+", rest):
        low = token.lower()
        if low in _USERNAME_SYMBOLS:
            parts.append(_USERNAME_SYMBOLS[low])
        elif low in _NUMBERS and low not in ("a",):
            parts.append(str(_NUMBERS[low]))
        elif low in ("capital", "uppercase", "lowercase", "letter", "the", "please", "thanks"):
            continue
        else:
            parts.append(low)
    name = "".join(parts)
    return name if _USERNAME_RE.match(name) else None


def extract_args(action_id: str, text: str, catalog: ActionCatalog) -> dict:
    """Pull argument VALUES out of the sentence. Anything not found is left out (and reported as missing)."""
    params = catalog.get(action_id).params
    tokens = _tokens(text)
    args: dict = {}
    if action_id == "open_note_explanation":
        # Two list indices must be attached to their named subjects, not just
        # assigned in utterance order: "explanation one for note two".
        def mentioned_number(nouns):
            for position, token in enumerate(tokens):
                if token not in nouns:
                    continue
                tail = tokens[position + 1:position + 4]
                if tail[:1] == ["number"]:
                    tail = tail[1:]
                if tail and tail[0] in _ORDINALS:
                    return _ORDINALS[tail[0]]
                if tail and (tail[0].isdigit() or tail[0] in _NUMBERS):
                    return _numbers(tail[:2])[0]
                before = tokens[position - 1] if position else ""
                if before in _ORDINALS:
                    return _ORDINALS[before]
            return None
        for name, nouns in {"index": {"note", "notes", "recording", "recordings"}, "explanation": {"explanation", "explanations"}}.items():
            value = mentioned_number(nouns)
            if value is not None:
                args[name] = value
        return args
    for name in params:
        if name == "index":
            ordinal = next((_ORDINALS[t] for t in tokens if t in _ORDINALS), None)
            numbers = _numbers(tokens)
            if action_id == "quiz_answer":
                # "answer B", "option c": a lone letter counts only right after one of these words.
                letter = next((_LETTERS[tokens[i + 1]] for i, t in enumerate(tokens[:-1])
                               if t in _LETTER_LEADS and tokens[i + 1] in _LETTERS), None)
                if letter and not (ordinal or numbers):
                    ordinal = letter
                if tokens[-1:] and tokens[-1] in _LETTERS and not (ordinal or numbers) and len(tokens) <= 3:
                    ordinal = _LETTERS[tokens[-1]]
            if not (ordinal or numbers) and action_id in ("open_item", "play_item_audio")                     and set(tokens) & {"recent", "latest", "newest", "last"}:
                ordinal = 1  # lists are newest first
            if ordinal or numbers:
                args["index"] = ordinal or numbers[0]
        elif name == "seconds":
            numbers = _numbers(tokens)
            minutes = "minute" in tokens or "minutes" in tokens
            amount = numbers[0] if numbers else (1 if minutes or "second" in tokens else None)
            if amount is not None:
                args["seconds"] = amount * (60 if minutes else 1)
        elif name == "direction" and action_id == "seek_lecture":
            if set(tokens) & {"back", "backward", "backwards", "rewind", "before", "earlier"}:
                args["direction"] = "backward"
            elif set(tokens) & {"forward", "ahead", "skip", "fast", "jump", "forwards", "advance"}:
                args["direction"] = "forward"
        elif name == "query":
            query = _QUERY_LEAD.sub("", text.strip()).strip(" .?!")
            if action_id == "filter_saved_items":
                query = re.sub(r"^(?:please\s+)?(?:filter|search|find|show)\s+(?:(?:my|the)\s+)?(?:(?:saved\s+)?(?:items|explanations|lectures)|library|bookmarks)\s*(?:by|for|on|about|matching|to)?\s*", "", text.strip(), flags=re.I)
                query = _QUERY_LEAD.sub("", query)
                query = re.sub(r"\s+in\s+(?:my\s+)?(?:saved\s+(?:items|explanations|lectures)|library|bookmarks)\s*$", "", query, flags=re.I).strip(" .?!")
                if query:
                    args["query"] = query
                continue
            args["query"] = query or text.strip()
        elif name == "question":
            args["question"] = text.strip()
        elif name == "name" and action_id == "fill_username":
            username = _spoken_username(text)
            if username:
                args["name"] = username
        elif name == "count" and action_id == "quiz_start":
            numbers = [n for n in _numbers(tokens) if n > 0]
            if numbers:
                # Keep within what quizzes allow ("twenty questions" -> the most there can be); the spoken reply
                # says the count used, so the learner hears exactly what was set up.
                spec = params["count"]
                args["count"] = max(spec.min, min(spec.max, numbers[0]))
        elif (action_id, name) in _ENUM_WORDS:
            value = _last_enum(tokens, _ENUM_WORDS[(action_id, name)])
            if value:
                args[name] = value
    return {k: v for k, v in args.items() if k in params}


class Interpreter:
    def __init__(self, catalog: Optional[ActionCatalog] = None, embedder=None,
                 execute_min=EXECUTE_MIN, confirm_min=CONFIRM_MIN, margin=MARGIN):
        self.catalog = catalog or ActionCatalog.load()
        self._embedder = embedder
        self._index = None
        self._cache: OrderedDict[str, Dict[str, float]] = OrderedDict()
        self.execute_min, self.confirm_min, self.margin = execute_min, confirm_min, margin

    # ── Embeddings ──────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            local_only = os.environ.get(
                "SYNAPSE_LOCAL_MODELS_ONLY", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            model = SentenceTransformer(
                EMBED_MODEL, device="cpu", local_files_only=local_only
            )
            batch_size = max(1, int(os.environ.get("SYNAPSE_VOICE_EMBED_BATCH_SIZE", "16")))
            self._embedder = lambda items: model.encode(
                items, normalize_embeddings=True, batch_size=batch_size
            )
        return np.asarray(self._embedder(texts), dtype=np.float32)

    def warm(self) -> None:
        self._build()

    def close(self) -> None:
        self._embedder = None
        self._index = None
        self._cache.clear()

    def _build(self):
        if self._index is None:
            ids, sentences = [], []
            for lang in self.catalog.languages:
                for action_id, items in self.catalog.examples(lang).items():
                    ids += [action_id] * len(items)
                    sentences += items
            self._index = (np.array(ids), self._encode(sentences))
        return self._index

    def scores(self, text: str) -> Dict[str, float]:
        """Best similarity of the sentence to each action's examples (cached per sentence)."""
        key = text.strip().lower()
        if key not in self._cache:
            ids, matrix = self._build()
            sims = matrix @ self._encode([text])[0]
            self._cache[key] = {a: float(sims[ids == a].max()) for a in dict.fromkeys(ids.tolist())}
            while len(self._cache) > MAX_SCORE_CACHE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    # ── Decision ────────────────────────────────────────────────────────────

    def allowed(self, ctx: VoiceContext) -> set:
        ok = {a.id for a in self.catalog.for_screen(ctx.screen)}
        if ctx.available_actions:
            ok &= set(ctx.available_actions) | _VOICE_LAYER | {"confirm_yes", "confirm_no"}
        if not ctx.explanation_question:
            ok.discard("answer_explanation")
        if not ctx.pending_confirmation:
            ok -= {"confirm_yes", "confirm_no"}
        # What is playing decides between the lecture and the explanation ("pause" means whichever is sounding).
        if ctx.playing in ("lecture", "explanation", "none"):
            if ctx.playing != "lecture":
                ok -= {"pause_lecture"}
            else:
                ok -= {"play_lecture", "pause_explanation", "play_explanation"}
            if ctx.playing != "explanation":
                ok -= {"pause_explanation"}
            else:
                ok -= {"play_explanation", "pause_lecture"}
        return ok

    def interpret(self, req: InterpretRequest) -> InterpretResponse:
        ctx, text = req.context, req.text
        language = req.language
        allowed = self.allowed(ctx)
        if language and language.split("-")[0] != "en":
            return self._reply("unsupported", "english_only", language=language)
        question = ctx.explanation_question
        requested = explanation_request(text) if "request_explanation" in allowed else None
        if "request_explanation" in allowed and requested is None and re.search(
            r"\b(?:don't|do not|never)\s+(?:explain|prepare|generate)\b|\b(?:brief|standard|detailed)\s+or\s+(?:brief|standard|detailed)\b", text.lower()):
            return self._reply("unsupported", "answer_unclear", language=language)
        if requested is not None and (not question or requested.get("depth")):
            depth = requested.get("depth")
            if question and depth in question.choices and "answer_explanation" in allowed:
                return self._reply("execute", "explanation_answer", action="answer_explanation",
                    args={"choice": depth}, confidence=1.0, language=language)
            return self._reply("execute", "explanation_requested", action="request_explanation",
                args=requested, confidence=1.0, language=language)
        if question and "answer_explanation" in allowed and not ctx.pending_confirmation:
            choice, handled = exact_answer(text, question.choices)
            if choice:
                return self._reply("execute", "explanation_answer", action="answer_explanation",
                                   args={"choice": choice}, confidence=1.0, language=language)
            if handled:
                return self._reply("unsupported", "answer_unclear", language=language)
            # Reuse the current embedding model for natural English paraphrases.
            options, examples = [], []
            for option in question.choices:
                options.extend([option] * len(EXAMPLES[option]))
                examples.extend(EXAMPLES[option])
            similarities = self._encode(examples) @ self._encode([text])[0]
            ranked_answers = sorted(
                ((option, max(float(s) for o, s in zip(options, similarities) if o == option))
                 for option in question.choices), key=lambda item: -item[1]
            )
            best, score = ranked_answers[0]
            runner = ranked_answers[1][1] if len(ranked_answers) > 1 else 0
            if score >= max(self.execute_min, 0.78) and score - runner >= max(self.margin, 0.12):
                return self._reply("execute", "explanation_answer", action="answer_explanation",
                                   args={"choice": best}, confidence=min(score, 1.0), language=language)
            allowed.discard("answer_explanation")
        if not _APP_QUESTION_FORM.search(text.lower()):
            allowed.discard("app_question")
        # Asking HOW to do something mentions the same topic as doing it, so a command would win on topic alone.
        # When the sentence has the shape of such a question AND already resembles one, it wins over commands
        # (but still competes with a real content question).
        scores = dict(self.scores(text))
        if _APP_QUESTION_FORM.search(text.lower()) and scores.get("app_question", 0.0) >= self.confirm_min - 0.1:
            commands = [s for a, s in scores.items() if a in allowed and a not in ("app_question", "ask_question")]
            scores["app_question"] = max(scores["app_question"] + 0.15, max(commands, default=0.0) + 0.02)
        words = set(_tokens(text))
        best_now = max((a for a in scores if a in allowed), key=lambda a: scores[a], default=None)
        for forward, backward in _PAIRS:
            if best_now not in (forward, backward):
                continue  # only settles a close call between the pair; a sentence about something else is left alone
            wants_back, wants_forward = bool(words & _BACK_WORDS), bool(words & _FORWARD_WORDS)
            if wants_back != wants_forward:
                win, lose = (backward, forward) if wants_back else (forward, backward)
                scores[win] = max(scores.get(win, 0.0), scores.get(lose, 0.0) + 0.02)
        ranked = sorted(((a, s) for a, s in scores.items() if a in allowed), key=lambda x: -x[1])
        # A number or name in the sentence is evidence for the actions that need one ("moment 3" -> go to moment).
        # The few best are nudged: up when their required values are all present, down when one is missing.
        nudged = []
        for action, score in ranked[:6]:
            needed = [n for n, prm in self.catalog.get(action).params.items() if prm.required]
            if needed:
                found = extract_args(action, text, self.catalog)
                score += 0.09 if all(n in found for n in needed) else -0.04
            nudged.append((action, score))
        ranked = sorted(nudged + ranked[6:], key=lambda x: -x[1])
        if not ranked:
            return self._reply("unsupported", "not_understood", language=language)
        candidates = [ActionCandidate(action=a, score=max(0.0, min(1.0, s))) for a, s in ranked[:3]]
        best_score = ranked[0][1]
        if best_score < self.confirm_min:
            return self._reply("unsupported", "not_understood", language=language, confidence=max(0.0, best_score))

        # The best match whose required values can actually be found: "find lectures on X" may sit just
        # under "pick result N", which cannot run without a number.
        picked = None
        for action, score in ranked[:3]:
            if best_score - score > 0.10:
                break
            found = extract_args(action, text, self.catalog)
            if all(n in found for n, p in self.catalog.get(action).params.items() if p.required):
                picked = (action, score, found)
                break
        if picked is None:
            top = ranked[0][0]
            missing = [n for n, p in self.catalog.get(top).params.items() if p.required and n not in extract_args(top, text, self.catalog)]
            return self._reply("unsupported", f"missing_{missing[0]}", language=language,
                               confidence=max(0.0, min(1.0, best_score)), candidates=candidates[:1])
        top, top_score, args = picked
        conf = max(0.0, min(1.0, top_score))
        second = next(((a, s) for a, s in ranked if a != top), None)
        if (second and top_score - second[1] < self.margin and top_score < 0.85
                and {top, second[0]}.isdisjoint({"confirm_yes", "confirm_no"})):
            return self._reply("clarify", "which_one", language=language, confidence=conf, candidates=candidates[:2])

        spec = self.catalog.get(top)
        try:
            args = self.catalog.validate_args(top, args)
        except CatalogError:
            return self._reply("unsupported", "not_understood", language=language, confidence=conf)

        if top == "app_question":
            # What is the question about? The best-matching real action anywhere in the app, even if it is
            # not on this screen: the frontend says how to do it, or where it works.
            about = [(a, sc) for a, sc in sorted(self.scores(text).items(), key=lambda x: -x[1])
                     if self.catalog.get(a).category not in ("voice", "dialog", "question")][:1]
            extra = [ActionCandidate(action=a, score=max(0.0, min(1.0, sc))) for a, sc in about]
            return self._reply("execute", "done_app_question", action=top, args={}, language=language,
                               confidence=conf, candidates=[candidates[0], *extra][:2])
        if top == "ask_question":
            return self._reply("ask_question", "asking", action=top, args=args, language=language, confidence=conf, candidates=candidates[:1])
        if top_score < self.execute_min or spec.risk == "confirm":
            return self._reply("confirm", f"confirm_{top}", action=top, args=args, language=language, confidence=conf, candidates=candidates[:1])
        return self._reply("execute", f"done_{top}", action=top, args=args, language=language, confidence=conf, candidates=candidates[:1])

    @staticmethod
    def _reply(decision, message_key, **fields) -> InterpretResponse:
        return InterpretResponse(decision=decision, message_key=message_key, **fields)
