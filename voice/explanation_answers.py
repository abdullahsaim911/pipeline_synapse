"""English answers to a bounded active question; uncertain meaning never acts."""
import re

EXAMPLES = {
    "listen": ("listen", "yes", "yes please", "sure", "go ahead", "play it", "listen now", "please play it"),
    "skip": ("skip", "no", "no thanks", "not now", "skip it", "continue the lecture", "move on", "I do not want to listen"),
    "yes": ("yes", "I understand", "that makes sense", "it is clear", "I understand now", "resume", "I am satisfied"),
    "no": ("no", "still confused", "I do not understand", "I don't understand", "it is not clear", "another version"),
    "brief": ("brief", "shorter", "just the main point", "make it shorter", "a short explanation", "summarize it"),
    "detailed": ("detailed", "explain more", "more detail", "step by step", "a fuller explanation", "explain in detail"),
    "standard": ("standard", "the standard version", "normal explanation", "original explanation", "the original", "standard explanation"),
    "wait": ("wait", "wait here", "I will wait", "stay here", "keep it paused", "do not resume", "don't resume"),
    "resume": ("resume", "continue", "continue the lecture", "keep going while it prepares", "resume the lecture", "go on", "listen later", "i will listen later", "continue while it prepares", "later", "not now"),
    "later": ("later", "not now", "continue", "resume", "save it for later", "continue the lecture", "listen later", "i will listen later", "no", "no thanks"),
    "replay": ("replay", "play it again", "repeat the explanation", "I understand but play it again", "listen again", "replay it"),
    "retry": ("retry", "try again", "retry the explanation", "try preparing it again", "please retry", "retry generation"),
    "next": ("next", "next one", "skip this one", "next explanation", "move to the next one", "another item"),
    "finish": ("finish review", "finish", "done", "end review", "I am done", "stop reviewing"),
    "process": ("process", "start processing", "prepare that version", "prepare it", "switch to processing", "generate that version"),
}


def explanation_request(text):
    text = " ".join(normalize(text).split())
    if "actually " in text:
        text = text.rsplit("actually ", 1)[1]
    defaults = {"i don't understand", "i do not understand", "i don't understand this", "i do not understand this",
        "i don't understand this explain it", "i do not understand this explain it", "explain this", "explain this part",
        "please explain this", "please explain this part", "i need an explanation", "help me understand this"}
    if text in defaults:
        return {}
    requests = {
        "brief": {"brief", "brief please", "brief version", "make it shorter", "give me a brief explanation", "give me the brief version"},
        "standard": {"standard", "standard please", "standard version", "normal explanation", "give me the standard explanation", "give me the standard version"},
        "detailed": {"detailed", "detailed please", "detailed version", "explain in detail", "give me more detail", "give me the detailed version", "give me a detailed explanation"},
    }
    for depth, phrases in requests.items():
        if text in phrases:
            return {"depth": depth}
    return None

def normalize(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.replace("’", "'").lower()).strip()

def exact_answer(text, choices):
    """Return (choice, handled). Negation/conflict cannot fall through to embeddings."""
    text = " ".join(normalize(text).split())
    # A clear self-correction supersedes the earlier clause.
    if "actually " in text:
        text = text.rsplit("actually ", 1)[1]
    matches = {choice for choice in choices if text in {normalize(example) for example in EXAMPLES[choice]}}
    if len(matches) == 1:
        return matches.pop(), True
    if matches:
        return None, True
    if text in {"yes", "no", "okay", "ok", "maybe", "i don't know", "i am not sure"}:
        return None, True
    if re.search(r"\b(not|don't|do not|never|but|or)\b", text):
        return None, True
    return None, False
