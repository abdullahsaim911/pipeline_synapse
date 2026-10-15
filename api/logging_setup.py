"""
Make the Ask a Question diagnostics visible.

uvicorn configures only its own loggers, so INFO lines from `qa` and `llm_interface`
(token usage per call, retrieval decisions, downgrades) would otherwise be dropped and only
warnings would reach the console. Nothing logged there contains prompts, answers, questions or
keys; see qa/service.py and llm_interface/gemini_provider.py.
"""

import logging

_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_ask_logging() -> None:
    """Send INFO and above from the Ask code to the console, once, unless logging is already set up."""
    if logging.getLogger().handlers:      # the host application configured logging; respect it
        return
    for name in ("qa", "llm_interface"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            log.addHandler(handler)
