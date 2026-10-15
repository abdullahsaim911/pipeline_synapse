"""
Local token estimation for Ask a Question prompts.

Why an estimate and not Gemini's count_tokens: that is a separate network request,
so counting on every question would double the request count against a free-tier
quota that is usually limited by requests per minute/day. Instead:

  * the prompt budget is enforced with this local estimate (conservative), and
  * the ACTUAL token counts Gemini reports (`usage_metadata`) are logged next to
    the estimate for every call, so the ratio can be calibrated from real traffic
    via ASK_CHARS_PER_TOKEN.

3.5 characters per token is deliberately below the ~4 typical for English prose:
lecture captions contain numbers, symbols and short math tokens that tokenize into
more pieces, and underestimating tokens is the unsafe direction.
"""

import logging
import math
import os

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ENV = "ASK_CHARS_PER_TOKEN"
DEFAULT_CHARS_PER_TOKEN = 3.5
_MIN_CHARS_PER_TOKEN, _MAX_CHARS_PER_TOKEN = 1.0, 8.0


def chars_per_token_from_env() -> float:
    raw = os.environ.get(CHARS_PER_TOKEN_ENV)
    if raw:
        try:
            value = float(raw)
            if _MIN_CHARS_PER_TOKEN <= value <= _MAX_CHARS_PER_TOKEN:
                return value
        except ValueError:
            pass
        logger.warning("Ignoring invalid %s", CHARS_PER_TOKEN_ENV)
    return DEFAULT_CHARS_PER_TOKEN


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimated token count of `text` (rounded up)."""
    return math.ceil(len(text) / chars_per_token) if text else 0


def tokens_to_chars(tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    return max(0, int(tokens * chars_per_token))
