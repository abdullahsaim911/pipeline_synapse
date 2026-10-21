"""
Feature flag for POST /video/quiz.

An EXPLICIT QUIZ_ENABLED always decides: the quiz is on only when the value is exactly "1" or "true"
(surrounding whitespace ignored); "", "0", "TRUE", "yes" and anything else mean off.

When QUIZ_ENABLED is not set at all, the default follows the environment: ON for development (SYNAPSE_ENV unset or
anything but "production") and OFF when SYNAPSE_ENV is exactly "production" (case and whitespace ignored). Set
SYNAPSE_ENV=production for any deployment that must keep the quiz dark. Read on every request, so it can be switched
without importing application code.
"""

import os

QUIZ_ENABLED_ENV_VAR = "QUIZ_ENABLED"
SYNAPSE_ENV_VAR = "SYNAPSE_ENV"
_ON_VALUES = ("1", "true")


def quiz_enabled() -> bool:
    explicit = os.environ.get(QUIZ_ENABLED_ENV_VAR)
    if explicit is not None:
        return explicit.strip() in _ON_VALUES
    return os.environ.get(SYNAPSE_ENV_VAR, "").strip().lower() != "production"
