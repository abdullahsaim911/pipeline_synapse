"""
Action catalog: the controlled set of things voice control may do.

Data lives next to this file:
  actions.json        language-neutral actions (ids, parameters, screens, risk)
  examples/<lang>.json  prototype sentences per action
  eval/<lang>.json      held-out sentences used only for measuring accuracy

Adding a language means adding two JSON files; no code changes.
"""

from .catalog import (
    ActionCatalog,
    ActionSpec,
    CatalogError,
    EvalItem,
    ParamSpec,
    normalize_sentence,
)

__all__ = [
    "ActionCatalog",
    "ActionSpec",
    "CatalogError",
    "EvalItem",
    "ParamSpec",
    "normalize_sentence",
]
