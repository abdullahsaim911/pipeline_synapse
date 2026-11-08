"""
Loader and validator for the voice action catalog.

Everything here is data checking: it decides nothing about what a user meant.
Loading fails loudly (CatalogError) on inconsistent data, so a typo in a JSON
file is found by the tests and never reaches the matcher.
"""

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from voice.schemas import Decision, Playing, Screen

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENS = list(get_args(Screen))
DECISIONS = list(get_args(Decision))
PLAYING = list(get_args(Playing))
CATEGORIES = (
    "navigation", "settings", "discovery", "lecture",
    "collections", "question", "voice", "dialog",
)
UNSUPPORTED = "unsupported"
MIN_EXAMPLES_PER_ACTION = 5

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)


class CatalogError(ValueError):
    """The catalog data is inconsistent."""


def normalize_sentence(text: str) -> str:
    """Lowercase, drop punctuation and collapse spaces, for duplicate detection."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


class ParamSpec(BaseModel):
    type: Literal["int", "string", "enum"]
    required: bool = True
    min: Optional[int] = None
    max: Optional[int] = None
    max_length: Optional[int] = Field(default=None, ge=1)
    values: Optional[List[str]] = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.type == "enum" and not self.values:
            raise ValueError("enum parameter needs values")
        if self.type == "int" and (self.min is None or self.max is None or self.min > self.max):
            raise ValueError("int parameter needs min <= max")
        if self.type == "string" and self.max_length is None:
            raise ValueError("string parameter needs max_length")
        return self


class ActionSpec(BaseModel):
    id: str
    category: Literal[*CATEGORIES]
    description: str = Field(..., min_length=20)
    screens: List[str] = Field(..., min_length=1)
    params: Dict[str, ParamSpec] = Field(default_factory=dict)
    risk: Literal["low", "confirm"] = "low"
    binding: str = Field(..., min_length=1)
    binding_status: Literal["existing", "wrapper", "voice"]

    @field_validator("id")
    @classmethod
    def _id(cls, value):
        if not _ID_RE.match(value):
            raise ValueError(f"invalid action id: {value!r}")
        return value

    @field_validator("screens")
    @classmethod
    def _screens(cls, value):
        for screen in value:
            if screen != "*" and screen not in SCREENS:
                raise ValueError(f"unknown screen: {screen!r}")
        return value

    def allowed_on(self, screen: str) -> bool:
        return "*" in self.screens or screen in self.screens


class EvalItem(BaseModel):
    text: str = Field(..., min_length=1)
    expect: str
    args: Dict[str, Any] = Field(default_factory=dict)
    screen: Screen
    playing: Playing = "unknown"
    pending: Optional[str] = None
    note: str = ""


class ActionCatalog:
    """The validated catalog plus its per-language examples and evaluation sets."""

    def __init__(self, actions: Dict[str, ActionSpec], examples: Dict[str, Dict[str, List[str]]],
                 evals: Dict[str, List[EvalItem]]):
        self._actions = actions
        self._examples = examples
        self._evals = evals

    # ── Loading ─────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, data_dir: Optional[str] = None, languages: Optional[List[str]] = None) -> "ActionCatalog":
        base = data_dir or DATA_DIR
        raw = _read_json(os.path.join(base, "actions.json"))
        actions: Dict[str, ActionSpec] = {}
        for entry in raw.get("actions", []):
            try:
                spec = ActionSpec(**entry)
            except ValidationError as exc:
                raise CatalogError(f"action {entry.get('id', '?')!r}: {exc}") from exc
            if spec.id in actions:
                raise CatalogError(f"duplicate action id: {spec.id}")
            actions[spec.id] = spec
        if not actions:
            raise CatalogError("actions.json defines no actions")

        if languages is None:
            languages = sorted(
                os.path.splitext(name)[0]
                for name in os.listdir(os.path.join(base, "examples"))
                if name.endswith(".json")
            )

        examples: Dict[str, Dict[str, List[str]]] = {}
        evals: Dict[str, List[EvalItem]] = {}
        for lang in languages:
            examples[lang] = cls._load_examples(base, lang, actions)
            evals[lang] = cls._load_eval(base, lang, actions, examples[lang])
        return cls(actions, examples, evals)

    @staticmethod
    def _load_examples(base: str, lang: str, actions: Dict[str, ActionSpec]) -> Dict[str, List[str]]:
        raw = _read_json(os.path.join(base, "examples", f"{lang}.json"))
        data = raw.get("examples", {})
        unknown = sorted(set(data) - set(actions))
        if unknown:
            raise CatalogError(f"examples/{lang}.json names unknown actions: {unknown}")
        seen: Dict[str, str] = {}
        for action_id, sentences in data.items():
            if not isinstance(sentences, list) or not all(isinstance(s, str) and s.strip() for s in sentences):
                raise CatalogError(f"examples/{lang}.json: {action_id} must be a list of non-empty strings")
            for sentence in sentences:
                key = normalize_sentence(sentence)
                if key in seen:
                    raise CatalogError(
                        f"examples/{lang}.json: {sentence!r} appears under both {seen[key]} and {action_id}"
                    )
                seen[key] = action_id
        if raw.get("complete"):
            thin = [a for a in actions if len(data.get(a, [])) < MIN_EXAMPLES_PER_ACTION]
            if thin:
                raise CatalogError(
                    f"examples/{lang}.json is marked complete but these actions have fewer than "
                    f"{MIN_EXAMPLES_PER_ACTION} examples: {thin}"
                )
        return data

    @classmethod
    def _load_eval(cls, base: str, lang: str, actions: Dict[str, ActionSpec],
                   examples: Dict[str, List[str]]) -> List[EvalItem]:
        path = os.path.join(base, "eval", f"{lang}.json")
        if not os.path.exists(path):
            return []
        raw = _read_json(path)
        example_keys = {normalize_sentence(s) for sentences in examples.values() for s in sentences}
        items: List[EvalItem] = []
        seen = set()
        probe = cls(actions, {}, {})
        for entry in raw.get("items", []):
            try:
                item = EvalItem(**entry)
            except ValidationError as exc:
                raise CatalogError(f"eval/{lang}.json item {entry.get('text')!r}: {exc}") from exc
            if item.expect != UNSUPPORTED and item.expect not in actions:
                raise CatalogError(f"eval/{lang}.json: {item.text!r} expects unknown action {item.expect!r}")
            if item.pending is not None and item.pending not in actions:
                raise CatalogError(f"eval/{lang}.json: {item.text!r} has unknown pending action {item.pending!r}")
            if item.expect != UNSUPPORTED:
                if not actions[item.expect].allowed_on(item.screen):
                    raise CatalogError(
                        f"eval/{lang}.json: {item.text!r} expects {item.expect} on '{item.screen}', "
                        f"where that action is not available"
                    )
                if item.args:
                    try:
                        probe.validate_args(item.expect, item.args)
                    except CatalogError as exc:
                        raise CatalogError(f"eval/{lang}.json: {item.text!r}: {exc}") from exc
            if normalize_sentence(item.text) in example_keys:
                raise CatalogError(
                    f"eval/{lang}.json: {item.text!r} is also an example sentence; "
                    f"held-out sentences must not be used as examples"
                )
            key = (normalize_sentence(item.text), item.screen, item.playing, item.pending)
            if key in seen:
                raise CatalogError(f"eval/{lang}.json: duplicate item {item.text!r}")
            seen.add(key)
            items.append(item)
        return items

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def languages(self) -> List[str]:
        return sorted(self._examples)

    def ids(self) -> List[str]:
        return list(self._actions)

    def get(self, action_id: str) -> ActionSpec:
        try:
            return self._actions[action_id]
        except KeyError:
            raise CatalogError(f"unknown action: {action_id!r}") from None

    def __contains__(self, action_id: str) -> bool:
        return action_id in self._actions

    def for_screen(self, screen: str) -> List[ActionSpec]:
        return [a for a in self._actions.values() if a.allowed_on(screen)]

    def examples(self, lang: str) -> Dict[str, List[str]]:
        return self._examples[lang]

    def eval_items(self, lang: str) -> List[EvalItem]:
        return self._evals.get(lang, [])

    def unknown_actions(self, action_ids: List[str]) -> List[str]:
        return [a for a in action_ids if a not in self._actions]

    # ── Argument validation ─────────────────────────────────────────────────

    def validate_args(self, action_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check args against the action's parameter spec and return a clean copy.
        Raises CatalogError for missing, unexpected or out-of-range values.
        """
        spec = self.get(action_id)
        if not isinstance(args, dict):
            raise CatalogError(f"{action_id}: args must be an object")
        unexpected = sorted(set(args) - set(spec.params))
        if unexpected:
            raise CatalogError(f"{action_id}: unexpected arguments {unexpected}")
        clean: Dict[str, Any] = {}
        for name, param in spec.params.items():
            if name not in args or args[name] is None:
                if param.required:
                    raise CatalogError(f"{action_id}: missing argument '{name}'")
                continue
            value = args[name]
            if param.type == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise CatalogError(f"{action_id}.{name}: must be an integer")
                if not param.min <= value <= param.max:
                    raise CatalogError(f"{action_id}.{name}: must be between {param.min} and {param.max}")
            elif param.type == "string":
                if not isinstance(value, str) or not value.strip():
                    raise CatalogError(f"{action_id}.{name}: must be a non-empty string")
                value = value.strip()
                if len(value) > param.max_length:
                    raise CatalogError(f"{action_id}.{name}: longer than {param.max_length} characters")
            else:
                if value not in param.values:
                    raise CatalogError(f"{action_id}.{name}: must be one of {param.values}")
            clean[name] = value
        return clean

    # ── Frontend export ─────────────────────────────────────────────────────

    def to_frontend_js(self) -> str:
        """JavaScript module the frontend imports; generated, never edited by hand."""
        actions = {}
        for spec in self._actions.values():
            actions[spec.id] = {
                "category": spec.category,
                "screens": spec.screens,
                "params": {name: p.model_dump(exclude_none=True) for name, p in spec.params.items()},
                "risk": spec.risk,
                "binding": spec.binding,
                "bindingStatus": spec.binding_status,
            }
        lines = [
            "// GENERATED by the Synapse backend: python -m voice.catalog export-js --out <this file>",
            "// Do not edit by hand. Source of truth: voice/catalog/actions.json in the backend repository.",
            "",
            f"export const SCREENS = {json.dumps(SCREENS)};",
            f"export const DECISIONS = {json.dumps(DECISIONS)};",
            f"export const PLAYING_STATES = {json.dumps(PLAYING)};",
            f"export const ACTIONS = {json.dumps(actions, indent=2, sort_keys=False)};",
            "",
        ]
        return "\n".join(lines)


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise CatalogError(f"missing catalog file: {path}") from None
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path} is not valid JSON: {exc}") from exc
