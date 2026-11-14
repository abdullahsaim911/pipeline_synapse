"""Authenticated final-answer-only adapter for the pinned Modal Mistral worker."""

from functools import lru_cache
import math
import copy
import json
import os
import time
from urllib.parse import urlsplit

import requests

from modal_inference.mistral_config import MODEL_ID, MODEL_REVISION, MAX_MODEL_LEN, MAX_OUTPUT_TOKENS
from .providers import (
    GenerationResult, GenerationSettings, TokenUsage, TextGenerationProvider,
    LLMConfigurationError, LLMAuthenticationError, LLMConnectionError,
    LLMTimeoutError, LLMRateLimitError, LLMUnavailableError, LLMRequestError,
    LLMMalformedResponseError, LLMEmptyResponseError,
)


class ModalMistralProvider(TextGenerationProvider):
    name = "modal_mistral"
    model = MODEL_ID
    model_revision = MODEL_REVISION
    generation_temperature = 0.1
    max_output_tokens = MAX_OUTPUT_TOKENS
    supports_structured_output = True

    def __init__(self, endpoint_url, api_key, *, timeout_seconds=300, session=None):
        parsed = urlsplit(endpoint_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise LLMConfigurationError("SYNAPSE_MISTRAL_MODAL_URL must be an HTTPS base URL", provider=self.name)
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("SYNAPSE_MODAL_API_KEY is required", provider=self.name)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise LLMConfigurationError("Mistral timeout must be positive and finite", provider=self.name)
        self._base_url = endpoint_url.rstrip("/")
        self._api_key = api_key.strip()
        self._timeout = timeout_seconds
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("SYNAPSE_MISTRAL_MODAL_URL", ""), os.environ.get("SYNAPSE_MODAL_API_KEY", ""))

    def __repr__(self):
        return f"ModalMistralProvider(model={self.model!r})"

    @staticmethod
    def _messages(prompt, system_instruction=None):
        return [
            {"role": "system", "content": system_instruction or (
                "You explain educational visuals for learners who cannot see them. "
                "Follow the evidence and the requested detail level. Distinguish visible facts from "
                "teacher speech, questions, hypotheses and refuted misconceptions. "
                "Give only the final spoken explanation, without headings, Markdown, LaTeX, "
                "private reasoning, or commentary about your instructions. "
                "Accuracy takes priority over supplying a complete-looking story. A graphic's meaning is "
                "unknown unless an explicit label, legend, source relationship or relevant teacher statement "
                "establishes it. Report missing mappings as uncertainty; never fill them with guesses, "
                "including guesses introduced by 'likely', 'may', 'suggests', or 'could'. A shared topic "
                "does not establish what a particular triangle, ruler, grid or marker means. "
                "A rate is a quantity per unit time, not a fixed workload. At a fixed rate, accumulated "
                "quantity is proportional to duration; that does not establish increasing computational "
                "power, exponential growth, or the meaning of a plotted mark. "
                "When the user requests structured fields, keep visual observations separate from "
                "teacher-grounded educational explanations and leave unsupported connections empty."
            )},
            {"role": "user", "content": prompt},
        ]

    def context_limit(self):
        return MAX_MODEL_LEN

    def prompt_input_tokens(self, prompt, system_instruction=None):
        return self._token_count(prompt, system_instruction)

    @lru_cache(maxsize=12)
    def _token_count(self, prompt, system_instruction):
        data = self._post("/tokenize", {"model": self.model, "messages": self._messages(prompt, system_instruction), "add_generation_prompt": True})
        count = data.get("count")
        if type(count) is not int or count < 1:
            raise LLMMalformedResponseError("Mistral token counter returned no valid count", provider=self.name)
        return count

    def _post(self, path, payload):
        # One bounded retry only for availability errors; no retry for malformed
        # answers, missing final text, authentication, or exhausted generation.
        deadline = time.monotonic() + self._timeout
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LLMTimeoutError("Mistral request timed out", provider=self.name)
            try:
                response = self._session.post(
                    self._base_url + path, json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=remaining, allow_redirects=False,
                )
            except requests.Timeout:
                raise LLMTimeoutError("Mistral request timed out", provider=self.name) from None
            except requests.RequestException:
                raise LLMConnectionError("Could not reach Mistral worker", provider=self.name) from None
            status = response.status_code
            if status in {429, 502, 503, 504} and attempt == 0:
                continue
            if status in {401, 403}:
                raise LLMAuthenticationError("Mistral worker rejected authentication", provider=self.name)
            if status == 429:
                raise LLMRateLimitError("Mistral worker is busy", provider=self.name)
            if status >= 500:
                raise LLMUnavailableError("Mistral worker is unavailable", provider=self.name)
            if status != 200:
                raise LLMRequestError("Mistral worker rejected the request", provider=self.name, status_code=status)
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError
                return data
            except ValueError:
                raise LLMMalformedResponseError("Mistral worker returned invalid JSON", provider=self.name) from None

    def generate(self, prompt, settings=None):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        settings = settings or GenerationSettings(temperature=self.generation_temperature, max_output_tokens=self.max_output_tokens)
        budget = settings.max_output_tokens
        if budget > self.max_output_tokens or self.prompt_input_tokens(prompt, settings.system_instruction) + budget > self.context_limit():
            raise LLMRequestError("Mistral prompt and completion exceed the serving budget", provider=self.name)
        started = time.perf_counter()
        wire_schema = copy.deepcopy(settings.response_schema)
        def preserve_length_bounds(node):
            if isinstance(node, dict):
                # xgrammar does not combine pattern with length reliably. The
                # schema enforces finite text; native bad_words prevents speech
                # markup. The original schema is validated by the caller too.
                if "maxLength" in node and "pattern" in node:
                    node.pop("pattern")
                for value in node.values():
                    preserve_length_bounds(value)
            elif isinstance(node, list):
                for value in node:
                    preserve_length_bounds(value)
        preserve_length_bounds(wire_schema)
        data = self._post("/v1/chat/completions", {
            "model": self.model, "messages": self._messages(prompt, settings.system_instruction),
            "temperature": settings.temperature, "top_p": 0.95, "max_tokens": budget,
            "repetition_penalty": 1.05,
            # Do not ban underscores/backslashes: JSON schema keys contain
            # underscores and JSON quoting requires escapes. A native bad-word
            # ban on them can deadlock constrained decoding at a required key.
            "bad_words": ["*", "`", "$", "∫", "∑"],
            # This endpoint produces spoken scripts, not Markdown or code
            # listings. Constrain markup during decoding rather than stripping
            # asterisks/subscripts afterwards and risking a changed expression.
            "structured_outputs": ({"json": wire_schema, "disable_any_whitespace": True} if wire_schema is not None
                                   else {"regex": r"[^*`$\\#_∫∑]*"}),
        })
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content")
            finish = choice["finish_reason"]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMMalformedResponseError("Mistral response has no valid final message", provider=self.name) from None
        if finish == "length" and settings.response_schema is not None and isinstance(text, str):
            # Some constrained decoders emit formatting whitespace after a
            # completed JSON object until the token ceiling. Accept only an
            # independently schema-valid COMPLETE document, never a prefix,
            # partial string, repaired JSON, or plain-text length termination.
            import jsonschema
            try:
                parsed = json.loads(text)
                jsonschema.validate(parsed, settings.response_schema)
            except (ValueError, jsonschema.ValidationError):
                raise LLMMalformedResponseError("Mistral returned incomplete structured output", provider=self.name) from None
            finish = "complete_json_at_limit"
        elif finish != "stop":
            raise LLMMalformedResponseError(f"Mistral did not finish the final explanation (finish={finish})", provider=self.name)
        if not isinstance(text, str) or not text.strip():
            raise LLMEmptyResponseError("Mistral returned no final explanation", provider=self.name)
        if any(marker in text for marker in ("[THINK]", "[/THINK]", "<think>", "</think>")):
            raise LLMMalformedResponseError("Mistral reasoning was not separated from its final answer", provider=self.name)
        usage = data.get("usage") or {}
        def count(key):
            value = usage.get(key)
            return value if type(value) is int and value >= 0 else None
        return GenerationResult(
            text=text.strip(), provider=self.name, model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000), finish_reason=finish,
            usage=TokenUsage(prompt_tokens=count("prompt_tokens"), output_tokens=count("completion_tokens"), total_tokens=count("total_tokens"),
                            thoughts_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
        )
