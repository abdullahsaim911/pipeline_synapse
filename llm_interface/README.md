# LLM providers

`llm_interface` contains two independent ways to call a language model:

| Component | Used by | Notes |
|---|---|---|
| `OllamaClient` (`ollama_client.py`) | Existing explanation pipeline (`LLMSynthesizer`) | Unchanged. Local Ollama/Mistral. |
| `TextGenerationProvider` (`providers.py`) + `GeminiProvider` (`gemini_provider.py`) | Future Ask a Question service | Hosted Gemini API, selected by configuration. |

The visual pipeline (Qwen2-VL via `llama-cpp-python`) is not touched.

## Using a provider

```python
from llm_interface import get_text_provider, GenerationSettings, LLMProviderError

provider = get_text_provider()          # reads LLM_PROVIDER (default "gemini")
try:
    result = provider.generate(
        prompt,
        GenerationSettings(temperature=0.2, max_output_tokens=600,
                           system_instruction="Answer only from the lecture."),
    )
    print(result.text, result.model, result.latency_ms)
except LLMProviderError as e:
    print(e.code, e.retryable, e)       # safe to show; never contains keys or prompts
```

## Configuration

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | yes (for Gemini) | none | API key. Never commit it. |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` | Gemini model identifier. |
| `GEMINI_TIMEOUT_SECONDS` | no | `60` | Per-request timeout. |
| `LLM_PROVIDER` | no | `gemini` | Provider selector. Only `gemini` exists today. |

If the key is missing, selecting Gemini raises `LLMConfigurationError`. There is no
silent fallback to a local or paid model.

### Getting a key

1. Sign in at [Google AI Studio](https://aistudio.google.com/apikey) and create an API key.
2. Install the SDK: `pip install google-genai` (also listed in `requirements.txt`).
3. Set the variable for the shell that starts the API.

PowerShell (current session):

```powershell
$env:GEMINI_API_KEY = "your-key"
# optional
$env:GEMINI_MODEL = "gemini-3.5-flash-lite"
```

bash:

```bash
export GEMINI_API_KEY="your-key"
```

The repo does not load `.env` files automatically. `.env`, `.env.local` and `.env.*`
are git-ignored (`.env.example` is allowed) so a local file will not be committed.

### Free tier caveats

Free-tier access, quotas, model availability and data-use terms are set by Google and
can change or differ by account and region. Free quotas are limited (expect HTTP 429
under load), and Google's terms for the free tier may allow use of submitted content to
improve its products, so do not send private data you would not want shared. Check
Google's current [pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) (view your limits in
AI Studio) and terms before relying on it. Model identifiers are retired over time; if
the default stops working, set `GEMINI_MODEL`.

## Errors

All provider failures derive from `LLMProviderError` (`code`, `retryable`, `status_code`).

| Class | `code` | Retryable | Typical cause |
|---|---|---|---|
| `LLMConfigurationError` | `configuration_error` | no | Missing key, bad timeout, SDK not installed, unknown provider |
| `LLMAuthenticationError` | `authentication_error` | no | Invalid key or permission (HTTP 400 "API key", 401, 403) |
| `LLMRateLimitError` | `rate_limited` | yes | Quota exceeded (HTTP 429) |
| `LLMTimeoutError` | `timeout` | yes | Timeout (HTTP 408/504 or client timeout) |
| `LLMConnectionError` | `connection_error` | yes | Network/DNS failure |
| `LLMUnavailableError` | `provider_unavailable` | yes | HTTP 5xx |
| `LLMRequestError` | `invalid_request` | no | Other 4xx (bad request, unknown model) |
| `LLMEmptyResponseError` | `empty_response` | no | No text returned (blocked or truncated) |
| `LLMMalformedResponseError` | `malformed_response` | yes | The model's reply broke the output contract (raised by the Ask service, not by a provider) |

The `POST /video/ask` endpoint maps these as: configuration -> 500, rate limit -> 429,
timeout -> 504, connection/unavailable -> 503, authentication -> 502, malformed reply -> 502,
everything else -> 502
(see `_llm_http_error` in `api/routers/video.py`).

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Tests use fakes; they need no key, network, or Gemini quota. The API and database tests
need `fastapi`, `httpx`, `sqlalchemy` and `pydantic` (`pip install -r api/requirements.txt httpx`);
they are skipped automatically if those are missing. `google-genai` is not needed to run the tests.
