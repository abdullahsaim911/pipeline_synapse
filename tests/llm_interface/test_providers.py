"""Tests for the provider layer. No network, API key or google-genai install needed."""

import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_interface import (  # noqa: E402
    GenerationSettings,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    OllamaClient,
    TokenUsage,
    get_text_provider,
)
from llm_interface.gemini_provider import DEFAULT_MODEL, GeminiProvider  # noqa: E402
from tests.llm_interface import fake_genai  # noqa: E402

SECRET = "test-secret-key-123"


def make_provider(result=None, **kwargs):
    client = fake_genai.FakeClient(result)
    provider = GeminiProvider(api_key=SECRET, client=client, **kwargs)
    return provider, client


class ConfigurationTests(unittest.TestCase):
    def test_missing_key_is_a_configuration_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError) as ctx:
                get_text_provider("gemini")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_blank_key_is_rejected(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "   "}, clear=True):
            with self.assertRaises(LLMConfigurationError):
                get_text_provider()

    def test_defaults_when_only_key_is_set(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": SECRET}, clear=True):
            provider = get_text_provider()
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.model, DEFAULT_MODEL)
        self.assertEqual(provider.timeout_seconds, 60.0)

    def test_model_and_timeout_come_from_environment(self):
        env = {"GEMINI_API_KEY": SECRET, "GEMINI_MODEL": "my-model",
               "GEMINI_TIMEOUT_SECONDS": "12.5"}
        with mock.patch.dict(os.environ, env, clear=True):
            provider = get_text_provider()
        self.assertEqual(provider.model, "my-model")
        self.assertEqual(provider.timeout_seconds, 12.5)

    def test_invalid_timeout_is_a_configuration_error(self):
        env = {"GEMINI_API_KEY": SECRET, "GEMINI_TIMEOUT_SECONDS": "soon"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(LLMConfigurationError):
                get_text_provider()

    def test_provider_name_can_come_from_llm_provider_variable(self):
        env = {"LLM_PROVIDER": "Gemini", "GEMINI_API_KEY": SECRET}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_text_provider().name, "gemini")

    def test_unknown_provider_does_not_fall_back(self):
        env = {"GEMINI_API_KEY": SECRET}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(LLMConfigurationError) as ctx:
                get_text_provider("openai")
        self.assertIn("openai", str(ctx.exception))

    def test_repr_does_not_contain_key(self):
        provider, _ = make_provider()
        self.assertNotIn(SECRET, repr(provider))

    def test_missing_sdk_is_a_configuration_error(self):
        provider = GeminiProvider(api_key=SECRET)
        with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaises(LLMConfigurationError) as ctx:
                provider.generate("hello")
        self.assertIn("google-genai", str(ctx.exception))


class GenerationTests(unittest.TestCase):
    def test_returns_text_and_metadata(self):
        with fake_genai.installed_fake_sdk():
            provider, client = make_provider(fake_genai.FakeResponse("  An answer.  "))
            result = provider.generate("What is a derivative?")
        self.assertEqual(result.text, "An answer.")
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, DEFAULT_MODEL)
        self.assertEqual(client.models.calls[0]["model"], DEFAULT_MODEL)

    def test_settings_are_passed_to_sdk(self):
        settings = GenerationSettings(
            temperature=0.1, max_output_tokens=300, system_instruction="Be brief.")
        with fake_genai.installed_fake_sdk():
            provider, client = make_provider(fake_genai.FakeResponse("ok"))
            provider.generate("hi", settings)
        config = client.models.calls[0]["config"].kwargs
        self.assertEqual(config, {"temperature": 0.1, "max_output_tokens": 300,
                                  "system_instruction": "Be brief."})

    def test_client_is_built_with_key_and_millisecond_timeout(self):
        provider = GeminiProvider(api_key=SECRET, timeout_seconds=7)
        with fake_genai.installed_fake_sdk(fake_genai.FakeResponse("ok")) as created:
            provider.generate("hi")
            provider.generate("again")
        self.assertEqual(len(created), 1)  # client is reused
        self.assertEqual(created[0].init_kwargs["api_key"], SECRET)
        self.assertEqual(created[0].init_kwargs["http_options"].timeout, 7000)

    def test_empty_prompt_is_rejected_before_any_call(self):
        provider, client = make_provider(fake_genai.FakeResponse("x"))
        with self.assertRaises(ValueError):
            provider.generate("   ")
        self.assertEqual(client.models.calls, [])

    def test_empty_response_is_an_error(self):
        for text in (None, "", "   "):
            with fake_genai.installed_fake_sdk():
                provider, _ = make_provider(fake_genai.FakeResponse(text))
                with self.assertRaises(LLMEmptyResponseError):
                    provider.generate("hi")

    def test_token_usage_is_captured_from_response_metadata(self):
        metadata = SimpleNamespace(prompt_token_count=1700, candidates_token_count=60,
                                   thoughts_token_count=12, total_token_count=1772,
                                   cached_content_token_count=None)
        with fake_genai.installed_fake_sdk():
            provider, _ = make_provider(fake_genai.FakeResponse("An answer.", usage_metadata=metadata))
            result = provider.generate("What is a derivative?")
        self.assertEqual(result.usage, TokenUsage(prompt_tokens=1700, output_tokens=60, thoughts_tokens=12,
                                                  total_tokens=1772, cached_tokens=None))

    def test_missing_or_malformed_usage_is_none_not_invented(self):
        garbage = SimpleNamespace(prompt_token_count="1700", candidates_token_count=True,
                                  thoughts_token_count=None, total_token_count=1.5)
        for metadata in (None, SimpleNamespace(), garbage):
            with fake_genai.installed_fake_sdk():
                provider, _ = make_provider(fake_genai.FakeResponse("An answer.", usage_metadata=metadata))
                self.assertIsNone(provider.generate("hi").usage)

    def test_partial_usage_keeps_only_what_was_reported(self):
        with fake_genai.installed_fake_sdk():
            provider, _ = make_provider(fake_genai.FakeResponse(
                "An answer.", usage_metadata=SimpleNamespace(prompt_token_count=300)))
            usage = provider.generate("hi").usage
        self.assertEqual(usage, TokenUsage(prompt_tokens=300))

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            GenerationSettings(temperature=5)
        with self.assertRaises(ValueError):
            GenerationSettings(max_output_tokens=0)


class ErrorMappingTests(unittest.TestCase):
    def assert_maps(self, exc, expected, retryable, status=None):
        with fake_genai.installed_fake_sdk():
            provider, _ = make_provider(exc)
            with self.assertRaises(LLMProviderError) as ctx:
                provider.generate("hi")
        error = ctx.exception
        self.assertIs(type(error), expected)
        self.assertEqual(error.retryable, retryable)
        self.assertEqual(error.status_code, status)
        self.assertNotIn(SECRET, str(error))
        return error

    def test_http_status_mapping(self):
        cases = [
            (fake_genai.ClientError(401, "nope"), LLMAuthenticationError, False, 401),
            (fake_genai.ClientError(403, "denied"), LLMAuthenticationError, False, 403),
            (fake_genai.ClientError(400, "API key not valid. Please pass a valid API key."),
             LLMAuthenticationError, False, 400),
            (fake_genai.ClientError(429, "RESOURCE_EXHAUSTED"), LLMRateLimitError, True, 429),
            (fake_genai.ServerError(504, "DEADLINE_EXCEEDED"), LLMTimeoutError, True, 504),
            (fake_genai.ServerError(503, "overloaded"), LLMUnavailableError, True, 503),
            (fake_genai.ServerError(500, "internal"), LLMUnavailableError, True, 500),
            (fake_genai.ClientError(400, "prompt too long"), LLMRequestError, False, 400),
            (fake_genai.ClientError(404, "model not found"), LLMRequestError, False, 404),
        ]
        for exc, expected, retryable, status in cases:
            with self.subTest(code=exc.code, message=exc.message):
                self.assert_maps(exc, expected, retryable, status)

    def test_provider_message_text_is_not_echoed(self):
        error = self.assert_maps(
            fake_genai.ClientError(400, f"bad request containing {SECRET}"),
            LLMRequestError, False, 400)
        self.assertNotIn("bad request", str(error))

    def test_transport_errors(self):
        class ReadTimeout(Exception):
            pass

        class ConnectError(Exception):
            pass

        self.assert_maps(ReadTimeout("slow"), LLMTimeoutError, True)
        self.assert_maps(TimeoutError("slow"), LLMTimeoutError, True)
        self.assert_maps(ConnectError("dns"), LLMConnectionError, True)
        self.assert_maps(ConnectionResetError("reset"), LLMConnectionError, True)

    def test_unknown_exception_becomes_generic_provider_error(self):
        error = self.assert_maps(RuntimeError("boom"), LLMProviderError, False)
        self.assertNotIn("boom", str(error))

    def test_logs_do_not_contain_key_or_prompt(self):
        prompt = "PRIVATE-LECTURE-QUESTION"
        with fake_genai.installed_fake_sdk():
            provider, _ = make_provider(fake_genai.ClientError(429, "quota"))
            with self.assertLogs("llm_interface.gemini_provider", level=logging.DEBUG) as logs:
                with self.assertRaises(LLMRateLimitError):
                    provider.generate(prompt)
        combined = "\n".join(logs.output)
        self.assertNotIn(SECRET, combined)
        self.assertNotIn(prompt, combined)


class OllamaCompatibilityTests(unittest.TestCase):
    """The Ollama client must keep its public surface and behaviour."""

    def test_public_names_still_exported(self):
        import llm_interface
        for name in ("OllamaClient", "OllamaError", "OllamaResponse",
                     "OllamaConnectionError", "OllamaTimeoutError",
                     "OllamaServerError", "OllamaClientError", "FUSION_AVAILABLE"):
            self.assertTrue(hasattr(llm_interface, name), name)

    def test_constructor_signature_used_by_synthesizer(self):
        client = OllamaClient(base_url="http://localhost:11434", model="mistral",
                              timeout=120, max_retries=3)
        self.assertEqual(client.model, "mistral")
        self.assertEqual(client.timeout, 120)

    def test_generate_still_uses_ollama_http_path(self):
        client = OllamaClient(model="mistral")
        fake_json = {"response": "hello", "model": "mistral"}
        with mock.patch("llm_interface.ollama_client.FUSION_AVAILABLE", False), \
                mock.patch.object(OllamaClient, "_make_request", return_value=fake_json):
            response = client.generate("prompt")
        self.assertEqual(response.text, "hello")

    def test_ollama_errors_are_independent_of_provider_errors(self):
        from llm_interface import OllamaError
        self.assertFalse(issubclass(OllamaError, LLMProviderError))

    def test_synthesizer_still_imports(self):
        from synthesizer import LLMSynthesizer  # noqa: F401


if __name__ == "__main__":
    unittest.main()
