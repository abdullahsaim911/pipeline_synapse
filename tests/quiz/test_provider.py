"""Quiz provider subclass: structured-output config, inherited behaviour, protected files untouched."""

import hashlib
import os
import unittest
from unittest import mock

from tests.llm_interface import fake_genai
from tests.quiz.helpers import PROJECT_ROOT
from llm_interface.gemini_provider import GeminiProvider
from llm_interface.providers import GenerationSettings, LLMConfigurationError, LLMRateLimitError
from quiz.prompt import response_schema
from quiz.provider import QuizGeminiProvider, QuizSettings, get_quiz_provider

SECRET = "test-secret-key-123"


def make(result=None):
    client = fake_genai.FakeClient(result)
    return QuizGeminiProvider(api_key=SECRET, client=client), client


class StructuredConfigTests(unittest.TestCase):
    def test_schema_adds_json_mime_type_and_schema_to_the_request(self):
        schema = response_schema()
        settings = QuizSettings(temperature=0.4, max_output_tokens=900, system_instruction="Be exact.",
                                response_json_schema=schema)
        with fake_genai.installed_fake_sdk():
            provider, client = make(fake_genai.FakeResponse('{"questions": []}'))
            result = provider.generate("prompt", settings)
        config = client.models.calls[0]["config"].kwargs
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertEqual(config["response_json_schema"], schema)
        self.assertEqual((config["temperature"], config["max_output_tokens"], config["system_instruction"]),
                         (0.4, 900, "Be exact."))
        self.assertEqual(result.text, '{"questions": []}')

    def test_without_a_schema_the_request_matches_the_shared_provider(self):
        for settings in (QuizSettings(max_output_tokens=300), GenerationSettings(max_output_tokens=300)):
            with fake_genai.installed_fake_sdk():
                provider, client = make(fake_genai.FakeResponse("ok"))
                provider.generate("prompt", settings)
            config = client.models.calls[0]["config"].kwargs
            self.assertNotIn("response_mime_type", config)
            self.assertNotIn("response_json_schema", config)

    def test_inherited_error_mapping_and_key_secrecy(self):
        with fake_genai.installed_fake_sdk():
            provider, _ = make(fake_genai.ClientError(429, "quota"))
            with self.assertRaises(LLMRateLimitError) as caught:
                provider.generate("prompt", QuizSettings(response_json_schema={"type": "object"}))
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertNotIn(SECRET, repr(provider))

    def test_schema_is_a_plain_json_schema_with_a_questions_array(self):
        schema = response_schema()
        self.assertEqual(schema["required"], ["questions"])
        item = schema["properties"]["questions"]["items"]
        self.assertEqual(item["properties"]["type"]["enum"], ["mcq", "tf"])
        self.assertNotIn("start_seconds", item["properties"])      # the model is never asked for times
        self.assertNotIn("quote", item["properties"])

    def test_quiz_settings_keep_generation_setting_validation(self):
        with self.assertRaises(ValueError):
            QuizSettings(temperature=9)
        self.assertTrue(isinstance(QuizSettings(), GenerationSettings))


class FactoryTests(unittest.TestCase):
    def test_from_env_builds_the_quiz_provider(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": SECRET}):
            os.environ.pop("LLM_PROVIDER", None)
            provider = get_quiz_provider()
        self.assertIsInstance(provider, QuizGeminiProvider)
        self.assertNotIn(SECRET, repr(provider))

    def test_missing_key_and_unsupported_provider_are_configuration_errors(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            with self.assertRaises(LLMConfigurationError):
                get_quiz_provider()
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "GEMINI_API_KEY": SECRET}):
            with self.assertRaises(LLMConfigurationError):
                get_quiz_provider()


class ProtectedFilesTests(unittest.TestCase):
    """The provider modules stay untouched: subclassing, not editing."""

    def test_shared_provider_has_no_structured_output_fields(self):
        self.assertFalse(hasattr(GenerationSettings(), "response_json_schema"))
        self.assertIs(GeminiProvider._build_config, QuizGeminiProvider.__mro__[1]._build_config)

    def test_pinned_code_files_still_match_the_archival_hashes(self):
        """config_pf.json pins llm_interface/* and qa/*; the quiz feature must not change any of them."""
        import json
        path = os.path.join(PROJECT_ROOT, "experiments", "pipeline_flags_v1", "config_pf.json")
        with open(path, encoding="utf-8") as handle:
            pinned = json.load(handle)["pins"]["code"]
        self.assertTrue(any(name.startswith("llm_interface/") for name in pinned))
        for relative, digest in pinned.items():
            with open(os.path.join(PROJECT_ROOT, relative), "rb") as handle:
                self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), digest, relative)


if __name__ == "__main__":
    unittest.main()
