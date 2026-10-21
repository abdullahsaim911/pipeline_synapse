"""Minimal stand-in for the google-genai SDK so tests need no install, key or network."""

import sys
import types
from contextlib import contextmanager
from unittest import mock


class APIError(Exception):
    def __init__(self, code, message="", status=None):
        super().__init__(f"{code} {message}")
        self.code = code
        self.message = message
        self.status = status


class ClientError(APIError):
    pass


class ServerError(APIError):
    pass


# The provider recognises SDK errors by their module name.
for _cls in (APIError, ClientError, ServerError):
    _cls.__module__ = "google.genai.errors"


class FakeResponse:
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result=None, **kwargs):
        self.init_kwargs = kwargs
        self.models = FakeModels(result)


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@contextmanager
def installed_fake_sdk(result=None):
    """Install fake `google.genai` modules; yields the list of created clients."""
    created = []

    def make_client(**kwargs):
        client = FakeClient(result, **kwargs)
        created.append(client)
        return client

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    sdk_types = types.ModuleType("google.genai.types")
    sdk_errors = types.ModuleType("google.genai.errors")

    genai.Client = make_client
    genai.types = sdk_types
    genai.errors = sdk_errors
    sdk_types.GenerateContentConfig = FakeConfig
    sdk_types.HttpOptions = lambda **kw: types.SimpleNamespace(**kw)
    sdk_errors.APIError = APIError
    sdk_errors.ClientError = ClientError
    sdk_errors.ServerError = ServerError
    google.genai = genai

    with mock.patch.dict(sys.modules, {
        "google": google,
        "google.genai": genai,
        "google.genai.types": sdk_types,
        "google.genai.errors": sdk_errors,
    }):
        yield created
