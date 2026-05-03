import importlib
import os
import sys
import types as pytypes

import pytest


class FakeHttpRetryOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeHttpOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakePart:
    @staticmethod
    def from_text(text):
        return {"text": text}

    @staticmethod
    def from_bytes(data, mime_type):
        return {"data": data, "mime_type": mime_type}


class FakeGenAIClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.http_options = kwargs.get("http_options")
        self.aio = pytypes.SimpleNamespace(
            models=pytypes.SimpleNamespace(generate_content=self.generate_content)
        )
        FakeGenAIClient.instances.append(self)

    async def generate_content(self, **kwargs):
        part = pytypes.SimpleNamespace(text="ok", inline_data=None)
        content = pytypes.SimpleNamespace(parts=[part])
        candidate = pytypes.SimpleNamespace(content=content)
        return pytypes.SimpleNamespace(candidates=[candidate])


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.candidate_count = kwargs.get("candidate_count", 1)
        for key, value in kwargs.items():
            setattr(self, key, value)


def install_dependency_stubs(monkeypatch):
    fake_google = pytypes.ModuleType("google")
    fake_genai = pytypes.ModuleType("google.genai")
    fake_types = pytypes.SimpleNamespace(
        HttpRetryOptions=FakeHttpRetryOptions,
        HttpOptions=FakeHttpOptions,
        Part=FakePart,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_genai.Client = FakeGenAIClient
    fake_genai.types = fake_types
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    fake_pil = pytypes.ModuleType("PIL")
    fake_pil.Image = object()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil.Image)

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_anthropic = pytypes.ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    fake_openai = pytypes.ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    FakeGenAIClient.instances.clear()


def import_generation_utils(monkeypatch):
    install_dependency_stubs(monkeypatch)
    sys.modules.pop("utils.generation_utils", None)
    return importlib.import_module("utils.generation_utils")


def test_reinitialize_clients_configures_gemini_timeout_and_retry(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    module = import_generation_utils(monkeypatch)

    initialized = module.reinitialize_clients()

    assert "Gemini" in initialized
    client = FakeGenAIClient.instances[-1]
    assert client.http_options.timeout == 600_000
    assert client.http_options.retry_options.attempts == 5
    assert client.http_options.retry_options.http_status_codes == [408, 429, 500, 502, 503, 504]


@pytest.mark.asyncio
async def test_gemini_call_uses_wall_clock_timeout(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_HARD_TIMEOUT_SEC", "123")
    module = import_generation_utils(monkeypatch)

    recorded_timeouts = []
    original_wait_for = module.asyncio.wait_for

    async def recording_wait_for(awaitable, timeout):
        recorded_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(module.asyncio, "wait_for", recording_wait_for)

    result = await module.call_gemini_with_retry_async(
        model_name="gemini-test",
        contents=[{"type": "text", "text": "hello"}],
        config=FakeGenerateContentConfig(candidate_count=1),
        max_attempts=1,
    )

    assert result == ["ok"]
    assert recorded_timeouts == [123.0]
    monkeypatch.setattr(module.asyncio, "wait_for", original_wait_for)


def test_priority_service_tier_is_applied_when_enabled(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_SERVICE_TIER", "priority")
    module = import_generation_utils(monkeypatch)

    config = FakeGenerateContentConfig(candidate_count=1)

    stabilized = module._apply_gemini_stability_config(config)

    assert stabilized.service_tier == "priority"
