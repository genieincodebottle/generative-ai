"""The shared provider layer. No API key, no network."""

import pytest

from services import providers
from services.providers import ProviderError, create_llm


class TestCatalogue:
    def test_every_provider_declares_the_same_shape(self):
        for name, spec in providers.PROVIDERS.items():
            assert spec["models"], f"{name} has no models"
            assert spec["key_url"], f"{name} has no key url"
            assert "env_key" in spec

    def test_gemini_lists_a_rolling_alias_first(self):
        # Pinned IDs rot. Every gemini-2.0-* ID this repo used has been
        # retired; the alias is what keeps a fresh clone working.
        assert providers.models_for("Gemini")[0] == "gemini-flash-latest"

    def test_anthropic_lists_the_claude_5_family(self):
        models = providers.models_for("Anthropic")
        assert models[0] == "claude-opus-5"
        assert not any(m.startswith("claude-3") for m in models)

    def test_unknown_provider_has_no_models(self):
        assert providers.models_for("Nope") == []


class TestKeys:
    def test_ollama_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert providers.api_key_for("Ollama") == "local"

    def test_available_providers_reflects_the_environment(self, monkeypatch):
        for spec in providers.PROVIDERS.values():
            if spec["env_key"]:
                monkeypatch.delenv(spec["env_key"], raising=False)
        # Ollama survives because it needs no key.
        assert providers.available_providers() == ["Ollama"]

        monkeypatch.setenv("GEMINI_API_KEY", "abc")
        assert "Gemini" in providers.available_providers()


class TestCreateLLM:
    def test_unknown_provider_raises_rather_than_returning_none(self):
        """The original had no else branch.

        An unknown provider returned None, and the caller then failed with
        `AttributeError: 'NoneType' object has no attribute 'invoke'` - a long
        way from the actual mistake.
        """
        with pytest.raises(ProviderError, match="Unknown provider"):
            create_llm("Nope", "some-model")

    def test_missing_key_names_the_variable_and_where_to_get_one(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        create_llm.cache_clear()
        with pytest.raises(ProviderError) as exc:
            create_llm("Anthropic", "claude-opus-5")
        assert "ANTHROPIC_API_KEY" in str(exc.value)
        assert "console.anthropic.com" in str(exc.value)


class TestOllamaReachable:
    def test_returns_false_instead_of_raising_when_nothing_answers(self):
        # Called on every page load; it must never raise.
        assert providers.ollama_reachable("http://127.0.0.1:1") is False


class TestHealthProbeSpeed:
    """A health endpoint must not block on a third-party service.

    The uncached Ollama probe used a 2-second timeout, which made /health take
    just over two seconds - longer than the launcher's own poll timeout. The
    launcher therefore timed out on every attempt and reported that the API
    had never started, while the API was serving requests happily.
    """

    def test_probe_result_is_cached(self, monkeypatch):
        calls = []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            raise OSError("nothing listening")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        providers._OLLAMA_PROBE.clear()

        assert providers.ollama_reachable("http://127.0.0.1:1") is False
        assert providers.ollama_reachable("http://127.0.0.1:1") is False
        assert providers.ollama_reachable("http://127.0.0.1:1") is False
        assert len(calls) == 1, "the probe should be cached, not repeated"

    def test_probe_timeout_is_short(self, monkeypatch):
        seen = {}

        def fake_urlopen(url, timeout=None):
            seen["timeout"] = timeout
            raise OSError("nothing listening")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        providers._OLLAMA_PROBE.clear()
        providers.ollama_reachable("http://127.0.0.1:2")
        assert seen["timeout"] <= 1.0, "health must stay well under a second"
