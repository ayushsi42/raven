"""Tests validating credential handling for external integrations."""
from __future__ import annotations

import pytest
from composio.exceptions import ApiKeyNotProvidedError

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.db import firebase as firebase_module
from hypothesis_agent.llm import LLMError, OpenAILLM
from hypothesis_agent.orchestration.langgraph_pipeline import _LazyComposioToolSet


def test_openai_llm_requires_api_key() -> None:
    """OpenAI adapter should reject empty API keys."""

    with pytest.raises(LLMError):
        OpenAILLM(api_key="", model="gpt-4o-mini")


def test_composio_toolset_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Composio tool access should fail when no API key is configured."""

    class _FailingComposio:
        def __init__(self) -> None:
            raise ApiKeyNotProvidedError()

    monkeypatch.setattr(
        "hypothesis_agent.orchestration.langgraph_pipeline.Composio",
        _FailingComposio,
    )
    toolset = _LazyComposioToolSet(user_id=None)

    with pytest.raises(RuntimeError, match="Composio API key must be provided"):
        toolset.get_tool("ALPHA_VANTAGE_TIME_SERIES_MONTHLY_ADJUSTED")


def test_initialize_firebase_uses_explicit_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Firebase initialisation should leverage provided certificate and project ID."""

    captured: dict[str, object] = {}

    def _raise_value_error(name: str | None = None):
        raise ValueError("no app")

    monkeypatch.setattr(firebase_module.firebase_admin, "get_app", _raise_value_error)

    fake_app = object()

    def _fake_initialize_app(*, credential=None, options=None, name="[DEFAULT]"):
        captured["credential"] = credential
        captured["options"] = options
        captured["name"] = name
        return fake_app

    monkeypatch.setattr(firebase_module.firebase_admin, "initialize_app", _fake_initialize_app)

    def _fake_certificate(path: str):
        captured["certificate_path"] = path
        return object()

    monkeypatch.setattr(firebase_module.credentials, "Certificate", _fake_certificate)

    class _FakeFirestoreClient:
        def __init__(self, app):
            self.app = app

    def _fake_client(*, app):
        captured["firestore_app"] = app
        return _FakeFirestoreClient(app)

    monkeypatch.setattr(firebase_module.firestore, "client", _fake_client)

    credential_path = tmp_path / "service-account.json"
    settings = AppSettings(
        firebase_project_id="raven-6324f",
        firebase_credentials_path=str(credential_path),
        firebase_app_name="raven-app",
    )

    handle = firebase_module.initialize_firebase(settings)

    assert captured["certificate_path"] == str(credential_path)
    assert captured["options"] == {"projectId": "raven-6324f"}
    assert captured["name"] == "raven-app"
    assert captured["firestore_app"] is fake_app
    assert handle.collection == settings.firebase_collection


def test_settings_read_authentication_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should populate authentication-related settings."""

    monkeypatch.setenv("RAVEN_OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("RAVEN_ALPHA_VANTAGE_API_KEY", "test-vantage")
    monkeypatch.setenv("RAVEN_FIREBASE_PROJECT_ID", "raven-6324f")
    monkeypatch.setenv("RAVEN_FIREBASE_CREDENTIALS_PATH", "/tmp/cred.json")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_API_KEY", "firebase-web")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_AUTH_DOMAIN", "auth.example.com")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_STORAGE_BUCKET", "bucket")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_MESSAGING_SENDER_ID", "sender")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_APP_ID", "app")
    monkeypatch.setenv("RAVEN_FIREBASE_WEB_MEASUREMENT_ID", "measure")

    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = get_settings()

    assert settings.openai_api_key == "test-openai"
    assert settings.alpha_vantage_api_key == "test-vantage"
    assert settings.firebase_project_id == "raven-6324f"
    assert settings.firebase_credentials_path == "/tmp/cred.json"
    assert settings.firebase_web_api_key == "firebase-web"
    assert settings.firebase_web_auth_domain == "auth.example.com"
    assert settings.firebase_web_storage_bucket == "bucket"
    assert settings.firebase_web_messaging_sender_id == "sender"
    assert settings.firebase_web_app_id == "app"
    assert settings.firebase_web_measurement_id == "measure"

    get_settings.cache_clear()  # type: ignore[attr-defined]