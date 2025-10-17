"""Tests for configuration helpers."""
from __future__ import annotations

from pytest import MonkeyPatch

from hypothesis_agent.config import get_settings


def test_get_settings_reads_environment(monkeypatch: MonkeyPatch) -> None:
    """Ensure settings respect environment overrides."""

    assert hasattr(get_settings, "cache_clear")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    monkeypatch.setenv("RAVEN_LOG_LEVEL", "debug")
    monkeypatch.setenv("RAVEN_API_PREFIX", "/internal")

    settings = get_settings()

    assert settings.log_level == "debug"
    assert settings.api_prefix == "/internal"

    get_settings.cache_clear()  # type: ignore[attr-defined]
