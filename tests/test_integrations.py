"""Tests validating credential handling for external integrations."""
from __future__ import annotations

import pytest

from hypothesis_agent.config import get_settings
from hypothesis_agent.llm import LLMError, OpenAILLM
from hypothesis_agent.orchestration.yfinance_tools import YFinanceToolSet


def test_openai_llm_requires_api_key() -> None:
    """OpenAI adapter should reject empty API keys."""

    with pytest.raises(LLMError):
        OpenAILLM(api_key="", model="gpt-4o-mini")


def test_yfinance_toolset_initialization() -> None:
    """YFinance toolset should initialize without API keys."""

    toolset = YFinanceToolSet()
    tool = toolset.get_tool("YFINANCE_COMPANY_INFO")
    assert tool is not None


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should populate application settings."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = get_settings()

    assert settings.openai_api_key == "test-openai"
    assert settings.log_level == "DEBUG"

    get_settings.cache_clear()  # type: ignore[attr-defined]