"""Shared pytest configuration for the hypothesis-agent test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hypothesis_agent.config import get_settings

# Ensure tests run in isolated, deterministic mode without external network traffic.
os.environ.setdefault("ENVIRONMENT", "test")
# Use a dedicated artifact directory under the project root for test outputs.
_DEFAULT_ARTIFACT_ROOT = Path(".pytest_artifacts").resolve()
os.environ.setdefault("ARTIFACT_STORE_PATH", str(_DEFAULT_ARTIFACT_ROOT))


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Reset cached settings around each test to honor environment changes."""

    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True, scope="session")
def _ensure_artifact_dir() -> None:
    """Guarantee the artifact directory exists for the duration of test runs."""

    _DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    yield