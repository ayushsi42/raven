"""Opt-in integration test against the real Yahoo Finance API.

The unit suite stubs out `YFinanceToolSet` everywhere else, so a real
`yfinance` regression (API shape change, rate limiting, network outage)
would currently only surface at runtime in production. This test makes
one real network call to prove the data layer still works end-to-end
against live Yahoo Finance data, without making it part of every
default `pytest` run (Yahoo Finance rate-limits aggressively and CI
should not depend on it being reachable/fast).

Skipped unless RUN_INTEGRATION_TESTS=1 is set, e.g.:

    RUN_INTEGRATION_TESTS=1 PYTHONPATH=src pytest -q -m integration
"""
from __future__ import annotations

import os

import pytest

from hypothesis_agent.orchestration.yfinance_tools import YFinanceToolSet

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set RUN_INTEGRATION_TESTS=1 to exercise the real Yahoo Finance API.",
)
def test_real_yfinance_company_info_for_aapl() -> None:
    """A well-known, stable ticker should return recognizable company info."""

    toolset = YFinanceToolSet()
    tool = toolset.get_tool("YFINANCE_COMPANY_INFO")
    response = tool.invoke({"symbol": "AAPL"})

    # `_get_company_info` returns a single-element list of dicts.
    assert isinstance(response, list)
    assert len(response) == 1
    info = response[0]
    # Don't over-assert on exact values (Yahoo's data shifts over time);
    # just confirm we got a real, non-empty, ticker-shaped payload back.
    assert info.get("symbol") == "AAPL"
    assert info.get("shortName") or info.get("longName")


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set RUN_INTEGRATION_TESTS=1 to exercise the real Yahoo Finance API.",
)
def test_real_yfinance_historical_prices_for_aapl() -> None:
    """Historical price fetch should return a non-empty time series."""

    toolset = YFinanceToolSet()
    tool = toolset.get_tool("YFINANCE_HISTORICAL_PRICES")
    response = tool.invoke({"symbol": "AAPL", "period": "5d", "interval": "1d"})

    # `_get_historical_prices` returns a list of daily OHLCV records.
    assert isinstance(response, list)
    assert len(response) > 0
    first_record = response[0]
    assert "date" in first_record
    assert "close" in first_record
