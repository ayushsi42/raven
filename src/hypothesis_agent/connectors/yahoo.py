"""Client for Yahoo Finance historical price data."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

_YAHOO_DOWNLOAD_URL = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}"


class YahooFinanceError(RuntimeError):
    """Raised when Yahoo Finance data cannot be retrieved."""


@dataclass(slots=True)
class PriceSeries:
    """Container for historical price data."""

    ticker: str
    prices: List[Dict[str, float]]


@dataclass(slots=True)
class YahooFinanceClient:
    """Retrieve historical OHLC data from Yahoo Finance."""

    timeout_seconds: float = 10.0

    def fetch_daily_prices(self, ticker: str, start: date, end: date) -> PriceSeries:
        """Fetch daily OHLC prices for the given ticker between the provided dates."""

        params = {
            "period1": int(start.strftime("%s")),
            "period2": int(end.strftime("%s")),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }

        url = _YAHOO_DOWNLOAD_URL.format(ticker=ticker)
        try:
            response = httpx.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Yahoo Finance request failed for ticker=%s", ticker, exc_info=True)
            raise YahooFinanceError(f"Unable to fetch Yahoo Finance data for {ticker}") from exc

        content = response.text
        reader = csv.DictReader(StringIO(content))
        prices: List[Dict[str, float]] = []
        for row in reader:
            try:
                prices.append(
                    {
                        "date": row["Date"],
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "adj_close": float(row["Adj Close"]),
                        "volume": float(row["Volume"]),
                    }
                )
            except (KeyError, ValueError):
                continue

        if not prices:
            raise YahooFinanceError(f"Yahoo Finance returned no data for {ticker}")

        return PriceSeries(ticker=ticker, prices=prices)
