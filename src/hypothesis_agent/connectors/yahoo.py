"""Client for Yahoo Finance historical price data."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used for type hints only
    import yfinance as yf  # noqa: F401

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = "1d"


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

        if not ticker or ticker.strip() == "":
            raise ValueError("ticker must be provided for Yahoo Finance lookup")
        if start > end:
            raise ValueError("start date must be on or before end date")

        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())

        try:
            import yfinance as yf  # type: ignore import
        except ImportError as exc:  # pragma: no cover - dependency missing at runtime
            raise YahooFinanceError(
                "yfinance is not installed; install the optional dependency to use Yahoo Finance data"
            ) from exc

        try:
            frame = yf.download(
                tickers=ticker,
                start=start_dt,
                end=end_dt,
                interval=_DEFAULT_INTERVAL,
                progress=False,
                actions=False,
                auto_adjust=False,
                threads=False,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - library failure surface
            logger.warning("yfinance download failed for ticker=%s", ticker, exc_info=True)
            raise YahooFinanceError(f"Unable to fetch Yahoo Finance data for {ticker}") from exc

        if frame.empty:
            raise YahooFinanceError(f"Yahoo Finance returned no data for {ticker}")

        prices: List[Dict[str, float]] = []
        for timestamp, row in frame.iterrows():
            try:
                open_price = float(row["Open"])
                high_price = float(row["High"])
                low_price = float(row["Low"])
                close_price = float(row["Close"])
                adj_close = float(row["Adj Close"])
                volume = float(row["Volume"])
            except (KeyError, TypeError, ValueError):
                continue

            if any(math.isnan(value) for value in (open_price, high_price, low_price, close_price, adj_close, volume)):
                continue

            if hasattr(timestamp, "to_pydatetime"):
                date_value = timestamp.to_pydatetime().date().isoformat()
            else:
                date_value = str(timestamp)

            prices.append(
                {
                    "date": date_value,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "adj_close": adj_close,
                    "volume": volume,
                }
            )

        if not prices:
            raise YahooFinanceError(f"Yahoo Finance returned no usable rows for {ticker}")

        return PriceSeries(ticker=ticker, prices=prices)
