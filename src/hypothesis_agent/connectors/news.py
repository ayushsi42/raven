"""Client for Alpha Vantage news sentiment API."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class NewsClientError(RuntimeError):
    """Raised when news sentiment data cannot be retrieved."""


@dataclass(slots=True)
class NewsArticle:
    """Minimal representation of a news article."""

    title: str
    summary: str
    url: str
    sentiment: float


@dataclass(slots=True)
class NewsClient:
    """Retrieve news sentiment using Alpha Vantage."""

    api_key: str
    timeout_seconds: float = 10.0

    def fetch_sentiment(self, tickers: List[str], limit: int = 5) -> List[NewsArticle]:
        topics = ",".join(tickers[:5]) or "market"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": topics,
            "sort": "RELEVANCE",
            "apikey": self.api_key,
        }

        try:
            response = httpx.get(_ALPHA_VANTAGE_URL, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Alpha Vantage request failed for tickers=%s", topics, exc_info=True)
            raise NewsClientError("Unable to fetch news sentiment data") from exc

        payload = response.json()
        feed = payload.get("feed", [])
        articles: List[NewsArticle] = []
        for item in feed:
            title = item.get("title", "")
            summary = item.get("summary", "")
            url = item.get("url", "")
            sentiment_str = item.get("overall_sentiment_score", "0")
            try:
                sentiment = float(sentiment_str)
            except (TypeError, ValueError):
                sentiment = 0.0
            if title and url:
                articles.append(NewsArticle(title=title, summary=summary, url=url, sentiment=sentiment))
            if len(articles) >= limit:
                break

        if not articles:
            raise NewsClientError("Alpha Vantage returned no news results")

        return articles
