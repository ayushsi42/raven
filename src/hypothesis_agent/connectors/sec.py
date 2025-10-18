"""Client for SEC filings metadata."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/include/ticker.txt"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"


class SecFilingsError(RuntimeError):
    """Raised when SEC filings cannot be retrieved."""


@dataclass(slots=True)
class FilingRecord:
    """Minimal filing metadata."""

    accession: str
    filing_type: str
    filed: str
    url: str
    company_name: str | None = None


@dataclass(slots=True)
class SecFilingsClient:
    """Retrieve filings data published by the SEC."""

    user_agent: str
    timeout_seconds: float = 10.0

    def fetch_recent_filings(self, ticker: str, limit: int = 5) -> List[FilingRecord]:
        cik = self._lookup_cik(ticker)
        if cik is None:
            raise SecFilingsError(f"Unknown ticker {ticker} for SEC filings")

        headers = {"User-Agent": self.user_agent}
        url = _SUBMISSIONS_URL.format(cik=int(cik))
        try:
            response = httpx.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("SEC submissions request failed for ticker=%s", ticker, exc_info=True)
            raise SecFilingsError(f"Unable to fetch SEC filings for {ticker}") from exc

        data = response.json()
        filings_data = data.get("filings", {}).get("recent", {})
        company_name = data.get("entityName")
        records: List[FilingRecord] = []
        accessions = filings_data.get("accessionNumber", [])
        forms = filings_data.get("form", [])
        filing_dates = filings_data.get("filingDate", [])
        primary_docs = filings_data.get("primaryDocument", [])

        for accession, filing_type, filing_date, primary_doc in zip(accessions, forms, filing_dates, primary_docs):
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{primary_doc}"
            records.append(
                FilingRecord(
                    accession=accession,
                    filing_type=filing_type,
                    filed=filing_date,
                    url=url,
                    company_name=company_name,
                )
            )
            if len(records) >= limit:
                break

        if not records:
            raise SecFilingsError(f"SEC returned no filings for {ticker}")

        return records

    def get_cik(self, ticker: str) -> str | None:
        """Expose ticker-to-CIK resolution for downstream consumers."""

        try:
            return self._lookup_cik(ticker)
        except SecFilingsError:
            return None

    @staticmethod
    @lru_cache(maxsize=1)
    def _ticker_map() -> Dict[str, str]:
        try:
            response = httpx.get(_TICKER_MAP_URL, timeout=10.0)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("SEC ticker map request failed", exc_info=True)
            raise SecFilingsError("Unable to download SEC ticker map") from exc

        mapping: Dict[str, str] = {}
        for line in response.text.splitlines():
            if "|" not in line:
                continue
            ticker, cik = line.split("|", 1)
            mapping[ticker.strip().upper()] = cik.strip()
        return mapping

    @classmethod
    def _lookup_cik(cls, ticker: str) -> str | None:
        mapping = cls._ticker_map()
        return mapping.get(ticker.upper())
