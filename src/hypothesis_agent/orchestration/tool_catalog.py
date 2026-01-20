"""Tool definitions catalog for YFinance data sources.

Provides structured metadata for the available Yahoo Finance data tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(slots=True)
class ToolDefinition:
    """Structured metadata about a Yahoo Finance tool."""

    slug: str
    description: str
    inputs: List[str]
    outputs: List[str]


# YFinance Tools Definitions
YFINANCE_TOOLS: Dict[str, ToolDefinition] = {
    "YFINANCE_COMPANY_INFO": ToolDefinition(
        slug="YFINANCE_COMPANY_INFO",
        description="Get comprehensive company information including fundamentals, ratios, and key metrics.",
        inputs=["symbol"],
        outputs=["shortName", "sector", "industry", "marketCap", "profitMargins", "operatingMargins", "revenueGrowth"],
    ),
    "YFINANCE_HISTORICAL_PRICES": ToolDefinition(
        slug="YFINANCE_HISTORICAL_PRICES",
        description="Get historical OHLCV price data for a given period and interval.",
        inputs=["symbol", "period", "interval"],
        outputs=["open", "high", "low", "close", "volume"],
    ),
    "YFINANCE_FINANCIALS": ToolDefinition(
        slug="YFINANCE_FINANCIALS",
        description="Get income statement / financials data (annual or quarterly).",
        inputs=["symbol", "quarterly"],
        outputs=["Total Revenue", "Net Income", "Operating Income", "Gross Profit"],
    ),
    "YFINANCE_BALANCE_SHEET": ToolDefinition(
        slug="YFINANCE_BALANCE_SHEET",
        description="Get balance sheet data (annual or quarterly).",
        inputs=["symbol", "quarterly"],
        outputs=["Total Assets", "Total Liabilities", "Stockholders Equity", "Total Debt"],
    ),
    "YFINANCE_CASH_FLOW": ToolDefinition(
        slug="YFINANCE_CASH_FLOW",
        description="Get cash flow statement data (annual or quarterly).",
        inputs=["symbol", "quarterly"],
        outputs=["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"],
    ),
    "YFINANCE_EARNINGS": ToolDefinition(
        slug="YFINANCE_EARNINGS",
        description="Get annual and quarterly earnings data.",
        inputs=["symbol"],
        outputs=["Revenue", "Earnings"],
    ),
    "YFINANCE_RECOMMENDATIONS": ToolDefinition(
        slug="YFINANCE_RECOMMENDATIONS",
        description="Get analyst recommendations and rating changes.",
        inputs=["symbol"],
        outputs=["date", "firm", "toGrade", "fromGrade", "action"],
    ),
    "YFINANCE_NEWS": ToolDefinition(
        slug="YFINANCE_NEWS",
        description="Get recent news articles for the ticker.",
        inputs=["symbol"],
        outputs=["title", "publisher", "link", "providerPublishTime"],
    ),
    "YFINANCE_HOLDERS": ToolDefinition(
        slug="YFINANCE_HOLDERS",
        description="Get institutional and major holders information.",
        inputs=["symbol"],
        outputs=["major_holders", "institutional_holders"],
    ),
    "YFINANCE_DIVIDENDS": ToolDefinition(
        slug="YFINANCE_DIVIDENDS",
        description="Get dividend history.",
        inputs=["symbol"],
        outputs=["date", "dividend"],
    ),
    "YFINANCE_SPLITS": ToolDefinition(
        slug="YFINANCE_SPLITS",
        description="Get stock split history.",
        inputs=["symbol"],
        outputs=["date", "split_ratio"],
    ),
    "YFINANCE_OPTIONS": ToolDefinition(
        slug="YFINANCE_OPTIONS",
        description="Get available options expiration dates.",
        inputs=["symbol"],
        outputs=["expiration_dates"],
    ),
}


def load_tool_catalog() -> Dict[str, ToolDefinition]:
    """Load the YFinance tool catalog.
    
    Returns:
        Dictionary mapping tool slugs to their definitions.
    """
    return YFINANCE_TOOLS.copy()
