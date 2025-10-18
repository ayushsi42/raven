"""Unit tests for Temporal workflow activities."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.connectors.news import NewsArticle
from hypothesis_agent.connectors.sec import FilingRecord
from hypothesis_agent.connectors.yahoo import PriceSeries
from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.storage.artifact_store import ArtifactStore
from hypothesis_agent.workflows.activities import validation
from hypothesis_agent.workflows.activities.validation import (
    await_human_review,
    run_analysis_planning,
    run_data_collection,
    run_delivery,
    run_detailed_analysis,
    run_hybrid_analysis,
    run_plan_generation,
    run_report_generation,
)


class _StubLLM(BaseLLM):
    def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
        return [
            "Fetch historical prices",
            "Pull SEC filings",
            "Aggregate relevant news",
        ]

    def generate_analysis_plan(self, request: HypothesisRequest, data_overview: Dict[str, Any]) -> List[str]:
        return [
            "Calculate returns and volatility",
            "Summarize recent filings",
            "Score sentiment dispersion",
        ]

    def generate_detailed_analysis(self, request: HypothesisRequest, metrics_overview: Dict[str, Any]) -> str:
        return "The hypothesis remains plausible given momentum, filings cadence, and sentiment balance."

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: Dict[str, Any],
        analysis_summary: str,
        artifact_paths: List[str],
    ) -> Dict[str, Any]:
        return {
            "executive_summary": "Moderate support for the investment thesis.",
            "key_findings": ["Momentum positive", "Filings cadence stable", "Sentiment constructive"],
            "risks": ["Macro slowdown"],
            "next_steps": ["Monitor earnings guidance"],
        }


class _StubYahoo:
    def fetch_daily_prices(self, ticker: str, start: date, end: date) -> PriceSeries:
        prices = [
            {"date": "2025-01-01", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "adj_close": 101.0, "volume": 1_000_000},
            {"date": "2025-01-02", "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.5, "adj_close": 102.5, "volume": 1_200_000},
            {"date": "2025-01-03", "open": 102.5, "high": 104.0, "low": 101.5, "close": 103.5, "adj_close": 103.5, "volume": 1_100_000},
        ]
        return PriceSeries(ticker=ticker, prices=prices)


class _StubSec:
    def fetch_recent_filings(self, ticker: str, limit: int = 5) -> List[FilingRecord]:
        return [
            FilingRecord(accession="0000001", filing_type="10-K", filed="2025-01-10", url="https://example.com/10k", company_name=f"{ticker} Inc"),
            FilingRecord(accession="0000002", filing_type="10-Q", filed="2024-11-01", url="https://example.com/10q", company_name=f"{ticker} Inc"),
        ]


class _StubNews:
    def fetch_sentiment(self, tickers: List[str], limit: int = 5) -> List[NewsArticle]:
        return [
            NewsArticle(title="Growth outlook brightens", summary="", url="https://example.com/a", sentiment=0.4),
            NewsArticle(title="New product launch", summary="", url="https://example.com/b", sentiment=0.2),
        ]


class _StubTool:
    def __init__(self, sink: List[Dict[str, Any]]) -> None:
        self._sink = sink

    def invoke(self, payload: Dict[str, Any]) -> None:
        self._sink.append(payload)


class _StubToolSet:
    def __init__(self) -> None:
        self.invocations: List[Dict[str, Any]] = []

    def get_tool(self, name: str) -> _StubTool:
        return _StubTool(self.invocations)


@pytest.fixture
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tuple[LangGraphValidationOrchestrator, _StubToolSet]:
    settings = AppSettings(
        notification_email="reports@example.com",
        artifact_store_path=str(tmp_path / "artifacts"),
    )
    artifact_store = ArtifactStore.from_path(settings.artifact_store_path)
    toolset = _StubToolSet()
    orchestrator = LangGraphValidationOrchestrator(
        settings=settings,
        llm=_StubLLM(),
        yahoo_client=_StubYahoo(),
        sec_client=_StubSec(),
        news_client=_StubNews(),
        artifact_store=artifact_store,
        toolset=toolset,
    )
    monkeypatch.setattr(validation, "_ORCHESTRATOR", orchestrator)
    return orchestrator, toolset


@pytest.mark.asyncio
async def test_pipeline_stages_execute(stub_orchestrator) -> None:
    _orchestrator, toolset = stub_orchestrator
    payload_request = HypothesisRequest(
        user_id="user-789",
        hypothesis_text="Short hypothesis",
        entities=["AAPL"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    context: Dict[str, Any] = {"metadata": {"workflow_id": "wf-local", "requires_human_review": False}}
    milestones: List[Dict[str, Any]] = []

    for activity in [
        run_plan_generation,
        run_data_collection,
        run_analysis_planning,
        run_hybrid_analysis,
        run_detailed_analysis,
    ]:
        stage_result = await activity({
            "request": payload_request.model_dump(mode="json"),
            "context": context,
            "milestones": milestones,
        })
        context = stage_result["context"]
        milestones.append(stage_result["milestone"])

    report_stage = await run_report_generation({
        "request": payload_request.model_dump(mode="json"),
        "context": context,
        "milestones": milestones,
    })
    summary = report_stage["summary"]
    assert summary is not None
    assert summary["current_stage"] == "report_generation"
    context = report_stage["context"]
    milestones.append(report_stage["milestone"])

    human_stage = await await_human_review({
        "request": payload_request.model_dump(mode="json"),
        "context": context,
        "milestones": milestones,
    })
    assert human_stage["milestone"]["status"] == "completed"
    context = human_stage["context"]
    milestones.append(human_stage["milestone"])

    delivery_stage = await run_delivery({
        "request": payload_request.model_dump(mode="json"),
        "context": context,
        "milestones": milestones,
    })
    context = delivery_stage["context"]
    milestones.append(delivery_stage["milestone"])

    pdf_path = Path(context["report"]["pdf_path"].replace("file://", ""))
    assert pdf_path.exists()
    assert toolset.invocations and toolset.invocations[0]["attachments"][0] == str(pdf_path)
    assert summary["score"] >= 0.0
    assert summary["confidence"] >= 0.0
    assert milestones[-1]["name"] == "delivery"
