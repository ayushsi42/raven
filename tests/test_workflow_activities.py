"""Unit tests for workflow activities."""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.storage.artifact_store import ArtifactStore
from hypothesis_agent.workflows.activities import validation
from hypothesis_agent.workflows.activities.validation import (
    await_human_review,
    run_data_collection,
    run_delivery,
    run_detailed_analysis,
    run_hybrid_analysis,
    run_plan_generation,
    run_report_generation,
)


class _StubLLM(BaseLLM):
        def generate_data_plan(self, request: HypothesisRequest) -> List[str]:
            return ["Collect stub data"]

        def generate_analysis_plan(self, request: HypothesisRequest, data_overview: Dict[str, Any]) -> List[str]:
            return ["Run stub analysis"]

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

        def generate_analysis_code(
            self,
            *,
            request: HypothesisRequest,
            analysis_plan: List[Dict[str, Any]],
            data_artifacts: Dict[str, str],
            data_format: Dict[str, str],
            attempt: int,
            history: List[Dict[str, str]],
        ) -> str:
            return (
                """```python\n"""
                "result = {\n"
                "    \"steps\": [\n"
                "        {\n"
                "            \"name\": \"metric_snapshot\",\n"
                "            \"outputs\": [\n"
                "                {\"label\": \"Revenue Growth\", \"value\": 0.12},\n"
                "                {\"label\": \"Sentiment Score\", \"value\": 0.2}\n"
                "            ]\n"
                "        }\n"
                "    ],\n"
                "    \"aggregated\": {\n"
                "        \"revenue_growth\": 0.12,\n"
                "        \"sentiment_score\": 0.2\n"
                "    },\n"
                "    \"insights\": [\"Revenue acceleration remains healthy.\"],\n"
                "    \"artifacts\": []\n"
                "}\n"
                "print(\"RESULT::\" + json.dumps(result))\n"
                "```"""
            )


STUB_TOOL_RESPONSES: Dict[str, Dict[str, Any]] = {
    "YFINANCE_HISTORICAL_PRICES": {
        "symbol": "AAPL",
        "period": "1y",
        "interval": "1d",
        "data": {
            "2024-01-31": {"close": 100.0},
            "2024-02-29": {"close": 102.0},
            "2024-03-31": {"close": 104.5},
            "2024-04-30": {"close": 106.0},
            "2024-05-31": {"close": 108.5},
            "2024-06-30": {"close": 112.0},
        }
    },
    "YFINANCE_COMPANY_INFO": {
        "symbol": "AAPL",
        "operatingMargins": 0.21,
        "profitMargins": 0.15,
    },
    "YFINANCE_NEWS": {
        "symbol": "AAPL",
        "news": [
            {"title": "Positive News", "publisher": "Reuters", "providerPublishTime": 1700000000},
        ]
    },
    "YFINANCE_CASH_FLOW": {
        "symbol": "AAPL",
        "cash_flow": {
            "2023-12-31": {"Operating Cash Flow": 1200000.0},
        }
    },
    "YFINANCE_BALANCE_SHEET": {
        "symbol": "AAPL",
        "balance_sheet": {
            "2023-12-31": {"Total Assets": 5000000.0, "Total Liabilities": 2100000.0},
        }
    },
}


class _StubTool:
    def __init__(self, sink: List[Dict[str, Any]], slug: str, responses: Dict[str, Dict[str, Any]]) -> None:
        self._sink = sink
        self._slug = slug
        self._responses = responses

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {"slug": self._slug, "arguments": copy.deepcopy(payload)}
        self._sink.append(record)
        response = self._responses.get(self._slug, {"ok": True})
        return copy.deepcopy(response)


class _StubToolSet:
    def __init__(self, responses: Dict[str, Dict[str, Any]]) -> None:
        self.invocations: List[Dict[str, Any]] = []
        self._responses = responses

    def get_tool(self, name: str) -> _StubTool:
        return _StubTool(self.invocations, name, self._responses)


@pytest.fixture
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tuple[LangGraphValidationOrchestrator, _StubToolSet]:
    settings = AppSettings(
        artifact_store_path=str(tmp_path / "artifacts"),
    )
    artifact_store = ArtifactStore.from_path(settings.artifact_store_path)
    toolset = _StubToolSet(copy.deepcopy(STUB_TOOL_RESPONSES))
    orchestrator = LangGraphValidationOrchestrator(
        settings=settings,
        llm=_StubLLM(),
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
    fetch_slugs = {call["slug"] for call in toolset.invocations if call["slug"].startswith("YFINANCE")}
    assert "YFINANCE_HISTORICAL_PRICES" in fetch_slugs
    assert "YFINANCE_COMPANY_INFO" in fetch_slugs
    assert summary["score"] >= 0.0
    assert summary["confidence"] >= 0.0
    assert milestones[-1]["name"] == "delivery"
    assert milestones[-1]["detail"] == "Report available for download."
