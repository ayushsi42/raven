"""Tests for workflow client shims."""
from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from hypothesis_agent.config import AppSettings
from hypothesis_agent.connectors.news import NewsArticle
from hypothesis_agent.connectors.sec import FilingRecord
from hypothesis_agent.connectors.yahoo import PriceSeries
from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.models.hypothesis import HypothesisRequest, MilestoneStatus, TimeHorizon
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator
from hypothesis_agent.storage.artifact_store import ArtifactStore
from hypothesis_agent.workflows.hypothesis_workflow import (
    HypothesisWorkflowClient,
    StageExecutionResult,
    WorkflowExecutionDetails,
)


class _StubLLM(BaseLLM):
    def generate_data_plan(self, request: HypothesisRequest) -> list[str]:
        return [
            "Fetch historical prices",
            "Pull SEC filings",
            "Aggregate relevant news",
        ]

    def generate_analysis_plan(self, request: HypothesisRequest, data_overview: dict[str, object]) -> list[str]:
        return [
            "Calculate returns and volatility",
            "Summarize recent filings",
            "Score sentiment dispersion",
        ]

    def generate_detailed_analysis(self, request: HypothesisRequest, metrics_overview: dict[str, object]) -> str:
        return "The hypothesis remains plausible given momentum, filings cadence, and sentiment balance."

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: dict[str, object],
        analysis_summary: str,
        artifact_paths: list[str],
    ) -> dict[str, object]:
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
    def fetch_recent_filings(self, ticker: str, limit: int = 5) -> list[FilingRecord]:
        return [
            FilingRecord(accession="0000001", filing_type="10-K", filed="2025-01-10", url="https://example.com/10k", company_name=f"{ticker} Inc"),
            FilingRecord(accession="0000002", filing_type="10-Q", filed="2024-11-01", url="https://example.com/10q", company_name=f"{ticker} Inc"),
        ]


class _StubNews:
    def fetch_sentiment(self, tickers: list[str], limit: int = 5) -> list[NewsArticle]:
        return [
            NewsArticle(title="Growth outlook brightens", summary="", url="https://example.com/a", sentiment=0.4),
            NewsArticle(title="New product launch", summary="", url="https://example.com/b", sentiment=0.2),
        ]


class _StubTool:
    def __init__(self, sink: list[dict[str, object]]) -> None:
        self._sink = sink

    def invoke(self, payload: dict[str, object]) -> None:
        self._sink.append(payload)


class _StubToolSet:
    def __init__(self) -> None:
        self.invocations: list[dict[str, object]] = []

    def get_tool(self, name: str) -> _StubTool:
        return _StubTool(self.invocations)


@pytest.fixture
def stub_toolset() -> _StubToolSet:
    return _StubToolSet()


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_toolset: _StubToolSet) -> None:
    settings = AppSettings(
        notification_email="reports@example.com",
        artifact_store_path=str(tmp_path / "artifacts"),
    )

    def _factory() -> LangGraphValidationOrchestrator:
        artifact_store = ArtifactStore.from_path(settings.artifact_store_path)
        return LangGraphValidationOrchestrator(
            settings=settings,
            llm=_StubLLM(),
            yahoo_client=_StubYahoo(),
            sec_client=_StubSec(),
            news_client=_StubNews(),
            artifact_store=artifact_store,
            toolset=stub_toolset,
        )

    monkeypatch.setattr(
        "hypothesis_agent.workflows.hypothesis_workflow.LangGraphValidationOrchestrator",
        _factory,
    )


@pytest.mark.asyncio
async def test_workflow_client_runs_pipeline_without_temporal(stub_toolset: _StubToolSet) -> None:
    """The workflow client should run the LangGraph pipeline and persist milestones locally."""

    client = HypothesisWorkflowClient(
        namespace="test",
        task_queue="queue",
        workflow="TestWorkflow",
        address="test-address",
    )
    hypothesis = HypothesisRequest(
        user_id="user-123",
        hypothesis_text="Revenue grows 10%",
        entities=["Company X"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    hypothesis_id = uuid4()
    result = await client.submit(hypothesis_id, hypothesis)
    assert result.validation.current_stage == "plan_generation"
    assert result.validation.milestones[0].status == MilestoneStatus.PENDING

    execution = await _wait_for_status(client, result.workflow_id, result.workflow_run_id, "COMPLETED")
    final_summary = await client.fetch_summary(result.workflow_id, result.workflow_run_id)
    assert result.workflow_id == f"hypothesis-{hypothesis_id}"
    assert result.workflow_run_id.startswith(f"{result.workflow_id}-run")
    assert execution.status == "COMPLETED"
    assert execution.history_length == len(final_summary.milestones)
    assert execution.milestones is not None
    assert execution.milestones[0].name == "plan_generation"
    assert final_summary.milestones[-1].name == "delivery"
    assert final_summary.current_stage == "delivery"
    assert stub_toolset.invocations, "Delivery stage should invoke the notification tool"

    await client.close()


@pytest.mark.asyncio
async def test_workflow_client_handles_human_review() -> None:
    """Workflows that require human review should pause and resume locally."""

    client = HypothesisWorkflowClient(
        namespace="test",
        task_queue="queue",
        workflow="TestWorkflow",
        address="test-address",
    )
    hypothesis = HypothesisRequest(
        user_id="user-456",
        hypothesis_text="Margins expand 5%",
        entities=["Company Y"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
        requires_human_review=True,
    )

    hypothesis_id = uuid4()
    result = await client.submit(hypothesis_id, hypothesis)

    awaiting_execution = await _wait_for_status(client, result.workflow_id, result.workflow_run_id, "AWAITING_REVIEW")
    assert awaiting_execution.awaiting_review is True
    assert awaiting_execution.milestones[-1].status == MilestoneStatus.WAITING_REVIEW

    final_summary = await client.resume(result.workflow_id, result.workflow_run_id, "approved")
    assert final_summary.current_stage == "delivery"
    assert final_summary.milestones[-1].name == "delivery"
    assert final_summary.milestones[-2].status == MilestoneStatus.COMPLETED

    execution_after = await _wait_for_status(client, result.workflow_id, result.workflow_run_id, "COMPLETED")
    assert execution_after.awaiting_review is False
    assert execution_after.milestones[-1].name == "delivery"

    await client.close()


@pytest.mark.asyncio
async def test_workflow_client_can_cancel_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A running workflow should support user-triggered cancellation."""

    import time
    import hypothesis_agent.workflows.hypothesis_workflow as wf_module

    base_factory = wf_module.LangGraphValidationOrchestrator

    def slow_factory() -> LangGraphValidationOrchestrator:
        orchestrator = base_factory()
        original_run_stage = orchestrator.run_stage

        def slow_run_stage(stage: str, request: HypothesisRequest, context: dict[str, object]) -> StageExecutionResult:
            time.sleep(0.05)
            return original_run_stage(stage, request, context)

        orchestrator.run_stage = slow_run_stage  # type: ignore[assignment]
        return orchestrator

    monkeypatch.setattr(wf_module, "LangGraphValidationOrchestrator", slow_factory)

    client = HypothesisWorkflowClient(
        namespace="test",
        task_queue="queue",
        workflow="TestWorkflow",
        address="test-address",
    )
    hypothesis = HypothesisRequest(
        user_id="user-cancel",
        hypothesis_text="Cancel scenario",
        entities=["Company Z"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    hypothesis_id = uuid4()
    submission = await client.submit(hypothesis_id, hypothesis)
    await asyncio.sleep(0.01)

    summary = await client.cancel(submission.workflow_id, submission.workflow_run_id)
    assert summary.current_stage == "cancelled"
    assert any(m.status == MilestoneStatus.CANCELLED for m in summary.milestones)

    execution = await client.describe(submission.workflow_id, submission.workflow_run_id)
    assert execution.status == "CANCELLED"
    assert execution.awaiting_review is False

    await client.close()


async def _wait_for_status(
    client: HypothesisWorkflowClient,
    workflow_id: str,
    workflow_run_id: str,
    expected_status: str,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> WorkflowExecutionDetails:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_execution: WorkflowExecutionDetails | None = None
    while loop.time() < deadline:
        execution = await client.describe(workflow_id, workflow_run_id)
        last_execution = execution
        if execution.status == expected_status:
            return execution
        await asyncio.sleep(poll_interval)
    raise AssertionError(
        f"Workflow {workflow_id}/{workflow_run_id} did not reach status {expected_status}. Last seen: {last_execution}"
    )
