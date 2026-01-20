"""Tests for workflow client shims."""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from hypothesis_agent.config import AppSettings
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
        return ["Collect historical prices", "Gather filings", "Summarise news sentiment"]

    def generate_analysis_plan(self, request: HypothesisRequest, data_overview: dict[str, object]) -> list[str]:
        return ["Run price momentum", "Aggregate fundamentals", "Compile sentiment"]

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

    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: list[dict[str, object]],
        data_artifacts: dict[str, str],
        data_format: dict[str, str],
        attempt: int,
        history: list[dict[str, str]],
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


STUB_TOOL_RESPONSES: dict[str, dict[str, object]] = {
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
    def __init__(self, sink: list[dict[str, object]], slug: str, responses: dict[str, dict[str, object]]) -> None:
        self._sink = sink
        self._slug = slug
        self._responses = responses

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        record = {"slug": self._slug, "arguments": copy.deepcopy(payload)}
        self._sink.append(record)
        response = self._responses.get(self._slug, {"ok": True})
        return copy.deepcopy(response)


class _StubToolSet:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.invocations: list[dict[str, object]] = []
        self._responses = responses

    def configure(self, responses: dict[str, dict[str, object]]) -> None:
        self._responses = responses

    def get_tool(self, name: str) -> _StubTool:
        return _StubTool(self.invocations, name, self._responses)


@pytest.fixture
def stub_toolset() -> _StubToolSet:
    return _StubToolSet(copy.deepcopy(STUB_TOOL_RESPONSES))


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_toolset: _StubToolSet) -> None:
    settings = AppSettings(
        artifact_store_path=str(tmp_path / "artifacts"),
    )

    def _factory() -> LangGraphValidationOrchestrator:
        artifact_store = ArtifactStore.from_path(settings.artifact_store_path)
        stub_toolset.invocations.clear()
        stub_toolset.configure(copy.deepcopy(STUB_TOOL_RESPONSES))
        return LangGraphValidationOrchestrator(
            settings=settings,
            llm=_StubLLM(),
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
    assert final_summary.milestones[-1].detail == "Report available for download."

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
    human_milestone = next(m for m in awaiting_execution.milestones if m.name == "human_review")
    assert human_milestone.status == MilestoneStatus.WAITING_REVIEW

    final_summary = await client.resume(result.workflow_id, result.workflow_run_id, "approved")
    assert final_summary.current_stage == "delivery"
    assert final_summary.milestones[-1].name == "delivery"
    assert final_summary.milestones[-1].detail == "Report available for download."
    human_milestone_after = next(m for m in final_summary.milestones if m.name == "human_review")
    assert human_milestone_after.status == MilestoneStatus.COMPLETED

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
