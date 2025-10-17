"""Tests for workflow client shims."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    MilestoneStatus,
    TimeHorizon,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


def _make_validation_summary() -> ValidationSummary:
    milestones = [
        WorkflowMilestone(name="data_ingest", status=MilestoneStatus.COMPLETED, detail="Data collected."),
        WorkflowMilestone(name="preprocessing", status=MilestoneStatus.COMPLETED, detail="Data normalized."),
        WorkflowMilestone(name="analysis", status=MilestoneStatus.COMPLETED, detail="Diagnostics computed."),
        WorkflowMilestone(name="sentiment", status=MilestoneStatus.COMPLETED, detail="Sentiment scored."),
        WorkflowMilestone(name="modeling", status=MilestoneStatus.COMPLETED, detail="Scenarios modeled."),
        WorkflowMilestone(name="report_generation", status=MilestoneStatus.COMPLETED, detail="Report compiled."),
    ]
    return ValidationSummary(
        score=0.61,
        conclusion="Partially supported",
        confidence=0.57,
        evidence=[],
        current_stage="report_generation",
        milestones=milestones,
    )


class _StubWorkflowHandle:
    def __init__(self, workflow_id: str, run_id: str, summary: ValidationSummary) -> None:
        self.id = workflow_id
        self.run_id = run_id
        self._description = type(
            "Description",
            (),
            {
                "workflow_execution_info": type(
                    "Info",
                    (),
                    {
                        "status": type("Status", (), {"name": "RUNNING"})(),
                        "history_length": 3,
                    },
                )()
            },
        )()
        self._summary = summary.model_dump(mode="json")
        self._milestones = [milestone.model_dump(mode="json") for milestone in summary.milestones]

    async def describe(self):
        return self._description

    async def result(self):
        return self._summary

    async def query(self, name: str):
        if name == "milestones":
            return self._milestones
        raise ValueError(f"Unknown query {name}")


class _StubTemporalClient:
    def __init__(self) -> None:
        self._summaries: dict[str, ValidationSummary] = {}

    async def start_workflow(self, workflow: str, payload: dict, *, id: str, task_queue: str):
        summary = _make_validation_summary()
        self._summaries[id] = summary
        return _StubWorkflowHandle(id, f"{id}-run", summary)

    async def close(self) -> None:  # pragma: no cover - stub close
        return None

    def get_workflow_handle(self, workflow_id: str, run_id: str) -> _StubWorkflowHandle:
        summary = self._summaries.get(workflow_id, _make_validation_summary())
        return _StubWorkflowHandle(workflow_id, run_id, summary)


@pytest.mark.asyncio
async def test_workflow_client_returns_placeholder_validation() -> None:
    """Ensure the stub client returns the expected placeholder validation summary."""

    client = HypothesisWorkflowClient(
        namespace="test",
        task_queue="queue",
        workflow="TestWorkflow",
        address="test-address",
        temporal_client=_StubTemporalClient(),
    )
    hypothesis = HypothesisRequest(
        user_id="user-123",
        hypothesis_text="Revenue grows 10%",
        entities=["Company X"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    hypothesis_id = uuid4()
    result = await client.submit(hypothesis_id, hypothesis)
    execution = await client.describe(result.workflow_id, result.workflow_run_id)

    assert result.validation.conclusion == "Partially supported"
    assert result.validation.score == pytest.approx(0.61)
    assert result.validation.confidence == pytest.approx(0.57)
    assert result.validation.evidence == []
    assert result.validation.milestones[-1].name == "report_generation"
    assert result.workflow_id == f"hypothesis-{hypothesis_id}"
    assert result.workflow_run_id == f"{result.workflow_id}-run"
    assert execution.status == "RUNNING"
    assert execution.history_length == 3
    assert execution.milestones is not None
    assert len(execution.milestones) == 6
    assert execution.milestones[0].name == "data_ingest"
