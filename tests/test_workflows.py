"""Tests for workflow client shims."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


class _StubWorkflowHandle:
    def __init__(self, workflow_id: str, run_id: str) -> None:
        self.id = workflow_id
        self.run_id = run_id


class _StubTemporalClient:
    async def start_workflow(self, workflow: str, payload: dict, *, id: str, task_queue: str):
        return _StubWorkflowHandle(id, f"{id}-run")

    async def close(self) -> None:  # pragma: no cover - stub close
        return None


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

    assert result.validation.conclusion == "Pending analysis"
    assert result.validation.score == 0.0
    assert result.validation.confidence == 0.0
    assert result.validation.evidence == []
    assert result.workflow_id == f"hypothesis-{hypothesis_id}"
    assert result.workflow_run_id == f"{result.workflow_id}-run"
