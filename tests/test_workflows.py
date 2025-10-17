"""Tests for workflow client shims."""
from __future__ import annotations

from datetime import date

import pytest

from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


@pytest.mark.asyncio
async def test_workflow_client_returns_placeholder_validation() -> None:
    """Ensure the stub client returns the expected placeholder validation summary."""

    client = HypothesisWorkflowClient(namespace="test", task_queue="queue")
    hypothesis = HypothesisRequest(
        user_id="user-123",
        hypothesis_text="Revenue grows 10%",
        entities=["Company X"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    result = await client.submit(hypothesis)

    assert result.conclusion == "Pending analysis"
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.evidence == []
