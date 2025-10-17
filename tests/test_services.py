"""Tests for service layer behaviour."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.repositories.hypothesis_repository import InMemoryHypothesisRepository
from hypothesis_agent.services.hypothesis_service import HypothesisService
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


@pytest.mark.asyncio
async def test_service_submit_persists_and_returns_response() -> None:
    """Submission should persist record and echo the workflow output."""

    service = HypothesisService(
        repository=InMemoryHypothesisRepository(),
        workflow_client=HypothesisWorkflowClient(namespace="ns", task_queue="queue"),
    )
    request = HypothesisRequest(
        user_id="user-123",
        hypothesis_text="Margin expansion",
        entities=["Company X"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    response = await service.submit(request)
    fetched = await service.get(response.hypothesis_id)

    assert fetched.hypothesis_id == response.hypothesis_id
    assert fetched.status == "accepted"
    assert fetched.validation.conclusion == "Pending analysis"


@pytest.mark.asyncio
async def test_service_get_missing_raises_key_error() -> None:
    """Requesting an unknown hypothesis ID should raise a KeyError."""

    service = HypothesisService(
        repository=InMemoryHypothesisRepository(),
        workflow_client=HypothesisWorkflowClient(namespace="ns", task_queue="queue"),
    )

    with pytest.raises(KeyError):
        await service.get(uuid4())
