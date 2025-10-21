"""Tests for service layer behaviour."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    MilestoneStatus,
    ResumeRequest,
    TimeHorizon,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.repositories.hypothesis_repository import InMemoryHypothesisRepository
from hypothesis_agent.services.hypothesis_service import HypothesisService
from hypothesis_agent.workflows.hypothesis_workflow import (
    WorkflowExecutionDetails,
    WorkflowSubmissionResult,
)


def _make_validation_summary() -> ValidationSummary:
    milestones = [
        WorkflowMilestone(name="plan_generation", status=MilestoneStatus.COMPLETED, detail="Plan generated."),
        WorkflowMilestone(name="data_collection", status=MilestoneStatus.COMPLETED, detail="Data collected."),
        WorkflowMilestone(name="hybrid_analysis", status=MilestoneStatus.COMPLETED, detail="Hybrid analytics complete."),
        WorkflowMilestone(name="detailed_analysis", status=MilestoneStatus.COMPLETED, detail="Narrative synthesized."),
        WorkflowMilestone(name="report_generation", status=MilestoneStatus.COMPLETED, detail="Report compiled."),
        WorkflowMilestone(name="human_review", status=MilestoneStatus.COMPLETED, detail="Review skipped."),
        WorkflowMilestone(name="delivery", status=MilestoneStatus.COMPLETED, detail="Delivered to stakeholders."),
    ]
    return ValidationSummary(
        score=0.61,
        conclusion="Partially supported",
        confidence=0.57,
        evidence=[],
        current_stage="delivery",
        milestones=milestones,
    )


class _StubWorkflowClient:
    async def submit(self, hypothesis_id, hypothesis):
        return WorkflowSubmissionResult(
            workflow_id=f"wf-{hypothesis_id}",
            workflow_run_id=f"run-{hypothesis_id}",
            validation=_make_validation_summary(),
        )

    async def describe(self, workflow_id: str, workflow_run_id: str):
        return WorkflowExecutionDetails(
            status="RUNNING",
            history_length=10,
            milestones=_make_validation_summary().milestones,
        )

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved"):
        return _make_validation_summary()

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str):
        return _make_validation_summary()


@pytest.mark.asyncio
async def test_service_submit_persists_and_returns_response() -> None:
    """Submission should persist record and echo the workflow output."""

    service = HypothesisService(
        repository=InMemoryHypothesisRepository(),
        workflow_client=_StubWorkflowClient(),
    )
    request = HypothesisRequest(
        user_id="user-123",
        hypothesis_text="Margin expansion",
        entities=["Company X"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    response = await service.submit(request)
    fetched = await service.get(response.hypothesis_id)

    assert response.workflow_id.startswith("wf-")
    assert response.workflow_run_id.startswith("run-")
    assert fetched.hypothesis_id == response.hypothesis_id
    assert fetched.workflow_id.startswith("wf-")
    assert fetched.workflow_run_id.startswith("run-")
    assert fetched.status == "accepted"
    assert fetched.validation.conclusion == "Partially supported"
    assert fetched.validation.milestones[0].name == "plan_generation"

    status = await service.get_status(response.hypothesis_id)

    assert status.workflow_status == "RUNNING"
    assert status.workflow_history_length == 10
    assert status.validation.current_stage == "delivery"
    assert len(status.validation.milestones) == 7


@pytest.mark.asyncio
async def test_service_get_missing_raises_key_error() -> None:
    """Requesting an unknown hypothesis ID should raise a KeyError."""

    service = HypothesisService(
        repository=InMemoryHypothesisRepository(),
        workflow_client=_StubWorkflowClient(),
    )

    with pytest.raises(KeyError):
        await service.get(uuid4())


@pytest.mark.asyncio
async def test_service_resume_updates_status() -> None:
    repository = InMemoryHypothesisRepository()
    service = HypothesisService(
        repository=repository,
        workflow_client=_StubWorkflowClient(),
    )
    request = HypothesisRequest(
        user_id="user-456",
        hypothesis_text="Revenue acceleration",
        entities=["Company Z"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
        requires_human_review=True,
    )

    submission = await service.submit(request)
    resume_response = await service.resume(submission.hypothesis_id, ResumeRequest())

    assert resume_response.status == "completed"
    stored = await service.get(submission.hypothesis_id)
    assert stored.status == "completed"
