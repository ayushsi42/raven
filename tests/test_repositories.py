"""Tests for repository implementations."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from hypothesis_agent.models.hypothesis import (
    HypothesisRecord,
    HypothesisRequest,
    MilestoneStatus,
    TimeHorizon,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.repositories.hypothesis_repository import (
    InMemoryHypothesisRepository,
)


def _make_validation_summary() -> ValidationSummary:
    milestones = [
        WorkflowMilestone(name="plan_generation", status=MilestoneStatus.COMPLETED, detail="Plan generated."),
        WorkflowMilestone(name="data_collection", status=MilestoneStatus.COMPLETED, detail="Data collected."),
        WorkflowMilestone(name="hybrid_analysis", status=MilestoneStatus.COMPLETED, detail="Hybrid analytics complete."),
        WorkflowMilestone(name="detailed_analysis", status=MilestoneStatus.COMPLETED, detail="Narrative synthesized."),
        WorkflowMilestone(name="report_generation", status=MilestoneStatus.COMPLETED, detail="Report compiled."),
        WorkflowMilestone(name="human_review", status=MilestoneStatus.COMPLETED, detail="Review skipped."),
        WorkflowMilestone(name="delivery", status=MilestoneStatus.COMPLETED, detail="Report available for download."),
    ]
    return ValidationSummary(
        score=0.61,
        conclusion="Partially supported",
        confidence=0.57,
        evidence=[],
        current_stage="delivery",
        milestones=milestones,
    )


@pytest.mark.asyncio
async def test_in_memory_repository_persists_and_retrieves() -> None:
    """Ensure the in-memory repository can round-trip records."""

    repository = InMemoryHypothesisRepository()

    hypothesis_request = HypothesisRequest(
        user_id="user-456",
        hypothesis_text="Revenue growth stabilizes",
        entities=["Company Y"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )
    record = HypothesisRecord(
        hypothesis_id=uuid4(),
        workflow_id="wf-123",
        workflow_run_id="run-123",
        request=hypothesis_request,
        status="accepted",
        validation=_make_validation_summary(),
    )

    await repository.save(record)
    fetched = await repository.get(record.hypothesis_id)

    assert fetched is not None
    assert fetched.hypothesis_id == record.hypothesis_id
    assert fetched.workflow_id == record.workflow_id
    assert fetched.workflow_run_id == record.workflow_run_id
    assert fetched.request == record.request
    assert fetched.user_id == "user-456"


@pytest.mark.asyncio
async def test_in_memory_repository_lists_by_user() -> None:
    """Ensure records can be filtered by user ID."""

    repository = InMemoryHypothesisRepository()
    uid = "user-789"
    
    for i in range(3):
        req = HypothesisRequest(
            user_id=uid if i < 2 else "other-user",
            hypothesis_text=f"Hypothesis {i}",
            time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
        )
        rec = HypothesisRecord(
            hypothesis_id=uuid4(),
            workflow_id=f"wf-{i}",
            workflow_run_id=f"run-{i}",
            request=req,
            status="accepted",
            validation=_make_validation_summary(),
        )
        await repository.save(rec)

    user_records = await repository.list_by_user(uid)
    assert len(user_records) == 2
    assert all(r.user_id == uid for r in user_records)
