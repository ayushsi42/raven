"""Tests for repository implementations."""
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
from hypothesis_agent.repositories.hypothesis_repository import (
    FirestoreHypothesisRepository,
    HypothesisRecord,
)


class _FakeSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return self._data


class _FakeDocumentReference:
    def __init__(self, storage: dict, key: str) -> None:
        self._storage = storage
        self._key = key

    def set(self, data: dict) -> None:
        self._storage[self._key] = data

    def get(self) -> _FakeSnapshot:
        return _FakeSnapshot(self._storage.get(self._key))


class _FakeCollection:
    def __init__(self, storage: dict) -> None:
        self._storage = storage

    def document(self, key: str) -> _FakeDocumentReference:
        return _FakeDocumentReference(self._storage, key)


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._storage: dict[str, dict[str, dict]] = {}

    def collection(self, name: str) -> _FakeCollection:
        bucket = self._storage.setdefault(name, {})
        return _FakeCollection(bucket)


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
async def test_firestore_repository_persists_and_retrieves() -> None:
    """Ensure the Firestore repository can round-trip records."""

    firestore_client = _FakeFirestoreClient()
    repository = FirestoreHypothesisRepository(firestore_client, "hypotheses")

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
