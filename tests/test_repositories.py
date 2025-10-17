"""Tests for repository implementations."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from pytest import TempPathFactory

from hypothesis_agent.config import AppSettings
from hypothesis_agent.db.migrations import upgrade_database
from hypothesis_agent.db.session import Database
from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    MilestoneStatus,
    TimeHorizon,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.repositories.hypothesis_repository import HypothesisRecord, SqlAlchemyHypothesisRepository


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


@pytest.mark.asyncio
async def test_sqlalchemy_repository_persists_and_retrieves(tmp_path_factory: TempPathFactory) -> None:
    """Ensure the SQLAlchemy repository can round-trip records."""

    db_path = tmp_path_factory.mktemp("db") / "repo.db"
    settings = AppSettings(database_url=f"sqlite+aiosqlite:///{db_path}")
    database = Database.from_settings(settings)
    await upgrade_database(settings.database_url)
    repository = SqlAlchemyHypothesisRepository(database.session_factory)

    try:
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
    finally:
        await database.dispose()
