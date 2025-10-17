"""Application service for managing hypothesis submissions."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from hypothesis_agent.models.hypothesis import HypothesisRequest, HypothesisResponse
from hypothesis_agent.repositories.hypothesis_repository import (
    HypothesisRecord,
    HypothesisRepository,
)
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


@dataclass(slots=True)
class HypothesisService:
    """Coordinate hypothesis submissions and persistence."""

    repository: HypothesisRepository
    workflow_client: HypothesisWorkflowClient

    async def submit(self, hypothesis: HypothesisRequest) -> HypothesisResponse:
        """Submit a hypothesis, persist it, and return the response contract."""

        hypothesis_id = uuid4()
        validation = await self.workflow_client.submit(hypothesis)
        record = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            request=hypothesis,
            status="accepted",
            validation=validation,
        )
        await self.repository.save(record)
        return HypothesisResponse(
            hypothesis_id=hypothesis_id,
            status=record.status,
            validation=record.validation,
        )

    async def get(self, hypothesis_id: UUID) -> HypothesisResponse:
        """Retrieve a previously submitted hypothesis record."""

        record = await self.repository.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        return HypothesisResponse(
            hypothesis_id=record.hypothesis_id,
            status=record.status,
            validation=record.validation,
        )
