"""Repository abstractions for hypothesis records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hypothesis_agent.db.models import HypothesisRecordModel
from hypothesis_agent.models.hypothesis import HypothesisRequest, ValidationSummary


@dataclass(slots=True)
class HypothesisRecord:
    """Stored representation of a hypothesis submission."""

    hypothesis_id: UUID
    workflow_id: str
    workflow_run_id: str
    request: HypothesisRequest
    status: str
    validation: ValidationSummary


class HypothesisRepository:
    """Abstract persistence interface for hypothesis records."""

    async def save(self, record: HypothesisRecord) -> None:  # pragma: no cover - interface stub
        raise NotImplementedError

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:  # pragma: no cover - interface stub
        raise NotImplementedError


class InMemoryHypothesisRepository(HypothesisRepository):
    """In-memory repository useful for testing and early iterations."""

    def __init__(self) -> None:
        self._storage: Dict[UUID, HypothesisRecord] = {}

    async def save(self, record: HypothesisRecord) -> None:
        self._storage[record.hypothesis_id] = record

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        return self._storage.get(hypothesis_id)


class SqlAlchemyHypothesisRepository(HypothesisRepository):
    """SQLAlchemy-backed repository for hypothesis records."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save(self, record: HypothesisRecord) -> None:
        async with self._session_factory() as session:
            await session.merge(
                HypothesisRecordModel(
                    id=str(record.hypothesis_id),
                    user_id=record.request.user_id,
                    payload=record.request.model_dump(),
                    status=record.status,
                    validation=record.validation.model_dump(),
                    workflow_id=record.workflow_id,
                    workflow_run_id=record.workflow_run_id,
                )
            )
            await session.commit()

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        async with self._session_factory() as session:
            stmt = select(HypothesisRecordModel).where(HypothesisRecordModel.id == str(hypothesis_id))
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return HypothesisRecord(
                hypothesis_id=hypothesis_id,
                workflow_id=row.workflow_id,
                workflow_run_id=row.workflow_run_id,
                request=HypothesisRequest.model_validate(row.payload),
                status=row.status,
                validation=ValidationSummary.model_validate(row.validation),
            )
