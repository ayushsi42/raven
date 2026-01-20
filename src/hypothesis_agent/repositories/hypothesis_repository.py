"""Storage abstractions for hypothesis records and their associated metadata."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from uuid import UUID

from hypothesis_agent.models.hypothesis import HypothesisRecord


class HypothesisRepository(ABC):
    """Protocol for persisting and retrieving hypothesis records."""

    @abstractmethod
    async def save(self, record: HypothesisRecord) -> None:
        """Persist a hypothesis record."""

    @abstractmethod
    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        """Retrieve a hypothesis record by its unique identifier."""

    @abstractmethod
    async def list_by_user(self, user_id: str) -> List[HypothesisRecord]:
        """List all hypotheses submitted by a specific user."""


class InMemoryHypothesisRepository(HypothesisRepository):
    """Thread-safe in-memory cache for hypothesis records."""

    def __init__(self) -> None:
        self._storage: Dict[UUID, HypothesisRecord] = {}

    async def save(self, record: HypothesisRecord) -> None:
        """Store the record in the internal dictionary."""
        self._storage[record.hypothesis_id] = record

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        """Fetch the record from the internal dictionary."""
        return self._storage.get(hypothesis_id)

    async def list_by_user(self, user_id: str) -> List[HypothesisRecord]:
        """Filter records by user ID."""
        return [r for r in self._storage.values() if r.user_id == user_id]
