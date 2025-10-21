"""Repository abstractions for hypothesis records."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

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


class HypothesisRepository(ABC):
    """Abstract persistence interface for hypothesis records."""

    @abstractmethod
    async def save(self, record: HypothesisRecord) -> None:  # pragma: no cover - interface stub
        """Persist or update a hypothesis record."""
        ...

    @abstractmethod
    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:  # pragma: no cover - interface stub
        """Retrieve a hypothesis record by its identifier."""
        ...


class InMemoryHypothesisRepository(HypothesisRepository):
    """In-memory repository useful for testing and early iterations."""

    def __init__(self) -> None:
        self._storage: Dict[UUID, HypothesisRecord] = {}

    async def save(self, record: HypothesisRecord) -> None:
        self._storage[record.hypothesis_id] = record

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        return self._storage.get(hypothesis_id)


class FirestoreHypothesisRepository(HypothesisRepository):
    """Firestore-backed repository for hypothesis records."""

    def __init__(self, client: Any, collection: str = "hypotheses") -> None:
        self._client = client
        self._collection = collection

    async def save(self, record: HypothesisRecord) -> None:
        payload = {
            "workflow_id": record.workflow_id,
            "workflow_run_id": record.workflow_run_id,
            "request": record.request.model_dump(mode="json"),
            "status": record.status,
            "validation": record.validation.model_dump(mode="json"),
        }

        def _write() -> None:
            self._client.collection(self._collection).document(str(record.hypothesis_id)).set(payload)

        await asyncio.to_thread(_write)

    async def get(self, hypothesis_id: UUID) -> Optional[HypothesisRecord]:
        def _read() -> Any:
            return self._client.collection(self._collection).document(str(hypothesis_id)).get()

        snapshot = await asyncio.to_thread(_read)
        if snapshot is None or not getattr(snapshot, "exists", False):
            return None

        data = snapshot.to_dict() or {}
        if not data:
            return None

        request_payload = data.get("request")
        validation_payload = data.get("validation")
        if not request_payload or not validation_payload:
            return None

        return HypothesisRecord(
            hypothesis_id=hypothesis_id,
            workflow_id=data.get("workflow_id", ""),
            workflow_run_id=data.get("workflow_run_id", ""),
            request=HypothesisRequest.model_validate(request_payload),
            status=data.get("status", "unknown"),
            validation=ValidationSummary.model_validate(validation_payload),
        )
