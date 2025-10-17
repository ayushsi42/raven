"""Temporal workflow client interface for hypothesis validation."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from temporalio.client import Client as TemporalClient

from hypothesis_agent.models.hypothesis import HypothesisRequest, ValidationSummary

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkflowSubmissionResult:
    """Result of initiating a validation workflow."""

    workflow_id: str
    workflow_run_id: str
    validation: ValidationSummary


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Facade over Temporal workflow interactions for hypothesis validation."""

    namespace: str
    task_queue: str
    workflow: str
    address: str
    temporal_client: Optional[TemporalClient] = None

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        """Submit a hypothesis to the Temporal workflow and return tracking metadata."""

        workflow_id = f"hypothesis-{hypothesis_id}"
        logger.info(
            "Submitting hypothesis=%s workflow_id=%s user=%s",
            hypothesis_id,
            workflow_id,
            hypothesis.user_id,
        )

        client = await self._ensure_client()

        try:
            run = await client.start_workflow(
                self.workflow,
                hypothesis.model_dump(mode="json"),
                id=workflow_id,
                task_queue=self.task_queue,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.exception("Failed to start Temporal workflow %s", workflow_id)
            raise

        # TODO: when workflow returns meaningful state, fetch it here.
        validation = ValidationSummary(
            score=0.0,
            conclusion="Pending analysis",
            confidence=0.0,
            evidence=[],
        )
        return WorkflowSubmissionResult(
            workflow_id=run.id,
            workflow_run_id=run.run_id,
            validation=validation,
        )

    async def _ensure_client(self) -> TemporalClient:
        if self.temporal_client is not None:
            return self.temporal_client

        logger.debug(
            "Connecting to Temporal server at %s (namespace=%s)",
            self.address,
            self.namespace,
        )
        self.temporal_client = await TemporalClient.connect(
            self.address,
            namespace=self.namespace,
        )
        return self.temporal_client

    async def close(self) -> None:
        """Close the underlying Temporal client if present."""

        if self.temporal_client is not None:
            await self.temporal_client.close()
            self.temporal_client = None
