"""Temporal workflow client interface for hypothesis validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from hypothesis_agent.models.hypothesis import HypothesisRequest, ValidationSummary

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Facade over Temporal workflow interactions for hypothesis validation."""

    namespace: str
    task_queue: str

    async def submit(self, hypothesis: HypothesisRequest) -> ValidationSummary:
        """Submit a hypothesis to the Temporal workflow.

        This implementation is a stub that will be replaced with actual Temporal
        SDK integration. For now, it logs the submission and returns a placeholder
        validation summary.
        """

        logger.info(
            "Submitting hypothesis for user=%s to namespace=%s queue=%s",
            hypothesis.user_id,
            self.namespace,
            self.task_queue,
        )
        return ValidationSummary(
            score=0.0,
            conclusion="Pending analysis",
            confidence=0.0,
            evidence=[],
        )
