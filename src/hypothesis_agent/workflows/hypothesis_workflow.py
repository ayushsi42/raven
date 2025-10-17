"""Temporal workflow client interface for hypothesis validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from temporalio.client import Client as TemporalClient

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkflowSubmissionResult:
    """Result of initiating a validation workflow."""

    workflow_id: str
    workflow_run_id: str
    validation: ValidationSummary


@dataclass(slots=True)
class WorkflowExecutionDetails:
    """Live execution metadata for a workflow."""

    status: str
    history_length: int | None = None
    milestones: List[WorkflowMilestone] | None = None


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Facade over Temporal workflow interactions for hypothesis validation."""

    namespace: str
    task_queue: str
    workflow: str
    address: str
    temporal_client: Optional[TemporalClient] = None
    _local_runs: Dict[Tuple[str, str], ValidationSummary] = field(default_factory=dict, init=False, repr=False)

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        """Submit a hypothesis to the Temporal workflow and return tracking metadata."""

        workflow_id = f"hypothesis-{hypothesis_id}"
        logger.info(
            "Submitting hypothesis=%s workflow_id=%s user=%s",
            hypothesis_id,
            workflow_id,
            hypothesis.user_id,
        )

        try:
            client = await self._ensure_client()
        except Exception:
            logger.warning(
                "Temporal connection unavailable for workflow_id=%s, running local pipeline instead",
                workflow_id,
                exc_info=True,
            )
            return self._record_local_run(workflow_id, hypothesis)

        try:
            handle = await client.start_workflow(
                self.workflow,
                hypothesis.model_dump(mode="json"),
                id=workflow_id,
                task_queue=self.task_queue,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "Failed to start Temporal workflow %s, running local pipeline instead",
                workflow_id,
                exc_info=True,
            )
            return self._record_local_run(workflow_id, hypothesis)

        workflow_run_id = handle.run_id

        try:
            payload = await handle.result()
            validation = ValidationSummary.model_validate(payload)
        except Exception:  # pragma: no cover - network dependent
            logger.warning(
                "Falling back to local validation summary for workflow_id=%s",
                workflow_id,
                exc_info=True,
            )
            validation = self._run_local_pipeline(hypothesis)
            self._local_runs[(workflow_id, workflow_run_id)] = validation
        return WorkflowSubmissionResult(
            workflow_id=handle.id,
            workflow_run_id=workflow_run_id,
            validation=validation,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        """Return execution details for a workflow."""

        local_key = (workflow_id, workflow_run_id)
        if local_key in self._local_runs:
            summary = self._local_runs[local_key]
            history_length = len(summary.milestones) if summary.milestones else None
            return WorkflowExecutionDetails(
                status="COMPLETED",
                history_length=history_length,
                milestones=summary.milestones,
            )

        client = await self._ensure_client()
        handle = client.get_workflow_handle(workflow_id, run_id=workflow_run_id)
        description = await handle.describe()
        info = description.workflow_execution_info
        status = getattr(info.status, "name", "STATUS_UNSPECIFIED")
        history_length = getattr(info, "history_length", None)
        milestones: List[WorkflowMilestone] | None = None
        try:
            milestone_payload = await handle.query("milestones")
        except Exception:  # pragma: no cover - query support optional
            logger.debug("Milestone query unavailable for %s", workflow_id, exc_info=True)
        else:
            if milestone_payload:
                milestones = [WorkflowMilestone.model_validate(m) for m in milestone_payload]
        return WorkflowExecutionDetails(
            status=status,
            history_length=history_length,
            milestones=milestones,
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

    def _fallback_summary(self, hypothesis: HypothesisRequest) -> ValidationSummary:
        """Synthesize a best-effort summary locally when Temporal result retrieval fails."""

        return self._run_local_pipeline(hypothesis)

    def _record_local_run(self, workflow_id: str, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        validation = self._run_local_pipeline(hypothesis)
        workflow_run_id = f"{workflow_id}-local"
        self._local_runs[(workflow_id, workflow_run_id)] = validation
        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            validation=validation,
        )

    def _run_local_pipeline(self, hypothesis: HypothesisRequest) -> ValidationSummary:
        orchestrator = LangGraphValidationOrchestrator()
        context: Dict[str, Any] = {}
        milestones: List[WorkflowMilestone] = []

        for stage in ["data_ingest", "preprocessing", "analysis", "sentiment", "modeling"]:
            stage_result = orchestrator.run_stage(stage, hypothesis, context)
            context = stage_result.context
            milestones.append(stage_result.milestone)

        final_result = orchestrator.run_stage("report_generation", hypothesis, context)
        summary = final_result.summary
        if summary is None:
            summary = ValidationSummary(
                score=0.5,
                conclusion="Pending synthesis",
                confidence=0.5,
                evidence=[],
                current_stage="report_generation",
                milestones=[],
            )
        milestones.append(final_result.milestone)
        return summary.model_copy(update={"milestones": milestones})
