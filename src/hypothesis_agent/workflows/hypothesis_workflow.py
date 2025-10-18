"""Temporal workflow client interface for hypothesis validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from temporalio.client import Client as TemporalClient

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    MilestoneStatus,
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
    awaiting_review: bool = False


@dataclass(slots=True)
class LocalWorkflowRun:
    """State retained for locally simulated workflow executions."""

    hypothesis: HypothesisRequest
    context: Dict[str, Any]
    summary: ValidationSummary
    awaiting_review: bool
    pending_summary_base: ValidationSummary | None = None
    report_milestone: WorkflowMilestone | None = None


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Facade over Temporal workflow interactions for hypothesis validation."""

    namespace: str
    task_queue: str
    workflow: str
    address: str
    temporal_client: Optional[TemporalClient] = None
    _local_runs: Dict[Tuple[str, str], LocalWorkflowRun] = field(default_factory=dict, init=False, repr=False)

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
            preview_run = self._execute_local_pipeline(workflow_id, hypothesis)
            validation = preview_run.summary
        except Exception:  # pragma: no cover - preview failures should not block submission
            logger.warning(
                "Failed to generate preview summary for workflow_id=%s",
                workflow_id,
                exc_info=True,
            )
            validation = ValidationSummary(
                score=0.0,
                conclusion="Workflow started",
                confidence=0.0,
                evidence=[],
                current_stage="data_ingest",
                milestones=[],
            )
        return WorkflowSubmissionResult(
            workflow_id=handle.id,
            workflow_run_id=workflow_run_id,
            validation=validation,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        """Return execution details for a workflow."""

        local_key = (workflow_id, workflow_run_id)
        if local_key in self._local_runs:
            local_run = self._local_runs[local_key]
            summary = local_run.summary
            history_length = len(summary.milestones) if summary.milestones else None
            status = "AWAITING_REVIEW" if local_run.awaiting_review else "COMPLETED"
            return WorkflowExecutionDetails(
                status=status,
                history_length=history_length,
                milestones=summary.milestones,
                awaiting_review=local_run.awaiting_review,
            )

        client = await self._ensure_client()
        handle = client.get_workflow_handle(workflow_id, run_id=workflow_run_id)
        description = await handle.describe()
        info = description.workflow_execution_info
        status = getattr(info.status, "name", "STATUS_UNSPECIFIED")
        history_length = getattr(info, "history_length", None)
        milestones: List[WorkflowMilestone] | None = None
        awaiting_review = False
        try:
            milestone_payload = await handle.query("milestones")
        except Exception:  # pragma: no cover - query support optional
            logger.debug("Milestone query unavailable for %s", workflow_id, exc_info=True)
        else:
            if milestone_payload:
                milestones = [WorkflowMilestone.model_validate(m) for m in milestone_payload]
                awaiting_review = any(m.status == MilestoneStatus.WAITING_REVIEW for m in milestones)
        return WorkflowExecutionDetails(
            status=status,
            history_length=history_length,
            milestones=milestones,
            awaiting_review=awaiting_review,
        )

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved") -> ValidationSummary:
        """Resume a paused workflow after a human review decision."""

        local_key = (workflow_id, workflow_run_id)
        if local_key in self._local_runs:
            local_run = self._local_runs[local_key]
            if not local_run.awaiting_review:
                return local_run.summary

            final_milestones: List[WorkflowMilestone] = []
            for milestone in local_run.summary.milestones:
                if milestone.name == "human_review":
                    final_milestones.append(
                        milestone.model_copy(
                            update={
                                "status": MilestoneStatus.COMPLETED,
                                "detail": f"Human decision: {decision}.",
                            }
                        )
                    )
                else:
                    final_milestones.append(milestone)

            if local_run.report_milestone:
                final_milestones.append(local_run.report_milestone)

            base_summary = local_run.pending_summary_base or local_run.summary
            final_summary = base_summary.model_copy(
                update={
                    "current_stage": "report_generation",
                    "milestones": final_milestones,
                }
            )

            local_run.summary = final_summary
            local_run.awaiting_review = False
            metadata = local_run.context.setdefault("metadata", {})
            metadata["human_review"] = {"required": True, "decision": decision}
            self._local_runs[local_key] = local_run
            return final_summary

        client = await self._ensure_client()
        handle = client.get_workflow_handle(workflow_id, run_id=workflow_run_id)
        await handle.signal("resume_human_review", decision)
        payload = await handle.result()
        return ValidationSummary.model_validate(payload)

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Retrieve the final validation summary for a completed workflow."""

        local_key = (workflow_id, workflow_run_id)
        if local_key in self._local_runs:
            return self._local_runs[local_key].summary

        client = await self._ensure_client()
        handle = client.get_workflow_handle(workflow_id, run_id=workflow_run_id)
        payload = await handle.result()
        return ValidationSummary.model_validate(payload)

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

    def _record_local_run(self, workflow_id: str, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        local_run = self._execute_local_pipeline(workflow_id, hypothesis)
        workflow_run_id = f"{workflow_id}-local"
        self._local_runs[(workflow_id, workflow_run_id)] = local_run
        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            validation=local_run.summary,
        )

    def _execute_local_pipeline(self, workflow_id: str, hypothesis: HypothesisRequest) -> LocalWorkflowRun:
        orchestrator = LangGraphValidationOrchestrator()
        context: Dict[str, Any] = {
            "metadata": {
                "workflow_id": workflow_id,
                "requires_human_review": hypothesis.requires_human_review,
            }
        }
        milestones: List[WorkflowMilestone] = []

        for stage in [
            "data_ingest",
            "entity_resolution",
            "preprocessing",
            "analysis",
            "sentiment",
            "modeling",
            "advanced_modeling",
        ]:
            stage_result = orchestrator.run_stage(stage, hypothesis, context)
            context = stage_result.context
            milestones.append(stage_result.milestone)

        report_stage = orchestrator.run_stage("report_generation", hypothesis, context)
        if report_stage.summary is None:
            summary_base = ValidationSummary(
                score=0.5,
                conclusion="Pending synthesis",
                confidence=0.5,
                evidence=[],
                current_stage="report_generation",
                milestones=[],
            )
        else:
            summary_base = report_stage.summary

        report_milestone = report_stage.milestone
        awaiting_review = bool(hypothesis.requires_human_review)
        metadata = context.setdefault("metadata", {})
        if awaiting_review:
            human_milestone = WorkflowMilestone(
                name="human_review",
                status=MilestoneStatus.WAITING_REVIEW,
                detail="Awaiting human reviewer decision.",
            )
            metadata["human_review"] = {"required": True, "decision": None}
            waiting_summary = summary_base.model_copy(
                update={
                    "conclusion": "Pending human review",
                    "current_stage": "human_review",
                    "milestones": milestones + [human_milestone],
                }
            )
            return LocalWorkflowRun(
                hypothesis=hypothesis,
                context=context,
                summary=waiting_summary,
                awaiting_review=True,
                pending_summary_base=summary_base,
                report_milestone=report_milestone,
            )

        metadata["human_review"] = {"required": False, "decision": "auto"}
        final_milestones = milestones + [report_milestone]
        summary = summary_base.model_copy(update={"milestones": final_milestones})
        return LocalWorkflowRun(
            hypothesis=hypothesis,
            context=context,
            summary=summary,
            awaiting_review=False,
            pending_summary_base=None,
            report_milestone=report_milestone,
        )
