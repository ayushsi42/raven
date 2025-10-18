"""Local workflow runner for hypothesis validation using LangGraph."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from uuid import UUID

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
    """Execute the LangGraph pipeline without relying on Temporal."""

    namespace: str
    task_queue: str
    workflow: str
    address: str

    _local_runs: Dict[Tuple[str, str], LocalWorkflowRun] = field(default_factory=dict, init=False, repr=False)

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        """Execute the validation pipeline and return tracking metadata."""

        workflow_id = f"hypothesis-{hypothesis_id}"
        workflow_run_id = self._generate_run_id(workflow_id)
        logger.info(
            "Executing local workflow hypothesis=%s workflow_id=%s run_id=%s user=%s",
            hypothesis_id,
            workflow_id,
            workflow_run_id,
            hypothesis.user_id,
        )

        local_run = self._execute_local_pipeline(workflow_id, workflow_run_id, hypothesis)
        self._local_runs[(workflow_id, workflow_run_id)] = local_run
        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            validation=local_run.summary,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        """Return execution details for a workflow."""

        local_key = (workflow_id, workflow_run_id)
        if local_key not in self._local_runs:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")

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

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved") -> ValidationSummary:
        """Resume a paused workflow after a human review decision."""

        local_key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(local_key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
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

        orchestrator = LangGraphValidationOrchestrator()
        delivery_stage = orchestrator.run_stage("delivery", local_run.hypothesis, local_run.context)
        local_run.context = delivery_stage.context
        final_milestones.append(delivery_stage.milestone)

        base_summary = local_run.pending_summary_base or local_run.summary
        final_summary = base_summary.model_copy(
            update={
                "current_stage": "delivery",
                "milestones": final_milestones,
            }
        )

        local_run.summary = final_summary
        local_run.awaiting_review = False
        metadata = local_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": decision}
        self._local_runs[local_key] = local_run
        return final_summary

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Retrieve the final validation summary for a completed workflow."""

        local_key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(local_key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        return local_run.summary

    async def close(self) -> None:
        """Placeholder for compatibility with previous Temporal client."""

        return None

    def _generate_run_id(self, workflow_id: str) -> str:
        run_id = f"{workflow_id}-run"
        counter = 1
        while (workflow_id, run_id) in self._local_runs:
            run_id = f"{workflow_id}-run-{counter}"
            counter += 1
        return run_id

    def _execute_local_pipeline(
        self,
        workflow_id: str,
        workflow_run_id: str,
        hypothesis: HypothesisRequest,
    ) -> LocalWorkflowRun:
        orchestrator = LangGraphValidationOrchestrator()
        context: Dict[str, Any] = {"metadata": {"workflow_id": workflow_id, "workflow_run_id": workflow_run_id}}
        milestones: List[WorkflowMilestone] = []

        for stage in [
            "plan_generation",
            "data_collection",
            "analysis_planning",
            "hybrid_analysis",
            "detailed_analysis",
        ]:
            stage_result = orchestrator.run_stage(stage, hypothesis, context)
            context = stage_result.context
            milestones.append(stage_result.milestone)

        report_stage = orchestrator.run_stage("report_generation", hypothesis, context)
        context = report_stage.context
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
        milestones.append(report_milestone)

        awaiting_review = bool(hypothesis.requires_human_review)
        metadata = context.setdefault("metadata", {})
        metadata["human_review"] = {"required": awaiting_review, "decision": None}

        if awaiting_review:
            human_milestone = WorkflowMilestone(
                name="human_review",
                status=MilestoneStatus.WAITING_REVIEW,
                detail="Awaiting human reviewer decision.",
            )
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

        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.COMPLETED,
            detail="Human review skipped or auto-approved.",
        )
        milestones.append(human_milestone)

        delivery_stage = orchestrator.run_stage("delivery", hypothesis, context)
        context = delivery_stage.context
        milestones.append(delivery_stage.milestone)

        summary = summary_base.model_copy(update={"milestones": milestones, "current_stage": "delivery"})
        return LocalWorkflowRun(
            hypothesis=hypothesis,
            context=context,
            summary=summary,
            awaiting_review=False,
            pending_summary_base=None,
            report_milestone=report_milestone,
        )
