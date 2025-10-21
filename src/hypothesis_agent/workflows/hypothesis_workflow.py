"""Local workflow runner for hypothesis validation using LangGraph."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from uuid import UUID

from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.langgraph_pipeline import (
    LangGraphValidationOrchestrator,
    StageExecutionResult,
)

logger = logging.getLogger(__name__)

_PIPELINE_STAGES: List[str] = [
    "plan_generation",
    "data_collection",
    "hybrid_analysis",
    "detailed_analysis",
    "report_generation",
    "delivery",
]


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
    completed: bool = False
    cancelled: bool = False


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Execute the LangGraph pipeline without relying on Temporal."""

    namespace: str
    task_queue: str
    workflow: str
    address: str

    _local_runs: Dict[Tuple[str, str], LocalWorkflowRun] = field(default_factory=dict, init=False, repr=False)
    _tasks: Dict[Tuple[str, str], asyncio.Task[None]] = field(default_factory=dict, init=False, repr=False)

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        """Schedule the validation pipeline and return tracking metadata."""

        workflow_id = f"hypothesis-{hypothesis_id}"
        workflow_run_id = self._generate_run_id(workflow_id)
        logger.info(
            "Executing local workflow hypothesis=%s workflow_id=%s run_id=%s user=%s",
            hypothesis_id,
            workflow_id,
            workflow_run_id,
            hypothesis.user_id,
        )

        context: Dict[str, Any] = {
            "metadata": {
                "workflow_id": workflow_id,
                "workflow_run_id": workflow_run_id,
                "stages": {},
            }
        }
        initial_summary = self._build_initial_summary(hypothesis.requires_human_review)
        local_run = LocalWorkflowRun(
            hypothesis=hypothesis,
            context=context,
            summary=initial_summary,
            awaiting_review=False,
            pending_summary_base=None,
            report_milestone=None,
            completed=False,
            cancelled=False,
        )

        key = (workflow_id, workflow_run_id)
        self._local_runs[key] = local_run

        task = asyncio.create_task(self._run_pipeline_async(workflow_id, workflow_run_id, hypothesis))
        self._tasks[key] = task
        task.add_done_callback(lambda t, task_key=key: self._on_task_complete(task_key, t))

        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            validation=initial_summary,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        """Return execution details for a workflow."""

        key = (workflow_id, workflow_run_id)
        if key not in self._local_runs:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")

        local_run = self._local_runs[key]
        summary = local_run.summary
        history_length = None
        if summary.milestones:
            history_length = sum(
                1 for milestone in summary.milestones if milestone.status != MilestoneStatus.PENDING
            )

        if local_run.awaiting_review:
            status = "AWAITING_REVIEW"
        elif local_run.cancelled:
            status = "CANCELLED"
        elif any(m.status == MilestoneStatus.BLOCKED for m in summary.milestones):
            status = "FAILED"
        elif local_run.completed:
            status = "COMPLETED"
        else:
            status = "RUNNING"

        return WorkflowExecutionDetails(
            status=status,
            history_length=history_length,
            milestones=summary.milestones,
            awaiting_review=local_run.awaiting_review,
        )

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved") -> ValidationSummary:
        """Resume a paused workflow after a human review decision."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        if not local_run.awaiting_review:
            return local_run.summary

        base_summary = local_run.pending_summary_base or local_run.summary
        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.COMPLETED,
            detail=f"Human decision: {decision}.",
        )
        milestones = self._merge_milestone(local_run, human_milestone, include_human=True)
        summary_after_review = base_summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": "human_review",
            }
        )
        local_run.summary = summary_after_review
        local_run.awaiting_review = False
        local_run.cancelled = False
        metadata = local_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": decision}
        self._local_runs[key] = local_run

        orchestrator = LangGraphValidationOrchestrator()
        self._mark_stage_running(key, "delivery")
        await asyncio.sleep(0)
        try:
            delivery_stage = await asyncio.to_thread(
                orchestrator.run_stage,
                "delivery",
                local_run.hypothesis,
                local_run.context,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_stage_failure(key, "delivery", exc)
            raise

        self._record_stage_metadata(delivery_stage.context, delivery_stage)
        self._apply_stage_result(key, delivery_stage, summary_override=summary_after_review)

        updated_run = self._local_runs[key]
        updated_run.awaiting_review = False
        updated_run.completed = True
        updated_run.pending_summary_base = None
        updated_run.report_milestone = delivery_stage.milestone
        updated_run.cancelled = False
        metadata = updated_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": decision}
        self._local_runs[key] = updated_run
        self._tasks.pop(key, None)
        return updated_run.summary

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Retrieve the current validation summary for a workflow."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        return local_run.summary

    async def close(self) -> None:
        """Cancel any running local workflow tasks."""

        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def cancel(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Forcefully stop an in-flight workflow and mark milestones as cancelled."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        if local_run.cancelled:
            return local_run.summary
        if local_run.completed and not local_run.awaiting_review:
            raise RuntimeError("Workflow has already completed and cannot be cancelled.")

        task = self._tasks.pop(key, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        cancelled_milestones: List[WorkflowMilestone] = []
        for milestone in local_run.summary.milestones:
            if milestone.status in {MilestoneStatus.COMPLETED, MilestoneStatus.BLOCKED}:
                cancelled_milestones.append(milestone)
                continue
            detail = milestone.detail
            if milestone.status == MilestoneStatus.RUNNING:
                detail = "Stage cancelled by user."
            elif milestone.status == MilestoneStatus.WAITING_REVIEW:
                detail = "Human review bypassed due to cancellation."
            else:
                detail = detail or "Stage not executed due to cancellation."
            cancelled_milestones.append(
                WorkflowMilestone(name=milestone.name, status=MilestoneStatus.CANCELLED, detail=detail)
            )

        summary = local_run.summary.model_copy(
            update={
                "milestones": cancelled_milestones,
                "conclusion": "Validation cancelled by user request.",
                "current_stage": "cancelled",
            }
        )

        local_run.summary = summary
        local_run.awaiting_review = False
        local_run.completed = True
        local_run.cancelled = True
        local_run.pending_summary_base = None
        local_run.report_milestone = None
        metadata = local_run.context.setdefault("metadata", {})
        metadata["cancelled"] = True
        self._local_runs[key] = local_run
        return summary

    def _generate_run_id(self, workflow_id: str) -> str:
        run_id = f"{workflow_id}-run"
        counter = 1
        while (workflow_id, run_id) in self._local_runs:
            run_id = f"{workflow_id}-run-{counter}"
            counter += 1
        return run_id

    def _stage_sequence(self, include_human_review: bool) -> List[str]:
        stages = list(_PIPELINE_STAGES)
        if include_human_review and "human_review" not in stages:
            delivery_index = stages.index("delivery")
            stages.insert(delivery_index, "human_review")
        return stages

    def _build_initial_summary(self, include_human_review: bool) -> ValidationSummary:
        stages = self._stage_sequence(include_human_review)
        milestones = [
            WorkflowMilestone(name=stage, status=MilestoneStatus.PENDING, detail=None)
            for stage in stages
        ]
        current_stage = stages[0] if stages else "pending"
        return ValidationSummary(
            score=0.0,
            conclusion="Pending validation",
            confidence=0.0,
            evidence=[],
            current_stage=current_stage,
            milestones=milestones,
        )

    async def _run_pipeline_async(
        self,
        workflow_id: str,
        workflow_run_id: str,
        hypothesis: HypothesisRequest,
    ) -> None:
        key = (workflow_id, workflow_run_id)
        orchestrator = LangGraphValidationOrchestrator()

        local_run = self._local_runs.get(key)
        if local_run is None:
            return

        context = local_run.context

        for stage in _PIPELINE_STAGES[:-2]:
            context = await self._run_single_stage(key, orchestrator, stage, hypothesis, context)
            if context is None:
                return

        context = await self._run_single_stage(key, orchestrator, "report_generation", hypothesis, context)
        if context is None:
            return

        summary_after_report = self._local_runs[key].summary
        self._local_runs[key].report_milestone = next(
            (milestone for milestone in summary_after_report.milestones if milestone.name == "report_generation"),
            None,
        )

        if hypothesis.requires_human_review:
            self._prepare_human_review(key, summary_after_report)
            self._tasks.pop(key, None)
            return

        context = await self._run_single_stage(key, orchestrator, "delivery", hypothesis, context)
        if context is None:
            return

        final_run = self._local_runs[key]
        final_run.context = context
        final_run.completed = True
        final_run.awaiting_review = False
        final_run.pending_summary_base = None
        final_run.cancelled = False
        metadata = final_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": False, "decision": "auto"}
        self._local_runs[key] = final_run
        self._tasks.pop(key, None)

    async def _run_single_stage(
        self,
        key: Tuple[str, str],
        orchestrator: LangGraphValidationOrchestrator,
        stage: str,
        hypothesis: HypothesisRequest,
        context: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        self._mark_stage_running(key, stage)
        await asyncio.sleep(0)
        try:
            result = await asyncio.to_thread(orchestrator.run_stage, stage, hypothesis, context)
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_stage_failure(key, stage, exc)
            return None

        self._record_stage_metadata(result.context, result)
        self._apply_stage_result(key, result)
        await asyncio.sleep(0)
        return result.context

    def _mark_stage_running(self, key: Tuple[str, str], stage: str) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        milestone = WorkflowMilestone(
            name=stage,
            status=MilestoneStatus.RUNNING,
            detail=f"{stage.replace('_', ' ').title()} in progress.",
        )
        milestones = self._merge_milestone(local_run, milestone)
        summary = local_run.summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        local_run.summary = summary
        local_run.completed = False
        local_run.cancelled = False
        self._local_runs[key] = local_run

    def _handle_stage_failure(self, key: Tuple[str, str], stage: str, exc: Exception) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        detail = str(exc)
        milestone = WorkflowMilestone(name=stage, status=MilestoneStatus.BLOCKED, detail=detail)
        milestones = self._merge_milestone(local_run, milestone)
        summary = local_run.summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        local_run.summary = summary
        local_run.awaiting_review = False
        local_run.completed = True
        local_run.cancelled = False
        metadata = local_run.context.setdefault("metadata", {})
        metadata.setdefault("human_review", {"required": False, "decision": None})
        self._local_runs[key] = local_run
        self._tasks.pop(key, None)
        logger.exception(
            "Pipeline stage %s failed for workflow %s/%s",
            stage,
            key[0],
            key[1],
            exc_info=True,
        )

    def _apply_stage_result(
        self,
        key: Tuple[str, str],
        result: StageExecutionResult,
        *,
        summary_override: ValidationSummary | None = None,
    ) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return

        base_summary = summary_override or local_run.summary
        summary_source = result.summary or base_summary
        milestones = self._merge_milestone(local_run, result.milestone)
        evidence = self._merge_evidence(summary_source.evidence, result.evidence)
        summary = summary_source.model_copy(
            update={
                "milestones": milestones,
                "current_stage": result.milestone.name,
                "evidence": evidence,
            }
        )
        local_run.summary = summary
        local_run.context = result.context
        local_run.completed = False
        local_run.cancelled = False
        if result.milestone.name == "report_generation":
            local_run.pending_summary_base = summary
            local_run.report_milestone = result.milestone
        self._local_runs[key] = local_run

    def _merge_milestone(
        self,
        local_run: LocalWorkflowRun,
        milestone: WorkflowMilestone,
        *,
        include_human: bool | None = None,
    ) -> List[WorkflowMilestone]:
        include = include_human if include_human is not None else (
            local_run.hypothesis.requires_human_review or milestone.name == "human_review"
        )
        order = self._stage_sequence(include)
        milestone_map = {existing.name: existing for existing in local_run.summary.milestones}
        milestone_map[milestone.name] = milestone
        merged: List[WorkflowMilestone] = []
        for name in order:
            if name in milestone_map:
                merged.append(milestone_map.pop(name))
        merged.extend(milestone_map.values())
        return merged

    def _prepare_human_review(self, key: Tuple[str, str], summary_base: ValidationSummary) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.WAITING_REVIEW,
            detail="Awaiting human reviewer decision.",
        )
        milestones = self._merge_milestone(local_run, human_milestone, include_human=True)
        summary = summary_base.model_copy(
            update={
                "milestones": milestones,
                "current_stage": "human_review",
            }
        )
        local_run.summary = summary
        local_run.awaiting_review = True
        local_run.completed = False
        local_run.pending_summary_base = summary_base
        metadata = local_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": None}
        self._local_runs[key] = local_run

    @staticmethod
    def _merge_evidence(
        existing: List[EvidenceReference],
        additions: List[EvidenceReference],
    ) -> List[EvidenceReference]:
        merged = list(existing)
        seen = {(item.type, str(item.uri)) for item in merged}
        for reference in additions:
            key = (reference.type, str(reference.uri))
            if key not in seen:
                merged.append(reference)
                seen.add(key)
        return merged

    def _on_task_complete(self, key: Tuple[str, str], task: asyncio.Task[None]) -> None:
        self._tasks.pop(key, None)
        try:
            task.result()
        except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Workflow task failed for workflow %s/%s",
                key[0],
                key[1],
                exc_info=True,
            )

    def _record_stage_metadata(self, context: Dict[str, Any], result: StageExecutionResult) -> None:
        metadata = context.setdefault("metadata", {})
        stages = metadata.setdefault("stages", {})
        stages[result.milestone.name] = {
            "status": result.milestone.status.value,
            "detail": result.milestone.detail,
        }
