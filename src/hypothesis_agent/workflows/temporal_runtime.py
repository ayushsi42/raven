"""Temporal-based workflow runtime providing resumable execution semantics."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.langgraph_pipeline import (
    LangGraphValidationOrchestrator,
)
from hypothesis_agent.orchestration.langgraph_pipeline import StageExecutionResult
from hypothesis_agent.workflows.hypothesis_workflow import WorkflowSubmissionResult, WorkflowExecutionDetails

_PIPELINE_STAGES: List[str] = [
    "plan_generation",
    "data_collection",
    "hybrid_analysis",
    "detailed_analysis",
    "report_generation",
    "delivery",
]


def _get_orchestrator() -> LangGraphValidationOrchestrator:
    return LangGraphValidationOrchestrator()


@activity.defn
async def run_stage_activity(stage: str, request_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    orchestrator = _get_orchestrator()
    request = HypothesisRequest.model_validate(request_payload)
    result = await asyncio.to_thread(orchestrator.run_stage, stage, request, context)
    return {
        "context": result.context,
        "milestone": result.milestone.model_dump(mode="json"),
        "evidence": [reference.model_dump(mode="json") for reference in result.evidence],
        "summary": result.summary.model_dump(mode="json") if result.summary is not None else None,
    }


@dataclass(slots=True)
class WorkflowState:
    request: Dict[str, Any]
    context: Dict[str, Any]
    summary: Dict[str, Any]
    requires_human_review: bool
    stage_index: int = 0
    awaiting_review: bool = False
    completed: bool = False
    cancelled: bool = False
    failure: Optional[str] = None
    pending_summary_base: Optional[Dict[str, Any]] = None
    report_milestone: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "summary": self.summary,
            "context": self.context,
            "stage_index": self.stage_index,
            "stages": list(_PIPELINE_STAGES),
            "awaiting_review": self.awaiting_review,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failure": self.failure,
            "pending_summary_base": self.pending_summary_base,
        }


def _stage_sequence(include_human: bool) -> List[str]:
    stages = list(_PIPELINE_STAGES)
    if include_human and "human_review" not in stages:
        delivery_index = stages.index("delivery")
        stages.insert(delivery_index, "human_review")
    return stages


def _build_initial_summary(include_human_review: bool) -> ValidationSummary:
    stages = _stage_sequence(include_human_review)
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


def _merge_milestone(
    summary: ValidationSummary,
    milestone: WorkflowMilestone,
    *,
    requires_human_review: bool,
    include_human: Optional[bool] = None,
) -> List[WorkflowMilestone]:
    include = include_human if include_human is not None else (
        requires_human_review or milestone.name == "human_review"
    )
    order = _stage_sequence(include)
    milestone_map = {existing.name: existing for existing in summary.milestones}
    milestone_map[milestone.name] = milestone
    merged: List[WorkflowMilestone] = []
    for name in order:
        if name in milestone_map:
            merged.append(milestone_map.pop(name))
    merged.extend(milestone_map.values())
    return merged


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


def _record_stage_metadata(context: Dict[str, Any], milestone: WorkflowMilestone) -> None:
    metadata = context.setdefault("metadata", {})
    stages = metadata.setdefault("stages", {})
    stages[milestone.name] = {
        "status": milestone.status.value,
        "detail": milestone.detail,
    }


@workflow.defn
class HypothesisTemporalWorkflow:
    def __init__(self) -> None:
        self._state: Optional[WorkflowState] = None
        self._requires_human_review: bool = False
        self._hypothesis_payload: Dict[str, Any] | None = None
        self._cancel_requested = False

    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._requires_human_review = bool(payload.get("requires_human_review", False))
        request_payload = payload["hypothesis"]
        metadata = payload.get("metadata") or {}
        context = payload.get("context") or {"metadata": metadata}
        summary_payload = payload.get("summary")
        if summary_payload is not None:
            summary = ValidationSummary.model_validate(summary_payload)
            stage_index = payload.get("stage_index", 0)
        else:
            summary = _build_initial_summary(self._requires_human_review)
            stage_index = 0
        self._hypothesis_payload = request_payload
        self._state = WorkflowState(
            request=request_payload,
            context=context,
            summary=summary.model_dump(mode="json"),
            requires_human_review=self._requires_human_review,
            stage_index=stage_index,
            metadata=metadata,
        )

        workflow.logger.info(
            "Temporal workflow started", workflow_id=workflow.info().workflow_id, run_id=workflow.info().run_id
        )

        await self._run_pipeline()
        assert self._state is not None
        self._state.completed = not self._cancel_requested and self._state.failure is None
        return self._state.summary

    @workflow.signal
    def cancel(self) -> None:
        self._cancel_requested = True
        if self._state is not None:
            self._state.cancelled = True

    @workflow.signal
    def human_decision(self, decision: str) -> None:
        if self._state is None or not self._state.awaiting_review:
            return
        summary_base = (
            ValidationSummary.model_validate(self._state.pending_summary_base)
            if self._state.pending_summary_base is not None
            else ValidationSummary.model_validate(self._state.summary)
        )
        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.COMPLETED,
            detail=f"Human decision: {decision}.",
        )
        milestones = _merge_milestone(
            summary_base,
            human_milestone,
            requires_human_review=self._requires_human_review,
            include_human=True,
        )
        summary = summary_base.model_copy(
            update={
                "milestones": milestones,
                "current_stage": "human_review",
            }
        )
        self._state.summary = summary.model_dump(mode="json")
        self._state.awaiting_review = False
        metadata = self._state.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": decision}

    @workflow.query
    def snapshot(self) -> Dict[str, Any]:
        if self._state is None:
            return {}
        return self._state.snapshot()

    async def _run_pipeline(self) -> None:
        if self._state is None:
            return
        start_index = self._state.stage_index
        planned_stages = _PIPELINE_STAGES
        for idx in range(start_index, len(planned_stages) - 2):
            stage = planned_stages[idx]
            await self._execute_stage(stage)
            if self._cancel_requested:
                self._finalise_cancellation(stage)
                return
        if self._state.stage_index <= len(planned_stages) - 2:
            await self._execute_stage("report_generation")
        if self._cancel_requested:
            self._finalise_cancellation("report_generation")
            return
        if self._requires_human_review:
            self._prepare_human_review()
            await workflow.wait_condition(lambda: self._state is not None and not self._state.awaiting_review)
            if self._cancel_requested:
                self._finalise_cancellation("human_review")
                return
        if self._state.stage_index < len(planned_stages):
            await self._execute_stage("delivery")

    async def _execute_stage(self, stage: str) -> None:
        if self._state is None:
            return
        if self._cancel_requested:
            return
        self._mark_stage_running(stage)
        assert self._hypothesis_payload is not None
        activity_result = await workflow.execute_activity(
            run_stage_activity,
            stage,
            self._hypothesis_payload,
            self._state.context,
            start_to_close_timeout=timedelta(minutes=20),
            schedule_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        milestone = WorkflowMilestone.model_validate(activity_result["milestone"])  # type: ignore[arg-type]
        _record_stage_metadata(activity_result["context"], milestone)
        try:
            self._apply_stage_result(activity_result)
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_stage_failure(stage, str(exc))
            raise
        self._state.stage_index += 1

    def _apply_stage_result(self, activity_result: Dict[str, Any]) -> None:
        if self._state is None:
            return
        milestone = WorkflowMilestone.model_validate(activity_result["milestone"])  # type: ignore[arg-type]
        evidence_payload = activity_result.get("evidence") or []
        evidence = [EvidenceReference.model_validate(item) for item in evidence_payload]
        summary_payload = activity_result.get("summary")
        base_summary = ValidationSummary.model_validate(self._state.summary)
        summary_source = (
            ValidationSummary.model_validate(summary_payload)
            if summary_payload is not None
            else base_summary
        )
        milestones = _merge_milestone(
            summary_source,
            milestone,
            requires_human_review=self._requires_human_review,
        )
        merged_evidence = _merge_evidence(summary_source.evidence, evidence)
        summary = summary_source.model_copy(
            update={
                "milestones": milestones,
                "current_stage": milestone.name,
                "evidence": merged_evidence,
            }
        )
        self._state.summary = summary.model_dump(mode="json")
        self._state.context = activity_result["context"]
        if milestone.name == "report_generation":
            self._state.pending_summary_base = summary.model_dump(mode="json")
            self._state.report_milestone = milestone.model_dump(mode="json")

    def _prepare_human_review(self) -> None:
        if self._state is None:
            return
        summary_base = ValidationSummary.model_validate(self._state.summary)
        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.WAITING_REVIEW,
            detail="Awaiting human reviewer decision.",
        )
        milestones = _merge_milestone(
            summary_base,
            human_milestone,
            requires_human_review=self._requires_human_review,
            include_human=True,
        )
        summary = summary_base.model_copy(
            update={
                "milestones": milestones,
                "current_stage": "human_review",
            }
        )
        self._state.summary = summary.model_dump(mode="json")
        self._state.awaiting_review = True
        self._state.pending_summary_base = summary_base.model_dump(mode="json")
        metadata = self._state.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": None}

    def _mark_stage_running(self, stage: str) -> None:
        if self._state is None:
            return
        summary = ValidationSummary.model_validate(self._state.summary)
        milestone = WorkflowMilestone(
            name=stage,
            status=MilestoneStatus.RUNNING,
            detail=f"{stage.replace('_', ' ').title()} in progress.",
        )
        milestones = _merge_milestone(
            summary,
            milestone,
            requires_human_review=self._requires_human_review,
        )
        updated = summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        self._state.summary = updated.model_dump(mode="json")

    def _handle_stage_failure(self, stage: str, detail: str) -> None:
        if self._state is None:
            return
        summary = ValidationSummary.model_validate(self._state.summary)
        milestone = WorkflowMilestone(name=stage, status=MilestoneStatus.BLOCKED, detail=detail)
        milestones = _merge_milestone(
            summary,
            milestone,
            requires_human_review=self._requires_human_review,
        )
        updated = summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        self._state.summary = updated.model_dump(mode="json")
        self._state.failure = detail
        self._state.awaiting_review = False

    def _finalise_cancellation(self, stage: str) -> None:
        if self._state is None:
            return
        summary = ValidationSummary.model_validate(self._state.summary)
        cancelled_milestones: List[WorkflowMilestone] = []
        for milestone in summary.milestones:
            if milestone.status in {MilestoneStatus.COMPLETED, MilestoneStatus.BLOCKED}:
                cancelled_milestones.append(milestone)
                continue
            detail = milestone.detail or "Stage not executed due to cancellation."
            if milestone.name == stage and milestone.status == MilestoneStatus.RUNNING:
                detail = "Stage cancelled by user."
            cancelled_milestones.append(
                WorkflowMilestone(name=milestone.name, status=MilestoneStatus.CANCELLED, detail=detail)
            )
        updated = summary.model_copy(
            update={
                "milestones": cancelled_milestones,
                "current_stage": "cancelled",
                "conclusion": "Validation cancelled by user request.",
            }
        )
        self._state.summary = updated.model_dump(mode="json")
        self._state.awaiting_review = False


class TemporalWorkflowRuntime:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._environment: WorkflowEnvironment | None = None
        self._client: Client | None = None
        self._worker: Worker | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._handles: Dict[Tuple[str, str], WorkflowHandle] = {}

    async def start(self) -> None:
        if self._environment is None:
            self._environment = await WorkflowEnvironment.start_local()
        if self._client is None:
            self._client = await self._environment.start_client()
        if self._worker is None:
            self._worker = Worker(
                self._client,
                task_queue=self.settings.temporal_task_queue,
                workflows=[HypothesisTemporalWorkflow],
                activities=[run_stage_activity],
            )
            self._worker_task = asyncio.create_task(self._worker.run())
            await asyncio.sleep(0)

    async def close(self) -> None:
        if self._worker is not None:
            await self._worker.shutdown()
        if self._worker_task is not None:
            await self._worker_task
        if self._environment is not None:
            await self._environment.close()
        self._environment = None
        self._client = None
        self._worker = None
        self._worker_task = None
        self._handles.clear()

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        await self.start()
        assert self._client is not None
        workflow_id = f"hypothesis-{hypothesis_id}"
        payload = {
            "hypothesis": hypothesis.model_dump(mode="json"),
            "metadata": {
                "workflow_id": workflow_id,
            },
            "context": {
                "metadata": {
                    "workflow_id": workflow_id,
                    "stages": {},
                }
            },
            "requires_human_review": hypothesis.requires_human_review,
        }
        handle = await self._client.start_workflow(
            HypothesisTemporalWorkflow.run,
            payload,
            id=workflow_id,
            task_queue=self.settings.temporal_task_queue,
        )
        self._handles[(workflow_id, handle.run_id)] = handle
        snapshot = await handle.query(HypothesisTemporalWorkflow.snapshot)
        summary = ValidationSummary.model_validate(snapshot["summary"])
        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=handle.run_id,
            validation=summary,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        handle = self._get_handle(workflow_id, workflow_run_id)
        snapshot = await handle.query(HypothesisTemporalWorkflow.snapshot)
        summary = ValidationSummary.model_validate(snapshot["summary"])
        history_length = None
        if summary.milestones:
            history_length = sum(1 for milestone in summary.milestones if milestone.status != MilestoneStatus.PENDING)
        if snapshot.get("cancelled"):
            status = "CANCELLED"
        elif snapshot.get("awaiting_review"):
            status = "AWAITING_REVIEW"
        elif snapshot.get("failure"):
            status = "FAILED"
        elif snapshot.get("completed"):
            status = "COMPLETED"
        else:
            status = "RUNNING"
        return WorkflowExecutionDetails(
            status=status,
            history_length=history_length,
            milestones=summary.milestones,
            awaiting_review=bool(snapshot.get("awaiting_review", False)),
        )

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        handle = self._get_handle(workflow_id, workflow_run_id)
        snapshot = await handle.query(HypothesisTemporalWorkflow.snapshot)
        return ValidationSummary.model_validate(snapshot["summary"])

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved") -> ValidationSummary:
        handle = self._get_handle(workflow_id, workflow_run_id)
        await handle.signal(HypothesisTemporalWorkflow.human_decision, decision)
        result_payload = await handle.result()
        summary = ValidationSummary.model_validate(result_payload)
        return summary

    async def cancel(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        handle = self._get_handle(workflow_id, workflow_run_id)
        await handle.signal(HypothesisTemporalWorkflow.cancel)
        result_payload = await handle.result()
        summary = ValidationSummary.model_validate(result_payload)
        return summary

    def _get_handle(self, workflow_id: str, workflow_run_id: str) -> WorkflowHandle:
        key = (workflow_id, workflow_run_id)
        handle = self._handles.get(key)
        if handle is not None:
            return handle
        if self._client is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        handle = self._client.get_workflow_handle_for_run_id(workflow_run_id)
        self._handles[key] = handle
        return handle