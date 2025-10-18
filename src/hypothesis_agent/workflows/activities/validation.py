"""Legacy activity wrappers now delegating directly to the LangGraph pipeline."""
from __future__ import annotations

from typing import Dict, List

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    MilestoneStatus,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.langgraph_pipeline import (
    LangGraphValidationOrchestrator,
    StageExecutionResult,
)

_ORCHESTRATOR: LangGraphValidationOrchestrator | None = None


def get_orchestrator() -> LangGraphValidationOrchestrator:
    """Return a cached orchestrator instance for activity executions."""

    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = LangGraphValidationOrchestrator()
    return _ORCHESTRATOR


def _parse_activity_payload(payload: Dict) -> tuple[HypothesisRequest, Dict, List[Dict]]:
    request_payload = payload.get("request", payload)
    request = HypothesisRequest.model_validate(request_payload)
    context = payload.get("context") or {}
    milestones: List[Dict] = payload.get("milestones") or []
    return request, context, milestones


def _serialize_stage_result(result: StageExecutionResult) -> Dict:
    return {
        "context": result.context,
        "milestone": result.milestone.model_dump(mode="json"),
        "evidence": [ref.model_dump(mode="json") for ref in result.evidence],
        "summary": result.summary.model_dump(mode="json") if result.summary else None,
    }


def _load_orchestrator_result(stage: str, payload: Dict) -> Dict:
    request, context, _ = _parse_activity_payload(payload)
    orchestrator = get_orchestrator()
    result = orchestrator.run_stage(stage, request, context)
    return _serialize_stage_result(result)


async def run_plan_generation(payload: Dict) -> Dict:
    """Generate the LLM-powered data acquisition plan."""

    return _load_orchestrator_result("plan_generation", payload)


async def run_data_collection(payload: Dict) -> Dict:
    """Collect live datasets using finance and news connectors."""

    return _load_orchestrator_result("data_collection", payload)


async def run_analysis_planning(payload: Dict) -> Dict:
    """Draft the quantitative analysis plan from collected data summaries."""

    return _load_orchestrator_result("analysis_planning", payload)


async def run_hybrid_analysis(payload: Dict) -> Dict:
    """Execute numerical analytics and persist derived artifacts."""

    return _load_orchestrator_result("hybrid_analysis", payload)


async def run_detailed_analysis(payload: Dict) -> Dict:
    """Produce the LLM-backed narrative explaining analysis outputs."""

    return _load_orchestrator_result("detailed_analysis", payload)


async def await_human_review(payload: Dict) -> Dict:
    """Pause workflow execution pending human approval when required."""

    request, context, _ = _parse_activity_payload(payload)
    metadata = dict(context.get("metadata") or {})
    human_state = dict(metadata.get("human_review") or {})
    requires_review = bool(
        metadata.get("requires_human_review")
        or request.requires_human_review
        or human_state.get("required")
    )

    if human_state.get("decision") is not None:
        requires_review = False

    if requires_review:
        metadata["human_review"] = {"required": True, "decision": None}
        metadata["awaiting_review"] = True
        milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.WAITING_REVIEW,
            detail="Awaiting human reviewer decision.",
        )
    else:
        metadata["human_review"] = {
            "required": False,
            "decision": human_state.get("decision", "auto"),
        }
        metadata["awaiting_review"] = False
        milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.COMPLETED,
            detail="Human review skipped or auto-approved.",
        )

    context["metadata"] = metadata
    result = StageExecutionResult(context=context, milestone=milestone)
    return _serialize_stage_result(result)


async def run_report_generation(payload: Dict) -> Dict:
    """Assemble the final validation report after upstream stages complete."""

    request, context, milestones_payload = _parse_activity_payload(payload)
    orchestrator = get_orchestrator()
    result = orchestrator.run_stage("report_generation", request, context)
    if result.summary is None:
        raise RuntimeError("Report generation stage failed to produce a validation summary")

    if milestones_payload:
        milestones_models = [WorkflowMilestone.model_validate(m) for m in milestones_payload]
    else:
        milestones_models = []

    if not milestones_models or milestones_models[-1].name != result.milestone.name:
        milestones_models.append(result.milestone)
    else:
        milestones_models[-1] = result.milestone

    result.summary = result.summary.model_copy(update={"milestones": milestones_models})
    return _serialize_stage_result(result)


async def run_delivery(payload: Dict) -> Dict:
    """Deliver the generated report via the configured Composio connector."""

    return _load_orchestrator_result("delivery", payload)
