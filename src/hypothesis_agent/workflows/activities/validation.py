"""Temporal activities orchestrating LangGraph + Composio validation stages."""
from __future__ import annotations

from typing import Dict, List

from temporalio import activity

from hypothesis_agent.models.hypothesis import HypothesisRequest, WorkflowMilestone
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


@activity.defn(name="run_data_ingestion")
async def run_data_ingestion(payload: Dict) -> Dict:
    """Collect raw datasets using LangGraph-driven Composio connectors."""

    return _load_orchestrator_result("data_ingest", payload)


@activity.defn(name="run_preprocessing")
async def run_preprocessing(payload: Dict) -> Dict:
    """Normalize raw datasets for downstream analytics."""

    return _load_orchestrator_result("preprocessing", payload)


@activity.defn(name="run_analysis")
async def run_analysis(payload: Dict) -> Dict:
    """Execute financial diagnostics on normalized data."""

    return _load_orchestrator_result("analysis", payload)


@activity.defn(name="run_sentiment")
async def run_sentiment(payload: Dict) -> Dict:
    """Score qualitative sentiment signals."""

    return _load_orchestrator_result("sentiment", payload)


@activity.defn(name="run_modeling")
async def run_modeling(payload: Dict) -> Dict:
    """Generate probabilistic modeling scenarios."""

    return _load_orchestrator_result("modeling", payload)


@activity.defn(name="perform_validation")
async def perform_validation(payload: Dict) -> Dict:
    """Assemble the final validation report after all upstream stages complete."""

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
