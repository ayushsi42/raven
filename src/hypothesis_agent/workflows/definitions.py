"""Temporal workflow definitions for RAVEN hypothesis validation."""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

from temporalio import workflow

from hypothesis_agent.models.hypothesis import (
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)


STAGES: List[tuple[str, str]] = [
    ("data_ingest", "run_data_ingestion"),
    ("preprocessing", "run_preprocessing"),
    ("analysis", "run_analysis"),
    ("sentiment", "run_sentiment"),
    ("modeling", "run_modeling"),
    ("report_generation", "perform_validation"),
]


@workflow.defn(name="HypothesisValidationWorkflow")
class HypothesisValidationWorkflow:
    """Orchestrate hypothesis validation activities."""

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}
        self._milestones: List[WorkflowMilestone] = []
        self._summary: ValidationSummary | None = None

    @workflow.run
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the validation workflow across sequential Temporal activities."""

        self._initialize_state(payload)
        self._publish_status(self._milestones[0].name)

        for index, (stage_name, activity_name) in enumerate(STAGES):
            activity_payload = {
                "request": payload,
                "context": self._context,
                "milestones": [milestone.model_dump(mode="json") for milestone in self._milestones],
            }

            result: Dict[str, Any] = await workflow.execute_activity(
                activity_name,
                activity_payload,
                schedule_to_close_timeout=timedelta(minutes=5),
            )
            self._apply_stage_result(stage_name, index, result)

        if self._summary is None:
            raise RuntimeError("Validation workflow completed without producing a summary")
        return self._summary.model_dump(mode="json")

    @workflow.query(name="milestones")
    def query_milestones(self) -> List[Dict[str, Any]]:
        """Expose milestone progression for live status checks."""

        return [milestone.model_dump(mode="json") for milestone in self._milestones]

    def _initialize_state(self, payload: Dict[str, Any]) -> None:
        self._context = {
            "request": payload,
            "data": {},
            "normalized": {},
            "analysis": {},
            "sentiment": {},
            "modeling": {},
            "insights": [],
            "evidence": [],
        }
        self._milestones = [
            WorkflowMilestone(name=name, status=MilestoneStatus.PENDING, detail=None)
            for name, _ in STAGES
        ]
        if self._milestones:
            self._milestones[0] = self._milestones[0].model_copy(update={"status": MilestoneStatus.RUNNING})
        self._summary = None

    def _apply_stage_result(self, stage_name: str, index: int, result: Dict[str, Any]) -> None:
        context_update = result.get("context") or {}
        if context_update:
            self._context = context_update

        milestone_payload = result.get("milestone") or {}
        milestone = WorkflowMilestone.model_validate(milestone_payload)
        self._milestones[index] = milestone
        self._publish_status(stage_name)

        stage_evidence = result.get("evidence") or []
        if stage_evidence:
            evidence_collection = self._context.setdefault("evidence", [])
            evidence_collection.extend(stage_evidence)

        summary_payload = result.get("summary")
        if summary_payload:
            summary = ValidationSummary.model_validate(summary_payload)
            summary = summary.model_copy(update={"milestones": self._milestones})
            self._summary = summary
        elif index + 1 < len(self._milestones):
            next_milestone = self._milestones[index + 1].model_copy(
                update={"status": MilestoneStatus.RUNNING}
            )
            self._milestones[index + 1] = next_milestone
            self._publish_status(next_milestone.name)

    def _publish_status(self, stage_name: str) -> None:
        try:
            workflow.upsert_search_attributes({"MilestoneStage": [stage_name]})
        except Exception:  # pragma: no cover - depends on Temporal cluster configuration
            workflow.logger.debug("Unable to upsert search attributes", exc_info=True)
