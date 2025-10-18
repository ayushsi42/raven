"""Application service for managing hypothesis submissions."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    HypothesisResponse,
    HypothesisStatusResponse,
    MilestoneStatus,
    ResumeRequest,
    ValidationSummary,
)
from hypothesis_agent.repositories.hypothesis_repository import (
    HypothesisRecord,
    HypothesisRepository,
)
from hypothesis_agent.workflows.hypothesis_workflow import (
    HypothesisWorkflowClient,
    WorkflowSubmissionResult,
)


@dataclass(slots=True)
class HypothesisService:
    """Coordinate hypothesis submissions and persistence."""

    repository: HypothesisRepository
    workflow_client: HypothesisWorkflowClient

    async def submit(self, hypothesis: HypothesisRequest) -> HypothesisResponse:
        """Submit a hypothesis, persist it, and return the response contract."""

        hypothesis_id = uuid4()
        submission: WorkflowSubmissionResult = await self.workflow_client.submit(hypothesis_id, hypothesis)
        status = "accepted"
        if any(m.status == MilestoneStatus.WAITING_REVIEW for m in submission.validation.milestones):
            status = "awaiting_review"

        record = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            workflow_id=submission.workflow_id,
            workflow_run_id=submission.workflow_run_id,
            request=hypothesis,
            status=status,
            validation=submission.validation,
        )
        await self.repository.save(record)
        return HypothesisResponse(
            hypothesis_id=hypothesis_id,
            workflow_id=submission.workflow_id,
            workflow_run_id=submission.workflow_run_id,
            status=status,
            validation=record.validation,
        )

    async def get(self, hypothesis_id: UUID) -> HypothesisResponse:
        """Retrieve a previously submitted hypothesis record."""

        record = await self.repository.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        return HypothesisResponse(
            hypothesis_id=record.hypothesis_id,
            workflow_id=record.workflow_id,
            workflow_run_id=record.workflow_run_id,
            status=record.status,
            validation=record.validation,
        )

    async def get_status(self, hypothesis_id: UUID) -> HypothesisStatusResponse:
        """Retrieve workflow execution status for a hypothesis."""

        record = await self.repository.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        execution = await self.workflow_client.describe(record.workflow_id, record.workflow_run_id)
        validation = record.validation
        status = record.status
        if execution.milestones:
            current_stage = validation.current_stage
            running = next((m.name for m in execution.milestones if m.status == MilestoneStatus.RUNNING), None)
            if running:
                current_stage = running
            else:
                completed = [m.name for m in execution.milestones if m.status == MilestoneStatus.COMPLETED]
                if completed:
                    current_stage = completed[-1]
            validation = validation.model_copy(
                update={
                    "milestones": execution.milestones,
                    "current_stage": current_stage,
                }
            )

        if execution.awaiting_review and status != "awaiting_review":
            status = "awaiting_review"
        elif execution.status == "COMPLETED" and not execution.awaiting_review:
            final_summary = await self.workflow_client.fetch_summary(
                workflow_id=record.workflow_id,
                workflow_run_id=record.workflow_run_id,
            )
            validation = final_summary
            if record.status not in {"needs_changes", "rejected"}:
                status = "completed"

        if status != record.status or validation != record.validation:
            updated = HypothesisRecord(
                hypothesis_id=record.hypothesis_id,
                workflow_id=record.workflow_id,
                workflow_run_id=record.workflow_run_id,
                request=record.request,
                status=status,
                validation=validation,
            )
            await self.repository.save(updated)
            record = updated

        return HypothesisStatusResponse(
            hypothesis_id=record.hypothesis_id,
            workflow_id=record.workflow_id,
            workflow_run_id=record.workflow_run_id,
            status=record.status,
            validation=validation,
            workflow_status=execution.status,
            workflow_history_length=execution.history_length,
        )

    async def get_report(self, hypothesis_id: UUID) -> ValidationSummary:
        """Return the most recently persisted validation report."""

        record = await self.repository.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        return record.validation

    async def resume(self, hypothesis_id: UUID, request: ResumeRequest) -> HypothesisResponse:
        """Resume a paused workflow after human review and persist the updated state."""

        record = await self.repository.get(hypothesis_id)
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")

        summary = await self.workflow_client.resume(record.workflow_id, record.workflow_run_id, request.decision)
        if request.decision == "approved":
            status = "completed"
        elif request.decision == "needs_changes":
            status = "needs_changes"
        else:
            status = "rejected"
        updated = HypothesisRecord(
            hypothesis_id=record.hypothesis_id,
            workflow_id=record.workflow_id,
            workflow_run_id=record.workflow_run_id,
            request=record.request,
            status=status,
            validation=summary,
        )
        await self.repository.save(updated)
        return HypothesisResponse(
            hypothesis_id=updated.hypothesis_id,
            workflow_id=updated.workflow_id,
            workflow_run_id=updated.workflow_run_id,
            status=updated.status,
            validation=summary,
        )
