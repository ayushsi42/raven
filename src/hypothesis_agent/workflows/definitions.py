"""Temporal workflow definitions for RAVEN hypothesis validation."""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

from hypothesis_agent.models.hypothesis import ValidationSummary


@workflow.defn(name="HypothesisValidationWorkflow")
class HypothesisValidationWorkflow:
    """Orchestrate hypothesis validation activities."""

    @workflow.run
    async def run(self, payload: dict) -> dict:
        """Entry point for the validation workflow.

        The workflow currently performs a single activity invocation that returns a
        placeholder validation summary. As new agents come online, this workflow will
        coordinate their execution.
        """

        summary: dict = await workflow.execute_activity(
            "perform_validation",
            payload,
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        return summary
