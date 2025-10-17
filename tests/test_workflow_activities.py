"""Unit tests for Temporal workflow activities."""
from __future__ import annotations

import pytest

from datetime import date

from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.workflows.activities.validation import (
    perform_validation,
    run_analysis,
    run_data_ingestion,
    run_modeling,
    run_preprocessing,
    run_sentiment,
)


@pytest.mark.asyncio
async def test_perform_validation_returns_deterministic_summary() -> None:
    request = HypothesisRequest(
        user_id="user-789",
        hypothesis_text="Short hypothesis",
        entities=["Company Z"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    payload = {"request": request.model_dump(mode="json"), "context": {}, "milestones": []}
    context = payload["context"]
    milestones = payload["milestones"]

    for activity in [
        run_data_ingestion,
        run_preprocessing,
        run_analysis,
        run_sentiment,
        run_modeling,
    ]:
        stage_result = await activity({"request": payload["request"], "context": context, "milestones": milestones})
        context = stage_result["context"]
        milestones.append(stage_result["milestone"])

    final_result = await perform_validation({"request": payload["request"], "context": context, "milestones": milestones})
    summary = final_result["summary"]

    assert summary["conclusion"] in {"Supported", "Partially supported", "Not supported"}
    assert 0.0 <= summary["score"] <= 1.0
    assert 0.0 <= summary["confidence"] <= 1.0
    assert summary["current_stage"] == "report_generation"
    assert summary["milestones"][-1]["name"] == "report_generation"
