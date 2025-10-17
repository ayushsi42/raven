"""Unit tests for Temporal workflow activities."""
from __future__ import annotations

import pytest

from datetime import date

from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon
from hypothesis_agent.workflows.activities.validation import perform_validation


@pytest.mark.asyncio
async def test_perform_validation_returns_deterministic_summary() -> None:
    request = HypothesisRequest(
        user_id="user-789",
        hypothesis_text="Short hypothesis",
        entities=["Company Z"],
        time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )

    result = await perform_validation(request.model_dump(mode="json"))

    assert result["conclusion"] == "Queued"
    assert 0.0 <= result["score"] <= 1.0
    assert result["confidence"] == 0.25
    assert result["evidence"] == []
