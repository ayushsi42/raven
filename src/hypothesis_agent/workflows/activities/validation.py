"""Temporal activities implementing placeholder hypothesis validation logic."""
from __future__ import annotations

from temporalio import activity

from hypothesis_agent.models.hypothesis import HypothesisRequest, ValidationSummary


@activity.defn(name="perform_validation")
async def perform_validation(payload: dict) -> dict:
    """Perform synchronous placeholder validation returning deterministic results.

    Temporal activities must be deterministic with respect to their inputs. This implementation
    computes a basic score using the payload size to demonstrate the wiring without relying on
    external services yet.
    """

    request = HypothesisRequest.model_validate(payload)
    # Deterministic pseudo-score based on hypothesis length.
    base_score = min(len(request.hypothesis_text) / 1000.0, 1.0)
    result = ValidationSummary(
        score=round(base_score, 4),
        conclusion="Pending human review" if request.requires_human_review else "Queued",
        confidence=0.25,
        evidence=[],
    )
    return result.model_dump(mode="json")
