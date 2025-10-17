"""API router wiring for hypothesis submission endpoints."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hypothesis_agent.models.hypothesis import HypothesisRequest, HypothesisResponse
from hypothesis_agent.services.hypothesis_service import HypothesisService

api_router = APIRouter()


def get_hypothesis_service(request: Request) -> HypothesisService:
    """Resolve the configured hypothesis service from the FastAPI application state."""

    try:
        return request.app.state.hypothesis_service
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Hypothesis service not configured on application state") from exc


@api_router.post("/hypotheses", response_model=HypothesisResponse, summary="Submit a hypothesis for validation")
async def submit_hypothesis(
    hypothesis: HypothesisRequest,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> HypothesisResponse:
    """Accept a hypothesis submission and dispatch it to the workflow service."""

    return await service.submit(hypothesis)


@api_router.get(
    "/hypotheses/{hypothesis_id}",
    response_model=HypothesisResponse,
    summary="Retrieve a hypothesis submission status",
)
async def get_hypothesis(
    hypothesis_id: UUID,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> HypothesisResponse:
    """Return the current status of a hypothesis submission."""

    try:
        return await service.get(hypothesis_id)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Hypothesis not found"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail) from exc
