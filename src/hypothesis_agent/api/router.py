"""API router wiring for hypothesis submission endpoints."""
from __future__ import annotations

from uuid import UUID

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hypothesis_agent.auth import require_authenticated_user
from hypothesis_agent.models.hypothesis import (
    HypothesisRequest,
    HypothesisResponse,
    HypothesisStatusResponse,
    ResumeRequest,
    ValidationSummary,
)
from hypothesis_agent.services.hypothesis_service import HypothesisService

api_router = APIRouter(dependencies=[Depends(require_authenticated_user)])


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

    try:
        return await service.submit(hypothesis)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Hypothesis submission failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


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
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Failed to retrieve hypothesis")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@api_router.get(
    "/hypotheses/{hypothesis_id}/report",
    response_model=ValidationSummary,
    summary="Retrieve the persisted validation report",
)
async def get_hypothesis_report(
    hypothesis_id: UUID,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> ValidationSummary:
    """Return the stored validation report for a hypothesis."""

    try:
        return await service.get_report(hypothesis_id)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Hypothesis not found"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Failed to retrieve hypothesis report")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@api_router.post(
    "/hypotheses/{hypothesis_id}/resume",
    response_model=HypothesisResponse,
    summary="Resume a hypothesis workflow after human review",
)
async def resume_hypothesis(
    hypothesis_id: UUID,
    request: ResumeRequest,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> HypothesisResponse:
    """Resume execution for a hypothesis requiring human approval."""

    try:
        return await service.resume(hypothesis_id, request)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Hypothesis not found"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Failed to resume hypothesis workflow")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@api_router.post(
    "/hypotheses/{hypothesis_id}/cancel",
    response_model=HypothesisStatusResponse,
    summary="Cancel a running hypothesis workflow",
)
async def cancel_hypothesis(
    hypothesis_id: UUID,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> HypothesisStatusResponse:
    """Cancel a running hypothesis workflow and return the updated status."""

    try:
        return await service.cancel(hypothesis_id)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Hypothesis not found"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Failed to cancel hypothesis workflow")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@api_router.get(
    "/hypotheses/{hypothesis_id}/status",
    response_model=HypothesisStatusResponse,
    summary="Retrieve live workflow execution status for a hypothesis",
)
async def get_hypothesis_status(
    hypothesis_id: UUID,
    service: HypothesisService = Depends(get_hypothesis_service),
) -> HypothesisStatusResponse:
    """Return workflow execution state for a hypothesis."""

    try:
        return await service.get_status(hypothesis_id)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Hypothesis not found"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).exception("Failed to retrieve hypothesis status")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
