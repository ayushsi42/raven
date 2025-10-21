"""FastAPI application entrypoint for the RAVEN hypothesis agent."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from hypothesis_agent.api.router import api_router
from hypothesis_agent.api.ui import ui_router
from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.db.firebase import FirebaseHandle, initialize_firebase
from hypothesis_agent.logging import configure_logging
from hypothesis_agent.metrics import record_request_metrics
from hypothesis_agent.repositories.hypothesis_repository import (
    FirestoreHypothesisRepository,
    HypothesisRepository,
    InMemoryHypothesisRepository,
)
from hypothesis_agent.services.hypothesis_service import HypothesisService
from hypothesis_agent.telemetry import RequestContextMiddleware
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


def _make_request_recorder(enabled: bool):
    def _record(request: Request, response: Response, latency: float) -> None:
        if not enabled:
            return
        route = request.scope.get("path") or request.url.path
        record_request_metrics(request.method, route, response.status_code, latency)

    return _record


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    firebase_handle: FirebaseHandle | None = None
    repository: HypothesisRepository
    if app_settings.use_firestore:
        firebase_handle = initialize_firebase(app_settings)
        repository = FirestoreHypothesisRepository(firebase_handle.client, firebase_handle.collection)
    else:
        repository = InMemoryHypothesisRepository()
    workflow_client = HypothesisWorkflowClient(
        namespace=app_settings.temporal_namespace,
        task_queue=app_settings.temporal_task_queue,
        workflow=app_settings.temporal_workflow,
        address=app_settings.temporal_address,
    )
    hypothesis_service = HypothesisService(
        repository=repository,
        workflow_client=workflow_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = app_settings
        app.state.firebase = firebase_handle
        app.state.workflow_client = workflow_client
        app.state.hypothesis_repository = repository
        app.state.hypothesis_service = hypothesis_service
        try:
            yield
        finally:
            await workflow_client.close()
            if firebase_handle is not None:
                await firebase_handle.dispose()

    application = FastAPI(
        title="RAVEN Hypothesis Validation API",
        version="0.1.0",
        description="Accepts hypotheses for validation and orchestrates the RAVEN workflow.",
        lifespan=lifespan,
    )
    application.add_middleware(
        RequestContextMiddleware,
        recorder=_make_request_recorder(app_settings.enable_prometheus),
    )

    application.include_router(ui_router)
    application.include_router(api_router, prefix=app_settings.api_prefix)

    # Expose core components immediately for compatibility with existing tests and tooling.
    application.state.settings = app_settings
    application.state.firebase = firebase_handle
    application.state.workflow_client = workflow_client
    application.state.hypothesis_repository = repository
    application.state.hypothesis_service = hypothesis_service

    if app_settings.enable_prometheus:
        @application.get("/metrics")
        async def metrics_endpoint() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
