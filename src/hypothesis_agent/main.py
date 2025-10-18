"""FastAPI application entrypoint for the RAVEN hypothesis agent."""
from __future__ import annotations

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from hypothesis_agent.api.router import api_router
from hypothesis_agent.api.ui import ui_router
from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.db.migrations import upgrade_database
from hypothesis_agent.db.session import Database
from hypothesis_agent.logging import configure_logging
from hypothesis_agent.metrics import record_request_metrics
from hypothesis_agent.repositories.hypothesis_repository import SqlAlchemyHypothesisRepository
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

    application = FastAPI(
        title="RAVEN Hypothesis Validation API",
        version="0.1.0",
        description="Accepts hypotheses for validation and orchestrates the RAVEN workflow.",
    )
    application.add_middleware(
        RequestContextMiddleware,
        recorder=_make_request_recorder(app_settings.enable_prometheus),
    )

    application.include_router(ui_router)
    application.include_router(api_router, prefix=app_settings.api_prefix)
    application.state.settings = app_settings

    database = Database.from_settings(app_settings)
    workflow_client = HypothesisWorkflowClient(
        namespace=app_settings.temporal_namespace,
        task_queue=app_settings.temporal_task_queue,
        workflow=app_settings.temporal_workflow,
        address=app_settings.temporal_address,
    )
    repository = SqlAlchemyHypothesisRepository(database.session_factory)

    application.state.database = database
    application.state.workflow_client = workflow_client
    application.state.hypothesis_repository = repository
    application.state.hypothesis_service = HypothesisService(
        repository=repository,
        workflow_client=workflow_client,
    )

    if app_settings.enable_prometheus:
        @application.get("/metrics")
        async def metrics_endpoint() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.on_event("startup")
    async def on_startup() -> None:
        await upgrade_database(app_settings.database_url)

    @application.on_event("shutdown")
    async def on_shutdown() -> None:
        await workflow_client.close()
        await database.dispose()

    return application


app = create_app()
