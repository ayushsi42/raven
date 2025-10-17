"""FastAPI application entrypoint for the RAVEN hypothesis agent."""
from __future__ import annotations

from fastapi import FastAPI

from hypothesis_agent.api.router import api_router
from hypothesis_agent.api.ui import ui_router
from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.db.migrations import upgrade_database
from hypothesis_agent.db.session import Database
from hypothesis_agent.logging import configure_logging
from hypothesis_agent.repositories.hypothesis_repository import SqlAlchemyHypothesisRepository
from hypothesis_agent.services.hypothesis_service import HypothesisService
from hypothesis_agent.workflows.hypothesis_workflow import HypothesisWorkflowClient


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    application = FastAPI(
        title="RAVEN Hypothesis Validation API",
        version="0.1.0",
        description="Accepts hypotheses for validation and orchestrates the RAVEN workflow.",
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

    @application.on_event("startup")
    async def on_startup() -> None:
        await upgrade_database(app_settings.database_url)

    @application.on_event("shutdown")
    async def on_shutdown() -> None:
        await workflow_client.close()
        await database.dispose()

    return application


app = create_app()
