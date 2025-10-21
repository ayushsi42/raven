"""Public-facing UI routes for the RAVEN hypothesis agent."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ui_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


@ui_router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request) -> HTMLResponse:
    """Render the primary hypothesis submission interface."""

    settings = request.app.state.settings
    firebase_config = {
        "apiKey": settings.firebase_web_api_key,
        "authDomain": settings.firebase_web_auth_domain,
        "projectId": settings.firebase_project_id,
        "storageBucket": settings.firebase_web_storage_bucket,
        "messagingSenderId": settings.firebase_web_messaging_sender_id,
        "appId": settings.firebase_web_app_id,
        "measurementId": settings.firebase_web_measurement_id,
    }
    firebase_config = {key: value for key, value in firebase_config.items() if value}
    return templates.TemplateResponse(
        request,
        "landing.html",
        {
            "api_prefix": settings.api_prefix,
            "firebase_config": firebase_config,
            "require_authentication": settings.require_authentication,
        },
    )