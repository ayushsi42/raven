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
    return templates.TemplateResponse(
        request,
        "landing.html",
        {
            "api_prefix": settings.api_prefix,
        },
    )