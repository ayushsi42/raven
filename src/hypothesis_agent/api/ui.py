"""UI route handlers for the RAVEN web interface."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from hypothesis_agent.config import get_settings

# Setup templates directory relative to this file
_BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

ui_router = APIRouter(tags=["UI"])


@ui_router.get("/", response_class=HTMLResponse)
async def get_landing_page(request: Request):
    """Render the application landing page."""
    settings = get_settings()
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
            "api_prefix": settings.api_prefix,
        },
    )