"""API key authentication utilities."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from hypothesis_agent.config import AppSettings, get_settings

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_api_key(
    request: Request,
    api_key_header: str | None = Depends(API_KEY_HEADER),
    settings: AppSettings | None = None,
) -> None:
    settings = settings or get_settings()
    configured_key = settings.api_key
    if configured_key is None:
        return
    provided = api_key_header or request.headers.get("authorization")
    if provided is None or provided != configured_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
