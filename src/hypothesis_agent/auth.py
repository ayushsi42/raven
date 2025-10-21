"""Authentication helpers for API and UI access control."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from firebase_admin import auth as firebase_auth

from hypothesis_agent.config import AppSettings, get_settings

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_authenticated_user(
    request: Request,
    api_key_header: str | None = Depends(API_KEY_HEADER),
    settings: AppSettings = Depends(get_settings),
) -> None:
    """Authorize a request using an API key or Firebase ID token."""

    configured_key = (settings.api_key or "").strip() or None
    provided_key = (api_key_header or "").strip() or None
    if not provided_key:
        auth_header = (request.headers.get("authorization") or "").strip()
        if auth_header.lower().startswith("apikey "):
            provided_key = auth_header[7:].strip() or None
    if configured_key is not None and provided_key == configured_key:
        request.state.auth_context = {"method": "api_key"}
        return

    token = _extract_bearer_token(request)
    if token:
        decoded = await _verify_id_token(token, request, settings)
        request.state.auth_context = {"method": "firebase", "uid": decoded.get("uid"), "claims": decoded}
        request.scope.setdefault("user", decoded)
        return

    if configured_key is not None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    if settings.require_authentication:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def _extract_bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


async def _verify_id_token(token: str, request: Request, settings: AppSettings) -> Dict[str, Any]:
    firebase_handle = getattr(request.app.state, "firebase", None)
    firebase_app = getattr(firebase_handle, "app", None)
    try:
        return await asyncio.to_thread(
            firebase_auth.verify_id_token,
            token,
            app=firebase_app,
            check_revoked=settings.firebase_auth_check_revoked,
        )
    except firebase_auth.ExpiredIdTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authentication token expired") from exc
    except firebase_auth.RevokedIdTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authentication token revoked") from exc
    except Exception as exc:  # pragma: no cover - defensive guard for other auth errors
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc
