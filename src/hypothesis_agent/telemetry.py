"""Telemetry helpers for observability."""
from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to responses and measure latency."""

    def __init__(
        self,
        app: FastAPI,
        recorder: Callable[[Request, Response, float], None],
    ) -> None:
        super().__init__(app)
        self._recorder = recorder

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start
        response.headers["x-request-id"] = request_id
        self._recorder(request, response, latency)
        return response
