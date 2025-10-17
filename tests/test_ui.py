"""Smoke tests for the public landing page."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from tests.test_hypothesis_endpoint import test_app  # noqa: F401


@pytest.mark.asyncio
async def test_landing_page_served(test_app) -> None:
    """The UI endpoint should respond with the branded landing page."""

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "RAVEN Hypothesis Studio" in response.text
    assert "hypothesis-form" in response.text