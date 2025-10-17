"""Tests covering the hypothesis API endpoints."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from hypothesis_agent.config import get_settings
from hypothesis_agent.main import create_app
from hypothesis_agent.services.hypothesis_service import HypothesisService


class _StubWorkflowHandle:
    def __init__(self, workflow_id: str) -> None:
        self.id = workflow_id
        self.run_id = f"{workflow_id}-run"


class _StubTemporalClient:
    async def start_workflow(self, workflow: str, payload: dict, *, id: str, task_queue: str):
        self.last_started = {
            "workflow": workflow,
            "payload": payload,
            "id": id,
            "task_queue": task_queue,
        }
        return _StubWorkflowHandle(id)

    async def close(self) -> None:  # pragma: no cover - stubbed close
        return None


@pytest.fixture
def test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """Provide an application instance backed by a temporary SQLite database."""

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("RAVEN_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("RAVEN_TEMPORAL_ADDRESS", "test-address")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    app = create_app()

    stub_client = _StubTemporalClient()
    app.state.workflow_client.temporal_client = stub_client
    app.state.hypothesis_service = HypothesisService(
        repository=app.state.hypothesis_repository,
        workflow_client=app.state.workflow_client,
    )

    yield app

    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_submit_and_retrieve_hypothesis(test_app) -> None:
    """Verify submission persists state and retrieval returns the stored record."""

    payload = {
        "user_id": "user-123",
        "hypothesis_text": "Company X EBITDA margin improves by 200 bps next year.",
        "entities": ["Company X"],
        "time_horizon": {
            "start": date(2025, 1, 1).isoformat(),
            "end": date(2025, 12, 31).isoformat(),
        },
        "risk_appetite": "moderate",
        "requires_human_review": False,
    }

    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        post_response = await client.post("/v1/hypotheses", json=payload)
        assert post_response.status_code == 200
        post_data = post_response.json()

        hypothesis_id = UUID(post_data["hypothesis_id"])
        assert post_data["workflow_id"].startswith("hypothesis-")
        assert post_data["workflow_run_id"].endswith("-run")
        get_response = await client.get(f"/v1/hypotheses/{hypothesis_id}")

    assert get_response.status_code == 200
    get_data = get_response.json()

    assert UUID(get_data["hypothesis_id"]) == hypothesis_id
    assert get_data["workflow_id"] == post_data["workflow_id"]
    assert get_data["workflow_run_id"] == post_data["workflow_run_id"]
    assert get_data["status"] == "accepted"
    assert get_data["validation"]["conclusion"] == "Pending analysis"
    assert get_data["validation"]["score"] == 0.0


@pytest.mark.asyncio
async def test_get_unknown_hypothesis_returns_404(test_app) -> None:
    """Ensure retrieving an unknown hypothesis returns a 404 error."""

    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        response = await client.get(f"/v1/hypotheses/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("Hypothesis")
