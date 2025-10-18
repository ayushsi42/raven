"""Tests covering the hypothesis API endpoints."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest_asyncio

from hypothesis_agent.config import get_settings
from hypothesis_agent.main import create_app
from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.services.hypothesis_service import HypothesisService

class _StubWorkflowExecutionInfo:
    def __init__(self, status: str, history_length: int) -> None:
        self.status = type("Status", (), {"name": status})()
        self.history_length = history_length


class _StubWorkflowDescription:
    def __init__(self, status: str, history_length: int) -> None:
        self.workflow_execution_info = _StubWorkflowExecutionInfo(status, history_length)


class _StubWorkflowHandle:
    def __init__(
        self,
        workflow_id: str,
        status: str,
        history_length: int,
        result_payload: dict,
        milestones: list[dict],
    ) -> None:
        self.id = workflow_id
        self.run_id = f"{workflow_id}-run"
        self._description = _StubWorkflowDescription(status, history_length)
        self._result_payload = result_payload
        self._milestones = milestones

    async def describe(self) -> _StubWorkflowDescription:
        return self._description

    async def result(self) -> dict:
        return self._result_payload

    async def query(self, name: str) -> list[dict]:
        if name == "milestones":
            return self._milestones
        raise ValueError(name)


class _StubTemporalClient:
    def __init__(self) -> None:
        self._summaries: dict[str, dict] = {}
        self._milestones: dict[str, list[dict]] = {}

    async def start_workflow(self, workflow: str, payload: dict, *, id: str, task_queue: str):
        self.last_started = {
            "workflow": workflow,
            "payload": payload,
            "id": id,
            "task_queue": task_queue,
        }
        summary, milestones = _build_stub_summary(HypothesisRequest.model_validate(payload))
        self._status = "COMPLETED"
        self._history_length = 42
        self._summaries[id] = summary
        self._milestones[id] = milestones
        return _StubWorkflowHandle(id, self._status, self._history_length, summary, milestones)

    def get_workflow_handle(self, workflow_id: str, run_id: str) -> _StubWorkflowHandle:
        summary = self._summaries.get(workflow_id, {})
        milestones = self._milestones.get(workflow_id, [])
        return _StubWorkflowHandle(
            workflow_id,
            getattr(self, "_status", "RUNNING"),
            getattr(self, "_history_length", 0),
            summary,
            milestones,
        )

    async def close(self) -> None:  # pragma: no cover - stubbed close
        return None

def _build_stub_summary(request: HypothesisRequest) -> tuple[dict, list[dict]]:
    milestones = [
        {"name": "data_ingest", "status": "completed", "detail": "Market, filings, news fetched."},
        {"name": "entity_resolution", "status": "completed", "detail": "Entities resolved."},
        {"name": "preprocessing", "status": "completed", "detail": "Normalized datasets."},
        {"name": "analysis", "status": "completed", "detail": "Financial diagnostics computed."},
        {"name": "sentiment", "status": "completed", "detail": "Sentiment scored."},
        {"name": "modeling", "status": "completed", "detail": "Scenario modeling finished."},
        {"name": "advanced_modeling", "status": "completed", "detail": "Advanced metrics computed."},
        {"name": "human_review", "status": "completed", "detail": "Human review skipped."},
        {"name": "report_generation", "status": "completed", "detail": "Report assembled."},
    ]
    summary = {
        "score": 0.63,
        "conclusion": "Partially supported",
        "confidence": 0.58,
        "evidence": [],
        "current_stage": "report_generation",
        "milestones": milestones,
    }
    return summary, milestones


@pytest_asyncio.fixture
async def test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
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

    await app.router.startup()
    try:
        yield app
    finally:
        await app.router.shutdown()
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

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        post_response = await client.post("/v1/hypotheses", json=payload)
        assert post_response.status_code == 200
        post_data = post_response.json()

        hypothesis_id = UUID(post_data["hypothesis_id"])
        get_response = await client.get(f"/v1/hypotheses/{hypothesis_id}")
        status_response = await client.get(f"/v1/hypotheses/{hypothesis_id}/status")
        assert post_data["workflow_id"].startswith("hypothesis-")
        assert post_data["workflow_run_id"].endswith("-run")

    assert get_response.status_code == 200
    get_data = get_response.json()

    assert UUID(get_data["hypothesis_id"]) == hypothesis_id
    assert get_data["workflow_id"] == post_data["workflow_id"]
    assert get_data["workflow_run_id"] == post_data["workflow_run_id"]
    assert get_data["status"] == "accepted"
    assert isinstance(get_data["validation"]["score"], float)
    assert isinstance(get_data["validation"]["confidence"], float)
    assert get_data["validation"]["current_stage"] in {"report_generation", "human_review"}
    milestones = get_data["validation"].get("milestones", [])
    assert any(m["name"] == "report_generation" for m in milestones)
    assert all("status" in m for m in milestones)

    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["workflow_status"] == "COMPLETED"
    assert status_data["workflow_history_length"] == 42
    assert status_data["validation"]["current_stage"] == "report_generation"
    assert len(status_data["validation"]["milestones"]) >= 1
    assert status_data["validation"]["milestones"][-1]["name"] == "report_generation"


@pytest.mark.asyncio
async def test_get_unknown_hypothesis_returns_404(test_app) -> None:
    """Ensure retrieving an unknown hypothesis returns a 404 error."""

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/v1/hypotheses/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("Hypothesis")


@pytest.mark.asyncio
async def test_get_unknown_hypothesis_status_returns_404(test_app) -> None:
    """Ensure status endpoint returns 404 for unknown hypotheses."""

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"/v1/hypotheses/{uuid4()}/status")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("Hypothesis")
