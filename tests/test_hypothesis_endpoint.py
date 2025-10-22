"""Tests covering the hypothesis API endpoints."""
from __future__ import annotations

import asyncio
import copy
from datetime import date
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest_asyncio

from hypothesis_agent.config import AppSettings, get_settings
from hypothesis_agent.llm import BaseLLM
from hypothesis_agent.main import create_app
from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.services.hypothesis_service import HypothesisService
from hypothesis_agent.orchestration.langgraph_pipeline import LangGraphValidationOrchestrator, StageExecutionResult
from hypothesis_agent.storage.artifact_store import ArtifactStore


class _StubLLM(BaseLLM):
    def generate_data_plan(self, request: HypothesisRequest) -> list[str]:
        return ["Collect historical prices", "Gather earnings transcripts", "Pull sentiment data"]

    def generate_analysis_plan(self, request: HypothesisRequest, data_overview: dict[str, object]) -> list[str]:
        return ["Calculate price momentum", "Summarise sentiment signals", "Evaluate margin trends"]

    def generate_detailed_analysis(self, request: HypothesisRequest, metrics_overview: dict[str, object]) -> str:
        return "The hypothesis remains plausible given momentum, filings cadence, and sentiment balance."

    def generate_report(
        self,
        request: HypothesisRequest,
        metrics_overview: dict[str, object],
        analysis_summary: str,
        artifact_paths: list[str],
    ) -> dict[str, object]:
        return {
            "executive_summary": "Moderate support for the investment thesis.",
            "key_findings": ["Momentum positive", "Filings cadence stable", "Sentiment constructive"],
            "risks": ["Macro slowdown"],
            "next_steps": ["Monitor earnings guidance"],
        }

    def generate_analysis_code(
        self,
        *,
        request: HypothesisRequest,
        analysis_plan: list[dict[str, object]],
        data_artifacts: dict[str, str],
        data_format: dict[str, str],
        attempt: int,
        history: list[dict[str, str]],
    ) -> str:
        return (
            """```python\n"""
            "result = {\n"
            "    \"steps\": [\n"
            "        {\n"
            "            \"name\": \"metric_snapshot\",\n"
            "            \"outputs\": [\n"
            "                {\"label\": \"Revenue Growth\", \"value\": 0.12},\n"
            "                {\"label\": \"Sentiment Score\", \"value\": 0.2}\n"
            "            ]\n"
            "        }\n"
            "    ],\n"
            "    \"aggregated\": {\n"
            "        \"revenue_growth\": 0.12,\n"
            "        \"sentiment_score\": 0.2\n"
            "    },\n"
            "    \"insights\": [\"Revenue acceleration remains healthy.\"],\n"
            "    \"artifacts\": []\n"
            "}\n"
            "print(\"RESULT::\" + json.dumps(result))\n"
            "```"""
        )


STUB_TOOL_RESPONSES: dict[str, dict[str, object]] = {
    "ALPHA_VANTAGE_TIME_SERIES_MONTHLY_ADJUSTED": {
        "Monthly Adjusted Time Series": {
            "2024-01-31": {"5. adjusted close": "100.0"},
            "2024-02-29": {"5. adjusted close": "102.0"},
            "2024-03-31": {"5. adjusted close": "104.5"},
            "2024-04-30": {"5. adjusted close": "106.0"},
            "2024-05-31": {"5. adjusted close": "108.5"},
            "2024-06-30": {"5. adjusted close": "112.0"},
        }
    },
    "ALPHA_VANTAGE_COMPANY_OVERVIEW": {
        "OperatingMarginTTM": "0.21",
        "ProfitMargin": "0.15",
    },
    "ALPHA_VANTAGE_NEWS_SENTIMENT": {
        "feed": [
            {"overall_sentiment_score": 0.3},
            {"overall_sentiment_score": 0.1},
        ]
    },
    "ALPHA_VANTAGE_CASH_FLOW": {
        "annualReports": [
            {"operatingCashflow": "1200000"},
            {"operatingCashflow": "900000"},
        ]
    },
    "ALPHA_VANTAGE_BALANCE_SHEET": {
        "annualReports": [
            {"totalAssets": "5000000", "totalLiabilities": "2100000"},
        ]
    },
}


class _StubTool:
    def __init__(self, sink: list[dict[str, object]], slug: str, responses: dict[str, dict[str, object]]) -> None:
        self._sink = sink
        self._slug = slug
        self._responses = responses

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        record = {"slug": self._slug, "arguments": copy.deepcopy(payload)}
        self._sink.append(record)
        response = self._responses.get(self._slug, {"ok": True})
        return copy.deepcopy(response)


class _StubToolSet:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.invocations: list[dict[str, object]] = []
        self._responses = responses

    def configure(self, responses: dict[str, dict[str, object]]) -> None:
        self._responses = responses

    def get_tool(self, name: str) -> _StubTool:
        return _StubTool(self.invocations, name, self._responses)


class _FakeSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return self._data


class _FakeDocumentReference:
    def __init__(self, storage: dict, key: str) -> None:
        self._storage = storage
        self._key = key

    def set(self, data: dict) -> None:
        self._storage[self._key] = data

    def get(self) -> _FakeSnapshot:
        return _FakeSnapshot(self._storage.get(self._key))


class _FakeCollection:
    def __init__(self, storage: dict) -> None:
        self._storage = storage

    def document(self, key: str) -> _FakeDocumentReference:
        return _FakeDocumentReference(self._storage, key)


class _FakeFirestoreClient:
    def __init__(self) -> None:
        self._storage: dict[str, dict[str, dict]] = {}

    def collection(self, name: str) -> _FakeCollection:
        bucket = self._storage.setdefault(name, {})
        return _FakeCollection(bucket)


class _FakeFirebaseHandle:
    def __init__(self, collection: str = "hypotheses") -> None:
        self.client = _FakeFirestoreClient()
        self.collection = collection

    async def dispose(self) -> None:
        pass


@pytest_asyncio.fixture
async def test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    """Provide an application instance backed by a fake Firestore datastore."""
    firebase_handle = _FakeFirebaseHandle()
    monkeypatch.setattr(
        "hypothesis_agent.db.firebase.initialize_firebase",
        lambda settings: firebase_handle,
    )
    monkeypatch.setattr(
        "hypothesis_agent.main.initialize_firebase",
        lambda settings: firebase_handle,
    )
    artifact_root = tmp_path / "artifacts"
    settings = AppSettings(
        artifact_store_path=str(artifact_root),
    )
    toolset = _StubToolSet(copy.deepcopy(STUB_TOOL_RESPONSES))

    def _factory() -> LangGraphValidationOrchestrator:
        artifact_store = ArtifactStore.from_path(settings.artifact_store_path)
        toolset.invocations.clear()
        toolset.configure(copy.deepcopy(STUB_TOOL_RESPONSES))
        return LangGraphValidationOrchestrator(
            settings=settings,
            llm=_StubLLM(),
            artifact_store=artifact_store,
            toolset=toolset,
        )

    monkeypatch.setattr(
        "hypothesis_agent.workflows.hypothesis_workflow.LangGraphValidationOrchestrator",
        _factory,
    )
    get_settings.cache_clear()  # type: ignore[attr-defined]
    app = create_app()
    app.state.stub_toolset = toolset
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
        assert post_data["workflow_id"].startswith("hypothesis-")
        assert post_data["workflow_run_id"].endswith("-run")
        assert post_data["validation"]["current_stage"] == "plan_generation"

        initial_get = await client.get(f"/v1/hypotheses/{hypothesis_id}")
        assert initial_get.status_code == 200
        initial_data = initial_get.json()
        assert initial_data["validation"]["current_stage"] == "plan_generation"

        status_data = await _poll_status_response(client, hypothesis_id)
        assert status_data["workflow_status"] in {"COMPLETED", "AWAITING_REVIEW"}

        get_response = await client.get(f"/v1/hypotheses/{hypothesis_id}")

    assert get_response.status_code == 200
    get_data = get_response.json()

    assert UUID(get_data["hypothesis_id"]) == hypothesis_id
    assert get_data["workflow_id"] == post_data["workflow_id"]
    assert get_data["workflow_run_id"] == post_data["workflow_run_id"]
    expected_status = "awaiting_review" if status_data["workflow_status"] == "AWAITING_REVIEW" else "completed"
    assert get_data["status"] == expected_status
    assert isinstance(get_data["validation"]["score"], float)
    assert isinstance(get_data["validation"]["confidence"], float)
    assert get_data["validation"]["current_stage"] in {"delivery", "human_review"}
    milestones = get_data["validation"].get("milestones", [])
    assert any(m["name"] == "report_generation" for m in milestones)
    assert all("status" in m for m in milestones)

    assert status_data["workflow_history_length"] <= len(status_data["validation"]["milestones"])
    if status_data["workflow_status"] == "COMPLETED":
        assert status_data["validation"]["current_stage"] == "delivery"
        assert status_data["validation"]["milestones"][-1]["name"] == "delivery"
        assert (
            status_data["validation"]["milestones"][-1]["detail"]
            == "Report available for download."
        )


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



@pytest.mark.asyncio
async def test_cancel_endpoint_stops_workflow(test_app, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a workflow should return a cancelled status payload."""

    import time
    import hypothesis_agent.workflows.hypothesis_workflow as wf_module

    base_factory = wf_module.LangGraphValidationOrchestrator

    def slow_factory() -> LangGraphValidationOrchestrator:
        orchestrator = base_factory()
        original_run_stage = orchestrator.run_stage

        def slow_run_stage(stage: str, request: HypothesisRequest, context: dict[str, object]) -> StageExecutionResult:
            time.sleep(0.05)
            return original_run_stage(stage, request, context)

        orchestrator.run_stage = slow_run_stage  # type: ignore[assignment]
        return orchestrator

    monkeypatch.setattr(wf_module, "LangGraphValidationOrchestrator", slow_factory)

    payload = {
        "user_id": "user-stop",
        "hypothesis_text": "Cancel via API",
        "entities": ["Company Stop"],
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
        hypothesis_id = UUID(post_response.json()["hypothesis_id"])

        await asyncio.sleep(0.01)

        cancel_response = await client.post(f"/v1/hypotheses/{hypothesis_id}/cancel")
        assert cancel_response.status_code == 200
        cancel_data = cancel_response.json()
        assert cancel_data["status"] == "cancelled"
        assert cancel_data["workflow_status"] == "CANCELLED"

        status_response = await client.get(f"/v1/hypotheses/{hypothesis_id}/status")

    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "cancelled"
    assert status_data["workflow_status"] == "CANCELLED"
    assert status_data["validation"]["current_stage"] == "cancelled"


async def _poll_status_response(
    client: AsyncClient,
    hypothesis_id: UUID,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> dict[str, object]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_data: dict[str, object] | None = None
    while loop.time() < deadline:
        response = await client.get(f"/v1/hypotheses/{hypothesis_id}/status")
        assert response.status_code == 200
        data = response.json()
        last_data = data
        if data.get("workflow_status") in {"COMPLETED", "AWAITING_REVIEW"}:
            return data
        await asyncio.sleep(poll_interval)
    raise AssertionError(
        f"Workflow {hypothesis_id} did not reach a terminal status. Last payload: {last_data}"
    )
