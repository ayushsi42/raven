"""Tests for repository implementations."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from pytest import TempPathFactory

from hypothesis_agent.config import AppSettings
from hypothesis_agent.db.session import Database
from hypothesis_agent.models.hypothesis import HypothesisRequest, TimeHorizon, ValidationSummary
from hypothesis_agent.repositories.hypothesis_repository import HypothesisRecord, SqlAlchemyHypothesisRepository

@pytest.mark.asyncio
async def test_sqlalchemy_repository_persists_and_retrieves(tmp_path_factory: TempPathFactory) -> None:
	"""Ensure the SQLAlchemy repository can round-trip records."""

	db_path = tmp_path_factory.mktemp("db") / "repo.db"
	settings = AppSettings(database_url=f"sqlite+aiosqlite:///{db_path}")
	database = Database.from_settings(settings)
	await database.create_all()
	repository = SqlAlchemyHypothesisRepository(database.session_factory)

	hypothesis_request = HypothesisRequest(
		user_id="user-456",
		hypothesis_text="Revenue growth stabilizes",
		entities=["Company Y"],
		time_horizon=TimeHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31)),
	)
	record = HypothesisRecord(
		hypothesis_id=uuid4(),
		request=hypothesis_request,
		status="accepted",
		validation=ValidationSummary(score=0.0, conclusion="Pending analysis", confidence=0.0, evidence=[]),
	)

	await repository.save(record)
	fetched = await repository.get(record.hypothesis_id)

	assert fetched is not None
	assert fetched.hypothesis_id == record.hypothesis_id
	assert fetched.request == record.request
	await database.dispose()
