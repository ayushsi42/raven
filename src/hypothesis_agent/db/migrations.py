"""Programmatic helpers for running Alembic migrations."""
from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_config(database_url: str) -> Config:
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "alembic.ini"
    if not config_path.exists():  # pragma: no cover - defensive guard for misconfiguration
        raise FileNotFoundError("alembic.ini not found at project root")

    alembic_config = Config(str(config_path))
    script_location = project_root / "alembic"
    alembic_config.set_main_option("script_location", str(script_location))
    alembic_config.set_main_option("sqlalchemy.url", database_url)
    return alembic_config


async def upgrade_database(database_url: str, revision: str = "head") -> None:
    """Run Alembic migrations to the requested revision asynchronously."""

    loop = asyncio.get_running_loop()
    config = _alembic_config(database_url)
    await loop.run_in_executor(None, command.upgrade, config, revision)