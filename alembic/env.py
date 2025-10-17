"""Alembic environment configuration."""
from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any, Dict

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from hypothesis_agent.config import get_settings
from hypothesis_agent.db.base import Base

# Interpret the config file for Python logging.
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None

target_metadata = Base.metadata


def _database_url() -> str:
    settings = get_settings()
    env_url = settings.database_url
    return env_url or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""

    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    configuration: Dict[str, Any] = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = AsyncEngine(
        create_async_engine(
            configuration["sqlalchemy.url"],
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async def do_run_migrations(connection: Connection) -> None:
        await connection.run_sync(
            lambda sync_conn: context.configure(connection=sync_conn, target_metadata=target_metadata)
        )
        await connection.run_sync(lambda sync_conn: context.run_migrations())

    async def run_async_migrations() -> None:
        async with connectable.connect() as connection:
            await do_run_migrations(connection)
        await connectable.dispose()

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
