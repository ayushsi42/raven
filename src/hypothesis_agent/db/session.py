"""Database session management."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (AsyncEngine, AsyncSession, async_sessionmaker,
                                    create_async_engine)

from hypothesis_agent.config import AppSettings
from hypothesis_agent.db.base import Base


@dataclass(slots=True)
class Database:
    """Manage database engine and session factory lifecycle."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "Database":
        """Build a database instance from application settings."""

        engine = create_async_engine(settings.database_url, future=True)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return cls(engine=engine, session_factory=session_factory)

    async def create_all(self) -> None:
        """Create all database tables defined in the metadata."""

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop all database tables; primarily useful for tests."""

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        """Dispose the database engine and release resources."""

        await self.engine.dispose()
