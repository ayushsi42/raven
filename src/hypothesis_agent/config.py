"""Application configuration models and access helpers."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "raven-hypothesis"
    temporal_address: str = "localhost:7233"
    temporal_workflow: str = "HypothesisValidationWorkflow"
    database_url: str = "sqlite+aiosqlite:///./raven.db"
    artifact_store_path: str = "./data/artifacts"
    alpha_vantage_api_key: str = "demo"
    sec_user_agent: str = "RAVEN/0.1 (support@example.com)"
    api_key: str | None = None
    enable_prometheus: bool = True

    model_config = SettingsConfigDict(env_prefix="RAVEN_", case_sensitive=False)


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached instance of the application settings."""

    return AppSettings()
