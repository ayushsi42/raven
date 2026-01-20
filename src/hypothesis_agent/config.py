"""Application configuration models and access helpers."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    artifact_store_path: str = "./data/artifacts"
    
    # Required API keys
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    
    # Optional settings
    sec_user_agent: str = "RAVEN/0.1 (support@example.com)"
    notification_email: str | None = None
    enable_prometheus: bool = True

    model_config = SettingsConfigDict(case_sensitive=False, env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached instance of the application settings."""

    return AppSettings()
