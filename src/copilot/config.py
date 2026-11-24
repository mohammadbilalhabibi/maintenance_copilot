"""Application settings, read from environment variables and .env."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project folder (data/, models/, index/, knowledge/). Set COPILOT_HOME when the package is
# installed outside the source tree (the Docker image sets /app).
PROJECT_ROOT = Path(os.environ.get("COPILOT_HOME") or Path(__file__).resolve().parents[2])


class Settings(BaseSettings):
    """All runtime configuration. Secrets come only from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database ---
    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'copilot.sqlite').as_posix()}",
        description="Read-write database (users, chats, tickets, fleet data loading).",
    )
    tools_database_url: str | None = Field(
        default=None,
        description="Read-only connection used by the agent's data tools. "
        "Defaults to DATABASE_URL (SQLite is then opened read-only).",
    )
    fleet_size: int = Field(100, description="Machines 1..N are loaded into the database.")
    sim_now: datetime = Field(
        datetime(2015, 11, 15, 8, 0),
        description="Simulated 'current time'. Tools never return data after it, "
        "so the agent cannot see the future of the 2015 sample data.",
    )

    # --- LLM (any OpenAI-compatible API) ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr = SecretStr("")
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_temperature: float | None = 0.0
    llm_max_output_tokens: int = 800
    agent_max_steps: int = Field(8, description="Max LLM turns per question (loop guard).")
    tool_timeout: float = 15.0
    checkpoint_durability: str = Field(
        "exit", description="LangGraph checkpoint writes: 'async' (after every step) or "
        "'exit' (only when a run ends or pauses for confirmation; fewer DB writes)."
    )
    # USD per 1M tokens for the configured model, used to report cost per request.
    llm_price_input_per_m: float = 0.15
    llm_price_cached_input_per_m: float = 0.075
    llm_price_output_per_m: float = 0.60

    # --- Models and documents ---
    model_path: Path = PROJECT_ROOT / "models" / "failure_risk.joblib"
    knowledge_dir: Path = PROJECT_ROOT / "knowledge"
    index_dir: Path = PROJECT_ROOT / "index"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None
    embedding_batch_size: int = Field(16, description="Texts per embedding batch (memory).")
    search_top_k: int = 4
    rrf_k: int = 60

    # --- Auth / API ---
    jwt_secret: SecretStr = SecretStr("change-me-in-env")
    jwt_expire_minutes: int = 12 * 60
    rate_limit_per_minute: int = 20
    allow_registration: bool = True
    daily_question_limit: int = Field(
        0, description="Questions per UTC day for all users together (protects the LLM budget "
        "of a public demo; counted from stored messages, so restarts do not reset it); 0 = off."
    )

    # --- Observability ---
    phoenix_collector_endpoint: str | None = Field(
        default=None, description="e.g. http://localhost:6006/v1/traces; unset = tracing off."
    )
    phoenix_project_name: str = "maintenance-copilot"
    phoenix_api_key: SecretStr | None = Field(
        default=None, description="API key for a Phoenix with login enabled (sent as Bearer)."
    )

    @property
    def tools_url(self) -> str:
        """Connection URL for read-only tool queries."""
        return self.tools_database_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
