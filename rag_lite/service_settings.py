"""
Service-layer settings for the RAG-Lite HTTP API.

These are *deployment* concerns (where the server binds, auth, CORS) and are kept
separate from the pipeline `Config` (models, retrieval, storage) which is loaded via
`Config.from_env()`. Both read from the same process environment / .env file.
"""

from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """HTTP service configuration, populated from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = Field("0.0.0.0", alias="API_HOST")
    port: int = Field(8000, alias="API_PORT")

    # When set, every request must send `X-API-Key: <api_key>`. Unset = open (dev only).
    api_key: Optional[str] = Field(None, alias="API_KEY")

    # Comma-separated list of allowed CORS origins, or "*" for all.
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")

    # Default collection served by the API when a request omits one.
    default_collection: str = Field("rag_lite", alias="PG_COLLECTION")

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        """Allow CORS_ORIGINS to be a comma-separated string in the environment."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
