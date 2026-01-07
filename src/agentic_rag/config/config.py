from __future__ import annotations

from pydantic import Field, SecretStr, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object.

    BaseSettings reads from environment variables (and optionally a .env file).
    This keeps secrets and machine-specific values out of source code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "gemma3:4b"
    EMBED_MODEL: str = "mxbai-embed-large:latest"

    # Storage
    CHROMA_DIR: str = "./.chroma"
    CHROMA_COLLECTION: str = "agentic_rag"
    
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"

    # Postgres
    POSTGRES_DSN: SecretStr | None = None
    
    
    # LLM Provider API Keys (for non-Ollama providers)
    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    
    
    # Langfuse
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_BASE_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )
    LANGFUSE_ENABLED: bool = True


settings = Settings()

