from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "anthropic"
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.5
    openai_model: str = "gpt-4o"
    openai_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-5-20251101"
    anthropic_api_key: str | None = None
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
