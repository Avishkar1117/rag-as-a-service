from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemma_model: str = "gemma-4-31b-it"
    ragas_judge_model: str = "gemma-4-31b-it"
    ocr_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "models/gemini-embedding-001"
    github_token: str = ""
    github_model: str = "gpt-4o"
    github_api_base: str = "https://models.inference.ai.azure.com"

    chunk_size: int = 1024
    chunk_overlap: int = 50
    top_k: int = 8

    chroma_persist_dir: str = "./chroma_store"

    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 86400
    answer_cache_ttl: int = 3600

    sentry_dsn: str = ""
    log_level: str = "INFO"


# Required fields (gemini_api_key) are populated from the environment / .env;
# mypy can't see that, hence the ignore.
settings = Settings()  # type: ignore[call-arg]
