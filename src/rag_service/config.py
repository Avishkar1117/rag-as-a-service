from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemma_model: str = "gemma-4-31b-it"
    ocr_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "models/gemini-embedding-001"
    github_token: str = ""
    github_model: str = "gpt-4o"
    github_api_base: str = "https://models.inference.ai.azure.com"

    # --- RAGAS judge ---------------------------------------------------------
    # `judge_provider` selects which provider's credentials + base URL +
    # default model the judge uses. Switching providers is a one-line .env
    # change; each provider's settings stay warm in code so you can flip
    # back and forth without re-discovering URLs and model slugs.
    #
    # Valid values: "deepseek" | "openrouter" | "gemini"
    judge_provider: str = "deepseek"

    # DeepSeek platform — https://platform.deepseek.com (OpenAI-compatible).
    # Available models per DeepSeek docs: deepseek-v4-flash (default),
    # deepseek-v4-pro, deepseek-chat / deepseek-reasoner (legacy, deprecated
    # 2026-07-24).
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_judge_model: str = "deepseek-v4-flash"

    # OpenRouter — https://openrouter.ai (OpenAI-compatible). Free tier
    # exists but is congested at peak times; useful as a backup.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_judge_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # Gemini-as-judge — same provider as the generator, so reintroduces
    # self-bias on faithfulness. Kept as an escape hatch for when external
    # providers are down.
    gemini_judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_judge_model: str = "gemma-4-31b-it"

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
