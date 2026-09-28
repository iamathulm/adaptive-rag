from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Adaptive Multimodal RAG"
    environment: str = "development"
    log_level: str = "INFO"
    qdrant_url: str = "http://localhost:6333"
    retrieval_top_k: int = 5
    rrf_k: int = 60
    generation_provider: str = "mock"
    generation_model: str = "gemini-2.0-flash"
    gemini_api_key: SecretStr | None = None
    generation_timeout_seconds: float = 30.0
    generation_max_tokens: int = 512
    generation_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()