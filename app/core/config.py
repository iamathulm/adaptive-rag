from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Adaptive Multimodal RAG"
    environment: str = "development"
    log_level: str = "INFO"
    qdrant_url: str = "http://localhost:6333"
    retrieval_top_k: int = 5
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()