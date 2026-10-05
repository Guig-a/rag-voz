from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="rag-voz")
    environment: str = Field(default="development")
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/rag"
    )
    llm_provider: str = Field(default="gemini")
    llm_api_key: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    llm_max_tokens: int = Field(default=1024, ge=1)
    embedding_model: str = Field(default="intfloat/multilingual-e5-small")
    embedding_dim: int = Field(default=384, ge=1)
    embedding_document_prefix: str = Field(default="passage:")
    embedding_query_prefix: str = Field(default="query:")
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)
    search_default_k: int = Field(default=5, ge=1)
    rag_min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_delay_seconds: float = Field(default=1.0, ge=0.0)
    eval_retry_attempts: int = Field(default=3, ge=1)
    eval_retry_delay_seconds: float = Field(default=2.0, ge=0.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chunk_step(self) -> int:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP precisa ser menor que CHUNK_SIZE.")

        return self.chunk_size - self.chunk_overlap


@lru_cache
def get_settings() -> Settings:
    return Settings()
