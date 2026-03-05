from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────
    app_env: str = "development"
    app_debug: bool = False
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── Database ─────────────────────────────────
    database_url: str  # asyncpg DSN

    # ── Supabase ─────────────────────────────────
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "uploads"

    # ── OCR ──────────────────────────────────────
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ── Ollama / LLM ─────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: float = 120.0  # seconds — LLMs can be slow on CPU
    ollama_max_retries: int = 2

    # ── Summary cache ─────────────────────────────
    summary_cache_ttl_seconds: int = 86_400  # 24 hours

    # ── CORS ─────────────────────────────────────
    allowed_origins: str = "http://localhost:3000"
    # Clinic backend base URL (used to resolve external tokens)
    clinic_backend_url: str = "http://localhost:4000"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
