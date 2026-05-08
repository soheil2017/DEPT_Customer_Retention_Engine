from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent.parent.parent  # retention_engine/


def _find_data_file(filename: str) -> Path:
    """Auto-detect data file location.

    Priority:
      1. data/  — bundled for Vercel / production deployments
      2. ../interview-case-files/  — local dev with original case files
    """
    candidates = [
        _HERE / "data" / filename,
        _HERE / "../interview-case-files" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_HERE / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #  LLM 
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"   # swap to gpt-4o for higher quality

    # Langfuse observability 
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    #  Paths 
    model_path: Path = _find_data_file("churn_model.pkl")
    customer_db_path: Path = _find_data_file("Vodafone_Customer_Database.csv")

    #  Business rules 
    churn_threshold: float = 0.5

    #  API 
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


settings = Settings()
