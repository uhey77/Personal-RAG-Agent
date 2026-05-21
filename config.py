# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"

BASE_COLLECTION_NAME = "personal_knowledge_base"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

EXCLUDED_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".crt",
}


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chat_provider: str = "google"
    chat_model: str = ""
    embedding_provider: str = "google"
    embedding_model: str = ""
    min_relevance_score: float = 0.2
    openai_api_key: str = Field(default="", repr=False)
    google_api_key: str = Field(default="", repr=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    @field_validator("chat_provider", "embedding_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator("chat_model", "embedding_model", mode="before")
    @classmethod
    def normalize_model(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("min_relevance_score", mode="before")
    @classmethod
    def parse_min_relevance_score(cls, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.2

    @model_validator(mode="after")
    def apply_model_defaults(self) -> "AppConfig":
        if not self.chat_model:
            self.chat_model = (
                "gemini-2.5-flash-lite"
                if self.chat_provider == "google"
                else "gpt-4.1-mini"
            )

        if not self.embedding_model:
            self.embedding_model = (
                "gemini-embedding-001"
                if self.embedding_provider == "google"
                else "text-embedding-3-small"
            )

        return self

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def docs_dir(self) -> Path:
        return DOCS_DIR

    @property
    def chroma_dir(self) -> Path:
        return CHROMA_DIR

    @property
    def base_collection_name(self) -> str:
        return BASE_COLLECTION_NAME

    @property
    def collection_name(self) -> str:
        raw_name = f"{self.base_collection_name}_{self.embedding_provider}_{self.embedding_model}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
        safe_name = re.sub(r"\.{2,}", ".", safe_name).strip("._-")

        if len(safe_name) < 3:
            safe_name = f"{safe_name}_kb".strip("._-")

        if len(safe_name) > 63:
            digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
            safe_name = f"{safe_name[:54].rstrip('._-')}_{digest}"

        if not safe_name[0].isalnum():
            safe_name = f"kb_{safe_name}"
        if not safe_name[-1].isalnum():
            safe_name = f"{safe_name}_kb"

        return safe_name[:63]

    def ensure_data_dirs(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def get_app_config() -> AppConfig:
    config = AppConfig()
    config.ensure_data_dirs()
    return config
