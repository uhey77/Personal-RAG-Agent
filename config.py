# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

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


def ensure_data_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_chat_provider() -> str:
    return get_env("CHAT_PROVIDER", "google").lower()


def get_embedding_provider() -> str:
    return get_env("EMBEDDING_PROVIDER", "google").lower()


def get_embedding_model() -> str:
    provider = get_embedding_provider()
    if provider == "google":
        return get_env("EMBEDDING_MODEL", "gemini-embedding-001")
    return get_env("EMBEDDING_MODEL", "text-embedding-3-small")


def get_min_relevance_score() -> float:
    raw_score = get_env("MIN_RELEVANCE_SCORE", "0.2")
    try:
        return float(raw_score)
    except ValueError:
        return 0.2


def get_collection_name() -> str:
    raw_name = f"{BASE_COLLECTION_NAME}_{get_embedding_provider()}_{get_embedding_model()}"
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


ensure_data_dirs()
