# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import AppConfig, get_app_config


class ProviderConfigurationError(RuntimeError):
    pass


DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash-lite",
}


def _require_api_key(value: str, name: str, provider: str) -> None:
    if not value:
        raise ProviderConfigurationError(
            f"{provider} を使うには .env に {name} を設定してください。"
        )


def get_chat_model_name(config: AppConfig | None = None) -> str:
    return (config or get_app_config()).chat_model


def create_chat_model(config: AppConfig | None = None) -> BaseChatModel:
    config = config or get_app_config()
    provider = config.chat_provider
    model = config.chat_model

    if provider == "openai":
        _require_api_key(config.openai_api_key, "OPENAI_API_KEY", "OpenAI")
        return ChatOpenAI(model=model, temperature=0)

    if provider == "google":
        _require_api_key(config.google_api_key, "GOOGLE_API_KEY", "Google")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    raise ProviderConfigurationError(
        "CHAT_PROVIDER は openai または google を指定してください。"
    )


def create_embeddings(config: AppConfig | None = None) -> Embeddings:
    config = config or get_app_config()
    provider = config.embedding_provider
    model = config.embedding_model

    if provider == "openai":
        _require_api_key(config.openai_api_key, "OPENAI_API_KEY", "OpenAI embeddings")
        return OpenAIEmbeddings(model=model)

    if provider == "google":
        _require_api_key(config.google_api_key, "GOOGLE_API_KEY", "Google embeddings")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise ProviderConfigurationError(
        "EMBEDDING_PROVIDER は openai または google を指定してください。"
    )


def get_provider_summary(config: AppConfig | None = None) -> dict[str, str]:
    config = config or get_app_config()
    return {
        "chat_provider": config.chat_provider,
        "chat_model": config.chat_model,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
    }
