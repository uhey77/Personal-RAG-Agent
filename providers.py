# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import get_chat_provider, get_embedding_model, get_embedding_provider, get_env


class ProviderConfigurationError(RuntimeError):
    pass


DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash-lite",
}


def _require_env(name: str, provider: str) -> None:
    if not get_env(name):
        raise ProviderConfigurationError(
            f"{provider} を使うには .env に {name} を設定してください。"
        )


def get_chat_model_name() -> str:
    provider = get_chat_provider()
    return get_env("CHAT_MODEL", DEFAULT_CHAT_MODELS.get(provider, "gemini-2.5-flash-lite"))


def create_chat_model() -> BaseChatModel:
    provider = get_chat_provider()
    model = get_chat_model_name()

    if provider == "openai":
        _require_env("OPENAI_API_KEY", "OpenAI")
        return ChatOpenAI(model=model, temperature=0)

    if provider == "google":
        _require_env("GOOGLE_API_KEY", "Google")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    raise ProviderConfigurationError(
        "CHAT_PROVIDER は openai または google を指定してください。"
    )


def create_embeddings() -> Embeddings:
    provider = get_embedding_provider()
    model = get_embedding_model()

    if provider == "openai":
        _require_env("OPENAI_API_KEY", "OpenAI embeddings")
        return OpenAIEmbeddings(model=model)

    if provider == "google":
        _require_env("GOOGLE_API_KEY", "Google embeddings")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise ProviderConfigurationError(
        "EMBEDDING_PROVIDER は openai または google を指定してください。"
    )


def get_provider_summary() -> dict[str, str]:
    return {
        "chat_provider": get_chat_provider(),
        "chat_model": get_chat_model_name(),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
    }
