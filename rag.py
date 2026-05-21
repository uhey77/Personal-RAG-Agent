# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import AppConfig, get_app_config
from providers import create_chat_model, create_embeddings

SYSTEM_PROMPT = (
    "あなたは個人用ナレッジベースの検索アシスタントです。\n\n"
    "ルール:\n"
    "- 必ず与えられたcontextだけを根拠に回答してください。\n"
    "- contextに複数の資料がある場合は、すべてを考慮して回答してください。\n"
    "- contextに質問の答えがない場合は、わからないと回答してください。\n"
    "- 推測で答えを補完しないでください。\n"
    "- 回答の最後に、参照したファイル名を列挙してください。\n"
    "- 回答は日本語で簡潔にしてください。\n"
)

HUMAN_PROMPT = (
    "質問:\n"
    "{question}\n\n"
    "context:\n"
    "{context}\n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


class RagAssistant:
    def __init__(
        self,
        config: AppConfig | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.config = config or get_app_config()
        self.collection_name = collection_name or self.config.collection_name

    def ask(self, question: str) -> dict:
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 15,
            },
        )
        docs = retriever.invoke(question)
        if not docs:
            return self.not_found_response()

        if not self.has_relevant_context(vectorstore, question):
            return self.not_found_response(docs)

        context = self.format_docs(docs)
        chain = PROMPT | create_chat_model(self.config)
        response = chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )
        sources = sorted({self.format_source(doc) for doc in docs})
        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_docs": docs,
        }

    def get_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=create_embeddings(self.config),
            persist_directory=str(self.config.chroma_dir),
        )

    def format_source(self, doc: Document) -> str:
        source = doc.metadata.get("file_path") or doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page")
        if page is not None:
            return f"{source} p.{int(page) + 1}"
        return str(source)

    def format_docs(self, docs: list[Document]) -> str:
        formatted = []
        for doc in docs:
            formatted.append(f"[source: {self.format_source(doc)}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def not_found_response(self, retrieved_docs: list[Document] | None = None) -> dict:
        return {
            "answer": "資料内では確認できません。\n\n参照:\n- なし",
            "sources": [],
            "retrieved_docs": retrieved_docs or [],
        }

    def has_relevant_context(self, vectorstore: Chroma, question: str) -> bool:
        threshold = self.config.min_relevance_score
        if threshold <= 0:
            return True

        scored_docs = vectorstore.similarity_search_with_relevance_scores(question, k=1)
        if not scored_docs:
            return False

        _, score = scored_docs[0]
        return score >= threshold
