# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import CHROMA_DIR, get_collection_name, get_min_relevance_score
from providers import create_chat_model, create_embeddings

SYSTEM_PROMPT = """
あなたは個人用ナレッジベースの検索アシスタントです。

ルール:
- 必ず与えられたcontextだけを根拠に回答してください。
- contextに答えがない場合は「資料内では確認できません」と答えてください。
- 推測で補完しないでください。
- 回答の最後に参照したファイル名を列挙してください。
- 日本語で簡潔に答えてください。
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
質問:
{question}

context:
{context}
""",
        ),
    ]
)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def format_source(doc: Document) -> str:
    source = doc.metadata.get("file_path") or doc.metadata.get("file_name", "unknown")
    page = doc.metadata.get("page")
    if page is not None:
        return f"{source} p.{int(page) + 1}"
    return str(source)


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for doc in docs:
        formatted.append(f"[source: {format_source(doc)}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def _not_found_response(retrieved_docs: list[Document] | None = None) -> dict:
    return {
        "answer": "資料内では確認できません。\n\n参照:\n- なし",
        "sources": [],
        "retrieved_docs": retrieved_docs or [],
    }


def _has_relevant_context(vectorstore: Chroma, question: str) -> bool:
    threshold = get_min_relevance_score()
    if threshold <= 0:
        return True

    scored_docs = vectorstore.similarity_search_with_relevance_scores(question, k=1)
    if not scored_docs:
        return False

    _, score = scored_docs[0]
    return score >= threshold


def ask(question: str) -> dict:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
        },
    )
    docs = retriever.invoke(question)
    if not docs:
        return _not_found_response()

    if not _has_relevant_context(vectorstore, question):
        return _not_found_response(docs)

    context = format_docs(docs)
    chain = PROMPT | create_chat_model()
    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )
    sources = sorted({format_source(doc) for doc in docs})
    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_docs": docs,
    }
