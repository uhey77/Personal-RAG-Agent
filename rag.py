# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

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

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたはRAG検索用のクエリを書き換えるアシスタントです。"
            "回答はせず、資料検索で見つかりやすい短い検索クエリだけを返してください。",
        ),
        (
            "human",
            "元の質問:\n{question}\n\n"
            "前回の検索クエリ:\n{search_query}\n\n"
            "検索しやすいクエリに書き換えてください。",
        ),
    ]
)


class RagGraphState(TypedDict, total=False):
    question: str
    search_query: str
    retry_count: int
    docs: list[Document]
    is_relevant: bool
    answer: str
    sources: list[str]
    retrieved_docs: list[Document]


class RagAssistant:
    def __init__(
        self,
        config: AppConfig | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.config = config or get_app_config()
        self.collection_name = collection_name or self.config.collection_name
        self.graph = self.build_graph()

    def ask(self, question: str) -> dict:
        result = self.graph.invoke(
            {
                "question": question,
                "search_query": question,
                "retry_count": 0,
            }
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_docs": result["retrieved_docs"],
            "search_query": result.get("search_query", question),
            "used_rewrite": result.get("retry_count", 0) > 0,
        }

    def build_graph(self):
        graph = StateGraph(RagGraphState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("rewrite_query", self.rewrite_query)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("not_found", self.not_found)
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self.route_after_retrieve,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "not_found": "not_found",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", END)
        graph.add_edge("not_found", END)
        return graph.compile()

    def retrieve(self, state: RagGraphState) -> RagGraphState:
        question = state["question"]
        search_query = state.get("search_query") or question
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 15,
            },
        )
        docs = retriever.invoke(search_query)
        is_relevant = bool(docs) and self.has_relevant_context(vectorstore, search_query)
        return {
            "search_query": search_query,
            "docs": docs,
            "is_relevant": is_relevant,
        }

    def route_after_retrieve(
        self,
        state: RagGraphState,
    ) -> Literal["generate_answer", "rewrite_query", "not_found"]:
        if state.get("docs") and state.get("is_relevant"):
            return "generate_answer"

        if state.get("retry_count", 0) < 1:
            return "rewrite_query"

        return "not_found"

    def rewrite_query(self, state: RagGraphState) -> RagGraphState:
        chain = REWRITE_PROMPT | create_chat_model(self.config)
        response = chain.invoke(
            {
                "question": state["question"],
                "search_query": state.get("search_query") or state["question"],
            }
        )
        rewritten_query = str(response.content).strip() or state["question"]
        return {
            "search_query": rewritten_query,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate_answer(self, state: RagGraphState) -> RagGraphState:
        docs = state.get("docs", [])
        context = self.format_docs(docs)
        chain = PROMPT | create_chat_model(self.config)
        response = chain.invoke(
            {
                "question": state["question"],
                "context": context,
            }
        )
        sources = sorted({self.format_source(doc) for doc in docs})
        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_docs": docs,
        }

    def not_found(self, state: RagGraphState) -> RagGraphState:
        response = self.not_found_response(state.get("docs", []))
        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "retrieved_docs": response["retrieved_docs"],
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
