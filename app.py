# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import re
import shutil
from pathlib import Path

import streamlit as st

from config import DOCS_DIR, get_collection_name
from indexer import rebuild_index
from loaders import SUPPORTED_EXTENSIONS, display_path, should_index_path
from providers import ProviderConfigurationError, get_provider_summary
from rag import RagAssistant


def sanitize_upload_name(file_name: str) -> str:
    original_name = Path(file_name).name.replace("\x00", "").strip()
    original_path = Path(original_name)
    suffix = original_path.suffix.lower()

    stem = re.sub(r"\s+", "_", original_path.stem)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    stem = stem.strip("._-")
    if not stem:
        stem = "upload"

    return f"{stem}{suffix}"


def unique_save_path(directory: Path, file_name: str) -> Path:
    path = directory / file_name
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError("保存先ファイル名を決定できませんでした。")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("設定")
        provider_summary = get_provider_summary()
        st.write(f"Chat: {provider_summary['chat_provider']} / {provider_summary['chat_model']}")
        st.write(
            "Embedding: "
            f"{provider_summary['embedding_provider']} / {provider_summary['embedding_model']}"
        )
        st.caption(f"Collection: {get_collection_name()}")

        st.header("ファイル追加")
        uploaded_files = st.file_uploader(
            "PDF / Markdown / txt / code",
            accept_multiple_files=True,
            type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        )

        if uploaded_files:
            saved_count = 0
            skipped_files = []
            for uploaded_file in uploaded_files:
                safe_name = sanitize_upload_name(uploaded_file.name)
                save_path = unique_save_path(DOCS_DIR, safe_name)
                if not should_index_path(save_path):
                    skipped_files.append(uploaded_file.name)
                    continue

                with save_path.open("wb") as file:
                    shutil.copyfileobj(uploaded_file, file)
                saved_count += 1

            if saved_count:
                st.success(f"{saved_count} files uploaded.")
            if skipped_files:
                st.warning("未対応または除外対象のファイルは保存しませんでした。")

        if st.button("インデックス作成 / 更新", type="primary"):
            with st.spinner("Indexing..."):
                try:
                    count = rebuild_index()
                    st.success(f"{count} chunks indexed.")
                except ProviderConfigurationError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)

        existing_files = sorted(
            path for path in DOCS_DIR.rglob("*") if path.is_file() and should_index_path(path)
        )
        if existing_files:
            st.header("保存済みファイル")
            for path in existing_files:
                st.write(f"- {display_path(path)}")


def render_answer(question: str) -> None:
    with st.spinner("Searching..."):
        try:
            result = RagAssistant().ask(question)
        except ProviderConfigurationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return

    st.subheader("回答")
    st.write(result["answer"])

    st.subheader("参照")
    if result["sources"]:
        for source in result["sources"]:
            st.write(f"- {source}")
    else:
        st.write("- なし")

    with st.expander("検索されたチャンク"):
        retrieved_docs = result["retrieved_docs"]
        if not retrieved_docs:
            st.write("なし")
            return

        for index, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"### Chunk {index}")
            st.caption(doc.metadata)
            st.write(doc.page_content[:1500])


def main() -> None:
    st.set_page_config(
        page_title="Personal RAG Assistant",
        layout="wide",
    )
    st.title("Personal RAG Assistant")
    st.caption("PDF・メモ・コードを検索できる個人用RAGアシスタント")

    render_sidebar()

    question = st.text_input(
        "質問",
        placeholder="例: このプロジェクトの認証処理はどこで行われている？",
    )
    if question:
        render_answer(question)


if __name__ == "__main__":
    main()
