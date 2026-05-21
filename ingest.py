# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, DOCS_DIR, get_collection_name
from loaders import load_documents_from_dir
from providers import create_embeddings


def _delete_collection_if_exists(collection_name: str) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" not in message and "not found" not in message:
            raise


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nfunction ",
            "\nexport ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def ingest_documents() -> int:
    collection_name = get_collection_name()
    documents = load_documents_from_dir(DOCS_DIR)
    if not documents:
        _delete_collection_if_exists(collection_name)
        return 0

    chunks = _split_documents(documents)
    embeddings = create_embeddings()

    _delete_collection_if_exists(collection_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Indexed {count} chunks.")
