# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, DOCS_DIR, get_collection_name
from loaders import load_documents_from_dir
from providers import create_embeddings


class DocumentIndexer:
    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str | None = None,
    ) -> None:
        self.docs_dir = docs_dir
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name or get_collection_name()

    def rebuild_index(self) -> int:
        documents = load_documents_from_dir(self.docs_dir)
        if not documents:
            self.delete_collection_if_exists()
            return 0

        chunks = self.split_documents(documents)
        embeddings = create_embeddings()

        self.delete_collection_if_exists()
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.chroma_dir),
        )
        vectorstore.add_documents(chunks)
        return len(chunks)

    def delete_collection_if_exists(self) -> None:
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            client.delete_collection(name=self.collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" not in message and "not found" not in message:
                raise

    def split_documents(self, documents: list[Document]) -> list[Document]:
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


def rebuild_index() -> int:
    return DocumentIndexer().rebuild_index()


if __name__ == "__main__":
    count = rebuild_index()
    print(f"Indexed {count} chunks.")
