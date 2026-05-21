# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import fitz
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from config import (
    DOCS_DIR,
    EXCLUDED_DIR_NAMES,
    EXCLUDED_FILE_NAMES,
    EXCLUDED_SUFFIXES,
)

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
}

SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf"}

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
}

PDF_METADATA_KEY_MAP = {
    "creationDate": "creationdate",
    "modDate": "moddate",
}


def display_path(path: Path, root: Path = DOCS_DIR) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def should_index_path(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    if path.name in EXCLUDED_FILE_NAMES or path.name.startswith(".env"):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _apply_common_metadata(doc: Document, path: Path) -> Document:
    relative_path = display_path(path)
    suffix = path.suffix.lower()
    doc.metadata.update(
        {
            "source": relative_path,
            "file_name": path.name,
            "file_path": relative_path,
            "extension": suffix,
            "language": LANGUAGE_BY_EXTENSION.get(suffix, "unknown"),
            "directory": str(Path(relative_path).parent),
        }
    )
    return doc


def _pdf_metadata(pdf: fitz.Document) -> dict:
    metadata = {}
    for key, value in pdf.metadata.items():
        if value:
            metadata[PDF_METADATA_KEY_MAP.get(key, key)] = value
    return metadata


def _load_pdf(path: Path) -> list[Document]:
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        total_pages = pdf.page_count
        base_metadata = _pdf_metadata(pdf)

        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue

            metadata = {
                **base_metadata,
                "page": page_index,
                "page_label": page.get_label(),
                "total_pages": total_pages,
            }
            docs.append(Document(page_content=text, metadata=metadata))

    return docs


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if not should_index_path(path):
        return []

    if suffix == ".pdf":
        docs = _load_pdf(path)
        return [_apply_common_metadata(doc, path) for doc in docs]

    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        try:
            docs = TextLoader(str(path), encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(
                str(path),
                encoding="utf-8",
                autodetect_encoding=True,
            ).load()
        return [_apply_common_metadata(doc, path) for doc in docs]

    return []


def load_documents_from_dir(directory: Path = DOCS_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in directory.rglob("*"):
        if path.is_file():
            documents.extend(load_file(path))
    return documents
