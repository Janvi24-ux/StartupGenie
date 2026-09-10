"""
RAG Module – Document Loader
Loads and chunks text documents from the knowledge base directory.
"""

import os
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a text chunk with its source metadata."""
    content: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def load_text_file(file_path: Path) -> str:
    """Read a plain text file and return its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping chunks by word boundaries.

    Args:
        text: Full document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks: List[str] = []
    current_chars = 0
    current_words: List[str] = []
    overlap_words: List[str] = []

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_chars + word_len > chunk_size and current_words:
            chunk_text_str = " ".join(current_words)
            chunks.append(chunk_text_str)
            # Build overlap from end of current chunk
            overlap_text = chunk_text_str[-overlap:] if len(chunk_text_str) > overlap else chunk_text_str
            overlap_words = overlap_text.split()
            current_words = overlap_words + [word]
            current_chars = sum(len(w) + 1 for w in current_words)
        else:
            current_words.append(word)
            current_chars += word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c.strip()]


def load_knowledge_base(
    kb_dir: str = "data/knowledge_base",
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Document]:
    """
    Load all .txt files from the knowledge base directory, chunk them,
    and return a list of Document objects.

    Args:
        kb_dir: Path to the knowledge base directory.
        chunk_size: Max characters per chunk.
        overlap: Overlap characters between chunks.

    Returns:
        List of Document objects ready for embedding.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        logger.warning("Knowledge base directory not found: %s", kb_dir)
        return []

    documents: List[Document] = []
    txt_files = sorted(kb_path.glob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in knowledge base directory: %s", kb_dir)

    for file_path in txt_files:
        try:
            raw_text = load_text_file(file_path)
            chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    source=file_path.name,
                    chunk_id=i,
                    metadata={"file": file_path.name, "chunk_index": i},
                )
                documents.append(doc)
            logger.info(
                "Loaded %d chunks from %s", len(chunks), file_path.name
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)

    logger.info("Total documents loaded: %d", len(documents))
    return documents
