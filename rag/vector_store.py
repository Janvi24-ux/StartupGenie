"""
RAG Module – Vector Store
Handles embedding generation and FAISS-based similarity search.
"""

import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy ML libraries at module import time
_sentence_transformer = None
_faiss = None


def _get_sentence_transformer(model_name: str):
    """Lazily load SentenceTransformer."""
    global _sentence_transformer
    from sentence_transformers import SentenceTransformer
    if _sentence_transformer is None:
        logger.info("Loading embedding model: %s", model_name)
        _sentence_transformer = SentenceTransformer(model_name)
    return _sentence_transformer


def _get_faiss():
    """Lazily load FAISS."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class VectorStore:
    """
    In-memory FAISS vector store for semantic similarity search.

    Embeds documents using SentenceTransformers and retrieves the top-k
    most semantically similar chunks for a given query.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._index = None
        self._documents: List[str] = []
        self._sources: List[str] = []
        self._dimension: int = 0

    def build(self, documents: List) -> None:
        """
        Embed all documents and build the FAISS index.

        Args:
            documents: List of Document objects from document_loader.
        """
        if not documents:
            logger.warning("No documents provided to VectorStore.build()")
            return

        model = _get_sentence_transformer(self.embedding_model_name)
        faiss = _get_faiss()

        texts = [doc.content for doc in documents]
        self._documents = texts
        self._sources = [doc.source for doc in documents]

        logger.info("Embedding %d document chunks...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        self._dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)
        logger.info("FAISS index built with %d vectors (dim=%d)", len(texts), self._dimension)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve the top-k most similar document chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of (content, source, score) tuples ordered by relevance.
        """
        if self._index is None or not self._documents:
            logger.warning("VectorStore is empty. Call build() first.")
            return []

        model = _get_sentence_transformer(self.embedding_model_name)

        query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((
                    self._documents[idx],
                    self._sources[idx],
                    float(score),
                ))
        return results

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and contains documents."""
        return self._index is not None and len(self._documents) > 0
