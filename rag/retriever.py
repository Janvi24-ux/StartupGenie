"""
RAG Module – Retriever
High-level interface for the RAG pipeline: load → embed → retrieve.
"""

import logging
from typing import List, Dict, Any

from rag.document_loader import load_knowledge_base
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Orchestrates the full RAG pipeline:
    1. Load and chunk documents from the knowledge base.
    2. Build an in-memory FAISS vector index.
    3. Retrieve relevant context for a given query.
    """

    def __init__(
        self,
        kb_dir: str = "data/knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        self.kb_dir = kb_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._vector_store = VectorStore(embedding_model=embedding_model)

    def initialize(self) -> None:
        """
        Load the knowledge base and build the vector index.
        Call once at application startup.
        """
        logger.info("Initialising RAG retriever...")
        documents = load_knowledge_base(
            kb_dir=self.kb_dir,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not documents:
            logger.error("No documents loaded – RAG retrieval will be unavailable.")
            return
        self._vector_store.build(documents)
        logger.info("RAG retriever ready.")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for the given query.

        Args:
            query: Natural-language query or description of the startup idea.
            top_k: Override the default number of results.

        Returns:
            List of dicts with keys: 'content', 'source', 'score'.
        """
        if not self._vector_store.is_ready:
            logger.warning("Vector store not ready. Returning empty context.")
            return []

        k = top_k if top_k is not None else self.top_k
        raw_results = self._vector_store.search(query, top_k=k)

        return [
            {
                "content": content,
                "source": source,
                "score": round(score, 4),
            }
            for content, source, score in raw_results
        ]

    def retrieve_for_startup(self, startup_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a composite query from the startup input fields and retrieve context.

        Args:
            startup_input: Dict with keys like idea, industry, location, budget, stage.

        Returns:
            Deduplicated list of relevant context chunks.
        """
        idea = startup_input.get("idea", "")
        industry = startup_input.get("industry", "")
        location = startup_input.get("location", "India")
        stage = startup_input.get("stage", "")
        budget = startup_input.get("budget", "")

        # Build a rich composite query that spans all blueprint sections
        composite_query = (
            f"Startup idea: {idea}. "
            f"Industry: {industry}. "
            f"Location: {location}. "
            f"Stage: {stage}. "
            f"Budget: {budget}. "
            "Government schemes funding incubators legal compliance business model "
            "revenue model competitor analysis go-to-market strategy investors."
        )

        # Also run targeted sub-queries for critical sections
        sub_queries = [
            f"{industry} startup government schemes funding India {location}",
            f"startup legal compliance registration India {industry}",
            f"incubators accelerators {industry} India",
            f"startup budget cost estimate {industry} {stage}",
            f"business model revenue {industry} startup",
        ]

        seen_contents = set()
        all_results: List[Dict[str, Any]] = []

        for q in [composite_query] + sub_queries:
            for result in self.retrieve(q, top_k=3):
                if result["content"] not in seen_contents:
                    seen_contents.add(result["content"])
                    all_results.append(result)

        # Sort by score descending and cap at top_k * 3 to keep context manageable
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[: self.top_k * 3]

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a readable context string for the LLM prompt.

        Args:
            retrieved_docs: Output of retrieve() or retrieve_for_startup().

        Returns:
            Formatted context string.
        """
        if not retrieved_docs:
            return "No relevant knowledge base context found."

        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_label = doc["source"].replace("_", " ").replace(".txt", "").title()
            parts.append(
                f"[Source {i}: {source_label} | Relevance: {doc['score']:.2f}]\n"
                f"{doc['content']}"
            )
        return "\n\n---\n\n".join(parts)
