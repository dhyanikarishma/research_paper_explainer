"""A lightweight in-memory vector store (the heart of RAG).

WHAT THIS DOES (in plain English):
  - We convert every chunk of the paper into an embedding (a vector of
    numbers that captures meaning).
  - When the user asks a question, we embed the question too.
  - We then find the chunks whose vectors are most "similar" to the
    question's vector using cosine similarity.
  - Those top chunks are the relevant context we hand to the LLM.

This is genuine vector search. We use NumPy instead of a heavier database
(like ChromaDB/FAISS) so the app is dependency-light and deploys without
the sqlite/native-build headaches that trip up beginners on free hosting.
The class interface is deliberately swappable if you upgrade later.
"""

import numpy as np

from src.config import TOP_K
from src.llm import embed_texts


class VectorStore:
    """Stores chunks + their embeddings and supports similarity search."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        # Shape will be (num_chunks, embedding_dim) once built.
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[str]) -> None:
        """Embed all chunks and store them as a normalized matrix.

        We L2-normalize each vector so cosine similarity becomes a simple
        dot product later (faster and numerically cleaner).
        """
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")

        self.chunks = chunks
        embeddings = embed_texts(chunks, task_type="retrieval_document")
        matrix = np.array(embeddings, dtype=np.float32)

        # Normalize each row to unit length (avoid divide-by-zero).
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = matrix / norms

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        """Return the `top_k` chunks most relevant to `query`."""
        if self._matrix is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        query_vec = np.array(
            embed_texts([query], task_type="retrieval_query")[0],
            dtype=np.float32,
        )
        norm = np.linalg.norm(query_vec)
        query_vec = query_vec / (norm if norm else 1e-10)

        # Cosine similarity == dot product of unit vectors.
        scores = self._matrix @ query_vec

        # Indices of the highest-scoring chunks, best first.
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    @property
    def is_ready(self) -> bool:
        return self._matrix is not None


class ChromaVectorStore:
    """Same interface as VectorStore, but backed by ChromaDB.

    We use an in-memory (ephemeral) Chroma client so there's no sqlite file
    to manage and nothing to clean up. We compute embeddings ourselves with
    Gemini (so Chroma never downloads its own embedding model), then hand the
    vectors to Chroma, which handles the similarity search.

    This demonstrates a "real" vector database backend; the NumPy store
    remains the default because it deploys with zero extra setup.
    """

    def __init__(self) -> None:
        import chromadb  # imported lazily so the app runs without chromadb

        self._client = chromadb.Client()
        # A fresh collection per store instance keeps papers isolated.
        self._collection = self._client.create_collection(
            name=f"paper_{id(self)}",
            metadata={"hnsw:space": "cosine"},
        )
        self.chunks: list[str] = []

    def build(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")
        self.chunks = chunks
        embeddings = embed_texts(chunks, task_type="retrieval_document")
        self._collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
        )

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        if not self.chunks:
            raise RuntimeError("Vector store is empty. Call build() first.")
        query_vec = embed_texts([query], task_type="retrieval_query")[0]
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, len(self.chunks)),
        )
        # Chroma returns a list-of-lists (one per query); we sent one query.
        return result["documents"][0]

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks)


def get_vector_store(backend: str | None = None):
    """Factory: return a vector store for the requested backend.

    Falls back to the NumPy store if ChromaDB isn't installed, so the app
    never crashes just because an optional dependency is missing.
    """
    from src.config import VECTOR_BACKEND

    backend = (backend or VECTOR_BACKEND).lower()
    if backend == "chroma":
        try:
            return ChromaVectorStore()
        except Exception:
            # ChromaDB unavailable -> degrade gracefully to NumPy.
            return VectorStore()
    return VectorStore()
