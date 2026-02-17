"""Unified retrieval over text and image embeddings using FAISS."""

from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from clip_multimodal_rag.embeddings import CLIPEmbedder


class MultimodalRetriever:
    """FAISS-backed retriever using precomputed CLIP embeddings."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        documents: list[Document],
        embeddings: list[np.ndarray],
    ):
        self.embedder = embedder
        self.documents = documents
        embeddings_array = np.array(embeddings).astype(np.float32)
        # LangChain FAISS expects (text, embedding) and an embedding function.
        # We pass a no-op embedder and precomputed text_embeddings.
        self._store = FAISS.from_embeddings(
            text_embeddings=[
                (doc.page_content, emb)
                for doc, emb in zip(documents, embeddings_array)
            ],
            embedding=_FakeEmbeddings(embedder.embedding_dim),
            metadatas=[doc.metadata for doc in documents],
        )

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Return top-k documents by CLIP similarity to the query."""
        query_embedding = self.embedder.embed_text(query)
        return self._store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
        )


class _FakeEmbeddings:
    """Minimal embedder so FAISS accepts our precomputed vectors; not used for indexing."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension
