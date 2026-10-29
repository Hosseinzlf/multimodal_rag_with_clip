"""Unified retrieval over text and image embeddings using FAISS."""

import faiss
import numpy as np
from langchain_core.documents import Document

from clip_multimodal_rag.embeddings import CLIPEmbedder


class MultimodalRetriever:
    """FAISS IndexFlatIP retriever using precomputed CLIP embeddings."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        documents: list[Document],
        embeddings: list[np.ndarray],
    ):
        self.embedder = embedder
        self.documents = documents
        embeddings_array = np.array(embeddings).astype("float32")
        dim = embeddings_array.shape[1] if embeddings_array.size else embedder.embedding_dim
        self._index = faiss.IndexFlatIP(dim)
        if len(embeddings_array):
            self._index.add(embeddings_array)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Return top-k documents by CLIP similarity to the query."""
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        query_embedding = self.embedder.embed_text(query).astype("float32")
        _scores, ids = self._index.search(query_embedding[None, :], k)
        return [self.documents[i] for i in ids[0] if i >= 0]
