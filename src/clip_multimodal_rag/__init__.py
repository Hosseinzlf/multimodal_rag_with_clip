"""
CLIP Multimodal RAG — Retrieval-Augmented Generation over PDFs with text and images.

Uses CLIP for unified text/image embeddings, FAISS for retrieval, and a LangChain
chat model (OpenAI by default) for answer generation.
"""

__version__ = "0.1.0"

from clip_multimodal_rag.embeddings import CLIPEmbedder
from clip_multimodal_rag.pdf_processor import PDFProcessor
from clip_multimodal_rag.pipeline import MultimodalRAGPipeline
from clip_multimodal_rag.retrieval import MultimodalRetriever

__all__ = [
    "__version__",
    "CLIPEmbedder",
    "PDFProcessor",
    "MultimodalRetriever",
    "MultimodalRAGPipeline",
]
