"""
CLIP Multimodal RAG — Retrieval-Augmented Generation over PDFs with text and images.

Uses CLIP for unified text/image embeddings, FAISS for retrieval, and Google Gemini
for answer generation.
"""

__version__ = "0.1.0"

from clip_multimodal_rag.embeddings import CLIPEmbedder
from clip_multimodal_rag.pdf_processor import PDFProcessor
from clip_multimodal_rag.retrieval import MultimodalRetriever
from clip_multimodal_rag.pipeline import MultimodalRAGPipeline

__all__ = [
    "__version__",
    "CLIPEmbedder",
    "PDFProcessor",
    "MultimodalRetriever",
    "MultimodalRAGPipeline",
]
