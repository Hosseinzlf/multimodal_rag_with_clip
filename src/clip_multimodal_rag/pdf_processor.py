"""PDF extraction: text chunks and images with metadata for multimodal RAG."""

import base64
import io
import warnings
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from clip_multimodal_rag.embeddings import CLIPEmbedder


def _image_id(page_index: int, img_index: int) -> str:
    return f"page_{page_index}_img_{img_index}"


class PDFProcessor:
    """Extract text and images from PDFs and produce documents with optional embeddings."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedder: CLIPEmbedder | None = None,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = embedder

    def process(
        self,
        pdf_path: str | Path,
    ) -> tuple[list[Document], list, dict[str, str]]:
        """
        Process a PDF into text chunks and image documents with CLIP embeddings.

        Requires embedder to be set. Returns documents, embeddings (numpy), and
        base64 image store for the LLM.

        Returns:
            all_docs: List of Document (text and image placeholders).
            all_embeddings: List of numpy arrays (same order as all_docs).
            image_data_store: Map image_id -> base64 PNG string for LLM.
        """
        if self.embedder is None:
            raise ValueError("PDFProcessor must be constructed with embedder for process().")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        all_docs: list[Document] = []
        all_embeddings: list = []
        image_data_store: dict[str, str] = {}

        with pymupdf.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    page_doc = Document(
                        page_content=text,
                        metadata={"page": page_index, "type": "text"},
                    )
                    for chunk in self.splitter.split_documents([page_doc]):
                        all_docs.append(chunk)
                        all_embeddings.append(
                            self.embedder.embed_text(chunk.page_content)
                        )

                for img_index, img_info in enumerate(page.get_images(full=True)):
                    try:
                        image_bytes = pdf.extract_image(img_info[0])["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        image_id = _image_id(page_index, img_index)
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format="PNG")
                        image_data_store[image_id] = base64.b64encode(
                            buffer.getvalue()
                        ).decode()

                        all_docs.append(
                            Document(
                                page_content=f"[Image: {image_id}]",
                                metadata={
                                    "page": page_index,
                                    "type": "image",
                                    "image_id": image_id,
                                },
                            )
                        )
                        all_embeddings.append(self.embedder.embed_image(pil_image))
                    except Exception as e:
                        warnings.warn(
                            f"Skip image {img_index} on page {page_index}: {e}"
                        )

        return all_docs, all_embeddings, image_data_store
