"""PDF extraction: text chunks and images with metadata for multimodal RAG."""

import base64
import io
import warnings
from pathlib import Path

import fitz
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
        doc = fitz.open(pdf_path)

        try:
            for page_index, page in enumerate(doc):
                # Text
                text = page.get_text()
                if text.strip():
                    temp_doc = Document(
                        page_content=text,
                        metadata={"page": page_index, "type": "text"},
                    )
                    chunks = self.splitter.split_documents([temp_doc])
                    for chunk in chunks:
                        all_docs.append(chunk)
                        emb = self.embedder.embed_text(chunk.page_content)
                        all_embeddings.append(emb)

                # Images
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(
                            io.BytesIO(image_bytes)
                        ).convert("RGB")

                        image_id = _image_id(page_index, img_index)
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_b64 = base64.b64encode(
                            buffered.getvalue()
                        ).decode()
                        image_data_store[image_id] = img_b64

                        image_doc = Document(
                            page_content=f"[Image: {image_id}]",
                            metadata={
                                "page": page_index,
                                "type": "image",
                                "image_id": image_id,
                            },
                        )
                        all_docs.append(image_doc)
                        emb = self.embedder.embed_image(pil_image)
                        all_embeddings.append(emb)
                    except Exception as e:
                        warnings.warn(
                            f"Skip image {img_index} on page {page_index}: {e}"
                        )
                        continue
        finally:
            doc.close()

        return all_docs, all_embeddings, image_data_store
