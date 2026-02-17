"""End-to-end multimodal RAG pipeline: retrieve + generate with Gemini."""

from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from clip_multimodal_rag.config import get_google_api_key
from clip_multimodal_rag.embeddings import CLIPEmbedder
from clip_multimodal_rag.pdf_processor import PDFProcessor
from clip_multimodal_rag.retrieval import MultimodalRetriever


def _build_multimodal_message(
    query: str,
    retrieved_docs: list,
    image_data_store: dict[str, str],
) -> HumanMessage:
    """Build a single HumanMessage with text context and inline base64 images."""
    content: list[dict] = []

    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n",
    })

    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join(
            f"[Page {d.metadata['page']}]: {d.page_content}"
            for d in text_docs
        )
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n",
        })

    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}",
                },
            })

    content.append({
        "type": "text",
        "text": "\n\nAnswer the question based on the provided text and images.",
    })

    return HumanMessage(content=content)


class MultimodalRAGPipeline:
    """Run multimodal RAG: index a PDF once, then query with natural language."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        retriever: MultimodalRetriever,
        image_data_store: dict[str, str],
        *,
        gemini_model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.image_data_store = image_data_store
        self.top_k = top_k
        self._llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=api_key or get_google_api_key(),
        )

    def query(self, question: str, verbose: bool = True) -> str:
        """Retrieve context and return Gemini's answer."""
        docs = self.retriever.retrieve(question, k=self.top_k)
        message = _build_multimodal_message(
            question, docs, self.image_data_store
        )
        response = self._llm.invoke([message])

        if verbose:
            print(f"\nRetrieved {len(docs)} documents:")
            for d in docs:
                doc_type = d.metadata.get("type", "unknown")
                page = d.metadata.get("page", "?")
                if doc_type == "text":
                    preview = (
                        d.page_content[:100] + "..."
                        if len(d.page_content) > 100
                        else d.page_content
                    )
                    print(f"  - Text (page {page}): {preview}")
                else:
                    print(f"  - Image (page {page})")
            print()

        return response.content
