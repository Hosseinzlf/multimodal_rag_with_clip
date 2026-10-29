"""End-to-end multimodal RAG pipeline: retrieve + generate with a chat model."""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from clip_multimodal_rag.config import CHAT_MODEL, get_openai_api_key
from clip_multimodal_rag.embeddings import CLIPEmbedder
from clip_multimodal_rag.retrieval import MultimodalRetriever


def _build_multimodal_message(
    query: str,
    retrieved_docs: list,
    image_data_store: dict[str, str],
) -> HumanMessage:
    """Build a single HumanMessage with text context and inline base64 images."""
    content: list[dict] = [
        {"type": "text", "text": f"Question: {query}\n\nContext:\n"}
    ]

    for doc in retrieved_docs:
        if doc.metadata.get("type") == "text":
            content.append({
                "type": "text",
                "text": f"[Text from page {doc.metadata['page']}]: {doc.page_content}\n",
            })
            continue

        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"[Image from page {doc.metadata['page']}]:",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}",
                },
            })

    content.append({
        "type": "text",
        "text": "\nAnswer the question using the text and images above.",
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
        chat_model: str = CHAT_MODEL,
        api_key: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.image_data_store = image_data_store
        self.top_k = top_k
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            get_openai_api_key()
        self._llm = init_chat_model(model=chat_model, temperature=temperature)

    def query(self, question: str, verbose: bool = True) -> str:
        """Retrieve context and return the chat model's answer."""
        docs = self.retriever.retrieve(question, k=self.top_k)
        message = _build_multimodal_message(
            question, docs, self.image_data_store
        )
        response = self._llm.invoke([message])

        if verbose:
            print(f"\nRetrieved {len(docs)} items:")
            for d in docs:
                doc_type = d.metadata.get("type", "unknown")
                page = d.metadata.get("page", "?")
                print(f"  - {doc_type} from page {page}")
            print()

        return response.content
