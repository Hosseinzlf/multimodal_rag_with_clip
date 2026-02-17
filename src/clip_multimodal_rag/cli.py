"""CLI for building and querying the multimodal RAG pipeline."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP Multimodal RAG: query PDFs with text and images."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the PDF file to index and query",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Question to ask about the document",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print retrieved context",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    from clip_multimodal_rag import (
        CLIPEmbedder,
        MultimodalRAGPipeline,
        MultimodalRetriever,
        PDFProcessor,
    )
    from clip_multimodal_rag.config import (
        get_google_api_key,
        CLIP_MODEL_ID,
        GEMINI_MODEL,
    )

    try:
        get_google_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Loading CLIP model...", flush=True)
    embedder = CLIPEmbedder(model_id=CLIP_MODEL_ID)
    processor = PDFProcessor(embedder=embedder)
    print("Processing PDF...", flush=True)
    docs, embeddings, image_store = processor.process(args.pdf)
    print(f"Indexed {len(docs)} chunks (text + images).", flush=True)

    retriever = MultimodalRetriever(embedder, docs, embeddings)
    pipeline = MultimodalRAGPipeline(
        embedder,
        retriever,
        image_store,
        gemini_model=GEMINI_MODEL,
        top_k=args.top_k,
    )
    answer = pipeline.query(args.query, verbose=not args.quiet)
    print(answer)


if __name__ == "__main__":
    main()
