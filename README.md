# CLIP Multimodal RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** system for PDFs that combines **text and images** in a single pipeline. It uses **CLIP** for unified embeddings, **FAISS** for similarity search, and **Google Gemini** for answer generation.

## Features

- **Multimodal indexing**: Extract and embed both text chunks and images from PDFs.
- **Unified retrieval**: CLIP places text and images in one vector space so queries retrieve the most relevant content regardless of type.
- **Production-oriented layout**: Installable package, config via environment variables, and a clean notebook for experimentation.

## Architecture

```
PDF → [PyMuPDF] → Text chunks + Images
                        ↓
              [CLIP] → Text & image embeddings
                        ↓
              [FAISS] → Vector index
                        ↓
Query → [CLIP] → query embedding → [FAISS] → top-k docs (text + images)
                        ↓
              [Gemini] ← context (text + base64 images) → Answer
```

## Requirements

- **Python** 3.10+
- **Google API key** (for Gemini)
- **~2GB RAM** for CLIP model and FAISS index (varies with PDF size)

## Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/YOUR_USERNAME/clip-multimodal-rag.git
cd clip-multimodal-rag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

### 3. Install the package and dependencies

```bash
pip install -e .
# or:  pip install -r requirements.txt
```

### 4. Environment variables

Copy the example env file and set your Google API key:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

Optional variables (see `.env.example`):

- `GEMINI_MODEL` – Gemini model name (default: `gemini-2.5-flash`)
- `CLIP_MODEL_ID` – CLIP model (default: `openai/clip-vit-base-patch32`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K` – RAG tuning

## Usage

### Option A: Jupyter notebook (recommended for exploration)

1. Place your PDF in the project root (or set `pdf_path` in the notebook).
2. Open and run the notebook:

```bash
jupyter notebook main.ipynb
```

3. Run all cells to load CLIP, process the PDF, build the index, and query with Gemini.

### Option B: Python API

```python
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from clip_multimodal_rag import CLIPEmbedder, PDFProcessor, MultimodalRetriever, MultimodalRAGPipeline

# 1. Load CLIP and process PDF
embedder = CLIPEmbedder()
processor = PDFProcessor(chunk_size=500, chunk_overlap=100, embedder=embedder)
pdf_path = Path("your_document.pdf")
docs, embeddings, image_store = processor.process(pdf_path)

# 2. Build retriever and pipeline
retriever = MultimodalRetriever(embedder, docs, embeddings)
pipeline = MultimodalRAGPipeline(
    embedder, retriever, image_store,
    gemini_model="gemini-2.5-flash",
    top_k=5,
)

# 3. Query
answer = pipeline.query("Summarize the main findings from the document.")
print(answer)
```

### Option C: CLI (optional)

```bash
python scripts/run_rag.py --pdf path/to/file.pdf --query "Your question here"
```

## Project structure

```
clip-multimodal-rag/
├── src/
│   └── clip_multimodal_rag/
│       ├── __init__.py      # Package exports
│       ├── config.py        # Env and defaults
│       ├── embeddings.py    # CLIP embedder
│       ├── pdf_processor.py # PDF → docs + embeddings
│       ├── retrieval.py     # FAISS retriever
│       └── pipeline.py      # RAG pipeline (retrieve + Gemini)
├── scripts/
│   └── run_rag.py           # CLI entrypoint
├── main.ipynb               # End-to-end demo notebook
├── requirements.txt
├── pyproject.toml
├── .env.example
├── README.md
└── LICENSE
```

## Technologies

| Component    | Technology |
|-------------|------------|
| PDF parsing | PyMuPDF (fitz) |
| Embeddings  | CLIP (OpenAI), Hugging Face Transformers |
| Vector store| FAISS (langchain-community) |
| LLM         | Google Gemini (LangChain) |
| Framework   | PyTorch, LangChain |


