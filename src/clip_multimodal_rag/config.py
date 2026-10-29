"""Configuration and environment settings for CLIP Multimodal RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Avoid OpenMP / FAISS duplicate-lib crashes on some macOS setups
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model defaults
CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-nano")

# RAG defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CLIP_MAX_LENGTH = 77  # CLIP's max token length


def get_openai_api_key() -> str:
    """Return OpenAI API key; raise if not set."""
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env or the environment."
        )
    return key
