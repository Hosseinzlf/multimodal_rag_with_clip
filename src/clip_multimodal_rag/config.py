"""Configuration and environment settings for CLIP Multimodal RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

# API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model defaults
CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# RAG defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CLIP_MAX_LENGTH = 77  # CLIP's max token length


def get_google_api_key() -> str:
    """Return Google API key; raise if not set."""
    key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it in .env or the environment."
        )
    return key
