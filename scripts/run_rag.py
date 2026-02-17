#!/usr/bin/env python3
"""
Convenience script to run the RAG pipeline from the project root.

Usage (from project root):
    python scripts/run_rag.py --pdf path/to/file.pdf --query "Your question"

Or install the package and use:
    clip-rag --pdf path/to/file.pdf --query "Your question"
"""

import sys
from pathlib import Path

# Ensure package is on path when run as script
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from clip_multimodal_rag.cli import main

if __name__ == "__main__":
    main()
