import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"

PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_META_PATH = INDEX_DIR / "chunks_metadata.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
