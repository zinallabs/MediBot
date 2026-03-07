"""
config/settings.py
------------------
Central configuration. All magic numbers live here.
To change any setting: edit this file OR set the environment variable.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")


# ── LLM & Embedding ───────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Vector Store ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME    = "medical_rag"

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# ── Retrieval ─────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))

# ── Data paths ────────────────────────────────────────────────────
RAW_DATA_DIR       = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
GOLDEN_DATA_DIR    = "./data/golden"

# ── Evaluation thresholds (CI gate will FAIL if below these) ──────
# These are the numbers you will put on your resume
FAITHFULNESS_THRESHOLD  = 0.80   # % of answer claims supported by retrieved docs
CONTEXT_RECALL_THRESHOLD = 0.75  # % of relevant docs actually retrieved
ANSWER_RELEVANCY_THRESHOLD = 0.70 # % of answer relevant to the question
