"""
src/build_index.py
------------------
Loads all processed chunks and indexes them into ChromaDB.

WHAT HAPPENS HERE:
  1. Load disease chunks + FDA chunks from data/processed/
  2. For each chunk, call OpenAI embeddings API → 1536-dimensional vector
  3. Store (vector, text, metadata) in ChromaDB on disk

WHY CHROMADB:
  - Runs locally, no server needed, free
  - Persists to disk so you don't re-embed every time
  - Supports metadata filtering (e.g. "only search FDA sources")
  - Same API as Pinecone/Weaviate — easy to swap later

COST ESTIMATE:
  - text-embedding-3-small: $0.02 per 1M tokens
  - 500 FDA labels × ~200 tokens/chunk × ~5 sections = ~500,000 tokens = $0.01
  - Plus disease chunks: ~41 × ~150 tokens = ~6,000 tokens = negligible
  - TOTAL: roughly $0.01-0.02 to build the entire index

HOW TO RUN:
  python src/build_index.py
  python src/build_index.py --reset    # wipe and rebuild from scratch
"""

import json
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def load_all_chunks() -> list[dict]:
    """Load all processed chunks from disk."""
    all_chunks = []

    for filename in ["fda_chunks.json", "disease_chunks.json"]:
        path = os.path.join(PROCESSED_DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  ⚠ Skipping {filename} (not found — run ingest scripts first)")
            continue
        with open(path) as f:
            chunks = json.load(f)
        print(f"  ✓ Loaded {len(chunks)} chunks from {filename}")
        all_chunks.extend(chunks)

    return all_chunks


def split_long_chunk(chunk: dict, max_chars: int = 2000) -> list[dict]:
    """Split long chunks into overlapping sub-chunks."""
    text = chunk["text"]
    if len(text) <= max_chars:
        return [chunk]

    sub_chunks = []
    overlap = 200
    start = 0
    part = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        sub_text = text[start:end]

        new_chunk = chunk.copy()
        new_chunk["text"] = sub_text
        new_chunk["chunk_id"] = f"{chunk['chunk_id']}_part{part}"
        sub_chunks.append(new_chunk)

        # FIX: advance by (max_chars - overlap), not end - overlap
        # Old code: start = end - overlap  ← caused infinite loop on short text
        next_start = start + max_chars - overlap
        if next_start <= start:  # safety guard — always move forward
            break
        start = next_start
        part += 1

    return sub_chunks


def build_index(chunks: list[dict], reset: bool = False) -> chromadb.Collection:
    """
    Embed all chunks and store in ChromaDB.

    Args:
        chunks: list of chunk dicts with 'text', 'chunk_id', and metadata
        reset: if True, delete existing collection and rebuild

    Returns:
        ChromaDB collection (ready for querying)
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    # Initialize ChromaDB (persistent — survives restarts)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection if reset requested
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    # Use local SentenceTransformers — FREE, no API key needed
    # all-MiniLM-L6-v2 is small (80MB), fast, and good quality
    openai_ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity (standard for text)
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection already has {existing_count} documents. Use --reset to rebuild.")
        return collection

    # Split any overly long chunks
    split_chunks = []
    for chunk in chunks:
        split_chunks.extend(split_long_chunk(chunk))

    print(f"\nIndexing {len(split_chunks)} chunks (after splitting)...")

    # Add in smaller batches to avoid memory issues
    batch_size = 25
    total_added = 0

    for i in range(0, len(split_chunks), batch_size):
        batch = split_chunks[i:i + batch_size]

        # Deduplicate IDs within batch (ChromaDB requires unique IDs)
        seen_ids = set()
        deduped_batch = []
        for idx, chunk in enumerate(batch):
            chunk_id = chunk.get("chunk_id", f"chunk_{i+idx}")
            if chunk_id in seen_ids:
                chunk_id = f"{chunk_id}_{idx}"
            seen_ids.add(chunk_id)
            chunk["chunk_id"] = chunk_id
            deduped_batch.append(chunk)

        try:
            collection.add(
                documents=[c["text"] for c in deduped_batch],
                ids=[c["chunk_id"] for c in deduped_batch],
                metadatas=[{
                    "source":    c.get("source", "unknown"),
                    "section":   c.get("section", "unknown"),
                    "drug_name": c.get("drug_name", ""),
                    "disease_name": c.get("disease_name", ""),
                } for c in deduped_batch],
            )
            total_added += len(deduped_batch)
            print(f"  ✓ Indexed {total_added}/{len(split_chunks)} chunks")

            # Small delay to be polite to OpenAI API
            time.sleep(0.2)

        except Exception as e:
            print(f"  ✗ Error at batch {i}: {e}")
            raise

    print(f"\n✓ Index built: {total_added} chunks in collection '{COLLECTION_NAME}'")
    return collection


def verify_index(collection: chromadb.Collection):
    """Quick smoke test: run a few queries to confirm the index works."""
    print("\n--- Index verification ---")

    test_queries = [
        "symptoms of diabetes",
        "ibuprofen dosage for adults",
        "malaria precautions",
    ]

    # We need a fresh client with embedding function for querying
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    for query in test_queries:
        results = col.query(query_texts=[query], n_results=2)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        print(f"\nQuery: '{query}'")
        for doc, meta in zip(docs, metas):
            print(f"  → [{meta['source']}] {doc[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild index")
    parser.add_argument("--limit", type=int, default=None, help="Only index first N chunks (for low-memory machines)")
    args = parser.parse_args()

    print("Loading chunks from data/processed/...")
    chunks = load_all_chunks()

    if not chunks:
        print("\nNo chunks found. Run these first:")
        print("  python src/ingest_disease.py --demo")
        print("  python src/ingest_fda.py --skip-fetch")
        sys.exit(1)

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Limited to first {args.limit} chunks")

    print(f"\nTotal chunks to index: {len(chunks)}")
    collection = build_index(chunks, reset=args.reset)
    verify_index(collection)


if __name__ == "__main__":
    main()