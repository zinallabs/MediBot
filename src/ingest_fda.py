"""
src/ingest_fda.py
-----------------
Downloads drug labels from the free openFDA API.

WHY THIS DATA:
  - 255,000+ official FDA drug labels (the same data pharmacists use)
  - Free, no signup required for basic use
  - Fields we care about: indications, warnings, dosage, active ingredients

HOW TO RUN:
  python src/ingest_fda.py --limit 500

OUTPUT:
  data/raw/fda_labels.json   (raw API responses)
  data/processed/fda_chunks.json  (one chunk per label section, ready for indexing)
"""

import requests
import json
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── Fields we extract into the knowledge base ────────────────────
# Each field becomes a SEPARATE chunk so retrieval is precise.
# e.g. "what's the dosage?" only retrieves the dosage chunk, not warnings.
FIELDS_TO_EXTRACT = [
    "indications_and_usage",      # what the drug treats
    "warnings",                   # danger signals — very important
    "dosage_and_administration",  # how much to take
    "active_ingredient",          # what's in it
    "purpose",                    # summary of what it does
    "stop_use",                   # when to stop taking
    "do_not_use",                 # contraindications
    "pregnancy_or_breast_feeding",# safety for pregnant users
]


def fetch_fda_labels(limit: int = 500, api_key: str = "") -> list[dict]:
    """
    Fetch `limit` drug labels from openFDA in batches of 100.

    Rate limits:
      - Without API key: 40 requests/minute, 1000/day
      - With free API key: 240 requests/minute, 120,000/day
    """
    base_url = "https://api.fda.gov/drug/label.json"
    all_results = []
    batch_size = 100  # openFDA max per request

    print(f"Fetching {limit} FDA drug labels...")

    for skip in range(0, limit, batch_size):
        current_batch = min(batch_size, limit - skip)
        params = {"limit": current_batch, "skip": skip}
        if api_key:
            params["api_key"] = api_key

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            all_results.extend(results)
            print(f"  ✓ Fetched {skip + len(results)}/{limit} labels")

            # Be polite to the API — avoid rate limiting
            time.sleep(0.5)

        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP error at skip={skip}: {e}")
            break
        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout at skip={skip}. Retrying once...")
            time.sleep(5)
            continue

    return all_results


def extract_chunks_from_label(label: dict) -> list[dict]:
    """
    Convert one FDA label (JSON object) into multiple text chunks.

    Each chunk = one section of the label.
    We store metadata (drug name, section, source) so we can cite sources.

    Example output chunk:
    {
        "text": "Adults: take 2 tablets every 4-6 hours...",
        "drug_name": "Ibuprofen",
        "section": "dosage_and_administration",
        "source": "FDA Drug Label",
        "chunk_id": "Ibuprofen_dosage_and_administration"
    }
    """
    chunks = []

    # Get the drug name — try brand name first, fall back to generic
    openfda = label.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    drug_name = brand_names[0] if brand_names else (generic_names[0] if generic_names else "Unknown Drug")
    drug_name = drug_name.title()  # normalize casing

    for field in FIELDS_TO_EXTRACT:
        if field not in label:
            continue  # skip missing sections

        # openFDA wraps everything in a list — take the first element
        raw_text = label[field]
        if isinstance(raw_text, list):
            raw_text = raw_text[0]

        # Skip very short sections (probably just headers)
        if len(raw_text.strip()) < 20:
            continue

        # Clean up the text slightly
        text = raw_text.strip()
        # Remove redundant section headers that openFDA sometimes includes
        for prefix in [field.upper().replace("_", " ") + " ", "USES ", "WARNINGS "]:
            if text.upper().startswith(prefix.upper()):
                text = text[len(prefix):]

        # Format the chunk with context so the LLM knows what it's reading
        formatted_text = f"Drug: {drug_name}\nSection: {field.replace('_', ' ').title()}\n\n{text}"

        chunks.append({
            "text": formatted_text,
            "drug_name": drug_name,
            "section": field,
            "source": "FDA Drug Label",
            "chunk_id": f"{drug_name}_{field}",
        })

    return chunks


def process_all_labels(raw_labels: list[dict]) -> list[dict]:
    """Convert raw FDA labels into chunks ready for vector indexing."""
    all_chunks = []
    skipped = 0

    for label in raw_labels:
        chunks = extract_chunks_from_label(label)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)

    print(f"\nProcessed {len(raw_labels)} labels → {len(all_chunks)} chunks ({skipped} labels skipped)")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA drug labels")
    parser.add_argument("--limit", type=int, default=500, help="Number of labels to fetch (default: 500)")
    parser.add_argument("--api-key", type=str, default="", help="Optional openFDA API key for higher rate limits")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip API fetch, reprocess existing raw data")
    args = parser.parse_args()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    raw_path = os.path.join(RAW_DATA_DIR, "fda_labels.json")
    processed_path = os.path.join(PROCESSED_DATA_DIR, "fda_chunks.json")

    # ── Step 1: Fetch raw data ────────────────────────────────────
    if args.skip_fetch and os.path.exists(raw_path):
        print(f"Loading existing raw data from {raw_path}")
        with open(raw_path) as f:
            raw_labels = json.load(f)
    else:
        raw_labels = fetch_fda_labels(args.limit, args.api_key)
        with open(raw_path, "w") as f:
            json.dump(raw_labels, f, indent=2)
        print(f"Saved {len(raw_labels)} raw labels → {raw_path}")

    # ── Step 2: Process into chunks ───────────────────────────────
    chunks = process_all_labels(raw_labels)
    with open(processed_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks → {processed_path}")

    # ── Step 3: Quick sanity check ────────────────────────────────
    print("\n--- Sample chunk (first one) ---")
    if chunks:
        print(json.dumps(chunks[0], indent=2))


if __name__ == "__main__":
    main()
