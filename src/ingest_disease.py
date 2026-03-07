"""
src/ingest_disease.py
---------------------
Loads the Kaggle Disease-Symptom dataset into chunks.

DATASET:
  Download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
  Files needed:
    - dataset.csv          (disease → symptoms mapping, 4920 rows)
    - symptom_Description.csv  (disease → description)
    - symptom_precaution.csv   (disease → 4 precautions)

HOW TO RUN:
  1. Download and place CSV files in data/raw/
  2. python src/ingest_disease.py

OUTPUT:
  data/processed/disease_chunks.json
"""

import pandas as pd
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_disease_symptoms(dataset_path: str) -> dict[str, list[str]]:
    """
    Load dataset.csv → {disease_name: [symptom1, symptom2, ...]}

    The CSV has columns: Disease, Symptom_1, Symptom_2, ..., Symptom_17
    Multiple rows per disease (each row has different symptom combinations)
    We merge all symptoms per disease.
    """
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()

    disease_symptoms = {}
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()
        symptoms = []
        for col in symptom_cols:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                # Clean up symptom text (remove underscores)
                val = val.replace("_", " ").strip()
                if val not in symptoms:
                    symptoms.append(val)

        if disease not in disease_symptoms:
            disease_symptoms[disease] = symptoms
        else:
            # Merge new symptoms
            for s in symptoms:
                if s not in disease_symptoms[disease]:
                    disease_symptoms[disease].append(s)

    return disease_symptoms


def load_descriptions(description_path: str) -> dict[str, str]:
    """Load symptom_Description.csv → {disease: description}"""
    df = pd.read_csv(description_path)
    df.columns = df.columns.str.strip()
    return dict(zip(df["Disease"].str.strip(), df["Description"].str.strip()))


def load_precautions(precaution_path: str) -> dict[str, list[str]]:
    """Load symptom_precaution.csv → {disease: [precaution1, ..., precaution4]}"""
    df = pd.read_csv(precaution_path)
    df.columns = df.columns.str.strip()

    precautions = {}
    prec_cols = [c for c in df.columns if "Precaution" in c]
    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()
        precs = [str(row[c]).strip() for c in prec_cols
                 if str(row[c]).strip().lower() not in ("nan", "none", "")]
        precautions[disease] = precs

    return precautions


def build_disease_chunks(
    symptoms: dict[str, list[str]],
    descriptions: dict[str, str],
    precautions: dict[str, list[str]],
) -> list[dict]:
    """
    Combine all three sources into one rich chunk per disease.

    WHY ONE CHUNK PER DISEASE (not split by section):
    The disease dataset is small (~41 diseases). Each disease entry is
    short enough (~200 tokens) to fit in one chunk. Keeping it together
    means a single retrieval gives the full picture — symptoms + description
    + precautions — which is what a user actually needs.

    For large datasets, you would split by section (like FDA labels).
    """
    all_diseases = set(symptoms.keys()) | set(descriptions.keys()) | set(precautions.keys())
    chunks = []

    for disease in sorted(all_diseases):
        parts = [f"Disease: {disease}"]

        # Symptoms
        syms = symptoms.get(disease, [])
        if syms:
            parts.append(f"Symptoms: {', '.join(syms)}")

        # Description
        desc = descriptions.get(disease, "")
        if desc:
            parts.append(f"Description: {desc}")

        # Precautions
        precs = precautions.get(disease, [])
        if precs:
            numbered = "; ".join(f"{i+1}. {p}" for i, p in enumerate(precs))
            parts.append(f"Precautions: {numbered}")

        text = "\n".join(parts)

        # Skip if we have almost no information
        if len(text) < 50:
            continue

        chunks.append({
            "text": text,
            "disease_name": disease,
            "section": "full_disease_profile",
            "source": "Disease Symptom Dataset",
            "chunk_id": f"disease_{disease.lower().replace(' ', '_')}",
            "symptom_count": len(syms),
        })

    return chunks




def main():
    parser = argparse.ArgumentParser(description="Ingest disease-symptom dataset")
    parser.add_argument("--demo", action="store_true",
                        help="Use built-in demo data (no Kaggle download needed)")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "disease_chunks.json")

    # Expect files in data/raw/
    dataset_path    = os.path.join(RAW_DATA_DIR, "dataset.csv")
    description_path = os.path.join(RAW_DATA_DIR, "symptom_Description.csv")
    precaution_path  = os.path.join(RAW_DATA_DIR, "symptom_precaution.csv")

    missing = [p for p in [dataset_path, description_path, precaution_path]
                if not os.path.exists(p)]
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  {f}")
        print("\nTip: Use --demo flag to test with built-in data first.")
        print("     Download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        sys.exit(1)

    symptoms     = load_disease_symptoms(dataset_path)
    descriptions = load_descriptions(description_path)
    precautions  = load_precautions(precaution_path)
    chunks       = build_disease_chunks(symptoms, descriptions, precautions)

    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\n✓ {len(chunks)} disease chunks → {output_path}")
    print("\n--- Sample chunk ---")
    print(chunks[0]["text"][:400])


if __name__ == "__main__":
    main()
