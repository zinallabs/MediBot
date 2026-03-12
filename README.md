# Medical RAG System 

A production-style Retrieval-Augmented Generation (RAG) system for medical Q&A,
built to demonstrate real AI engineering skills for the healthcare domain.

---

## What This System Does

Ask any question about diseases, symptoms, or medications. The system:
1. Searches a knowledge base of FDA drug labels + disease profiles
2. Retrieves the most relevant information
3. Generates a grounded answer with citations
4. Refuses to answer when information is insufficient (no hallucination)

```
Q: What are the symptoms of diabetes?

A: According to the disease reference database [Source 1], symptoms of 
   diabetes include increased thirst, frequent urination, extreme hunger, 
   unexplained weight loss, fatigue, blurred vision, and slow healing sores.

Sources:
  [1] Disease Reference: Diabetes
```

---

## Architecture

```
User Question
     │
     ▼
[Embedding]  ← text-embedding-3-small (OpenAI)
     │
     ▼
[ChromaDB]   ← cosine similarity search → top-5 chunks
     │
     ▼
[Prompt]     ← system prompt + retrieved context + question
     │
     ▼
[GPT-4o-mini] ← temperature=0, max_tokens=600
     │
     ▼
Answer + Citations
```

**Knowledge Base:**
- **FDA Drug Labels** — 500 labels from openFDA API (free, official)
- **Disease-Symptom Dataset** — 41 diseases with symptoms, descriptions, precautions

**Tech Stack:**
| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | text-embedding-3-small | $0.02/1M tokens, strong quality |
| Vector Store | ChromaDB | Local, free, production-ready API |
| LLM | gpt-4o-mini | Fast, cheap ($0.15/1M input tokens) |
| Evaluation | RAGAS | Industry-standard RAG metrics |

---

## Quick Start

### 1. Clone and install
```bash
git clone <your-repo>
cd medical_rag
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run setup (downloads data + builds index)
```bash
python main.py --setup
```

### 4. Ask questions
```bash
# Interactive mode
python main.py

# Single question
python main.py --question "What are the symptoms of malaria?"
```

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [openFDA API](https://open.fda.gov/apis/drug/label/) | 255,000+ official FDA drug labels | Free, no signup |
| [Kaggle Disease Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | 41 diseases, symptoms, descriptions, precautions | Free, Kaggle account |
| [MIMIC-IV](https://physionet.org/content/mimiciv/) | Real de-identified ICU clinical notes | Free, 1-week approval |

The system ships with demo data (5 diseases) so you can test immediately
without any downloads.

---

## Evaluation

The system is evaluated on a golden dataset of 20 hand-curated Q&A pairs.

### Run evaluation
```bash
python main.py --eval --eval-sample 10
```

### Target scores
| Metric | Target | What it measures |
|--------|--------|-----------------|
| Faithfulness | ≥ 0.80 | Are claims supported by retrieved docs? |
| Context Recall | ≥ 0.75 | Were relevant chunks retrieved? |
| Answer Relevancy | ≥ 0.70 | Is the answer relevant to the question? |

### CI/CD Quality Gate
Every pull request triggers automated evaluation via GitHub Actions.
PRs that cause metric scores to drop below thresholds are **blocked**.

---

## Project Structure

```
medical_rag/
├── src/
│   ├── ingest_fda.py       # Pull drug labels from openFDA API
│   ├── ingest_disease.py   # Load disease-symptom dataset
│   ├── build_index.py      # Embed chunks → ChromaDB
│   └── rag_engine.py       # Core RAG pipeline (retrieve → generate)
├── evals/
│   └── golden_dataset.py   # 20 Q&A pairs + RAGAS evaluation
├── config/
│   └── settings.py         # All configuration in one place
├── data/
│   ├── raw/                # Downloaded data (gitignored)
│   ├── processed/          # Chunked data ready for indexing
│   └── golden/             # Evaluation dataset + results
├── .github/workflows/
│   └── rag_quality_gate.yml  # CI pipeline
├── main.py                 # Entry point
└── requirements.txt
```

---

## Design Decisions

**Why ChromaDB over Pinecone/Weaviate?**
For a portfolio project, local ChromaDB eliminates infrastructure cost and
complexity. The API is identical to Weaviate — switching takes ~10 lines.

**Why temperature=0?**
Medical information should be deterministic and reproducible. The same
question should produce the same answer every time.

**Why `source` citations in every answer?**
RAG without citations is not trustworthy in medical contexts. Every claim
must be traceable to its source document. This is a hard requirement for
any real healthcare application.

**Why abstention on unknown questions?**
A medical system that confidently answers questions it doesn't have data
for is dangerous. The "I don't have enough information" response is
explicitly tested in the adversarial section of the golden dataset.

---

## Limitations

- **Not a clinical tool.** Do not use for real medical decisions.
- **Knowledge cutoff.** FDA labels are from a specific download date; the
  system does not update in real time.
- **English only.** No multilingual support.
- **41 diseases.** The disease dataset is small; many conditions are not
  covered. MIMIC-IV integration would significantly expand coverage.
- **No HIPAA compliance.** This system is not suitable for real patient data.

---

## Next Steps (Project 2 Roadmap)

- [ ] Add hybrid retrieval (BM25 + vector) with RRF re-ranking
- [ ] Add Cohere Rerank for better retrieval precision
- [ ] Integrate MIMIC-IV discharge summaries (requires PhysioNet access)
- [ ] Add Langfuse observability (latency tracing, cost tracking)
- [ ] Build FastAPI endpoint with rate limiting

---

## Disclaimer

This system is for educational and portfolio demonstration purposes only.
It is NOT intended for clinical use. Always consult a qualified healthcare
professional for medical advice.
