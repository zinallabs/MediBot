"""
src/rag_engine.py
-----------------
The core RAG pipeline. This is the file interviewers will ask you to walk through.

PIPELINE FLOW:
  User question
      → embed question
      → ChromaDB: find top-K most similar chunks (vector search)
      → build prompt: system + context + question
      → OpenAI LLM: generate answer grounded in context
      → return answer + sources (citations)

KEY DESIGN DECISIONS (be ready to explain these in interviews):

1. WHY citations?
   Medical information MUST be traceable. "Ibuprofen causes stomach bleeding"
   is only useful if you can point to the FDA label that says so.

2. WHY the "I don't know" instruction?
   Without it, LLMs hallucinate. With it, the model says "I don't have
   enough information" when context doesn't support an answer — this is
   called "abstention" and it's critical in medical applications.

3. WHY temperature=0?
   Medical answers should be deterministic and reproducible. You want
   the same question to give the same answer every time.

4. WHY separate retrieval from generation?
   So you can improve them independently. Bad retrieval → fix chunking or
   embeddings. Good retrieval but bad answer → fix the prompt.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, COLLECTION_NAME, RETRIEVAL_TOP_K
)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its metadata."""
    text: str
    source: str          # e.g. "FDA Drug Label"
    section: str         # e.g. "dosage_and_administration"
    drug_name: str       # e.g. "Ibuprofen" (empty for disease chunks)
    disease_name: str    # e.g. "Diabetes" (empty for FDA chunks)
    distance: float      # cosine distance — lower = more similar

    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        if self.drug_name:
            return f"FDA Drug Label: {self.drug_name} ({self.section.replace('_', ' ').title()})"
        elif self.disease_name:
            return f"Disease Reference: {self.disease_name}"
        return self.source


@dataclass
class RAGResponse:
    """The full response from the RAG pipeline."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    retrieved_context: str   # the raw context fed to the LLM
    model_used: str

    def format(self) -> str:
        """Pretty-print the response with citations."""
        lines = [
            f"Answer: {self.answer}",
            "",
            "Sources:",
        ]
        for i, source in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {source.citation}")
        return "\n".join(lines)


# ── Prompt templates ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical information assistant that helps users understand health conditions and medications.

CRITICAL RULES:
1. Answer ONLY using the information provided in the [CONTEXT] below.
2. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in my knowledge base to answer this question. Please consult a healthcare professional."
3. ALWAYS cite your sources using the source labels provided (e.g. [Source 1], [Source 2]).
4. Never make up medical information. Never guess.
5. For any serious medical situation, always remind the user to consult a healthcare professional.
6. Keep answers clear and concise — avoid unnecessary medical jargon.

DISCLAIMER: This system is for informational purposes only and is NOT a substitute for professional medical advice."""

USER_PROMPT_TEMPLATE = """[CONTEXT]
{context}

[QUESTION]
{question}

Please answer the question using only the context above. Cite sources with [Source N] notation."""


# ── RAG Engine ───────────────────────────────────────────────────

class MedicalRAGEngine:
    """
    Production-style RAG engine for medical Q&A.

    Usage:
        engine = MedicalRAGEngine()
        response = engine.query("What are the symptoms of diabetes?")
        print(response.format())
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")

        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        # Local embeddings — FREE, no API key needed, same model as build_index.py
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        try:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            print(f"✓ Loaded index: {self.collection.count()} chunks")
        except Exception:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run: python src/build_index.py"
            )

    # ── Step 1: Retrieve ─────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
        """
        Find the most relevant chunks for a question.

        ChromaDB automatically embeds the question and does cosine similarity
        search against all indexed chunks.
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                section=meta.get("section", "unknown"),
                drug_name=meta.get("drug_name", ""),
                disease_name=meta.get("disease_name", ""),
                distance=dist,
            ))

        return chunks

    # ── Step 2: Build context ────────────────────────────────────

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into the context string for the prompt.

        We label each source [Source 1], [Source 2] so the LLM can
        reference them in its answer (enables citations).
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk.citation}\n{chunk.text}"
            )
        return "\n\n---\n\n".join(context_parts)

    # ── Step 3: Generate ─────────────────────────────────────────

    def generate(self, question: str, context: str) -> str:
        """
        Call OpenAI LLM with the retrieved context as grounding.

        temperature=0 → deterministic output (same question = same answer)
        This is what you want for medical info: reproducible, consistent.
        """
        prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,       # deterministic
            max_tokens=600,      # enough for a thorough answer
        )

        return response.choices[0].message.content.strip()

    # ── Full pipeline ────────────────────────────────────────────

    def query(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResponse:
        """
        Run the full RAG pipeline: retrieve → build context → generate.

        This is what gets called in tests, the API, and the UI.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k)

        # Step 2: Build context string
        context = self.build_context(chunks)

        # Step 3: Generate grounded answer
        answer = self.generate(question, context)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            retrieved_context=context,
            model_used=LLM_MODEL,
        )


# ── Interactive CLI ───────────────────────────────────────────────

def interactive_mode(engine: MedicalRAGEngine):
    """Simple command-line interface for testing the RAG system."""
    print("\n" + "="*60)
    print("Medical RAG System — Interactive Mode")
    print("Type 'quit' to exit, 'sources' to see retrieved chunks")
    print("="*60 + "\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        try:
            response = engine.query(question)
            print(f"\n{response.format()}\n")

            # Show retrieval quality (useful during development)
            print(f"[Debug] Retrieved {len(response.sources)} chunks, "
                  f"best distance: {response.sources[0].distance:.3f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    engine = MedicalRAGEngine()
    interactive_mode(engine)
