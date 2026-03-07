"""
main.py
-------
Entry point for the Medical RAG system.

Usage:
    python main.py                         # interactive mode
    python main.py --question "..."        # single question
    python main.py --setup                 # ingest data + build index
    python main.py --eval                  # run evaluation
"""

import argparse
import os
import sys


def setup(demo_mode: bool = True):
    """Run the full data pipeline: ingest → index."""
    print("="*50)
    print("Setting up Medical RAG System")
    print("="*50)

    # Step 1: Ingest disease data
    print("\n[1/2] Ingesting disease data...")
    from src.ingest_disease import create_demo_chunks, main as disease_main
    import sys as _sys
    if demo_mode:
        _sys.argv = ["ingest_disease.py", "--demo"]
    disease_main()

    # Step 2: Ingest FDA data (small sample)
    print("\n[2/2] Fetching FDA drug labels (100 labels)...")
    _sys.argv = ["ingest_fda.py", "--limit", "100"]
    from src.ingest_fda import main as fda_main
    fda_main()

    # Step 3: Build index
    print("\n[3/3] Building vector index...")
    _sys.argv = ["build_index.py"]
    from src.build_index import main as index_main
    index_main()

    print("\n✓ Setup complete! Run: python main.py")


def main():
    parser = argparse.ArgumentParser(description="Medical RAG System")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    parser.add_argument("--setup", action="store_true", help="Run full data pipeline")
    parser.add_argument("--eval", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--eval-sample", type=int, default=10, help="Questions to evaluate (default: 10)")
    args = parser.parse_args()

    if args.setup:
        setup()
        return

    if args.eval:
        from rag_engine import MedicalRAGEngine
        from eval.golden_dataset import run_evaluation, save_golden_dataset
        save_golden_dataset()
        engine = MedicalRAGEngine()
        run_evaluation(engine, sample_size=args.eval_sample)
        return

    # Load engine (requires index to exist)
    try:
        from rag_engine import MedicalRAGEngine, interactive_mode
        engine = MedicalRAGEngine()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nRun setup first: python main.py --setup")
        sys.exit(1)

    if args.question:
        response = engine.query(args.question)
        print(f"\n{response.format()}")
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
