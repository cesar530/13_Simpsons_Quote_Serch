#!/usr/bin/env python3
# =============================================================================
# Simpsons Quote Search Engine - Main Application
# =============================================================================
# Author: César Adrián Delgado Díaz
# Portfolio: https://tu-portfolio.com
# LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
# GitHub: https://github.com/cesar530
# License: MIT
# =============================================================================
"""
Simpsons Quote Search Engine - CLI Client

Este módulo proporciona un cliente de línea de comandos para interactuar
con el Simpsons Quote Search Engine, permitiendo:

- Buscar quotes usando BM25, semántico o híbrido
- Hacer preguntas con RAG
- Ingestar datos
- Evaluar el sistema

Usage:
    python simpsons_search.py search "homer beer"
    python simpsons_search.py ask "¿Cuál es la filosofía de Homer?"
    python simpsons_search.py ingest --source huggingface
    python simpsons_search.py serve --port 8000
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🍩 SIMPSONS QUOTE SEARCH ENGINE 🍩                         ║
    ║                                                              ║
    ║   RAG + Evaluación + Observabilidad                         ║
    ║                                                              ║
    ║   Autor: César Adrián Delgado Díaz                          ║
    ║   Licencia: MIT                                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def search_command(args):
    """Execute search command."""
    from config import get_settings
    from retrieval import HybridRetriever
    from utils import Timer
    
    settings = get_settings()
    
    print(f"\n🔍 Searching for: '{args.query}'")
    print(f"   Method: {args.method}")
    print(f"   Top K: {args.top_k}")
    print("-" * 60)
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Try to load existing indices
    index_dir = Path(args.index_dir or "data/indices")
    if index_dir.exists():
        try:
            retriever.load(str(index_dir))
            print(f"✅ Loaded indices from {index_dir}")
        except Exception as e:
            print(f"⚠️ Could not load indices: {e}")
            print("   Please run 'ingest' command first.")
            return
    else:
        print(f"❌ Index directory not found: {index_dir}")
        print("   Please run 'ingest' command first.")
        return
    
    # Execute search
    with Timer("Search") as timer:
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            method=args.method
        )
    
    # Display results
    print(f"\n📊 Found {len(results)} results in {timer.elapsed_ms:.2f}ms\n")
    
    for i, doc in enumerate(results, 1):
        print(f"[{i}] Score: {doc.get('score', 0):.4f}")
        print(f"    👤 {doc.get('character', 'Unknown')}:")
        text = doc.get('text', '')
        if len(text) > 100:
            print(f"    \"{text[:100]}...\"")
        else:
            print(f"    \"{text}\"")
        if doc.get('episode_code'):
            print(f"    📺 {doc.get('episode_code')}")
        print()
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to {args.output}")


def ask_command(args):
    """Execute ask (RAG) command."""
    from config import get_settings
    from retrieval import HybridRetriever
    from retrieval.generator import ResponseGenerator
    from utils import Timer
    
    settings = get_settings()
    
    print(f"\n❓ Question: {args.question}")
    print("-" * 60)
    
    # Check for API key
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not configured!")
        print("   Set it in .env file or as environment variable.")
        return
    
    # Initialize retriever
    retriever = HybridRetriever()
    index_dir = Path(args.index_dir or "data/indices")
    
    if index_dir.exists():
        try:
            retriever.load(str(index_dir))
        except Exception as e:
            print(f"❌ Could not load indices: {e}")
            return
    else:
        print(f"❌ Index directory not found: {index_dir}")
        return
    
    # Initialize generator
    generator = ResponseGenerator(
        model=args.model or settings.OPENAI_MODEL,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Retrieval
    print("\n📚 Retrieving relevant documents...")
    with Timer("Retrieval") as retrieval_timer:
        documents = retriever.search(
            query=args.question,
            top_k=args.top_k,
            method="hybrid"
        )
    print(f"   Found {len(documents)} documents in {retrieval_timer.elapsed_ms:.2f}ms")
    
    # Generation
    print("\n🤖 Generating response...")
    with Timer("Generation") as gen_timer:
        response = generator.generate(args.question, documents)
    
    # Display response
    print("\n💬 Response:")
    print("=" * 60)
    print(response.get('answer', 'No response generated'))
    print("=" * 60)
    
    # Citations
    if args.show_citations:
        print("\n📖 Sources:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"   [{i}] {doc.get('character', 'Unknown')}: \"{doc.get('text', '')[:50]}...\"")
    
    # Metrics
    print(f"\n📊 Metrics:")
    print(f"   - Retrieval time: {retrieval_timer.elapsed_ms:.2f}ms")
    print(f"   - Generation time: {gen_timer.elapsed_ms:.2f}ms")
    print(f"   - Total time: {retrieval_timer.elapsed_ms + gen_timer.elapsed_ms:.2f}ms")
    print(f"   - Tokens used: {response.get('tokens_used', 'N/A')}")


def ingest_command(args):
    """Execute data ingestion command."""
    from ingestion import DataIngester
    from utils import Timer
    
    print("\n📥 Starting data ingestion...")
    print(f"   Source: {args.source}")
    print(f"   Max samples: {args.max_samples}")
    print("-" * 60)
    
    # Initialize ingester
    ingester = DataIngester(
        data_dir=args.data_dir or "data",
        output_dir=args.output_dir or "data/processed"
    )
    
    # Ingest based on source
    with Timer("Ingestion") as timer:
        if args.source == "huggingface":
            count = ingester.ingest_from_huggingface(
                dataset_name=args.dataset or "jayantdocplix/simpsons-script-lines",
                max_samples=args.max_samples
            )
        elif args.source == "csv":
            if not args.file:
                print("❌ --file required for CSV source")
                return
            count = ingester.ingest_from_csv(args.file)
        elif args.source == "json":
            if not args.file:
                print("❌ --file required for JSON source")
                return
            count = ingester.ingest_from_json(args.file)
        else:
            print(f"❌ Unknown source: {args.source}")
            return
    
    print(f"\n✅ Loaded {count} documents in {timer.elapsed_seconds:.2f}s")
    
    # Deduplicate
    if not args.no_dedupe:
        dups = ingester.deduplicate()
        print(f"🔄 Removed {dups} duplicates")
    
    # Get statistics
    stats = ingester.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   - Total documents: {stats['total_documents']}")
    print(f"   - Unique characters: {stats['unique_characters']}")
    
    print(f"\n🌟 Top 5 Characters:")
    for char, count in stats['top_characters'][:5]:
        print(f"   - {char}: {count} quotes")
    
    # Save processed data
    output_file = ingester.save_processed("quotes_processed.json")
    print(f"\n💾 Saved to: {output_file}")
    
    # Build indices if requested
    if args.build_index:
        print("\n🔨 Building search indices...")
        from retrieval import HybridRetriever
        
        retriever = HybridRetriever()
        retriever.initialize(lazy=False)
        
        documents = [doc.to_dict() for doc in ingester.documents]
        
        with Timer("Indexing") as index_timer:
            num_indexed = retriever.index_documents(
                documents,
                text_field="text",
                id_field="id",
                show_progress=True
            )
        
        print(f"\n✅ Indexed {num_indexed} documents in {index_timer.elapsed_seconds:.2f}s")
        
        # Save indices
        index_dir = args.index_dir or "data/indices"
        retriever.save(index_dir)
        print(f"💾 Indices saved to: {index_dir}")


def serve_command(args):
    """Start the API server."""
    import uvicorn
    from config import get_settings
    
    settings = get_settings()
    
    print_banner()
    print(f"🚀 Starting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Reload: {args.reload}")
    print("-" * 60)
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


def evaluate_command(args):
    """Run evaluation."""
    from config import get_settings
    from retrieval import HybridRetriever
    from eval import RetrievalEvaluator
    from utils import Timer
    import json
    
    print("\n📊 Running evaluation...")
    print("-" * 60)
    
    # Load test queries
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_queries = json.load(f)
    else:
        # Default test queries
        test_queries = [
            {"id": "q1", "query": "Homer eating donuts", "relevant_chars": ["Homer Simpson"]},
            {"id": "q2", "query": "Bart skateboard trouble", "relevant_chars": ["Bart Simpson"]},
            {"id": "q3", "query": "Lisa saxophone music", "relevant_chars": ["Lisa Simpson"]},
            {"id": "q4", "query": "Mr Burns evil money", "relevant_chars": ["Mr. Burns"]},
            {"id": "q5", "query": "Ned Flanders neighborino", "relevant_chars": ["Ned Flanders"]},
        ]
    
    # Initialize retriever
    retriever = HybridRetriever()
    index_dir = Path(args.index_dir or "data/indices")
    
    if index_dir.exists():
        retriever.load(str(index_dir))
    else:
        print(f"❌ Index directory not found: {index_dir}")
        return
    
    # Run evaluation
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    
    methods_to_test = args.methods.split(",") if args.methods else ["bm25", "semantic", "hybrid"]
    
    for method in methods_to_test:
        print(f"\n🔍 Evaluating {method.upper()} retrieval...")
        
        for test in test_queries:
            results = retriever.search(
                query=test['query'],
                top_k=10,
                method=method
            )
            
            # Determine relevant based on character
            relevant_ids = [
                r['id'] for r in results
                if r.get('character') in test.get('relevant_chars', [])
            ]
            
            evaluator.add_query_result(
                query_id=test['id'],
                query_text=test['query'],
                retrieved_ids=[r['id'] for r in results],
                retrieved_scores=[r.get('score', 0) for r in results],
                relevant_ids=relevant_ids if relevant_ids else [results[0]['id']] if results else []
            )
        
        # Calculate and display metrics
        metrics = evaluator.evaluate()
        print(f"\n📈 {method.upper()} Results:")
        print(f"   - Recall@5: {metrics.recall_at_5:.4f}")
        print(f"   - MRR: {metrics.mrr:.4f}")
        print(f"   - NDCG@10: {metrics.ndcg_at_10:.4f}")
        
        # Reset for next method
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    
    if args.output:
        print(f"\n💾 Results saved to: {args.output}")


def interactive_mode():
    """Run interactive mode."""
    from config import get_settings
    
    print_banner()
    print("🎮 Interactive Mode")
    print("   Commands: search, ask, stats, help, quit")
    print("-" * 60)
    
    # Try to initialize retriever
    retriever = None
    try:
        from retrieval import HybridRetriever
        retriever = HybridRetriever()
        if Path("data/indices").exists():
            retriever.load("data/indices")
            print("✅ Indices loaded successfully")
        else:
            print("⚠️ No indices found. Run 'python simpsons_search.py ingest' first.")
    except Exception as e:
        print(f"⚠️ Could not initialize retriever: {e}")
    
    while True:
        try:
            user_input = input("\n🍩 > ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""
            
            if cmd in ("quit", "exit", "q"):
                print("👋 ¡Ay caramba! Goodbye!")
                break
            
            elif cmd == "help":
                print("\nAvailable commands:")
                print("  search <query>  - Search for quotes")
                print("  ask <question>  - Ask a question (RAG)")
                print("  stats           - Show statistics")
                print("  quit            - Exit")
            
            elif cmd == "search":
                if not query:
                    print("Usage: search <query>")
                    continue
                
                if retriever:
                    results = retriever.search(query, top_k=5, method="hybrid")
                    print(f"\n📊 Found {len(results)} results:\n")
                    for i, doc in enumerate(results, 1):
                        print(f"[{i}] {doc.get('character', 'Unknown')}:")
                        print(f"    \"{doc.get('text', '')[:80]}...\"")
                else:
                    print("❌ Retriever not initialized")
            
            elif cmd == "ask":
                if not query:
                    print("Usage: ask <question>")
                    continue
                
                settings = get_settings()
                if not settings.OPENAI_API_KEY:
                    print("❌ OPENAI_API_KEY not configured")
                    continue
                
                if retriever:
                    from retrieval.generator import ResponseGenerator
                    
                    docs = retriever.search(query, top_k=5, method="hybrid")
                    generator = ResponseGenerator()
                    response = generator.generate(query, docs)
                    
                    print(f"\n💬 {response.get('answer', 'No response')}")
                else:
                    print("❌ Retriever not initialized")
            
            elif cmd == "stats":
                if retriever:
                    print(f"\n📊 Index Statistics:")
                    # Add stats display
                else:
                    print("❌ Retriever not initialized")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🍩 Simpsons Quote Search Engine - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Search for quotes:
    python simpsons_search.py search "homer beer donuts"
    
  Ask a question (RAG):
    python simpsons_search.py ask "¿Cuál es la filosofía de Homer?"
    
  Ingest data:
    python simpsons_search.py ingest --source huggingface --max-samples 10000
    
  Start API server:
    python simpsons_search.py serve --port 8000
    
  Interactive mode:
    python simpsons_search.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for quotes")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--method", choices=["bm25", "semantic", "hybrid"], 
                               default="hybrid", help="Search method")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--index-dir", help="Index directory path")
    search_parser.add_argument("--output", "-o", help="Output file (JSON)")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question (RAG)")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    ask_parser.add_argument("--model", help="OpenAI model to use")
    ask_parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ask_parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens in response")
    ask_parser.add_argument("--index-dir", help="Index directory path")
    ask_parser.add_argument("--show-citations", action="store_true", help="Show source citations")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data")
    ingest_parser.add_argument("--source", choices=["huggingface", "csv", "json"], 
                               default="huggingface", help="Data source")
    ingest_parser.add_argument("--dataset", help="HuggingFace dataset name")
    ingest_parser.add_argument("--file", "-f", help="File path for CSV/JSON")
    ingest_parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to load")
    ingest_parser.add_argument("--data-dir", help="Data directory")
    ingest_parser.add_argument("--output-dir", help="Output directory")
    ingest_parser.add_argument("--index-dir", help="Index directory")
    ingest_parser.add_argument("--build-index", action="store_true", help="Build search indices")
    ingest_parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--log-level", default="INFO", help="Log level")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--test-file", help="Test queries JSON file")
    eval_parser.add_argument("--methods", help="Methods to test (comma-separated)")
    eval_parser.add_argument("--index-dir", help="Index directory path")
    eval_parser.add_argument("--output", "-o", help="Output file")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "search":
        search_command(args)
    elif args.command == "ask":
        ask_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "serve":
        serve_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "interactive":
        interactive_mode()
    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
