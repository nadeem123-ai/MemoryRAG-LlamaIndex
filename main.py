"""
main.py
-------
Terminal entry point for the LlamaIndex RAG pipeline.

Usage:
    python main.py --pdf docs/
    python main.py --pdf docs/resume.pdf docs/report.pdf
    python main.py --pdf docs/ --provider openai --model gpt-4o-mini
"""

import argparse
from rag import RAGPipeline

DEMO_QUESTIONS = [
    "What documents have been loaded and what are they about?",
    "What are the key topics or skills mentioned across all documents?",
    "Tell me more about the first topic you mentioned",
]


def print_result(result: dict) -> None:
    sources = result["sources"]
    print(f"\nRetrieved {len(sources)} chunks:")
    for i, s in enumerate(sources):
        print(f"  {i + 1}. [{s['file']} | Page {s['page']}] {s['preview']}...")
    print(f"\nAnswer:\n{result['answer']}")
    print("=" * 70)


def run_demo(pipeline: RAGPipeline) -> None:
    print("\n" + "=" * 70)
    print("DEMO MODE — watch how memory works across questions")
    print("=" * 70)
    for question in DEMO_QUESTIONS:
        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print("=" * 70)
        result = pipeline.ask(question)
        print_result(result)
        input("\nPress Enter for next question...")


def run_interactive(pipeline: RAGPipeline) -> None:
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE — type 'exit' to quit, 'clear' to reset memory")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if question.lower() == "clear":
            pipeline.clear_memory()
            print("✓ Memory cleared\n")
            continue
        if question.lower() == "history":
            history = pipeline.get_history()
            if not history:
                print("No history yet.\n")
            else:
                for msg in history:
                    role = "You" if str(msg.role) == "user" else "Assistant"
                    print(f"{role}: {str(msg.content)[:100]}...")
            continue

        try:
            print(f"\n{'=' * 70}\nQuestion: {question}\n{'=' * 70}")
            result = pipeline.ask(question)
            print_result(result)
        except Exception as e:
            print(f"Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex Multi-PDF RAG")
    parser.add_argument("--pdf", nargs="+", default=["docs/"], metavar="PATH")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-demo", action="store_true")
    args = parser.parse_args()

    pdf_input = args.pdf[0] if len(args.pdf) == 1 else args.pdf

    pipeline = RAGPipeline(
        pdf_path=pdf_input,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
    )

    if not args.no_demo:
        run_demo(pipeline)
    run_interactive(pipeline)


if __name__ == "__main__":
    main()