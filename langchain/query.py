from __future__ import annotations

import argparse

from lc_config import load_config
from pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LangChain RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    args = parser.parse_args()

    config = load_config()

    if args.question:
        result = answer_question(config, args.question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source}")
        print(f"Latency: {result['latency_s']}s")
        return

    print("Interactive mode. Press Ctrl+C to exit.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        result = answer_question(config, question)
        print(f"Answer: {result['answer']}")
        print(f"Latency: {result['latency_s']}s")
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    main()
