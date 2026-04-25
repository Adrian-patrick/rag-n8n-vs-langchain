from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_config import load_config
from pipeline import answer_question


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lc = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lc)
    return hits / len(expected_keywords)


def run_evaluation(input_file: Path, output_file: Path) -> None:
    config = load_config()
    items = json.loads(input_file.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    total_recall = 0.0
    total_latency = 0.0

    for item in items:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        result = answer_question(config, question)
        recall = keyword_recall(result["answer"], expected_keywords)

        total_recall += recall
        total_latency += float(result["latency_s"])

        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "latency_s": result["latency_s"],
                "keyword_recall": round(recall, 3),
            }
        )

    count = len(rows) if rows else 1
    report = {
        "summary": {
            "count": len(rows),
            "avg_keyword_recall": round(total_recall / count, 3),
            "avg_latency_s": round(total_latency / count, 3),
        },
        "results": rows,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluation complete. Saved report to: {output_file}")
    print(json.dumps(report["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic RAG evaluation")
    parser.add_argument(
        "--input",
        default="evaluation/qa_pairs.json",
        help="Path to evaluation question set",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        help="Path to write evaluation report",
    )
    args = parser.parse_args()

    run_evaluation(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
