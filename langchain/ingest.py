from __future__ import annotations

import argparse
import shutil

from lc_config import load_config
from pipeline import ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sitemap content into local vector DB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector store before ingestion",
    )
    args = parser.parse_args()

    config = load_config()

    if args.reset and config.vector_dir.exists():
        shutil.rmtree(config.vector_dir)

    stats = ingest_documents(config)
    print("Ingestion complete")
    print(f"URLs selected: {stats['urls']}")
    print(f"Pages loaded: {stats['pages']}")
    print(f"Chunks stored: {stats['chunks']}")


if __name__ == "__main__":
    main()
