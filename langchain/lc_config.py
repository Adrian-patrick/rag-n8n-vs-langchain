from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class RAGConfig:
    sitemap_url: str
    url_prefixes: list[str]
    max_urls: int
    request_timeout: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vector_dir: Path
    collection_name: str
    embedding_provider: str
    embedding_model_openai: str
    embedding_model_ollama: str
    llm_provider: str
    llm_model_openai: str
    llm_model_ollama: str
    openai_api_key: str


def _csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config() -> RAGConfig:
    load_dotenv()

    return RAGConfig(
        sitemap_url=os.getenv("SITEMAP_URL", "https://pytorch.org/sitemap.xml"),
        url_prefixes=_csv_to_list(
            os.getenv("URL_PREFIXES", "https://pytorch.org/docs/stable")
        ),
        max_urls=int(os.getenv("MAX_URLS", "60")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "5")),
        vector_dir=Path(os.getenv("VECTOR_DIR", "data/chroma")),
        collection_name=os.getenv("COLLECTION_NAME", "pytorch-docs"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").lower(),
        embedding_model_openai=os.getenv(
            "EMBEDDING_MODEL_OPENAI", "text-embedding-3-small"
        ),
        embedding_model_ollama=os.getenv(
            "EMBEDDING_MODEL_OLLAMA", "nomic-embed-text"
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
        llm_model_openai=os.getenv("LLM_MODEL_OPENAI", "gpt-4o-mini"),
        llm_model_ollama=os.getenv("LLM_MODEL_OLLAMA", "llama3.2"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    )
