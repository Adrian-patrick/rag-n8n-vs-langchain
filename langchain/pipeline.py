from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lc_config import RAGConfig


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant for PyTorch documentation Q&A.
Answer only from the context. If the answer is not in context, say you do not know.

Context:
{context}

Question:
{question}

Answer in a concise and practical way.
"""
)


def _require_openai_key(config: RAGConfig) -> None:
    if not config.openai_api_key or "YOUR_OPENAI_API_KEY" in config.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Use a real key or switch to OLLAMA providers."
        )


def get_embeddings(config: RAGConfig):
    if config.embedding_provider == "openai":
        _require_openai_key(config)
        return OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=config.embedding_model_openai,
        )

    if config.embedding_provider == "ollama":
        return OllamaEmbeddings(model=config.embedding_model_ollama)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {config.embedding_provider}. Use 'openai' or 'ollama'."
    )


def get_llm(config: RAGConfig):
    if config.llm_provider == "openai":
        _require_openai_key(config)
        return ChatOpenAI(api_key=config.openai_api_key, model=config.llm_model_openai)

    if config.llm_provider == "ollama":
        return ChatOllama(model=config.llm_model_ollama)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {config.llm_provider}. Use 'openai' or 'ollama'."
    )


def fetch_sitemap_urls(
    sitemap_url: str,
    max_urls: int,
    url_prefixes: list[str],
    timeout: int,
) -> list[str]:
    response = requests.get(sitemap_url, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    urls: list[str] = []

    for loc in root.findall(".//{*}loc"):
        if not loc.text:
            continue
        url = loc.text.strip()
        if url_prefixes and not any(url.startswith(prefix) for prefix in url_prefixes):
            continue
        urls.append(url)
        if len(urls) >= max_urls:
            break

    return urls


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_web_documents(urls: list[str], timeout: int) -> list[Document]:
    documents: list[Document] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            text = _html_to_text(response.text)
            if not text:
                continue
            documents.append(Document(page_content=text, metadata={"source": url}))
        except requests.RequestException:
            # Skip transient URL failures so ingestion can continue.
            continue

    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _document_id(document: Document, idx: int) -> str:
    raw = f"{document.metadata.get('source', 'unknown')}:{idx}:{document.page_content[:180]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_vector_store(config: RAGConfig, create: bool = True) -> Chroma:
    if create:
        config.vector_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(config)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.vector_dir),
    )


def ingest_documents(config: RAGConfig) -> dict[str, int]:
    urls = fetch_sitemap_urls(
        sitemap_url=config.sitemap_url,
        max_urls=config.max_urls,
        url_prefixes=config.url_prefixes,
        timeout=config.request_timeout,
    )
    pages = load_web_documents(urls=urls, timeout=config.request_timeout)
    chunks = chunk_documents(
        documents=pages,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    vector_store = get_vector_store(config)
    ids = [_document_id(doc, idx) for idx, doc in enumerate(chunks)]

    if chunks:
        vector_store.add_documents(chunks, ids=ids)

    return {
        "urls": len(urls),
        "pages": len(pages),
        "chunks": len(chunks),
    }


def answer_question(config: RAGConfig, question: str) -> dict[str, object]:
    vector_store = get_vector_store(config, create=False)
    retriever = vector_store.as_retriever(search_kwargs={"k": config.top_k})

    started = time.perf_counter()
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = RAG_PROMPT.format_messages(context=context, question=question)
    llm = get_llm(config)
    llm_response = llm.invoke(prompt)
    latency = time.perf_counter() - started

    answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
    sources = [doc.metadata.get("source", "") for doc in docs]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_s": round(latency, 3),
    }
