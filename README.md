# RAG: n8n vs LangChain

This project compares two Retrieval-Augmented Generation (RAG) implementations over PyTorch docs:

- n8n workflow: `n8n/n8n-RAG.json`
- LangChain Python pipeline: `langchain/*.py`

The LangChain implementation is now complete and includes ingestion, querying, and basic evaluation.

## Goal

- Crawl PyTorch doc pages from sitemap
- Chunk and embed documents
- Store embeddings in a vector database
- Answer questions with retrieved context
- Evaluate answer quality and latency

## Project Structure

```
rag-n8n-vs-langchain/
├── data/
│   └── chroma/                 # created after ingestion
├── evaluation/
│   ├── qa_pairs.json           # sample evaluation set
│   └── results.json            # generated after evaluation
├── langchain/
│   ├── lc_config.py            # environment-driven config
│   ├── pipeline.py             # ingestion + retrieval + generation core
│   ├── ingest.py               # CLI: build vector index
│   ├── query.py                # CLI: ask questions
│   └── evaluate.py             # CLI: run evaluation
├── n8n/
│   └── n8n-RAG.json
├── .env.example
├── requirements.txt
└── README.md
```

## LangChain Stack

- Vector DB: Chroma (local persistent)
- Embeddings: Ollama (default) or OpenAI
- LLM: Ollama (default) or OpenAI
- Data source: PyTorch sitemap + docs pages

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your env file from the template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

4. Keep `OPENAI_API_KEY` as placeholder unless using OpenAI.

## Configuration

All config is in `.env`.

Important fields:

- `EMBEDDING_PROVIDER`: `ollama` or `openai`
- `LLM_PROVIDER`: `ollama` or `openai`
- `URL_PREFIXES`: comma-separated URL filters
- `VECTOR_DIR`: local Chroma path

Default config runs fully local with Ollama models:

- Embeddings: `nomic-embed-text`
- LLM: `llama3.2`

If using OpenAI providers, replace:

- `OPENAI_API_KEY=YOUR_OPENAI_API_KEY`

## Usage

### 1) Ingest documents

```bash
python langchain/ingest.py --reset
```

What it does:

- Reads sitemap URLs
- Filters by prefixes
- Downloads pages
- Splits into chunks
- Stores in Chroma

### 2) Query the RAG system

Single question:

```bash
python langchain/query.py "What is torch.nn.Module?"
```

Interactive mode:

```bash
python langchain/query.py
```

### 3) Run evaluation

```bash
python langchain/evaluate.py --input evaluation/qa_pairs.json --output evaluation/results.json
```

Outputs summary metrics:

- `avg_keyword_recall`
- `avg_latency_s`

## n8n Workflow

Import `n8n/n8n-RAG.json` into n8n to run the workflow implementation.

## Notes

- Some sitemap URLs may fail to fetch; ingestion skips failed pages and continues.
- For fair comparison, use similar embedding/LLM settings across n8n and LangChain.
