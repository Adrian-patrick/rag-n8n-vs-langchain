# RAG: n8n vs LangChain

This project builds and compares a Retrieval-Augmented Generation (RAG) pipeline using:

- n8n (workflow automation)
- LangChain (Python framework)

Both implementations are kept similar to enable fair evaluation.

---

## 🎯 Goal

- Ingest PyTorch documentation
- Generate embeddings
- Store in vector database
- Enable semantic search + generation
- Compare performance

---

## 📁 Structure
```
rag-n8n-vs-langchain/
├── n8n/          # n8n workflow exports
├── langchain/    # LangChain implementation
├── data/         # Source & processed data
├── evaluation/   # Metrics & comparison
└── README.md
```

## 🚀 Current Status

### n8n
✅ Working RAG pipeline  
- Sitemap crawling  
- HTML extraction  
- Chunking  
- Embeddings  
- Retrieval + generation  

Currently supports PyTorch blogs and partial docs.

### LangChain
🚧 In progress

---

## ⚙️ Stack

- Embeddings: Local / API
- Vector DB: Supabase / Local
- LLM: Local / API
- Orchestration: n8n / LangChain

---

## 📌 Roadmap

- [x] Single-site RAG
- [x] Sitemap parsing
- [x] Chunking
- [ ] Full PyTorch docs
- [ ] LangChain pipeline
- [ ] Evaluation metrics

---

## 👤 Author

Adrian Patrick
