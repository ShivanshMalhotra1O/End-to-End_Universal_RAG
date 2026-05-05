# Universal RAG Agent

A production-grade Retrieval Augmented Generation (RAG) agent built with LlamaIndex, LangGraph, and Google Gemini / OpenAI. Swap the document, provider, and prompt via a single config file — no code changes needed.

---


## 🧠 What Is This?

A **Universal RAG Agent** that:
- Takes **any document** (PDF) as a knowledge source
- Answers questions **grounded in that document only**
- Remembers **conversation history** across turns
- Rewrites vague follow-up questions for accurate retrieval
- Caches repeated questions to save API tokens
- Supports **multiple LLM providers** (Gemini, OpenAI) via config

---

## 🏗️ Architecture

```
User Question
      ↓
[Rewrite Node]     → rewrites vague questions using chat history
      ↓
[Retrieve Node]    → LlamaIndex searches vector store for relevant chunks
      ↓
[Generate Node]    → LLM answers using chunks + conversation history
      ↓
Answer
```

Monitoring via **LangSmith** — every run is traced with token usage, latency, and retrieved chunks.

---

## 📁 Project Structure

```
universal-rag/
│
├── config/
│   └── rag_config.yaml          ← swap document, provider, prompt here
│
├── data/
│   └── Medical_book_small.pdf   ← your document
│
├── src/
│   ├── __init__.py
│   ├── state.py                 ← LangGraph shared state
│   ├── injestion.py             ← load, chunk, embed, persist index
│   ├── retriever.py             ← semantic search on vector store
│   ├── providers.py             ← factory: Gemini / OpenAI switcher
│   └── graph.py                 ← LangGraph agent with 3 nodes
│
├── app.py                       ← Streamlit UI
├── main.py                      ← terminal version
├── requirements.txt
├── packages.txt
└── .env                         ← API keys (never commit this)
```

---

## ⚙️ Configuration

Everything is controlled via `config/rag_config.yaml`:

```yaml
provider: "gemini"               # switch to "openai" anytime

document:
  path: "data/Medical_book_small.pdf"
  chunk_size: 512
  chunk_overlap: 50

llm:
  model: "gemini-2.0-flash"
  temperature: 0.2

embeddings:
  model: "gemini-embedding-001"

retriever:
  top_k: 4

storage:
  path: "storage/medical_index"  # cached embeddings, built once

prompt:
  system: |
    You are a medical information assistant...
```

**To use a different document:**
1. Drop your PDF into `data/`
2. Update `document.path` in config
3. Delete `storage/` folder to force re-embedding
4. Update `prompt.system` for your new domain

**To switch providers:**
1. Change `provider: "openai"`
2. Update `llm.model` and `embeddings.model`
3. Add `OPENAI_API_KEY` to `.env`

---

## 🛠️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/universal-rag.git
cd universal-rag
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the root:
```
GOOGLE_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key"
LANGSMITH_API_KEY="your-langsmith-key"
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-project
CONFIG_PATH=config/rag_config.yaml
```

Get your keys from:
- Gemini → [aistudio.google.com](https://aistudio.google.com)
- OpenAI → [platform.openai.com](https://platform.openai.com)
- LangSmith → [smith.langchain.com](https://smith.langchain.com)

### 5. Run the app

**Streamlit UI:**
```bash
streamlit run app.py
```

**Terminal version:**
```bash
python main.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| **LlamaIndex** | Document loading, chunking, embedding, vector store |
| **LangGraph** | Agent workflow orchestration |
| **LangSmith** | Monitoring, tracing, token usage |
| **Streamlit** | Web UI |
| **Google Gemini** | LLM + Embeddings (default) |
| **OpenAI** | LLM + Embeddings (alternative) |

---

## ✨ Features

- 🔄 **Universal** — swap document, provider, prompt via config only
- 💾 **Persistent Index** — embeddings cached to disk, never re-embedded
- 🧠 **Chat Memory** — remembers full conversation history
- ✏️ **Question Rewriting** — rewrites vague follow-ups before retrieval
- ⚡ **History Cache** — repeated questions answered instantly
- 🛡️ **Guardrails** — strict scope and rules in system prompt
- 📊 **Monitored** — full LangSmith tracing per run

---

## 📊 Monitoring

Every run is traced in LangSmith:

```
Run
├── rewrite_node   → input question → rewritten question
├── retrieve_node  → query → top-k chunks
└── generate_node  → prompt → answer + token usage
```

View traces at [smith.langchain.com](https://smith.langchain.com) → Projects → `rag-project`

---

## 🤝 Contributing

This is a learning project. Feel free to fork and extend it.

---

## 📄 License

MIT
