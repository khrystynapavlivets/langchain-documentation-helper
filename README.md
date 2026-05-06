# 📘 LangChain Documentation Helper

A high-performance RAG (Retrieval-Augmented Generation) application designed to crawl, index, and query complex documentation using advanced AI agents. Currently optimized for **LangChain** documentation.

Built with modern Python tooling and state-of-the-art LLMs, this project provides a seamless chat interface for developers to get accurate answers with direct source citations.

---

## 🚀 Overview

The system operates in two main phases:
1. **Ingestion Pipeline**: Crawls documentation using Tavily, splits text into chunks, generates embeddings with HuggingFace, and stores them in a Pinecone vector database.
2. **RAG Chat Interface**: A Streamlit-based UI where an AI agent (powered by Groq) retrieves relevant context from the vector store and generates answers with source attribution.

---

## ✨ Key Features

- **🔍 Intelligent Retrieval**: Semantic search powered by Pinecone and BGE-large embeddings.
- **🚀 Hardware Accelerated**: Automatic support for **CUDA** (Nvidia), **MPS** (Apple Silicon), or CPU for embedding generation.
- **🤖 Agentic RAG**: Uses LangChain agents to intelligently retrieve context before answering.
- **⚡ Ultra-Fast LLM**: Powered by **Llama 3.3-70b** via **Groq** for near-instant responses.
- **📚 Source Citing**: Every answer includes expandable sections with direct links to the source documentation.
- **🧹 Session Management**: Built-in chat history management with the ability to clear sessions.
- **📦 Modern Tooling**: Managed by `uv` for lightning-fast dependency management and reproducible environments.

---

## 🛠️ Tech Stack

- **Framework**: [LangChain](https://www.langchain.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Provider**: [Groq](https://groq.com/) (Llama 3.3 70B)
- **Embeddings**: [HuggingFace](https://huggingface.co/) (`BAAI/bge-large-en-v1.5`)
- **Vector DB**: [Pinecone](https://www.pinecone.io/)
- **Search/Crawl**: [Tavily AI](https://tavily.com/)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

---

## 📂 Project Structure

```text
├── backend/
│   ├── core.py         # RAG logic, Agent setup & LLM integration
│   └── __init__.py
├── main.py             # Streamlit UI & Chat Interface
├── ingestion.py        # Async data ingestion & vector indexing pipeline
├── consts.py           # Global constants & configurations
├── logger.py           # Custom logging utility
├── pyproject.toml      # Project metadata & dependencies
└── .env                # Environment variables (local only)
```

---

## 🏁 Getting Started

### Prerequisites

- **Python 3.14+** (as specified in `pyproject.toml`)
- **[uv](https://github.com/astral-sh/uv)** installed
- **Pinecone Index**: Create an index named `langchain-docs-helper-index` with **1024 dimensions** (for BGE-large).
- API Keys for:
  - **Groq** (for LLM)
  - **Pinecone** (for Vector DB)
  - **Tavily** (for crawling)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd documentation-helper
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key
```

### 3. Install Dependencies

```bash
uv sync
```

---

## 📖 Usage

### Phase 1: Ingest Data
Before chatting, you need to populate the vector store:

```bash
uv run ingestion.py
```
*This will crawl the LangChain documentation, split it into chunks (size: 2000, overlap: 200), generate embeddings, and upload them to Pinecone.*

### Phase 2: Run Chat UI
Launch the application:

```bash
uv run streamlit run main.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Development

- **Formatting**: Uses `black` and `isort`.
- **Logging**: Detailed color-coded logs are provided during ingestion for debugging.
- **Hardware**: Automatically detects and uses `CUDA`, `MPS` (Mac M-series), or `CPU` for embedding generation.

---

## 📜 License

MIT License - feel free to use and modify for your own projects.

