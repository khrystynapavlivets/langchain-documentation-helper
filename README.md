# 📘 Documentation Helper: LangChain RAG Assistant

A modern, high-performance RAG (Retrieval-Augmented Generation) application designed to help developers navigate and query complex documentation—starting with **LangChain**. Built with Python, Streamlit, and advanced vector search capabilities.

---

## ✨ Features

- **🔍 Smart Retrieval**: Uses Pinecone vector database for lightning-fast semantic search.
- **🤖 AI-Powered Answers**: Leverages state-of-the-art LLMs to provide concise, context-aware answers.
- **📚 Source Attribution**: Automatically cites sources from the documentation to ensure accuracy.
- **🕸️ Intelligent Ingestion**: Automated pipeline using Tavily for crawling and extracting documentation.
- **💬 Interactive UI**: Clean and intuitive chat interface built with Streamlit.
- **🚀 Modern Tooling**: Managed by `uv` for ultra-fast dependency resolution and environment management.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Embeddings**: OpenAI (`text-embedding-3-small`) / HuggingFace (`BAAI/bge-large-en-v1.5`)
- **Web Crawling**: [Tavily AI](https://tavily.com/)
- **Environment**: [uv](https://github.com/astral-sh/uv)

---

## 🚀 Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed on your system.
- API Keys for:
  - OpenAI
  - Pinecone
  - Tavily AI

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd documentation-helper
   ```

2. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   TAVILY_API_KEY=your_tavily_key
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

---

## 📖 Usage

### 1. Ingest Documentation
To populate the vector store with the latest documentation:
```bash
uv run ingestion.py
```

### 2. Run the Application
Launch the Streamlit chat interface:
```bash
uv run streamlit run main.py
```

---

## 📂 Project Structure

- `main.py`: Entry point for the Streamlit application.
- `ingestion.py`: Pipeline for crawling and indexing documentation.
- `backend/core.py`: Core RAG logic and LLM integration.
- `consts.py`: Global constants and configurations.
- `logger.py`: Custom logging utility for better visibility.

