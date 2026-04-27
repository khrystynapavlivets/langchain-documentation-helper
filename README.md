
# 🦜 LangChain Documentation Helper



## 🎯 Overview

The **LangChain Documentation Helper** is a sophisticated AI-powered web application that serves as a slim version of [chat.langchain.com](https://chat.langchain.com/). This intelligent documentation assistant provides accurate answers to questions about LangChain documentation using advanced Retrieval-Augmented Generation (RAG) techniques, enhanced with web crawling capabilities and conversational memory.

### ✨ Key Features

**RAG Pipeline Flow:**

1. 🌐 **Web Crawling**: Real-time web scraping and content extraction using Tavily's advanced crawling capabilities
2. 📚 **Document Processing**: Intelligent chunking and preprocessing of LangChain documentation
3. 🔍 **Vector Storage**: Advanced embedding and indexing using Pinecone for fast similarity search
4. 🎯 **Intelligent Retrieval**: Context-aware document retrieval based on user queries
5. 🧩 **Memory System**: Conversational memory for coreference resolution and context continuity
6. 🧠 **Context-Aware Generation**: Provides accurate, contextual answers with source citations
7. 💬 **Interactive Interface**: User-friendly chat interface powered by Streamlit
8. 🚀 **Real-time Processing**: Fast end-to-end pipeline from query to response


## 🛠️ Tech Stack

<div align="center">

| Component | Technology | Description |
|-----------|------------|-------------|
| 🖥️ **Frontend** | Streamlit | Interactive web interface |
| 🧠 **AI Framework** | LangChain 🦜🔗 | Orchestrates the AI pipeline |
| 🔍 **Vector Database** | Pinecone 🌲 | Stores and retrieves document embeddings |
| 🌐 **Web Crawling** | Tavily | Intelligent web scraping and content extraction |
| 🧩 **Memory** | Conversational Memory | Coreference resolution and context continuity |
| 🤖 **LLM** | OpenAI GPT | Powers the conversational AI |
| 🐍 **Backend** | Python | Core application logic |

</div>

## 📁 Project Structure

```
documentation-helper/
├── backend/                          # Core backend logic
│   ├── __init__.py
│   └── core.py
├── static/                           # Static assets (images, logos)
│   ├── banner.gif
│   ├── LangChain Logo.png
│   ├── Tavily Logo.png
│   ├── Tavily Logo Trimmed Padded.png
│   └── Trimmed Padded Langchain.png
├── chroma_db/                        # Local vector database
├── main.py                           # Streamlit application entry point
├── ingestion.py                      # Document ingestion pipeline
├── consts.py                         # Configuration constants
├── logger.py                         # Logging utilities
├── Tavily Demo Tutorial.ipynb        # 📚 Tutorial: Introduction to Tavily API
├── Tavily Crawl Demo Tutorial.ipynb  # 📚 Tutorial: Advanced Tavily crawling techniques
└── requirements files                # Pipfile, Pipfile.lock
```


