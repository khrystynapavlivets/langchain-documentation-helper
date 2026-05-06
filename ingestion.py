import asyncio
import ssl
import os
import certifi
import torch
from dotenv import load_dotenv


from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_info, log_warning, log_error, log_header, log_success
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-helper-index", embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: list[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: list[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/docs/introduction/",
            "max_depth": 3,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        url = tavily_crawl_result_item["url"]

        # Filter out invalid, template, or outdated URLs, and non-technical pages
        invalid_patterns = [
            "${",
            "CHAT_APP_URL",
            "/oss/",
            "docs.langchain.com",
            "/legal/",
            "/privacy",
            "/careers",
            "/forum",
            "/academy",
            "/changelog",
            "/support",
            "status.smith",
            "trust.langchain",
            "blog.langchain",
            "interrupt.langchain",
        ]
        if any(pattern in url for pattern in invalid_patterns):
            log_warning(f"TavilyCrawl: Skipping non-technical or outdated URL: {url}")
            continue

        log_info(f"TavilyCrawl: Successfully crawled {url} from documentation site")
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": url},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 2000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
