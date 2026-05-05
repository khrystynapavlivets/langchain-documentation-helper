from typing import Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-helper-index", embedding=embeddings
)

# Initialize chat model
model = ChatGroq(
    # model="llama-3.1-8b-instant",
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=512,
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 3 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=3)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "1. You are a professional technical assistant specializing in LangChain documentation."
        "2. You have access to a search tool (retrieval tool) and MUST use it before every response to obtain up-to-date data."
        "3. Your response must be based exclusively on the provided context; do not use external knowledge that might be outdated."
        "4. If the context does not contain the answer, clearly state: 'Unfortunately, the provided documentation does not contain enough information to answer.'"
        "5. When providing code examples, ensure they use current LangChain syntax and patterns found within the retrieved documents."
        "6. DO NOT include any URLs, source lines, or 'Source:' mentions within the main body of your answer."
        "7. Provide all references ONLY in a dedicated '### Sources' section at the very end of your response."
        "8. Use Markdown format for the sources list: * Page Name."
        "9. IT IS STRICTLY FORBIDDEN to hallucinate or manually construct URLs; use only the exact links provided by the search tool."
        "10. DO NOT include disclaimers, notes about being an AI, or warnings about the information being retrieved from documentation."
        "11. Maintain a professional, developer-oriented tone, being concise and avoiding all unnecessary filler or redundant text."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")
    for doc in result["context"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
