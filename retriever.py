"""
retriever.py
Handles the retrieval step of the RAG pipeline:
loads the FAISS vectorstore and returns a retriever object
that llm_chain.py can plug into the QA chain.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Path where the FAISS index is saved (adjust to match your project structure)
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def get_embeddings(model_name: str = "text-embedding-3-small"):
    """Initialize the embedding model used to build/query the FAISS index."""
    return OpenAIEmbeddings(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def load_vectorstore(index_path: str = FAISS_INDEX_PATH):
    """
    Load a previously saved FAISS index from disk.
    """
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # required for local FAISS pickle loading
    )
    return vectorstore


def get_retriever(vectorstore=None, k: int = 4):
    """
    Return a retriever object from the vectorstore.
    If no vectorstore is passed, loads it from disk.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_docs(query: str, k: int = 4):
    """
    Convenience function: run a query directly and return matching documents.
    Useful for quick testing without building the full QA chain.
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


if __name__ == "__main__":
    # Quick manual test
    test_query = input("Enter a test query: ")
    results = retrieve_docs(test_query)

    print(f"\nTop {len(results)} results:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:300], "...")
        print(f"Source: {doc.metadata.get('source', 'unknown')}\n")