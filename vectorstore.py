"""
vectorstore.py
Handles building and loading the FAISS vector store for the RAG pipeline:
- loads/splits PDF documents
- embeds them using OpenAI embeddings
- saves/loads the FAISS index to/from disk
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")          # folder with your PDFs
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_embeddings(model_name: str = "text-embedding-3-small"):
    """Initialize the embedding model."""
    return OpenAIEmbeddings(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def load_documents(data_dir: str = DATA_DIR):
    """Load all PDF documents from the data directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {data_dir}")
    return documents


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_vectorstore(data_dir: str = DATA_DIR, index_path: str = FAISS_INDEX_PATH):
    """
    Build a FAISS index from documents in data_dir and save it to index_path.
    Run this once (or whenever your source documents change).
    """
    documents = load_documents(data_dir)
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")

    return vectorstore


def load_vectorstore(index_path: str = FAISS_INDEX_PATH):
    """
    Load a previously built FAISS index from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            f"Run build_vectorstore() first to create it."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


if __name__ == "__main__":
    # Run this file directly to (re)build the index from your PDFs
    build_vectorstore()