"""
retrieval package
Handles the retrieval side of the RAG pipeline (vector store + retriever).
"""

from .vectorstore import build_vectorstore, load_vectorstore
from .retriever import get_retriever, retrieve_docs

__all__ = [
    "build_vectorstore",
    "load_vectorstore",
    "get_retriever",
    "retrieve_docs",
]