"""
ingestion/chunker.py
Splits loaded documents into smaller chunks suitable for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Split a list of LangChain Document objects into smaller chunks.

    Args:
        documents: list of Document objects (e.g. from a PDF loader)
        chunk_size: max characters per chunk
        chunk_overlap: overlap between consecutive chunks, to preserve context across splits

    Returns:
        list of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph breaks first, falls back to finer splits
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")

    return chunks


if __name__ == "__main__":
    # Quick manual test using the loader (adjust import if your loader file differs)
    from ingestion.loader import load_documents

    docs = load_documents()
    chunks = split_documents(docs)

    print(f"\nSample chunk:\n{'-' * 40}")
    print(chunks[0].page_content[:300])
    print(f"{'-' * 40}\nMetadata: {chunks[0].metadata}")