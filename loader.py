"""
ingestion/loader.py
Loads raw documents (PDFs) from the data directory for the RAG pipeline.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from config import DATA_DIR


def load_documents(data_dir: str = DATA_DIR):
    """
    Load all PDF documents from the given directory.

    Args:
        data_dir: path to the folder containing PDF files

    Returns:
        list of LangChain Document objects, one per page
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)/page(s) from '{data_dir}'")

    return documents


if __name__ == "__main__":
    # Quick manual test
    docs = load_documents()

    if docs:
        print(f"\nSample document:\n{'-' * 40}")
        print(docs[0].page_content[:300])
        print(f"{'-' * 40}\nMetadata: {docs[0].metadata}")
    else:
        print("No documents found. Check that your PDFs are in the DATA_DIR folder.")