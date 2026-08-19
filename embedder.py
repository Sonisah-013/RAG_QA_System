"""
ingestion/embedder.py
Handles creation of the embedding model used to convert text chunks into vectors.
"""

from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Initialize and return the OpenAI embeddings model.
    Used both when building the FAISS index and when querying it.
    """
    return OpenAIEmbeddings(
        model=model_name,
        api_key=OPENAI_API_KEY,
    )


if __name__ == "__main__":
    # Quick manual test — embed a sample string and check the vector shape
    embeddings = get_embeddings()
    sample_text = "This is a test sentence for embedding."
    vector = embeddings.embed_query(sample_text)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")