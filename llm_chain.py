"""
llm_chain.py
Handles the generation step of the RAG pipeline:
takes retrieved context + user query, sends to the LLM, returns an answer.
"""

import os
from langchain_openai import ChatOpenAI   
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

from prompt_templates import qa_prompt

load_dotenv()


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    """Initialize the LLM."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_qa_chain(vectorstore, model_name: str = "gpt-4o-mini", k: int = 4):
    """
    Build a RetrievalQA chain using the given FAISS vectorstore as retriever.
    """
    llm = get_llm(model_name=model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

    return qa_chain


def ask_question(qa_chain, query: str):
    """
    Run a query through the QA chain and return answer + sources.
    """
    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]

    return answer, sources


if __name__ == "__main__":
    from retrieval.vectorstore import load_vectorstore

    vs = load_vectorstore()
    chain = build_qa_chain(vs)

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        ans, srcs = ask_question(chain, q)
        print(f"\nAnswer: {ans}")
        print(f"Sources: {srcs}")