"""
prompt_templates.py
Central place for all prompt templates used in the RAG QA system.
Keeping prompts here makes them easy to iterate on without touching pipeline logic.
"""

from langchain_core.prompts import PromptTemplate


# --- Main QA prompt (used in llm_chain.py) ---
QA_PROMPT_TEMPLATE = """
You are a helpful assistant answering questions based only on the provided context.
Use ONLY the information in the context below to answer the question.
If the answer is not contained in the context, say "I don't know based on the provided documents."
Do not make up information that isn't in the context.

Context:
{context}

Question:
{question}

Answer:
"""

qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# --- Stricter variant: forces citations to source ---
QA_PROMPT_WITH_SOURCES_TEMPLATE = """
You are a helpful assistant answering questions based only on the provided context.
Answer the question using ONLY the context below. After your answer, briefly note
which part of the context supports it.
If the answer is not contained in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer (with brief source note):
"""

qa_prompt_with_sources = PromptTemplate(
    template=QA_PROMPT_WITH_SOURCES_TEMPLATE,
    input_variables=["context", "question"],
)


# --- Summarization prompt (optional, e.g. for summarizing retrieved chunks) ---
SUMMARY_PROMPT_TEMPLATE = """
Summarize the following text in 2-3 concise sentences, preserving key facts:

{text}

Summary:
"""

summary_prompt = PromptTemplate(
    template=SUMMARY_PROMPT_TEMPLATE,
    input_variables=["text"],
)


# --- Query rewriting prompt (optional, improves retrieval for vague questions) ---
QUERY_REWRITE_TEMPLATE = """
Rewrite the following user question to be more specific and searchable,
without changing its original intent. Return ONLY the rewritten question.

Original question: {question}

Rewritten question:
"""

query_rewrite_prompt = PromptTemplate(
    template=QUERY_REWRITE_TEMPLATE,
    input_variables=["question"],
)