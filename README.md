# PDF RAG Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions over PDF documents and get accurate, context-grounded answers. Built as a portfolio project to demonstrate practical LLM application development.

## Project Description

This project implements an end-to-end RAG system that:
- Ingests PDF documents and splits them into semantically meaningful chunks
- Converts chunks into vector embeddings using OpenAI's embedding models
- Stores and indexes embeddings in a FAISS vector store for fast similarity search
- Retrieves the most relevant chunks for a given query
- Passes retrieved context to an OpenAI LLM (via LangChain) to generate grounded, accurate answers

Unlike a plain LLM call, this approach reduces hallucination by grounding answers in the actual document content, making it suitable for Q&A over domain-specific or private PDF documents.

## Architecture

```
                ┌─────────────────┐
                │   PDF Document   │
                └────────┬─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Document Loader    │  (PyPDFLoader / LangChain)
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Text Splitter      │  (RecursiveCharacterTextSplitter)
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  OpenAI Embeddings   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   FAISS Vector Store │  (index + retriever)
              └──────────┬───────────┘
                         │
             User Query  │
                 │        │
                 ▼        ▼
              ┌─────────────────────┐
              │  Retriever (top-k)    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  LangChain RetrievalQA│
              │  Chain + OpenAI LLM   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     Final Answer      │
              └─────────────────────┘
```

**Flow summary:** PDF → chunk → embed → store in FAISS → retrieve relevant chunks on query → augment prompt with context → LLM generates answer.

## Tech Stack

| Component        | Tool/Library         |
|-------------------|-----------------------|
| LLM               | OpenAI (GPT models)   |
| Embeddings        | OpenAI Embeddings     |
| Vector Store      | FAISS                 |
| Orchestration     | LangChain             |
| Document Format   | PDF                   |
| Interface         | Jupyter Notebook / CLI |

## Setup Guide

### Prerequisites
- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Sonisah-013/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables

   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

## How to Run

1. Place the PDF document(s) you want to query inside the `data/` folder (or update the file path in the notebook/script).

2. Launch the notebook:
   ```bash
   jupyter notebook
   ```
   Then open the main `.ipynb` file and run the cells in order:
   - Load and split the PDF
   - Generate embeddings and build the FAISS index
   - Run queries against the index

   **OR**, if using the CLI script version:
   ```bash
   python main.py --pdf data/your_file.pdf --query "Your question here"
   ```

3. Ask questions interactively — the system will retrieve relevant chunks from the PDF and return an LLM-generated answer grounded in that content.

## Example

```
Query: "What are the key findings in section 3?"

Answer: Based on the retrieved context, the key findings include...
```

## Future Improvements

- [ ] Add a Streamlit UI for interactive querying
- [ ] Support multiple PDF uploads and multi-document retrieval
- [ ] Add source citation (which page/chunk the answer came from)
- [ ] Experiment with open-source embeddings/LLMs (HuggingFace) as a free alternative
- [ ] Deploy as a hosted demo (Streamlit Cloud / HuggingFace Spaces)

