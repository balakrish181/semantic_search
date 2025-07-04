# Academic Paper RAG System

A Retrieval-Augmented Generation (RAG) system for academic papers that enables semantic search and question answering using OpenAI embeddings and language models.

## Overview

This system allows you to search for academic papers on Google Scholar, download and process them, and then query them using natural language. The system uses OpenAI embeddings for semantic search and OpenAI's GPT models for answering questions based on the retrieved papers.

## Features

- Search for academic papers on Google Scholar using SerpAPI
- Download PDFs and convert them to Markdown using Docling
- Generate embeddings for papers using OpenAI's text-embedding-3-small model
- Store embeddings in a FAISS vector database for efficient retrieval
- Answer queries using RAG with OpenAI's GPT-4o-mini model

## Requirements

### API Keys

The following API keys are required and should be set in a `.env` file:

- `OPENAI_API_KEY`: Required for embedding generation and RAG
- `SERPAPI_API_KEY`: Required for Google Scholar search

### Environment Variables

Additional environment variables that can be set in the `.env` file:

- `PDF_STORAGE_PATH`: Directory to store downloaded PDFs (default: "./pdfs")
- `MARKDOWN_STORAGE_PATH`: Directory to store converted Markdown files (default: "./papers")
- `CHROMA_PERSIST_DIRECTORY`: Directory to store vector database (default: "./vector_db")

### Python Dependencies

```
pip install requests python-dotenv serpapi docling llama-index openai faiss-cpu
```

## Usage

### Command Line Interface

The system provides a command-line interface through `main.py`:

#### Ingesting Papers

```bash
python main.py ingest "quantum computing algorithms" --num-results 5
```

This command will:
1. Search for "quantum computing algorithms" on Google Scholar
2. Download the top 5 papers
3. Convert them to Markdown
4. Generate embeddings using OpenAI
5. Store the embeddings in the vector database

#### Querying Papers

```bash
python main.py query "What are the latest advancements in quantum computing algorithms?"
```

This command will:
1. Convert your query to an embedding using OpenAI
2. Find the most relevant papers in the vector database
3. Use RAG with GPT-4o-mini to generate an answer based on the retrieved papers

## Architecture

The system consists of the following components:

- `scholar_search.py`: Google Scholar search via SerpAPI
- `text_extraction.py`: PDF downloading and conversion to Markdown
- `embedding.py`: OpenAI embedding generation
- `vector_store.py`: FAISS vector database integration
- `rag_engine.py`: LlamaIndex + GPT-4o-mini for answering queries
- `main.py`: Main entry point with CLI

## Notes

- The system uses an in-memory FAISS vector store to avoid compatibility issues with ChromaDB Rust bindings on Windows.
- For persistent storage across runs, the system saves the FAISS index to disk.
- OpenAI embeddings provide high-quality semantic search capabilities.
