# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) demo application built with Python. It allows users to upload PDF documents and ask questions about them using natural language processing. The application uses Streamlit for the web interface, LangChain for document processing and RAG functionality, FAISS for vector storage, and Ollama for local LLM inference.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv for dependency management)
uv sync

# Alternative: Install with pip
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the Streamlit app (main application)
streamlit run app.py

# Run the basic main.py (just prints hello message)
python main.py
```

### Development Dependencies
The project uses `uv` for dependency management. Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Architecture

### Core Components

1. **app.py** - Main Streamlit application with RAG functionality
   - PDF upload and processing using PyMuPDF (fitz)
   - Document chunking with RecursiveCharacterTextSplitter
   - Vector embeddings using HuggingFace sentence-transformers
   - FAISS vector store for document retrieval
   - Ollama LLM integration for question answering

2. **main.py** - Simple entry point (minimal functionality)

### Key Dependencies

- **streamlit**: Web application framework
- **langchain**: Document processing and RAG chain orchestration
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **PyMuPDF**: PDF document parsing
- **ollama**: Local LLM inference (requires Ollama server running)

### Application Flow

1. User uploads PDF via Streamlit interface
2. PDF is processed and split into chunks (500 chars, 100 overlap)
3. Chunks are embedded using HuggingFace model "all-MiniLM-L6-v2"
4. Vector store is created with FAISS for retrieval
5. User asks questions which are processed through RetrievalQA chain
6. Relevant chunks are retrieved and sent to Ollama LLM for answering

## Requirements

- Python 3.13+
- Ollama server running locally with llama3.2 model available
- The application expects Ollama to be accessible at default endpoint

## File Structure

```
RagDemo/
├── app.py              # Main Streamlit RAG application
├── main.py             # Basic entry point
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Dependencies (legacy format)
├── uv.lock            # Dependency lock file
└── README.md          # (empty)
```