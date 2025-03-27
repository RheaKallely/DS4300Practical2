# DS 43000 Retrieval-Augmented Generation system

# Retrieval-Augmented Generation (RAG) System Setup Guide

## Table of Contents
1. [Overview](#overview)
2. [Required Downloads & Installations](#required-downloads--installations)
   - [Install Python](#1-install-python-if-not-already-installed)
   - [Install Required Python Libraries](#2-install-required-python-libraries)
   - [Install Ollama for Local LLMs](#3-install-ollama-for-local-llms)
   - [Set Up a Vector Database with Docker](#4-set-up-a-vector-database-with-docker)
   - [Download Embedding Models](#5-download-embedding-models)
   - [Clone the Project Repository](#6-clone-the-project-repository)
3. [Running the System](#7-run-the-system)

## Overview

The Retrieval-Augmented Generation (RAG) system enables efficient querying of a collective document repository using a locally hosted Large Language Model (LLM). This system allows users to:

- Ingest and index course notes and other documents.
- Utilize embedding models to create vectorized representations of text.
- Store and query these embeddings using a vector database.
- Generate responses using a locally running LLM.

This guide outlines the setup process for installing dependencies and running the system.

## Required Downloads & Installations

To set up the RAG system on your local machine, install the following dependencies:

### 1. Install Python (If Not Already Installed)
- Download and install Python 3.8 or higher from [python.org](https://www.python.org/)
- Verify installation:
  ```sh
  python --version
  ```

### 2. Install Required Python Libraries
- Use pip to install necessary libraries:
  ```sh
  pip install -r requirements.txt
  ```
- If `requirements.txt` is not available, install manually:
  ```sh
  pip install sentence-transformers redis chromadb milvus ollama
  ```

### 3. Install Ollama for Local LLMs
- Download and install Ollama from [ollama.ai](https://ollama.ai)
- Verify installation:
  ```sh
  ollama --version
  ```

### 4. Set Up a Vector Database with Docker
Use Docker to set up and run the vector databases:

#### Start Redis Vector DB
```sh
  docker run -d --name redis-vector -p 6379:6379 redis/redis-stack:latest
```

#### Start ChromaDB
```sh
  docker run -d --name chromadb -p 8000:8000 ghcr.io/chroma-core/chroma:latest
```

#### Start Milvus
```sh
  docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

### 5. Download Embedding Models
Ensure access to embedding models by running:
```sh
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 6. Clone the Project Repository
- If using Git:
  ```sh
  git clone <https://github.com/RheaKallely/DS4300Practical2.git>
  ```

## 7. Run the LLM 
- To retrieve the text chunks run:
  ```
  python text_preprocess.py
  ```
- To run the embedding model run: 
  ```
  python embedding.py
  ```
- To  process queries using the vector database and ask the model a question run:
  ```
  python vector_db.py 
  ```



