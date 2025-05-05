# UrbanClerk - NYC Payroll Assistant

Provides a command-line interface for querying New York City employee payroll data using Retrieval Augmented Generation (RAG). It leverages Ollama for language model inference and Pinecone as a vector database to store and retrieve relevant context. It also includes a python script to ingest the raw CSV data into Pinecone.

## Overview

The script allows users to ask questions about NYC employees, drawing information from a Pinecone index containing payroll records. It uses a system prompt to ensure the assistant only answers based on provided context and avoids making up information. The chat history is logged to a file for review. This project includes two main components:

1. **Interactive Chat Interface:** Allows users to query the data via command line.
2. **Data Ingestion Script:** Loads CSV payroll data, splits it into chunks, generates embeddings using Ollama, and upserts the data into Pinecone.

## Prerequisites

- **Python 3.8+**
- **Ollama:** [https://ollama.com/](https://ollama.com/) - Must be installed and running with the `nomic-embed-text` model downloaded.
- **Pinecone:** [https://www.pinecone.io/](https://www.pinecone.io/) - An account is required.
