# Knowledge Base Directory

This directory contains authoritative medical knowledge for the RAG system.

## Directory Structure

- `raw/`: Original medical documents (PDF, Word, etc.)
- `processed/`: Preprocessed text chunks ready for vectorization
- `scripts/`: Scripts for processing and building the knowledge base

## Adding Knowledge

1. Place original medical documents in the `raw/` directory
2. Run processing scripts to extract and chunk the content
3. Use the build script to create vector embeddings and store in Milvus

## Knowledge Sources

Recommended sources for medical knowledge:
- Medical textbooks
- Clinical guidelines
- Research papers
- Health authority publications
- Medical encyclopedias
