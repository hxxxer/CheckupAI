"""
Build Knowledge Base Script

This script processes medical documents and creates vector embeddings
for the RAG knowledge base using BGE-M3.
"""

import os
import sys
from pathlib import Path
import PyPDF2
from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))
from config.config import config


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def create_knowledge_collection():
    """Create Milvus collection for knowledge base"""
    
    # Connect to Milvus
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    
    # BGE-M3 uses 1024-dimensional embeddings
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000)
    ]
    
    schema = CollectionSchema(fields=fields, description="Medical knowledge base")
    collection = Collection(name="medical_knowledge", schema=schema)
    
    # Create index
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection


def process_documents(raw_dir, processed_dir):
    """Process raw documents into text chunks"""
    
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_data = []
    
    for file_path in Path(raw_dir).glob('**/*.pdf'):
        print(f"Processing {file_path}...")
        
        # Extract text
        text = extract_text_from_pdf(file_path)
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Store processed chunks
        for i, chunk in enumerate(chunks):
            processed_data.append({
                'text': chunk,
                'source': str(file_path.name),
                'chunk_id': i
            })
    
    # Save processed data
    output_file = Path(processed_dir) / 'processed_chunks.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(processed_data)} chunks from {len(list(Path(raw_dir).glob('**/*.pdf')))} documents")
    return processed_data


def build_vector_database(processed_data):
    """Build vector database from processed data using BGE-M3"""
    
    # Initialize BGE-M3 encoder
    encoder = BGEM3FlagModel(config.EMBEDDING_MODEL, use_fp16=True)
    
    # Create collection
    collection = create_knowledge_collection()
    
    # Prepare data for insertion
    texts = [item['text'] for item in processed_data]
    sources = [item['source'] for item in processed_data]
    metadata = [json.dumps({'chunk_id': item['chunk_id']}) for item in processed_data]
    
    # Generate embeddings with BGE-M3
    print("Generating embeddings with BGE-M3...")
    embeddings_output = encoder.encode(texts, batch_size=12, max_length=8192)
    # Use dense vectors for Milvus
    embeddings = embeddings_output['dense_vecs']
    
    # Insert into Milvus
    print("Inserting into Milvus...")
    entities = [
        embeddings.tolist(),
        texts,
        sources,
        metadata
    ]
    
    collection.insert(entities)
    collection.flush()
    
    print(f"Successfully inserted {len(texts)} documents into knowledge base")


def main():
    """Main execution"""
    
    # Paths
    raw_dir = Path(__file__).parent.parent / 'data' / 'knowledge_base' / 'raw'
    processed_dir = Path(__file__).parent.parent / 'data' / 'knowledge_base' / 'processed'
    
    print("Starting knowledge base construction...")
    
    # Check if raw directory has files
    pdf_files = list(raw_dir.glob('**/*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        print("Please add medical documents to the raw/ directory first.")
        return
    
    # Process documents
    processed_data = process_documents(raw_dir, processed_dir)
    
    # Build vector database
    build_vector_database(processed_data)
    
    print("Knowledge base construction completed!")


if __name__ == '__main__':
    main()
