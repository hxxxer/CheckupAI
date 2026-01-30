"""
User Profile Sync Script

This script syncs user profiles to Milvus vector database using BGE-M3
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))
from config.config import config


def create_profile_collection():
    """Create Milvus collection for user profiles"""
    
    # Connect to Milvus
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    
    # BGE-M3 uses 1024-dimensional embeddings
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=100)
    ]
    
    schema = CollectionSchema(fields=fields, description="User health profiles")
    collection = Collection(name="user_profiles", schema=schema)
    
    # Create index
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection


def sync_user_profile(user_id, profile_data):
    """
    Sync user profile to vector database
    
    Args:
        user_id: User identifier
        profile_data: User profile dictionary
    """
    
    # Initialize BGE-M3 encoder
    encoder = BGEM3FlagModel(config.EMBEDDING_MODEL, use_fp16=True)
    
    # Connect and get collection
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    
    try:
        collection = Collection("user_profiles")
    except:
        collection = create_profile_collection()
    
    # Prepare profile text for embedding
    profile_texts = []
    
    # Add test results
    if 'tests' in profile_data:
        for test in profile_data['tests']:
            text = f"{test.get('name', '')}: {test.get('value', '')} {test.get('unit', '')}"
            if 'reference_range' in test:
                text += f" (参考: {test['reference_range']})"
            profile_texts.append(text)
    
    # Add symptoms
    if 'symptoms' in profile_data:
        for symptom in profile_data['symptoms']:
            profile_texts.append(f"症状: {symptom}")
    
    # Add abnormal indicators
    if 'abnormal_indicators' in profile_data:
        for indicator in profile_data['abnormal_indicators']:
            text = f"异常: {indicator.get('test', '')} - {indicator.get('value', '')}"
            profile_texts.append(text)
    
    if not profile_texts:
        print(f"No profile data to sync for user {user_id}")
        return
    
    # Generate embeddings with BGE-M3
    embeddings_output = encoder.encode(profile_texts, batch_size=12, max_length=8192)
    # Use dense vectors for Milvus
    embeddings = embeddings_output['dense_vecs']
    
    # Prepare data for insertion
    user_ids = [user_id] * len(profile_texts)
    timestamps = [profile_data.get('timestamp', datetime.now().isoformat())] * len(profile_texts)
    report_types = [profile_data.get('report_type', 'general')] * len(profile_texts)
    
    # Insert into Milvus
    entities = [
        user_ids,
        embeddings.tolist(),
        profile_texts,
        timestamps,
        report_types
    ]
    
    collection.insert(entities)
    collection.flush()
    
    print(f"Successfully synced {len(profile_texts)} profile entries for user {user_id}")


def sync_from_processed_reports():
    """Sync profiles from processed reports directory"""
    
    processed_dir = Path(__file__).parent.parent / 'data' / 'sensitive' / 'processed'
    
    if not processed_dir.exists():
        print(f"Processed reports directory not found: {processed_dir}")
        return
    
    # Process each JSON file
    for json_file in processed_dir.glob('*.json'):
        print(f"Processing {json_file.name}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        user_id = report_data.get('user_id', json_file.stem)
        
        sync_user_profile(user_id, report_data)
    
    print("Profile sync completed!")


def main():
    """Main execution"""
    
    print("Starting user profile sync...")
    
    # Check if we should sync from files or use sample data
    processed_dir = Path(__file__).parent.parent / 'data' / 'sensitive' / 'processed'
    
    if processed_dir.exists() and list(processed_dir.glob('*.json')):
        sync_from_processed_reports()
    else:
        print("No processed reports found.")
        print("To sync profiles, place processed report JSON files in:")
        print(f"  {processed_dir}")


if __name__ == '__main__':
    main()
