"""
Configuration Module
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Milvus Configuration
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', './model/lora_checkpoint')
    EMBEDDING_MODEL = os.getenv(
        'EMBEDDING_MODEL',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    # OpenAI Configuration (for LLM validation)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', None)
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # OCR Configuration
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    
    # Data Paths
    DATA_DIR = os.getenv('DATA_DIR', './data')
    SENSITIVE_DIR = os.path.join(DATA_DIR, 'sensitive')
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, 'knowledge_base')
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))


config = Config()
