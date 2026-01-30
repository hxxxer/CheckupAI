"""
Dual-Path Retriever
Combines knowledge base retrieval with user profile retrieval
"""

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np


class DualPathRetriever:
    def __init__(self, 
                 milvus_host='localhost',
                 milvus_port='19530',
                 embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize dual-path retriever
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            embedding_model: Sentence embedding model
        """
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)
        
        # Collection names
        self.knowledge_collection_name = 'medical_knowledge'
        self.profile_collection_name = 'user_profiles'
        
        # Load collections
        try:
            self.knowledge_collection = Collection(self.knowledge_collection_name)
            self.profile_collection = Collection(self.profile_collection_name)
            self.knowledge_collection.load()
            self.profile_collection.load()
        except Exception as e:
            print(f"Collection loading error: {e}")
            self.knowledge_collection = None
            self.profile_collection = None
    
    def encode_query(self, query):
        """
        Encode query text to embedding vector
        
        Args:
            query: Query text
            
        Returns:
            numpy array: Embedding vector
        """
        embedding = self.encoder.encode([query])[0]
        return embedding
    
    def retrieve_from_knowledge_base(self, query, top_k=5):
        """
        Retrieve relevant documents from knowledge base
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            list: Retrieved documents
        """
        if not self.knowledge_collection:
            return []
        
        query_embedding = self.encode_query(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.knowledge_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source", "metadata"]
        )
        
        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    'text': hit.entity.get('text'),
                    'source': hit.entity.get('source'),
                    'metadata': hit.entity.get('metadata'),
                    'score': hit.score
                })
        
        return documents
    
    def retrieve_from_user_profile(self, user_id, query, top_k=3):
        """
        Retrieve relevant information from user profile
        
        Args:
            user_id: User identifier
            query: User query
            top_k: Number of results to return
            
        Returns:
            list: Retrieved profile information
        """
        if not self.profile_collection:
            return []
        
        query_embedding = self.encode_query(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        # Search with user_id filter
        expr = f'user_id == "{user_id}"'
        
        results = self.profile_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "timestamp", "report_type"]
        )
        
        profile_data = []
        for hits in results:
            for hit in hits:
                profile_data.append({
                    'text': hit.entity.get('text'),
                    'timestamp': hit.entity.get('timestamp'),
                    'report_type': hit.entity.get('report_type'),
                    'score': hit.score
                })
        
        return profile_data
    
    def dual_retrieve(self, query, user_id=None, knowledge_k=5, profile_k=3):
        """
        Perform dual-path retrieval
        
        Args:
            query: User query
            user_id: User identifier (optional)
            knowledge_k: Number of knowledge base results
            profile_k: Number of user profile results
            
        Returns:
            dict: Combined retrieval results
        """
        results = {
            'knowledge': self.retrieve_from_knowledge_base(query, knowledge_k),
            'profile': []
        }
        
        if user_id:
            results['profile'] = self.retrieve_from_user_profile(user_id, query, profile_k)
        
        return results
