"""
Dual-Path Retriever
Combines knowledge base retrieval with user profile retrieval
Uses BGE-M3 for embedding and BGE-reranker-v2-m3 for reranking
"""

from pymilvus import connections, Collection
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from .reranker import BGEReranker


class DualPathRetriever:
    def __init__(self, 
                 milvus_host='localhost',
                 milvus_port='19530',
                 embedding_model='BAAI/bge-m3',
                 reranker_model='BAAI/bge-reranker-v2-m3',
                 use_reranker=True):
        """
        Initialize dual-path retriever
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            embedding_model: BGE-M3 embedding model
            reranker_model: BGE reranker model
            use_reranker: Whether to use reranker for result refinement
        """
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        
        # Initialize BGE-M3 embedding model
        self.encoder = BGEM3FlagModel(embedding_model, use_fp16=True)
        
        # Initialize reranker
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = BGEReranker(reranker_model, use_fp16=True)
        else:
            self.reranker = None
        
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
        Encode query text to embedding vector using BGE-M3
        
        Args:
            query: Query text
            
        Returns:
            numpy array: Dense embedding vector
        """
        # BGE-M3 returns dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
        embeddings = self.encoder.encode([query])
        # Use dense vectors for Milvus search
        return embeddings['dense_vecs'][0]
    
    def retrieve_from_knowledge_base(self, query, top_k=5, rerank_top_k=None):
        """
        Retrieve relevant documents from knowledge base
        
        Args:
            query: User query
            top_k: Number of initial results to retrieve
            rerank_top_k: Number of results after reranking (None = same as top_k)
            
        Returns:
            list: Retrieved and optionally reranked documents
        """
        if not self.knowledge_collection:
            return []
        
        # Retrieve more candidates for reranking
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        
        query_embedding = self.encode_query(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.knowledge_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=retrieve_k,
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
        
        # Apply reranking if enabled
        if self.use_reranker and self.reranker and documents:
            final_k = rerank_top_k if rerank_top_k is not None else top_k
            documents = self.reranker.rerank(query, documents, top_k=final_k)
        
        return documents
    
    def retrieve_from_user_profile(self, user_id, query, top_k=3, rerank_top_k=None):
        """
        Retrieve relevant information from user profile
        
        Args:
            user_id: User identifier
            query: User query
            top_k: Number of initial results to retrieve
            rerank_top_k: Number of results after reranking
            
        Returns:
            list: Retrieved and optionally reranked profile information
        """
        if not self.profile_collection:
            return []
        
        # Retrieve more candidates for reranking
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        
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
            limit=retrieve_k,
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
        
        # Apply reranking if enabled
        if self.use_reranker and self.reranker and profile_data:
            final_k = rerank_top_k if rerank_top_k is not None else top_k
            profile_data = self.reranker.rerank(query, profile_data, top_k=final_k)
        
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
