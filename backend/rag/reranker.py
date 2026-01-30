"""
Reranker Module
Uses BGE-reranker-v2-m3 for result reranking
"""

from FlagEmbedding import FlagReranker
import numpy as np


class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', use_fp16=True):
        """
        Initialize BGE reranker
        
        Args:
            model_name: Reranker model name
            use_fp16: Whether to use FP16 for faster inference
        """
        self.reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16
        )
    
    def rerank(self, query, documents, top_k=None):
        """
        Rerank documents based on query
        
        Args:
            query: Query text
            documents: List of document dicts with 'text' field
            top_k: Number of top results to return (None returns all)
            
        Returns:
            list: Reranked documents with scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # Handle single document case (score is float, not list)
        if isinstance(scores, float):
            scores = [scores]
        
        # Combine documents with scores
        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            reranked.append(doc_copy)
        
        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top_k if specified
        if top_k:
            return reranked[:top_k]
        
        return reranked
    
    def rerank_with_threshold(self, query, documents, threshold=0.5, top_k=None):
        """
        Rerank documents and filter by threshold
        
        Args:
            query: Query text
            documents: List of document dicts
            threshold: Minimum rerank score to include
            top_k: Maximum number of results
            
        Returns:
            list: Filtered and reranked documents
        """
        reranked = self.rerank(query, documents, top_k=None)
        
        # Filter by threshold
        filtered = [doc for doc in reranked if doc['rerank_score'] >= threshold]
        
        # Apply top_k limit
        if top_k:
            return filtered[:top_k]
        
        return filtered
    
    def batch_rerank(self, queries, documents_list, top_k=None):
        """
        Rerank multiple query-document sets in batch
        
        Args:
            queries: List of query texts
            documents_list: List of document lists
            top_k: Number of top results per query
            
        Returns:
            list: List of reranked document lists
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)
        
        return results
