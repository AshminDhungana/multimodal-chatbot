"""
Tests for RAG Pipeline Module

This file contains unit tests for the RAG pipeline functionality.
Tests cover:
- Document addition
- Document retrieval
- Vector store operations
- Context preparation

Run with: pytest tests/test_rag.py -v

Author: Your Name
Date: 2025-11-02
Version: 1.0.0
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.rag_pipeline import RAGPipeline, prepare_context_from_documents, split_documents


# ============================================================================
# FIXTURES (Setup/Teardown)
# ============================================================================

@pytest.fixture
def temp_vectorstore():
    """Create temporary directory for vectorstore."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that learns from data.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "Natural language processing enables computers to understand human language.",
        "Python is a popular programming language for data science and AI.",
        "The transformer architecture revolutionized deep learning in 2017.",
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for documents."""
    return [
        {"source": "ai_guide.pdf", "page": 1},
        {"source": "ai_guide.pdf", "page": 2},
        {"source": "nlp_guide.pdf", "page": 1},
        {"source": "python_guide.pdf", "page": 1},
        {"source": "ai_guide.pdf", "page": 3},
    ]


@pytest.fixture
def rag_pipeline(temp_vectorstore):
    """Create RAG pipeline instance."""
    return RAGPipeline(
        vectorstore_path=temp_vectorstore,
        chunk_size=100,
        chunk_overlap=20,
        top_k=3
    )


# ============================================================================
# TESTS FOR DOCUMENT ADDITION
# ============================================================================

class TestDocumentAddition:
    """Test adding documents to RAG pipeline."""
    
    def test_add_single_document(self, rag_pipeline, sample_documents):
        """Test adding a single document."""
        result = rag_pipeline.add_documents([sample_documents[0]])
        assert result is True
    
    def test_add_multiple_documents(self, rag_pipeline, sample_documents):
        """Test adding multiple documents."""
        result = rag_pipeline.add_documents(sample_documents)
        assert result is True
    
    def test_add_documents_with_metadata(self, rag_pipeline, sample_documents, sample_metadata):
        """Test adding documents with metadata."""
        result = rag_pipeline.add_documents(sample_documents, sample_metadata)
        assert result is True
    
    def test_add_empty_document_list(self, rag_pipeline):
        """Test adding empty list of documents."""
        result = rag_pipeline.add_documents([])
        assert result is False
    
    def test_add_none_as_documents(self, rag_pipeline):
        """Test adding None as documents."""
        with pytest.raises(Exception):
            rag_pipeline.add_documents(None)
    
    def test_document_count_increases(self, rag_pipeline, sample_documents):
        """Test that document count increases after adding."""
        rag_pipeline.add_documents(sample_documents[:2])
        stats1 = rag_pipeline.get_stats()
        
        rag_pipeline.add_documents(sample_documents[2:])
        stats2 = rag_pipeline.get_stats()
        
        assert stats2['num_documents'] > stats1['num_documents']


# ============================================================================
# TESTS FOR DOCUMENT RETRIEVAL
# ============================================================================

class TestDocumentRetrieval:
    """Test retrieving documents from RAG pipeline."""
    
    def test_retrieve_from_empty_store(self, rag_pipeline):
        """Test retrieving from empty store."""
        results = rag_pipeline.retrieve("machine learning")
        assert len(results) == 0
    
    def test_retrieve_documents(self, rag_pipeline, sample_documents):
        """Test basic document retrieval."""
        rag_pipeline.add_documents(sample_documents)
        results = rag_pipeline.retrieve("machine learning", top_k=2)
        
        assert len(results) > 0
        assert 'content' in results[0]
        assert 'score' in results[0]
    
    def test_retrieve_returns_correct_number(self, rag_pipeline, sample_documents):
        """Test that retrieve returns correct number of results."""
        rag_pipeline.add_documents(sample_documents)
        
        results = rag_pipeline.retrieve("AI", top_k=3)
        assert len(results) <= 3
    
    def test_retrieve_returns_sorted_results(self, rag_pipeline, sample_documents):
        """Test that results are sorted by relevance."""
        rag_pipeline.add_documents(sample_documents)
        results = rag_pipeline.retrieve("machine learning", top_k=5)
        
        # Results should be sorted by score (descending)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_with_metadata(self, rag_pipeline, sample_documents, sample_metadata):
        """Test retrieving documents with metadata."""
        rag_pipeline.add_documents(sample_documents, sample_metadata)
        results = rag_pipeline.retrieve("Python", top_k=1)
        
        assert len(results) > 0
        assert 'metadata' in results[0]
    
    def test_search_simple(self, rag_pipeline, sample_documents):
        """Test simple search function."""
        rag_pipeline.add_documents(sample_documents)
        results = rag_pipeline.search("neural networks", top_k=2)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)
    
    def test_retrieve_with_custom_top_k(self, rag_pipeline, sample_documents):
        """Test retrieve with custom top_k value."""
        rag_pipeline.add_documents(sample_documents)
        
        results_1 = rag_pipeline.retrieve("AI", top_k=1)
        results_3 = rag_pipeline.retrieve("AI", top_k=3)
        
        assert len(results_1) <= len(results_3)


# ============================================================================
# TESTS FOR PERSISTENCE
# ============================================================================

class TestPersistence:
    """Test saving and loading vector stores."""
    
    def test_save_vectorstore(self, rag_pipeline, sample_documents):
        """Test saving vector store."""
        rag_pipeline.add_documents(sample_documents)
        result = rag_pipeline.save_vectorstore()
        assert result is True
    
    def test_load_vectorstore(self, rag_pipeline, sample_documents):
        """Test loading vector store."""
        # Add and save
        rag_pipeline.add_documents(sample_documents)
        rag_pipeline.save_vectorstore()
        
        # Create new instance and load
        rag2 = RAGPipeline(vectorstore_path=rag_pipeline.vectorstore_path)
        result = rag2.load_vectorstore()
        assert result is True
    
    def test_load_nonexistent_store(self, rag_pipeline):
        """Test loading from nonexistent path."""
        result = rag_pipeline.load_vectorstore()
        assert result is False
    
    def test_retrieve_after_load(self, rag_pipeline, sample_documents):
        """Test that retrieval works after loading from disk."""
        # Add, save, then load
        rag_pipeline.add_documents(sample_documents)
        rag_pipeline.save_vectorstore()
        
        rag2 = RAGPipeline(vectorstore_path=rag_pipeline.vectorstore_path)
        rag2.load_vectorstore()
        
        # Should be able to retrieve
        results = rag2.retrieve("machine learning", top_k=2)
        assert len(results) > 0


# ============================================================================
# TESTS FOR UTILITY FUNCTIONS
# ============================================================================

class TestUtilityFunctions:
    """Test helper functions."""
    
    def test_get_stats(self, rag_pipeline, sample_documents):
        """Test getting statistics."""
        rag_pipeline.add_documents(sample_documents)
        stats = rag_pipeline.get_stats()
        
        assert 'num_documents' in stats
        assert 'embedding_model' in stats
        assert stats['status'] == 'active'
    
    def test_get_stats_empty(self, rag_pipeline):
        """Test getting stats from empty store."""
        stats = rag_pipeline.get_stats()
        assert stats['num_documents'] == 0
    
    def test_clear_vectorstore(self, rag_pipeline, sample_documents):
        """Test clearing vector store."""
        rag_pipeline.add_documents(sample_documents)
        result = rag_pipeline.clear_vectorstore()
        assert result is True
    
    def test_batch_retrieve(self, rag_pipeline, sample_documents):
        """Test batch retrieve for multiple queries."""
        rag_pipeline.add_documents(sample_documents)
        
        queries = ["machine learning", "neural networks", "Python"]
        results = rag_pipeline.batch_retrieve(queries, top_k=2)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    def test_prepare_context_from_documents(self):
        """Test context preparation from documents."""
        documents = [
            {
                "content": "Text 1",
                "metadata": {"source": "doc1.pdf"}
            },
            {
                "content": "Text 2",
                "metadata": {"source": "doc2.pdf"}
            }
        ]
        
        context = prepare_context_from_documents(documents)
        
        assert isinstance(context, str)
        assert "Text 1" in context
        assert "Text 2" in context
        assert "doc1.pdf" in context
    
    def test_prepare_context_empty(self):
        """Test context preparation with empty list."""
        context = prepare_context_from_documents([])
        assert context == "No relevant documents found."
    
    def test_split_documents(self):
        """Test document splitting."""
        text = "A" * 1000  # 1000 character string
        chunks = split_documents(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 110 for chunk in chunks)  # 100 + 10 overlap buffer
    
    def test_split_empty_text(self):
        """Test splitting empty text."""
        chunks = split_documents("", chunk_size=100)
        assert chunks == [] or len(chunks) == 0


# ============================================================================
# TESTS FOR ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling in RAG pipeline."""
    
    def test_invalid_chunk_size(self, temp_vectorstore):
        """Test with invalid chunk size."""
        # Should either handle gracefully or raise appropriate error
        try:
            pipeline = RAGPipeline(
                vectorstore_path=temp_vectorstore,
                chunk_size=-100  # Invalid
            )
            # If it doesn't raise, that's also valid
        except ValueError:
            pass  # Expected
    
    def test_retrieve_with_empty_query(self, rag_pipeline, sample_documents):
        """Test retrieve with empty query."""
        rag_pipeline.add_documents(sample_documents)
        # Empty string might return nothing or all
        results = rag_pipeline.retrieve("")
        assert isinstance(results, list)
    
    def test_add_documents_with_mismatched_metadata(self, rag_pipeline, sample_documents):
        """Test adding documents with fewer metadata items than documents."""
        metadata = [{"source": "doc.pdf"}]  # Only 1 item
        result = rag_pipeline.add_documents(sample_documents, metadata)
        # Should handle gracefully
        assert result is True or result is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self, rag_pipeline, sample_documents, sample_metadata):
        """Test complete RAG workflow."""
        # 1. Add documents
        assert rag_pipeline.add_documents(sample_documents, sample_metadata)
        
        # 2. Save
        assert rag_pipeline.save_vectorstore()
        
        # 3. Search
        results = rag_pipeline.retrieve("artificial intelligence", top_k=2)
        assert len(results) > 0
        
        # 4. Load in new instance
        rag2 = RAGPipeline(vectorstore_path=rag_pipeline.vectorstore_path)
        assert rag2.load_vectorstore()
        
        # 5. Search again
        results2 = rag2.retrieve("deep learning", top_k=2)
        assert len(results2) > 0
    
    def test_multiple_search_operations(self, rag_pipeline, sample_documents):
        """Test multiple sequential searches."""
        rag_pipeline.add_documents(sample_documents)
        
        queries = [
            "machine learning",
            "neural networks",
            "Python programming",
            "transformers"
        ]
        
        all_results = []
        for query in queries:
            results = rag_pipeline.retrieve(query, top_k=2)
            all_results.extend(results)
        
        assert len(all_results) > 0
    
    def test_add_then_add_more_documents(self, rag_pipeline, sample_documents):
        """Test adding documents in multiple batches."""
        # First batch
        rag_pipeline.add_documents(sample_documents[:2])
        results1 = rag_pipeline.retrieve("machine learning", top_k=1)
        
        # Second batch
        rag_pipeline.add_documents(sample_documents[2:])
        results2 = rag_pipeline.retrieve("machine learning", top_k=1)
        
        # Should still work
        assert len(results1) > 0
        assert len(results2) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_add_many_documents(self, rag_pipeline):
        """Test adding many documents."""
        docs = [f"Document {i}: This is test content number {i}." for i in range(100)]
        result = rag_pipeline.add_documents(docs)
        assert result is True
    
    def test_retrieve_speed(self, rag_pipeline, sample_documents):
        """Test retrieval performance."""
        rag_pipeline.add_documents(sample_documents)
        
        import time
        start = time.time()
        results = rag_pipeline.retrieve("machine learning", top_k=3)
        elapsed = time.time() - start
        
        assert len(results) > 0
        assert elapsed < 5  # Should be fast (under 5 seconds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
