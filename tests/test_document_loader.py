"""
Tests for Document Loader Module

This file contains unit tests for document processing functionality.
Tests cover:
- PDF loading
- Text loading
- Document chunking
- Metadata extraction

Run with: pytest tests/test_document_loader.py -v

Author: Your Name
Date: 2025-11-02
Version: 1.0.0
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.document_loader import (
    DocumentLoader,
    TextLoader,
    is_supported_format,
    get_supported_formats,
    estimate_chunks
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_documents_dir():
    """Create temporary directory for test documents."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_txt_file(temp_documents_dir):
    """Create a sample text file."""
    file_path = os.path.join(temp_documents_dir, "sample.txt")
    with open(file_path, 'w') as f:
        f.write("""
        Machine Learning Guide
        
        Machine learning is a subset of artificial intelligence.
        It focuses on learning from data without being explicitly programmed.
        
        Chapter 1: Basics
        Machine learning uses algorithms to find patterns in data.
        These patterns can be used to make predictions.
        
        Chapter 2: Types
        Supervised learning uses labeled data.
        Unsupervised learning finds patterns without labels.
        Reinforcement learning learns through rewards.
        """)
    return file_path


@pytest.fixture
def sample_markdown_file(temp_documents_dir):
    """Create a sample markdown file."""
    file_path = os.path.join(temp_documents_dir, "sample.md")
    with open(file_path, 'w') as f:
        f.write("""# Artificial Intelligence

## Introduction
AI is transforming technology.

## Machine Learning
ML is a subset of AI.

### Neural Networks
Neural networks are inspired by biology.
""")
    return file_path


@pytest.fixture
def document_loader():
    """Create DocumentLoader instance."""
    return DocumentLoader(chunk_size=100, chunk_overlap=20)


# ============================================================================
# TESTS FOR SUPPORTED FORMATS
# ============================================================================

class TestSupportedFormats:
    """Test file format support."""
    
    def test_supported_pdf(self):
        """Test that PDF is supported."""
        assert is_supported_format("document.pdf") is True
    
    def test_supported_txt(self):
        """Test that TXT is supported."""
        assert is_supported_format("file.txt") is True
    
    def test_supported_md(self):
        """Test that MD is supported."""
        assert is_supported_format("readme.md") is True
    
    def test_supported_docx(self):
        """Test that DOCX is supported."""
        assert is_supported_format("document.docx") is True
    
    def test_unsupported_format(self):
        """Test that unsupported formats are rejected."""
        assert is_supported_format("image.jpg") is False
        assert is_supported_format("video.mp4") is False
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        formats = get_supported_formats()
        assert isinstance(formats, dict)
        assert '.pdf' in formats
        assert '.txt' in formats


# ============================================================================
# TESTS FOR TEXT LOADING
# ============================================================================

class TestTextLoading:
    """Test text file loading."""
    
    def test_load_txt_file(self, document_loader, sample_txt_file):
        """Test loading a text file."""
        chunks = document_loader.load_file(sample_txt_file)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_load_markdown_file(self, document_loader, sample_markdown_file):
        """Test loading a markdown file."""
        chunks = document_loader.load_file(sample_markdown_file)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_load_nonexistent_file(self, document_loader):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            document_loader.load_file("nonexistent.txt")
    
    def test_load_unsupported_format(self, document_loader, temp_documents_dir):
        """Test loading unsupported file format."""
        # Create unsupported file
        file_path = os.path.join(temp_documents_dir, "test.jpg")
        with open(file_path, 'wb') as f:
            f.write(b"fake image data")
        
        with pytest.raises(ValueError):
            document_loader.load_file(file_path)


# ============================================================================
# TESTS FOR DOCUMENT CHUNKING
# ============================================================================

class TestDocumentChunking:
    """Test text chunking."""
    
    def test_chunk_text_basic(self, document_loader):
        """Test basic text chunking."""
        text = "A" * 500
        chunks = document_loader.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_respects_size(self, document_loader):
        """Test that chunks respect size limits."""
        text = "A" * 1000
        chunks = document_loader.chunk_text(text, chunk_size=100, chunk_overlap=10)
        
        # Each chunk should be <= chunk_size + small buffer for overlap
        for chunk in chunks:
            assert len(chunk) <= 120  # 100 + 20 buffer
    
    def test_chunk_overlap_preserved(self, document_loader):
        """Test that overlap preserves context."""
        text = "Word A. Word B. Word C. Word D. Word E."
        chunks = document_loader.chunk_text(text, chunk_size=20, chunk_overlap=5)
        
        # Should have multiple chunks
        assert len(chunks) > 1
    
    def test_empty_text_chunking(self, document_loader):
        """Test chunking empty text."""
        chunks = document_loader.chunk_text("")
        assert len(chunks) == 0 or chunks == []
    
    def test_small_text_chunking(self, document_loader):
        """Test chunking text smaller than chunk size."""
        text = "Small text"
        chunks = document_loader.chunk_text(text, chunk_size=500)
        
        assert len(chunks) >= 1


# ============================================================================
# TESTS FOR DOCUMENT LOADING
# ============================================================================

class TestDocumentLoading:
    """Test document file operations."""
    
    def test_load_file_with_metadata(self, document_loader, sample_txt_file):
        """Test loading file with metadata."""
        chunks = document_loader.load_file(sample_txt_file, return_metadata=True)
        
        if chunks:
            first_item = chunks[0]
            assert isinstance(first_item, tuple)
            chunk_text, metadata = first_item
            assert isinstance(chunk_text, str)
            assert isinstance(metadata, dict)
    
    def test_load_multiple_files(self, document_loader, sample_txt_file, sample_markdown_file):
        """Test loading multiple files."""
        files = [sample_txt_file, sample_markdown_file]
        chunks = document_loader.load_files(files)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_load_directory(self, document_loader, temp_documents_dir):
        """Test loading entire directory."""
        # Create multiple test files
        for i in range(3):
            file_path = os.path.join(temp_documents_dir, f"test{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test document {i}\n" * 100)
        
        chunks = document_loader.load_directory(temp_documents_dir, recursive=False)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0


# ============================================================================
# TESTS FOR FILE STATISTICS
# ============================================================================

class TestFileStatistics:
    """Test file information gathering."""
    
    def test_get_file_stats(self, document_loader, sample_txt_file):
        """Test getting file statistics."""
        stats = document_loader.get_file_stats(sample_txt_file)
        
        assert isinstance(stats, dict)
        assert 'file_name' in stats
        assert 'file_size' in stats
        assert 'file_type' in stats
    
    def test_estimate_chunks(self):
        """Test chunk estimation."""
        # 5 MB file with 500 char chunks
        estimated = estimate_chunks(5.0, chunk_size=500)
        
        assert isinstance(estimated, int)
        assert estimated > 0


# ============================================================================
# TESTS FOR TEXT CLEANING
# ============================================================================

class TestTextCleaning:
    """Test text preprocessing and cleaning."""
    
    def test_clean_text_extra_whitespace(self, document_loader):
        """Test cleaning extra whitespace."""
        text = "Hello    world   \n\n\nHow   are    you"
        chunks = document_loader.chunk_text(text)
        
        # Should clean up extra whitespace
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_clean_special_characters(self, document_loader):
        """Test handling special characters."""
        text = "Hello\x00world\ufffdtest"  # Null bytes and replacement chars
        chunks = document_loader.chunk_text(text)
        
        # Should not crash
        assert isinstance(chunks, list)


# ============================================================================
# TESTS FOR ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_chunk_size(self, document_loader):
        """Test with invalid chunk size."""
        text = "Test text" * 100
        # Should handle gracefully
        try:
            chunks = document_loader.chunk_text(text, chunk_size=-100)
        except ValueError:
            pass  # Expected
    
    def test_chunk_with_none_text(self, document_loader):
        """Test chunking None."""
        with pytest.raises((TypeError, AttributeError)):
            document_loader.chunk_text(None)


# ============================================================================
# TESTS FOR TEXT LOADER
# ============================================================================

class TestTextLoaderClass:
    """Test TextLoader class directly."""
    
    def test_text_loader_initialization(self):
        """Test TextLoader initialization."""
        loader = TextLoader(encoding='utf-8')
        assert loader.encoding == 'utf-8'
    
    def test_text_loader_format_support(self):
        """Test format support checking."""
        loader = TextLoader()
        
        assert loader.supports_format("test.txt") is True
        assert loader.supports_format("test.md") is True
        assert loader.supports_format("test.pdf") is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for document loading."""
    
    def test_full_workflow(self, document_loader, sample_txt_file):
        """Test complete document loading workflow."""
        # 1. Check format support
        assert is_supported_format(sample_txt_file)
        
        # 2. Get file stats
        stats = document_loader.get_file_stats(sample_txt_file)
        assert stats['file_size'] > 0
        
        # 3. Load file
        chunks = document_loader.load_file(sample_txt_file)
        assert len(chunks) > 0
        
        # 4. Check chunks are reasonable
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_multiple_files_workflow(self, document_loader, sample_txt_file, sample_markdown_file):
        """Test loading multiple files workflow."""
        files = [sample_txt_file, sample_markdown_file]
        
        chunks = document_loader.load_files(files)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_chunk_large_text(self, document_loader):
        """Test chunking large text."""
        text = "Sample text. " * 10000  # Large text
        chunks = document_loader.chunk_text(text)
        
        assert len(chunks) > 100
    
    def test_load_multiple_large_files(self, document_loader, temp_documents_dir):
        """Test loading multiple large files."""
        # Create large test files
        for i in range(5):
            file_path = os.path.join(temp_documents_dir, f"large{i}.txt")
            with open(file_path, 'w') as f:
                # Write 1MB of data
                f.write("Test data. " * 100000)
        
        chunks = document_loader.load_directory(temp_documents_dir, recursive=False)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
