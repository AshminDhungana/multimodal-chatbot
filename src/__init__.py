"""
Package Initialization for src module

This file makes the src directory a Python package and provides
convenient imports for all submodules.

It allows users to import from src like:
    from src.rag_pipeline import RAGPipeline
    from src.llm_models import LLMManager
    from src.document_loader import DocumentLoader
    from src.embeddings import EmbeddingManager
    from src.vectorstore import VectorStore
"""

# Version info
__version__ = "1.0.0"
__author__ = "Ashmin Dhungana"
__description__ = "Multimodel Chatbot with Retrieval-Augmented Generation"

# Import main classes for easier access
try:
    from src.rag_pipeline import RAGPipeline, prepare_context_from_documents, split_documents
    print("✓ RAGPipeline loaded")
except ImportError as e:
    print(f"⚠ Could not import RAGPipeline: {e}")

try:
    from src.llm_models import LLMManager, LLMManager, OpenAILLM, HuggingFaceLLM, ConversationManager
    print("✓ LLM Models loaded")
except ImportError as e:
    print(f"⚠ Could not import LLM Models: {e}")

try:
    from src.document_loader import DocumentLoader, PDFLoader, TextLoader, DocxLoader
    print("✓ Document Loader loaded")
except ImportError as e:
    print(f"⚠ Could not import Document Loader: {e}")

try:
    from src.embeddings import EmbeddingManager, SentenceTransformerEmbedding, VectorOperations
    print("✓ Embeddings loaded")
except ImportError as e:
    print(f"⚠ Could not import Embeddings: {e}")

try:
    from src.vectorstore import VectorStore, VectorStoreManager, create_vector_store
    print("✓ Vector Store loaded")
except ImportError as e:
    print(f"⚠ Could not import Vector Store: {e}")

# Public API
__all__ = [
    # RAG
    "RAGPipeline",
    "prepare_context_from_documents",
    "split_documents",
    
    # LLM
    "LLMManager",
    "OpenAILLM",
    "HuggingFaceLLM",
    "ConversationManager",
    
    # Documents
    "DocumentLoader",
    "PDFLoader",
    "TextLoader",
    "DocxLoader",
    
    # Embeddings
    "EmbeddingManager",
    "SentenceTransformerEmbedding",
    "VectorOperations",
    
    # Vector Store
    "VectorStore",
    "VectorStoreManager",
    "create_vector_store",
]

print("\n" + "="*50)
print("Multimodel Chatbot RAG System")
print(f"Version: {__version__}")
print("="*50 + "\n")
