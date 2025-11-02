"""
Package Initialization for src module

This file makes the src directory a Python package and provides
convenient imports for all submodules.

It allows users to import from src like:
    from src.rag_pipeline import RAGPipelineLCEL
    from src.llm_models import LLMManager
    from src.document_loader import DocumentLoader
    from src.embeddings import EmbeddingManager
    from src.vectorstore import VectorStore
"""

import sys
import warnings

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "Ashmin Dhungana"
__description__ = "Multimodel Chatbot with Retrieval-Augmented Generation"

# ============================================================================
# STARTUP BANNER
# ============================================================================

print("\n" + "="*60)
print("🤖 Multimodel Chatbot with RAG System")
print(f"📦 Version: {__version__}")
print(f"👤 Author: {__author__}")
print("="*60 + "\n")

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

def _check_dependencies():
    """Check if all required LangChain packages are installed."""
    required_packages = {
        'langchain_text_splitters': 'Text Splitters',
        'langchain_community': 'Community Integrations',
        'langchain_huggingface': 'Hugging Face Integration',
        'langchain_core': 'LangChain Core',
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(f"  - {package} ({description})")
    
    if missing:
        warnings.warn(
            f"⚠️  Missing LangChain packages:\n{''.join(missing)}\n"
            f"Install with: pip install -r requirements.txt",
            RuntimeWarning
        )
        return False
    return True

# Check dependencies on import
_dependencies_ok = _check_dependencies()

# ============================================================================
# IMPORT MAIN CLASSES WITH ERROR HANDLING
# ============================================================================

# Track import status
_import_status = {
    'rag_pipeline': False,
    'llm_models': False,
    'document_loader': False,
    'embeddings': False,
    'vectorstore': False,
}

# --------------------------
# RAG PIPELINE
# --------------------------
try:
    from src.rag_pipeline import RAGPipelineLCEL, prepare_context_from_documents
    print("✅ RAG Pipeline loaded successfully")
    _import_status['rag_pipeline'] = True
except ImportError as e:
    print(f"⚠️  Could not import RAG Pipeline: {str(e)}")
    print("   Fix: Check imports in src/rag_pipeline.py")
except Exception as e:
    print(f"❌ Error loading RAG Pipeline: {str(e)}")

# --------------------------
# LLM MODELS
# --------------------------
try:
    from src.llm_models import (
        LLMManager,
        OpenAILLMChain,
        HuggingFaceLLMChain,
        ConversationManager
    )
    print("✅ LLM Models loaded successfully")
    _import_status['llm_models'] = True
except ImportError as e:
    print(f"⚠️  Could not import LLM Models: {str(e)}")
    print("   Fix: Check imports in src/llm_models.py")
except Exception as e:
    print(f"❌ Error loading LLM Models: {str(e)}")

# --------------------------
# DOCUMENT LOADERS
# --------------------------
try:
    from src.document_loader import (
        DocumentLoader,
        PDFLoader,
        TextLoader,
        DocxLoader
    )
    print("✅ Document Loaders loaded successfully")
    _import_status['document_loader'] = True
except ImportError as e:
    print(f"⚠️  Could not import Document Loaders: {str(e)}")
    print("   Fix: Check imports in src/document_loader.py")
except Exception as e:
    print(f"❌ Error loading Document Loaders: {str(e)}")

# --------------------------
# EMBEDDINGS
# --------------------------
try:
    from src.embeddings import (
        EmbeddingManager,
        SentenceTransformerEmbedding,
        VectorOperations
    )
    print("✅ Embeddings loaded successfully")
    _import_status['embeddings'] = True
except ImportError as e:
    print(f"⚠️  Could not import Embeddings: {str(e)}")
    print("   Fix: Check imports in src/embeddings.py")
except Exception as e:
    print(f"❌ Error loading Embeddings: {str(e)}")

# --------------------------
# VECTOR STORE
# --------------------------
try:
    from src.vectorstore import (
        VectorStore,
        VectorStoreManager,
        create_vector_store
    )
    print("✅ Vector Store loaded successfully")
    _import_status['vectorstore'] = True
except ImportError as e:
    print(f"⚠️  Could not import Vector Store: {str(e)}")
    print("   Fix: Check imports in src/vectorstore.py")
except Exception as e:
    print(f"❌ Error loading Vector Store: {str(e)}")

# ============================================================================
# IMPORT STATUS SUMMARY
# ============================================================================

_all_loaded = all(_import_status.values())
if _all_loaded:
    print("\n✅ All modules loaded successfully!\n")
else:
    failed = [k for k, v in _import_status.items() if not v]
    print(f"\n⚠️  {len(failed)} module(s) failed to load: {', '.join(failed)}")
    print("   Check the errors above and fix the imports.\n")

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # RAG Pipeline
    "RAGPipelineLCEL",
    "prepare_context_from_documents",

    # LLM Models
    "LLMManager",
    "OpenAILLMChain",
    "HuggingFaceLLMChain",
    "ConversationManager",

    # Document Loaders
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
    
    # Version info
    "__version__",
    "__author__",
    "__description__",
]

# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def get_import_status():
    """
    Return the status of module imports.
    
    Returns:
        dict: Status of each module import
    """
    return _import_status.copy()

def check_all_loaded():
    """
    Check if all modules are loaded successfully.
    
    Returns:
        bool: True if all modules loaded, False otherwise
    """
    return _all_loaded

# ============================================================================
# FRIENDLY ERROR MESSAGES
# ============================================================================

if not _dependencies_ok:
    print("\n⚠️  DEPENDENCY WARNING")
    print("   Some required LangChain packages are missing.")
    print("   Please install them with: pip install -r requirements.txt\n")

if not _all_loaded:
    print("\n⚠️  MODULE LOADING INCOMPLETE")
    print("   Some modules failed to import. This may be due to:")
    print("   1. Missing or incorrect imports in module files")
    print("   2. LangChain 1.0.3 compatibility issues")
    print("   3. Missing dependencies\n")
    print("   Action items:")
    print("   - Review the errors above")
    print("   - Check src/*.py files for correct imports")
    print("   - Ensure all packages are installed\n")

# ============================================================================
