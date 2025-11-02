"""
Vector Store Module for Multimodel Chatbot

This module provides FAISS vector database management with document storage,
semantic similarity search, and persistence capabilities.

Features:
- FAISS vector database wrapper
- GPU/CPU embedding support
- Document storage with metadata
- Semantic similarity search
- Batch operations
- Persistence (save/load)
- Index management
- Metadata tracking (created/updated timestamps)

LangChain 1.0.3 Compatible
"""

import os
import logging
import shutil
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

# ============================================================================
# LANGCHAIN 1.0.3 IMPORTS (COMPATIBLE)
# ============================================================================

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Optional imports
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# Index types
INDEX_TYPE_FLAT = "flat"
INDEX_TYPE_IVF = "ivf"
INDEX_TYPE_HNSW = "hnsw"

# Load from .env
DEFAULT_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """
    Wrapper around FAISS vector store for document management.
    
    Provides:
    - Document addition with metadata
    - Semantic similarity search
    - Vector-based search
    - Persistence (save/load)
    - Index management
    - Metadata tracking
    """

    def __init__(
        self,
        store_path: str = DEFAULT_VECTORSTORE_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        index_type: str = INDEX_TYPE_FLAT,
        device: Optional[str] = None,
    ):
        """
        Initialize VectorStore.
        
        Args:
            store_path: Path to store FAISS index
            embedding_model: HuggingFace embedding model
            index_type: Type of FAISS index (flat, ivf, hnsw)
            device: "cpu" or "cuda" (auto-detect if None)
        """
        self.store_path = store_path
        self.index_type = index_type
        self.embedding_model_name = embedding_model
        self.faiss_store: Optional[FAISS] = None
        self.document_count: int = 0

        # Detect device
        if device is None:
            device = "cuda" if (torch_available and torch.cuda.is_available()) else "cpu"
        self.device = device

        logger.info(
            f"✅ [VectorStore] Initializing: path={store_path}, "
            f"model={embedding_model}, device={device}"
        )

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
            )
            logger.info(f"✅ [Embeddings] Loaded: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ [Embeddings] Failed to load: {str(e)}")
            raise

        # Initialize metadata
        self.metadata = {
            "created": None,
            "updated": None,
            "model": embedding_model,
            "index_type": index_type,
            "device": device,
        }

        # Create store path
        Path(store_path).mkdir(parents=True, exist_ok=True)

        # Load existing index if available
        if self._index_exists():
            logger.info("[VectorStore] Existing index found, loading...")
            self.load()
        else:
            self.metadata["created"] = datetime.now().isoformat()
            logger.info("[VectorStore] New store initialized")

    # ========================================================================
    # INDEX MANAGEMENT
    # ========================================================================

    def _index_exists(self) -> bool:
        """Check if FAISS index exists."""
        index_path = os.path.join(self.store_path, "index.faiss")
        return os.path.exists(index_path)

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts
            
        Returns:
            True if successful
        """
        if not documents:
            logger.warning("[Add] No documents provided")
            return False

        try:
            logger.info(f"[Add] Adding {len(documents)} documents")

            # Create Document objects
            doc_objects = []
            for i, text in enumerate(documents):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                doc_objects.append(Document(page_content=text, metadata=meta))

            # Add to FAISS store
            if self.faiss_store is None:
                logger.info("[Add] Creating new FAISS index")
                self.faiss_store = FAISS.from_documents(doc_objects, self.embeddings)
            else:
                logger.info("[Add] Adding to existing index")
                self.faiss_store.add_documents(doc_objects)

            # Update count
            self.document_count = self.faiss_store.index.ntotal
            self.metadata["updated"] = datetime.now().isoformat()

            logger.info(f"✅ [Add] Total documents: {self.document_count}")
            return True

        except Exception as e:
            logger.error(f"❌ [Add] Error: {str(e)}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        return_raw_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using query text.
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score (0-1)
            return_raw_scores: Return raw distance scores
            
        Returns:
            List of search results with scores
        """
        if self.faiss_store is None:
            logger.warning("[Search] Vector store is empty")
            return []

        try:
            logger.debug(f"[Search] Query: {query[:50]}...")
            docs_and_scores = self.faiss_store.similarity_search_with_score(
                query, k=top_k
            )

            results = []
            for doc, distance in docs_and_scores:
                # Convert distance to similarity (0-1 scale)
                similarity = 1 / (1 + distance)

                if similarity >= score_threshold:
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": float(similarity),
                    }

                    if return_raw_scores:
                        result["distance"] = float(distance)

                    results.append(result)

            logger.info(f"[Search] Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ [Search] Error: {str(e)}")
            return []

    def search_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        return_raw_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed embedding vector.
        
        Args:
            vector: Embedding vector
            top_k: Number of results
            return_raw_scores: Return raw distance scores
            
        Returns:
            List of search results
        """
        if self.faiss_store is None:
            logger.warning("[SearchVector] Vector store is empty")
            return []

        try:
            logger.debug("[SearchVector] Searching by vector")

            # Reshape if needed
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)

            # Ensure float32
            vector = vector.astype(np.float32)

            # Search
            index = self.faiss_store.index
            distances, indices = index.search(vector, top_k)

            results = []
            docstore_dict = self.faiss_store.docstore._dict
            docstore_keys = list(docstore_dict.keys())

            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(docstore_keys):
                    key = docstore_keys[idx]
                    doc = self.faiss_store.docstore[key]

                    similarity = 1 / (1 + dist)
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": float(similarity),
                    }

                    if return_raw_scores:
                        result["distance"] = float(dist)

                    results.append(result)

            logger.info(f"[SearchVector] Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ [SearchVector] Error: {str(e)}")
            return []

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform multiple searches.
        
        Args:
            queries: List of search queries
            top_k: Results per query
            
        Returns:
            List of result lists
        """
        logger.info(f"[BatchSearch] Searching {len(queries)} queries")
        results = [self.search(q, top_k) for q in queries]
        return results

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def save(self) -> bool:
        """Save vector store to disk."""
        try:
            if self.faiss_store is None:
                logger.warning("[Save] No store to save")
                return False

            logger.info(f"[Save] Saving to {self.store_path}")

            # Save FAISS index
            self.faiss_store.save_local(self.store_path)

            # Save metadata
            self._save_metadata()

            logger.info("✅ [Save] Vector store saved successfully")
            return True

        except Exception as e:
            logger.error(f"❌ [Save] Error: {str(e)}")
            return False

    def load(self) -> bool:
        """Load vector store from disk."""
        try:
            if not self._index_exists():
                logger.warning(f"[Load] Index not found at {self.store_path}")
                return False

            logger.info(f"[Load] Loading from {self.store_path}")

            # Load FAISS index
            self.faiss_store = FAISS.load_local(
                self.store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load metadata
            self._load_metadata()

            # Update count
            self.document_count = self.faiss_store.index.ntotal

            logger.info(f"✅ [Load] Loaded with {self.document_count} documents")
            return True

        except Exception as e:
            logger.error(f"❌ [Load] Error: {str(e)}")
            return False

    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            self.metadata["updated"] = datetime.now().isoformat()
            if not self.metadata.get("created"):
                self.metadata["created"] = datetime.now().isoformat()

            meta_path = os.path.join(self.store_path, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

            logger.debug("[Metadata] Saved")
        except Exception as e:
            logger.error(f"❌ [Metadata] Save error: {str(e)}")

    def _load_metadata(self):
        """Load metadata from JSON file."""
        try:
            meta_path = os.path.join(self.store_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.debug("[Metadata] Loaded")
        except Exception as e:
            logger.error(f"⚠️  [Metadata] Load error: {str(e)}")

    # ========================================================================
    # MANAGEMENT
    # ========================================================================

    def clear(self) -> bool:
        """Clear all documents and delete store."""
        try:
            logger.info("[Clear] Clearing vector store")

            self.faiss_store = None
            self.document_count = 0
            self.metadata["created"] = None
            self.metadata["updated"] = None

            if os.path.exists(self.store_path):
                shutil.rmtree(self.store_path)
                logger.info(f"[Clear] Deleted {self.store_path}")

            Path(self.store_path).mkdir(parents=True, exist_ok=True)
            logger.info("✅ [Clear] Vector store cleared")
            return True

        except Exception as e:
            logger.error(f"❌ [Clear] Error: {str(e)}")
            return False

    def rebuild_index(self) -> bool:
        """Rebuild FAISS index from documents."""
        try:
            if self.faiss_store is None:
                logger.warning("[Rebuild] No store to rebuild")
                return False

            logger.info("[Rebuild] Rebuilding index")

            # Extract all documents
            docstore_dict = self.faiss_store.docstore._dict
            all_docs = [self.faiss_store.docstore[key] for key in docstore_dict.keys()]

            # Rebuild from documents
            self.faiss_store = FAISS.from_documents(all_docs, self.embeddings)
            self.metadata["updated"] = datetime.now().isoformat()

            logger.info("✅ [Rebuild] Index rebuilt successfully")
            return True

        except Exception as e:
            logger.error(f"❌ [Rebuild] Error: {str(e)}")
            return False

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def is_empty(self) -> bool:
        """Check if store is empty."""
        return self.faiss_store is None or self.document_count == 0

    def get_document_count(self) -> int:
        """Get number of documents."""
        return self.document_count

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        try:
            store_size = 0
            if os.path.exists(self.store_path):
                for dirpath, _, filenames in os.walk(self.store_path):
                    for fname in filenames:
                        fpath = os.path.join(dirpath, fname)
                        store_size += os.path.getsize(fpath)

            return {
                "status": "active" if not self.is_empty() else "empty",
                "document_count": self.document_count,
                "store_path": self.store_path,
                "store_size_mb": round(store_size / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "index_type": self.index_type,
                "device": self.device,
                "created": self.metadata.get("created"),
                "updated": self.metadata.get("updated"),
            }

        except Exception as e:
            logger.error(f"[Stats] Error: {str(e)}")
            return {"status": "error", "error": str(e)}


# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """
    Manage multiple named vector stores.
    
    Allows creation and management of separate vector stores
    for different document collections.
    """

    def __init__(self, base_path: str = "./vectorstores"):
        """
        Initialize VectorStoreManager.
        
        Args:
            base_path: Base directory for all stores
        """
        self.base_path = base_path
        self.stores: Dict[str, VectorStore] = {}
        self.default_store: Optional[str] = None

        Path(base_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ [Manager] Initialized at {base_path}")

    def create_store(
        self,
        name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        set_default: bool = True,
    ) -> VectorStore:
        """
        Create a new named vector store.
        
        Args:
            name: Store name
            embedding_model: Embedding model to use
            set_default: Set as default store
            
        Returns:
            VectorStore instance
        """
        try:
            store_path = os.path.join(self.base_path, name)
            logger.info(f"[Manager] Creating store: {name}")

            store = VectorStore(store_path, embedding_model)
            self.stores[name] = store

            if set_default or self.default_store is None:
                self.default_store = name
                logger.info(f"[Manager] Set default store: {name}")

            logger.info(f"✅ [Manager] Store created: {name}")
            return store

        except Exception as e:
            logger.error(f"❌ [Manager] Error creating store: {str(e)}")
            raise

    def get_store(self, name: Optional[str] = None) -> Optional[VectorStore]:
        """
        Get a store by name (or default).
        
        Args:
            name: Store name (uses default if None)
            
        Returns:
            VectorStore instance or None
        """
        if name is None:
            name = self.default_store

        if name is None:
            logger.warning("[Manager] No store specified and no default set")
            return None

        return self.stores.get(name)

    def delete_store(self, name: str) -> bool:
        """
        Delete a store.
        
        Args:
            name: Store name
            
        Returns:
            True if successful
        """
        try:
            if name not in self.stores:
                logger.warning(f"[Manager] Store not found: {name}")
                return False

            logger.info(f"[Manager] Deleting store: {name}")
            self.stores[name].clear()
            del self.stores[name]

            if self.default_store == name:
                self.default_store = None

            logger.info(f"✅ [Manager] Store deleted: {name}")
            return True

        except Exception as e:
            logger.error(f"❌ [Manager] Error deleting store: {str(e)}")
            return False

    def list_stores(self) -> List[str]:
        """Get list of all store names."""
        return list(self.stores.keys())

    def get_stats_all(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all stores."""
        return {name: store.get_stats() for name, store in self.stores.items()}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_vector_store(
    documents: List[str],
    store_path: str = DEFAULT_VECTORSTORE_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    save_immediately: bool = True,
) -> VectorStore:
    """
    Create and populate a vector store.
    
    Args:
        documents: List of document texts
        store_path: Path to store
        embedding_model: Embedding model
        save_immediately: Save after adding documents
        
    Returns:
        Populated VectorStore
    """
    logger.info(f"[Helper] Creating store with {len(documents)} documents")

    store = VectorStore(store_path, embedding_model)
    store.add_documents(documents)

    if save_immediately:
        store.save()

    return store


def get_vectorstore_size(store_path: str) -> int:
    """
    Get total size of vector store in bytes.
    
    Args:
        store_path: Path to store
        
    Returns:
        Size in bytes
    """
    try:
        total = 0
        if os.path.exists(store_path):
            for dirpath, _, filenames in os.walk(store_path):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    total += os.path.getsize(fpath)
        return total
    except Exception as e:
        logger.error(f"[Size] Error: {str(e)}")
        return 0


# ============================================================================
# END OF MODULE
# ============================================================================
