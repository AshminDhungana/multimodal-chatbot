"""
Vector Store Module for Multimodel Chatbot

This module manages the FAISS vector database for semantic search.
It provides:
- Vector store initialization and management
- Document storage and retrieval
- Semantic similarity search
- Batch operations
- Persistence (save/load)
- Index optimization

The vector store is the backbone of the RAG system, enabling
fast semantic search through millions of vectors.

"""

import os
import logging
import pickle
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# Third-party imports
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Index type constants
INDEX_TYPE_FLAT = "flat"           # Exact search
INDEX_TYPE_IVF = "ivf"             # Approximate search (faster)
INDEX_TYPE_HNSW = "hnsw"           # Hierarchical search


# ============================================================================
# VECTOR STORE WRAPPER
# ============================================================================

class VectorStore:
    """
    Wrapper around FAISS vector store for document management.
    
    This class provides a high-level interface for:
    - Adding documents to the vector database
    - Searching for similar documents
    - Managing the vector index
    - Persistence operations
    
    Attributes:
        store_path: Directory where index is saved
        embeddings: Embedding model instance
        faiss_store: FAISS instance from LangChain
        metadata: Additional store information
        document_count: Number of documents in store
    """
    
    def __init__(
        self,
        store_path: str = "./vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = INDEX_TYPE_FLAT
    ):
        """
        Initialize Vector Store.
        
        Args:
            store_path: Directory to save/load index
            embedding_model: HuggingFace embedding model
            index_type: Type of FAISS index
            
        Example:
            >>> store = VectorStore("./vectorstore")
            >>> store.add_documents(["Text 1", "Text 2"])
            >>> results = store.search("query")
        """
        self.store_path = store_path
        self.index_type = index_type
        self.faiss_store = None
        self.document_count = 0
        self.metadata = {
            "created": None,
            "updated": None,
            "model": embedding_model,
            "index_type": index_type
        }
        
        logger.info(f"Initializing VectorStore at {store_path}")
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"}
        )
        
        # Create directory if needed
        Path(store_path).mkdir(parents=True, exist_ok=True)
        
        # Try to load existing store
        if self._index_exists():
            self.load()
        else:
            logger.info("No existing index found. Store will be created on first add.")
    
    # ========================================================================
    # INDEX MANAGEMENT
    # ========================================================================
    
    def _index_exists(self) -> bool:
        """
        Check if vector store index exists.
        
        Returns:
            bool: True if index files exist
        """
        index_file = os.path.join(self.store_path, "index.faiss")
        return os.path.exists(index_file)
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
            ids: Optional IDs for documents
            
        Returns:
            bool: True if successful
            
        Example:
            >>> store = VectorStore()
            >>> docs = ["Hello world", "Machine learning"]
            >>> meta = [{"source": "file1"}, {"source": "file2"}]
            >>> store.add_documents(docs, meta)
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            logger.info(f"Adding {len(documents)} documents to store")
            
            # Create LangChain Document objects
            doc_objects = []
            for i, doc_text in enumerate(documents):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                doc_objects.append(Document(page_content=doc_text, metadata=meta))
            
            # Add to FAISS
            if self.faiss_store is None:
                logger.info("Creating new FAISS index")
                self.faiss_store = FAISS.from_documents(
                    doc_objects,
                    self.embeddings
                )
            else:
                logger.info("Adding to existing FAISS index")
                self.faiss_store.add_documents(doc_objects)
            
            self.document_count += len(documents)
            logger.info(f"Successfully added {len(documents)} documents. Total: {self.document_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query (natural language)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results with content and metadata
            
        Example:
            >>> store = VectorStore()
            >>> results = store.search("What is AI?", top_k=3)
            >>> for result in results:
            ...     print(result['content'])
            ...     print(f"Score: {result['score']}")
        """
        try:
            if self.faiss_store is None:
                logger.warning("Vector store is empty")
                return []
            
            logger.info(f"Searching for: {query[:50]}...")
            
            # Search with scores
            docs_and_scores = self.faiss_store.similarity_search_with_score(query, k=top_k)
            
            # Format results
            results = []
            for doc, score in docs_and_scores:
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 / (1 + score)
                
                if similarity >= score_threshold:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(similarity),
                        "distance": float(score)
                    })
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def similarity_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Simple similarity search returning just text.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of text contents
            
        Example:
            >>> store = VectorStore()
            >>> results = store.similarity_search("AI", top_k=2)
            >>> print(results[0])  # First result text
        """
        docs = self.search(query, top_k)
        return [doc["content"] for doc in docs]
    
    def search_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search using a pre-computed embedding vector.
        
        Useful when you already have embeddings from elsewhere.
        
        Args:
            vector: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of search results
            
        Example:
            >>> store = VectorStore()
            >>> query_vector = some_embedding  # Pre-computed
            >>> results = store.search_by_vector(query_vector)
        """
        try:
            if self.faiss_store is None:
                logger.warning("Vector store is empty")
                return []
            
            logger.info("Searching by vector")
            
            # Get the underlying FAISS index
            index = self.faiss_store.index
            
            # Search (FAISS expects 2D array)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            distances, indices = index.search(vector.astype(np.float32), top_k)
            
            # Get documents
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0:  # Valid index
                    doc = self.faiss_store.docstore.search(str(idx))
                    similarity = 1 / (1 + distance)
                    
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(similarity),
                        "distance": float(distance)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by vector: {str(e)}")
            return []
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Results per query
            
        Returns:
            List of result lists
            
        Example:
            >>> store = VectorStore()
            >>> queries = ["AI", "ML", "DL"]
            >>> results = store.batch_search(queries)
        """
        all_results = []
        for query in queries:
            results = self.search(query, top_k)
            all_results.append(results)
        return all_results
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self) -> bool:
        """
        Save vector store to disk.
        
        Returns:
            bool: True if successful
            
        Example:
            >>> store.add_documents(["text"])
            >>> store.save()  # Persists to disk
        """
        try:
            if self.faiss_store is None:
                logger.warning("No store to save")
                return False
            
            logger.info(f"Saving vector store to {self.store_path}")
            
            # Save FAISS index
            self.faiss_store.save_local(self.store_path)
            
            # Save metadata
            self._save_metadata()
            
            logger.info("Vector store saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        Load vector store from disk.
        
        Returns:
            bool: True if successful
            
        Example:
            >>> store = VectorStore()
            >>> store.load()  # Loads from disk
        """
        try:
            if not self._index_exists():
                logger.warning(f"No index found at {self.store_path}")
                return False
            
            logger.info(f"Loading vector store from {self.store_path}")
            
            # Load FAISS index
            self.faiss_store = FAISS.load_local(
                self.store_path,
                self.embeddings
            )
            
            # Load metadata
            self._load_metadata()
            
            # Count documents
            self.document_count = self.faiss_store.index.ntotal
            
            logger.info(f"Vector store loaded successfully ({self.document_count} documents)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            import json
            metadata_path = os.path.join(self.store_path, "metadata.json")
            
            from datetime import datetime
            self.metadata["updated"] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.debug("Metadata saved")
        except Exception as e:
            logger.warning(f"Could not save metadata: {str(e)}")
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        try:
            import json
            metadata_path = os.path.join(self.store_path, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug("Metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load metadata: {str(e)}")
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def clear(self) -> bool:
        """
        Clear all documents from store.
        
        Returns:
            bool: True if successful
            
        Example:
            >>> store.clear()  # Delete all documents
        """
        try:
            logger.info("Clearing vector store")
            
            self.faiss_store = None
            self.document_count = 0
            
            # Delete files
            if os.path.exists(self.store_path):
                import shutil
                shutil.rmtree(self.store_path)
                Path(self.store_path).mkdir(parents=True, exist_ok=True)
            
            logger.info("Vector store cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get store statistics.
        
        Returns:
            Dictionary with stats
            
        Example:
            >>> stats = store.get_stats()
            >>> print(f"Documents: {stats['document_count']}")
        """
        return {
            "document_count": self.document_count,
            "store_path": self.store_path,
            "index_type": self.index_type,
            "embedding_model": self.metadata.get("model", "unknown"),
            "created": self.metadata.get("created"),
            "updated": self.metadata.get("updated"),
            "is_empty": self.faiss_store is None
        }
    
    def get_document_count(self) -> int:
        """
        Get number of documents in store.
        
        Returns:
            int: Document count
        """
        if self.faiss_store is None:
            return 0
        return self.faiss_store.index.ntotal
    
    def delete_documents(self, indices: List[int]) -> bool:
        """
        Delete documents by index.
        
        Note: FAISS doesn't support efficient deletion.
        For frequent deletions, consider rebuilding the index.
        
        Args:
            indices: List of document indices to delete
            
        Returns:
            bool: True if successful
        """
        try:
            logger.warning("Document deletion is not efficient in FAISS. Consider rebuilding.")
            logger.info(f"Marking {len(indices)} documents for deletion")
            # FAISS deletion is complex - would need to rebuild index
            # For now, just log warning
            return False
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def is_empty(self) -> bool:
        """
        Check if store is empty.
        
        Returns:
            bool: True if empty
        """
        return self.faiss_store is None or self.document_count == 0
    
    # ========================================================================
    # ADVANCED OPERATIONS
    # ========================================================================
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index.
        
        Useful after many deletions or for optimization.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Rebuilding FAISS index")
            
            if self.faiss_store is None:
                logger.warning("No store to rebuild")
                return False
            
            # Get all documents
            all_docs = []
            for i in range(self.document_count):
                doc = self.faiss_store.docstore.search(str(i))
                all_docs.append(doc)
            
            # Recreate store
            self.faiss_store = FAISS.from_documents(all_docs, self.embeddings)
            
            logger.info("Index rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            return False
    
    def get_index_info(self) -> Dict:
        """
        Get detailed index information.
        
        Returns:
            Dictionary with index details
        """
        if self.faiss_store is None:
            return {
                "status": "empty",
                "documents": 0,
                "dimensions": 0
            }
        
        try:
            index = self.faiss_store.index
            return {
                "status": "loaded",
                "documents": index.ntotal,
                "dimensions": index.d,
                "type": type(index).__name__,
                "is_trained": index.is_trained
            }
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {"status": "error"}


# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """
    Manager for multiple vector stores.
    
    Useful when working with multiple collections of documents.
    
    Attributes:
        stores: Dictionary of named vector stores
        default_store: Default store for operations
    """
    
    def __init__(self, base_path: str = "./vectorstores"):
        """
        Initialize Vector Store Manager.
        
        Args:
            base_path: Base directory for all stores
        """
        self.base_path = base_path
        self.stores: Dict[str, VectorStore] = {}
        self.default_store = None
        
        Path(base_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"VectorStoreManager initialized at {base_path}")
    
    def create_store(
        self,
        name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> VectorStore:
        """
        Create a new named vector store.
        
        Args:
            name: Name for the store
            embedding_model: Embedding model to use
            
        Returns:
            VectorStore instance
            
        Example:
            >>> manager = VectorStoreManager()
            >>> store = manager.create_store("documents")
            >>> store.add_documents(["text"])
        """
        try:
            store_path = os.path.join(self.base_path, name)
            store = VectorStore(store_path, embedding_model)
            self.stores[name] = store
            
            if self.default_store is None:
                self.default_store = name
            
            logger.info(f"Created vector store: {name}")
            return store
            
        except Exception as e:
            logger.error(f"Error creating store: {str(e)}")
            return None
    
    def get_store(self, name: str) -> Optional[VectorStore]:
        """
        Get a named vector store.
        
        Args:
            name: Store name
            
        Returns:
            VectorStore instance or None
        """
        return self.stores.get(name)
    
    def delete_store(self, name: str) -> bool:
        """
        Delete a named vector store.
        
        Args:
            name: Store name
            
        Returns:
            bool: True if successful
        """
        try:
            if name in self.stores:
                self.stores[name].clear()
                del self.stores[name]
                logger.info(f"Deleted store: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting store: {str(e)}")
            return False
    
    def list_stores(self) -> List[str]:
        """
        List all available stores.
        
        Returns:
            List of store names
        """
        return list(self.stores.keys())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_vector_store(
    documents: List[str],
    store_path: str = "./vectorstore",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> VectorStore:
    """
    Create and populate a vector store in one step.
    
    Args:
        documents: List of documents to add
        store_path: Where to save the store
        embedding_model: Embedding model to use
        
    Returns:
        Initialized VectorStore
        
    Example:
        >>> docs = ["Hello", "World"]
        >>> store = create_vector_store(docs)
        >>> store.search("greeting")
    """
    try:
        store = VectorStore(store_path, embedding_model)
        store.add_documents(documents)
        store.save()
        return store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None


def get_vectorstore_size(store_path: str) -> int:
    """
    Get size of vector store on disk in bytes.
    
    Args:
        store_path: Path to store
        
    Returns:
        Size in bytes
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(store_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    except Exception as e:
        logger.error(f"Error getting store size: {str(e)}")
        return 0
