
"""
RAG Pipeline Module for Multimodel Chatbot

This module implements the core Retrieval-Augmented Generation (RAG) pipeline.
It handles:
- Document loading and chunking
- Vector embeddings creation
- Vector store management (FAISS)
- Semantic search and document retrieval
- Context preparation for LLM

The RAG approach enhances LLM responses by retrieving relevant documents
from a knowledge base, ensuring answers are grounded in actual data.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

# Third-party imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# RAG PIPELINE CLASS
# ============================================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    
    This class manages the complete RAG workflow:
    1. Accepts documents/text chunks
    2. Converts them to vector embeddings
    3. Stores in FAISS vector database
    4. Retrieves relevant documents based on queries
    
    Attributes:
        vectorstore_path (str): Path to store FAISS index
        chunk_size (int): Size of text chunks in characters
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of documents to retrieve
        embeddings: HuggingFace embeddings model
        vectorstore: FAISS vector store instance
    """
    
    def __init__(
        self,
        vectorstore_path: str = "./vectorstore",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            vectorstore_path: Directory to save/load FAISS index
            chunk_size: Characters per chunk (how to split documents)
            chunk_overlap: Characters overlap between chunks (for context)
            top_k: How many documents to retrieve per query
            embedding_model: HuggingFace model for embeddings
        """
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"}  # or "cuda" if GPU available
        )
        
        # Initialize empty vectorstore
        self.vectorstore: Optional[FAISS] = None
        
        logger.info("RAG Pipeline initialized successfully")
    
    # ========================================================================
    # DOCUMENT MANAGEMENT
    # ========================================================================
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text chunks to add
            metadata: Optional metadata for each document (e.g., source, page)
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            >>> chunks = ["Document text 1", "Document text 2"]
            >>> meta = [{"source": "file1.pdf"}, {"source": "file1.pdf"}]
            >>> pipeline.add_documents(chunks, meta)
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Create LangChain Document objects
            docs = []
            for i, doc in enumerate(documents):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                docs.append(Document(page_content=doc, metadata=meta))
            
            # Create or update vectorstore
            if self.vectorstore is None:
                logger.info("Creating new vector store")
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            else:
                logger.info("Adding to existing vector store")
                self.vectorstore.add_documents(docs)
            
            logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: User's search query (natural language)
            top_k: Override default number of results to return
            filters: Optional filters (not implemented in basic FAISS)
            
        Returns:
            List of retrieved documents with content and metadata
            
        Example:
            >>> results = pipeline.retrieve("What is machine learning?", top_k=3)
            >>> for result in results:
            ...     print(result['content'])
            ...     print(result['metadata'])
        """
        if self.vectorstore is None:
            logger.warning("Vector store is empty, no documents to retrieve")
            return []
        
        try:
            k = top_k if top_k else self.top_k
            logger.info(f"Retrieving {k} documents for query: {query[:50]}...")
            
            # Search vector store using similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            retrieved_docs = []
            for doc, score in results:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # Lower score = more similar
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Simple search function - returns just the text content.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of text contents
            
        Example:
            >>> results = pipeline.search("Python programming")
            >>> print(results[0])  # First result text
        """
        docs = self.retrieve(query, top_k)
        return [doc["content"] for doc in docs]
    
    # ========================================================================
    # PERSISTENCE (Saving/Loading)
    # ========================================================================
    
    def save_vectorstore(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            bool: True if successful
            
        Example:
            >>> pipeline.save_vectorstore()
            True
        """
        try:
            if self.vectorstore is None:
                logger.warning("No vector store to save")
                return False
            
            # Create directory if it doesn't exist
            Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving vector store to {self.vectorstore_path}")
            self.vectorstore.save_local(self.vectorstore_path)
            
            logger.info("Vector store saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vectorstore(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            bool: True if successful
            
        Example:
            >>> pipeline.load_vectorstore()
            True
        """
        try:
            if not os.path.exists(self.vectorstore_path):
                logger.warning(f"Vector store path does not exist: {self.vectorstore_path}")
                return False
            
            logger.info(f"Loading vector store from {self.vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings
            )
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def clear_vectorstore(self) -> bool:
        """
        Clear the vector store (remove all documents).
        
        Returns:
            bool: True if successful
            
        Example:
            >>> pipeline.clear_vectorstore()
            True
        """
        try:
            logger.info("Clearing vector store...")
            self.vectorstore = None
            
            # Also delete the saved files
            if os.path.exists(self.vectorstore_path):
                import shutil
                shutil.rmtree(self.vectorstore_path)
                logger.info(f"Deleted vector store directory: {self.vectorstore_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store information
            
        Example:
            >>> stats = pipeline.get_stats()
            >>> print(f"Total documents: {stats['num_documents']}")
        """
        if self.vectorstore is None:
            return {
                "num_documents": 0,
                "embedding_model": self.embedding_model_name,
                "chunk_size": self.chunk_size,
                "status": "empty"
            }
        
        try:
            # Get index size (approximate number of documents)
            num_docs = self.vectorstore.index.ntotal
            
            return {
                "num_documents": num_docs,
                "embedding_model": self.embedding_model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"status": "error"}
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries at once.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            
        Returns:
            List of result lists
            
        Example:
            >>> queries = ["Query 1", "Query 2"]
            >>> all_results = pipeline.batch_retrieve(queries)
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_context_from_documents(
    documents: List[Dict],
    include_metadata: bool = True
) -> str:
    """
    Prepare a formatted context string from retrieved documents.
    
    This is used to create the context that gets passed to the LLM.
    
    Args:
        documents: List of document dicts from retrieve()
        include_metadata: Whether to include source information
        
    Returns:
        Formatted context string
        
    Example:
        >>> docs = pipeline.retrieve("query")
        >>> context = prepare_context_from_documents(docs)
        >>> print(context)
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        part = f"Document {i}:\n{doc['content']}"
        
        if include_metadata and doc.get('metadata'):
            source = doc['metadata'].get('source', 'Unknown')
            part += f"\n[Source: {source}]"
        
        context_parts.append(part)
    
    return "\n\n---\n\n".join(context_parts)


def split_documents(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split a large text into smaller chunks.
    
    This is the preprocessing step before adding to vector store.
    
    Args:
        text: Full text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
        
    Example:
        >>> chunks = split_documents("Large text...", chunk_size=500)
        >>> print(len(chunks))  # Number of chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
# ============================================================================
# Are you learning or just viewing ? -The End
# ============================================================================