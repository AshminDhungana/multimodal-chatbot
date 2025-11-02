"""
Embeddings Module for Multimodel Chatbot

This module handles text embeddings and vector operations.
It provides:
- Loading and managing embedding models
- Converting text to vector embeddings
- Vector similarity calculations
- Batch embedding processing
- Model selection and switching

Embeddings are numerical representations of text that capture semantic meaning.
Similar texts have similar vector representations, enabling semantic search.

LangChain 1.0.3 Compatible
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod

# ============================================================================
# LANGCHAIN 1.0.3 IMPORTS (COMPATIBLE)
# ============================================================================

from langchain_huggingface import HuggingFaceEmbeddings

# Third-party imports
from sentence_transformers import SentenceTransformer, util
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Available embedding models with detailed information
EMBEDDING_MODELS = {
    # Small, fast models (good for production)
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM (Fast)",
        "dimensions": 384,
        "speed": "Very Fast",
        "quality": "Good",
        "memory": "Low",
        "recommended": "fast",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "name": "MPNet (Balanced)",
        "dimensions": 768,
        "speed": "Medium",
        "quality": "Excellent",
        "memory": "Medium",
        "recommended": "balanced",
    },
    "BAAI/bge-small-en-v1.5": {
        "name": "BGE-Small (Fast & Quality)",
        "dimensions": 384,
        "speed": "Very Fast",
        "quality": "Excellent",
        "memory": "Low",
        "recommended": "production",
    },
    "BAAI/bge-base-en-v1.5": {
        "name": "BGE-Base (Best Quality)",
        "dimensions": 768,
        "speed": "Medium",
        "quality": "Best",
        "memory": "Medium",
        "recommended": "best",
    },
    # Larger models (better quality, slower)
    "sentence-transformers/all-distilroberta-v1": {
        "name": "DistilRoBERTa (Good Balance)",
        "dimensions": 768,
        "speed": "Medium",
        "quality": "Very Good",
        "memory": "Medium",
        "recommended": "quality",
    },
}

# Default model from environment or fallback
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================================
# BASE EMBEDDING CLASS
# ============================================================================

class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models.
    
    Defines the interface that all embedding implementations must follow.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Convert single text to embedding."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embeddings."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        pass


# ============================================================================
# SENTENCE TRANSFORMER EMBEDDING
# ============================================================================

class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Embedding using Sentence Transformers with LangChain 1.0.3 support.
    
    Sentence Transformers are specialized models trained to produce
    meaningful embeddings for sentences and paragraphs.
    
    Can use either sentence-transformers directly or via LangChain's
    langchain_huggingface wrapper for better integration.
    
    Attributes:
        model_name: Identifier of the model
        model: Loaded SentenceTransformer instance
        device: CPU or CUDA device
        embedding_dimension: Size of output vectors
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = None,
        normalize_embeddings: bool = True,
        use_langchain: bool = False,
    ):
        """
        Initialize Sentence Transformer Embedding.
        
        Args:
            model_name: Model identifier from HuggingFace
            device: "cpu" or "cuda" (auto-detect if None)
            normalize_embeddings: Whether to normalize vectors to unit length
            use_langchain: Use LangChain's HuggingFaceEmbeddings wrapper
            
        Example:
            >>> embedding = SentenceTransformerEmbedding()
            >>> vector = embedding.embed("Hello world")
            >>> print(vector.shape)  # (384,)
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_langchain = use_langchain

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Loading embedding model: {model_name} on {self.device} "
            f"({'via LangChain' if use_langchain else 'direct'})"
        )

        try:
            if use_langchain:
                # Use LangChain's wrapper (LangChain 1.0.3 compatible)
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "normalize_embeddings": normalize_embeddings
                    }
                )
                # Get embedding dimension
                dummy_embedding = self.model.embed_query("test")
                self.embedding_dimension = len(dummy_embedding)
                logger.info(
                    f"✅ Model loaded via LangChain. Dimensions: {self.embedding_dimension}"
                )

            else:
                # Use sentence-transformers directly
                self.model = SentenceTransformer(
                    model_name,
                    device=self.device
                )

                # Get embedding dimension
                dummy_embedding = self.model.encode("test")
                self.embedding_dimension = len(dummy_embedding)
                logger.info(
                    f"✅ Model loaded directly. Dimensions: {self.embedding_dimension}"
                )

        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            raise

    def embed(self, text: str) -> np.ndarray:
        """
        Convert single text to embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Example:
            >>> embedding_model = SentenceTransformerEmbedding()
            >>> vector = embedding_model.embed("Hello world")
            >>> print(vector.shape)  # (384,)
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(self.embedding_dimension)

            # Encode text
            if self.use_langchain:
                embedding = self.model.embed_query(text)
            else:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings
                )

            return np.array(embedding)

        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Convert multiple texts to embeddings efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            2D numpy array of embeddings (N x D)
            
        Example:
            >>> embedding_model = SentenceTransformerEmbedding()
            >>> texts = ["Hello", "World", "Test"]
            >>> embeddings = embedding_model.embed_batch(texts)
            >>> print(embeddings.shape)  # (3, 384)
        """
        try:
            if not texts:
                logger.warning("Empty text list provided")
                return np.array([])

            logger.info(f"Embedding batch of {len(texts)} texts")

            # Filter empty texts
            texts = [t if t and t.strip() else "." for t in texts]

            # Encode batch
            if self.use_langchain:
                embeddings = self.model.embed_documents(texts)
                embeddings = np.array(embeddings)
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=show_progress
                )

            logger.info(f"✅ Successfully embedded {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dimension

    def get_model_name(self) -> str:
        """Return model name."""
        return self.model_name


# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    """
    Manages embedding models and operations.
    
    This is the main interface for working with embeddings.
    Handles:
    - Loading models (supports both direct and LangChain modes)
    - Switching between models
    - Embedding text
    - Vector operations (similarity, distance)
    - Batch processing
    - Caching
    
    Attributes:
        current_model: Currently active embedding model
        embedding_cache: Cache of recent embeddings
        available_models: Dictionary of available models
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = None,
        cache_size: int = 1000,
        normalize_embeddings: bool = True,
        use_langchain: bool = False,
    ):
        """
        Initialize Embedding Manager.
        
        Args:
            model_name: Model to load initially (from .env or default)
            device: "cpu" or "cuda"
            cache_size: Number of embeddings to cache
            normalize_embeddings: Normalize vectors to unit length
            use_langchain: Use LangChain's wrapper (LangChain 1.0.3)
            
        Example:
            >>> manager = EmbeddingManager()
            >>> vector = manager.embed("Hello world")
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_langchain = use_langchain

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size

        # Load initial model
        self.current_model: Optional[BaseEmbedding] = None
        self._load_model(model_name)

        logger.info(
            f"✅ EmbeddingManager initialized with {model_name} "
            f"(cache_size={cache_size})"
        )

    def _load_model(self, model_name: str) -> bool:
        """
        Load an embedding model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")

            self.current_model = SentenceTransformerEmbedding(
                model_name=model_name,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
                use_langchain=self.use_langchain,
            )

            self.model_name = model_name
            self.embedding_cache.clear()

            logger.info(f"✅ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text (with caching).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Example:
            >>> manager = EmbeddingManager()
            >>> vector = manager.embed("Hello world")
            >>> print(vector.shape)  # (384,)
        """
        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension())

        # Check cache
        if text in self.embedding_cache:
            logger.debug("Cache hit for text embedding")
            return self.embedding_cache[text]

        # Generate embedding
        embedding = self.current_model.embed(text)

        # Cache it (with size limit)
        if len(self.embedding_cache) < self.cache_size:
            self.embedding_cache[text] = embedding

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts
            batch_size: Texts to process at once
            show_progress: Show progress bar
            
        Returns:
            2D array of embeddings (N x D)
            
        Example:
            >>> manager = EmbeddingManager()
            >>> texts = ["Hello", "World"]
            >>> embeddings = manager.embed_batch(texts)
            >>> print(embeddings.shape)  # (2, 384)
        """
        return self.current_model.embed_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

    # ========================================================================
    # SIMILARITY AND DISTANCE OPERATIONS
    # ========================================================================

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1, where 1 is identical)
            
        Example:
            >>> manager = EmbeddingManager()
            >>> v1 = manager.embed("Hello")
            >>> v2 = manager.embed("Hi")
            >>> sim = manager.similarity(v1, v2)
            >>> print(f"Similarity: {sim:.2f}")  # 0.85
        """
        try:
            # Ensure 1D arrays
            if embedding1.ndim == 2:
                embedding1 = embedding1[0]
            if embedding2.ndim == 2:
                embedding2 = embedding2[0]

            # Calculate cosine similarity
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

            # Convert to scalar
            if isinstance(similarity_score, torch.Tensor):
                return float(similarity_score.item())
            return float(similarity_score)

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_similar(
        self,
        query_embedding: np.ndarray,
        embeddings_list: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query vector
            embeddings_list: Array of vectors to search
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
            
        Example:
            >>> manager = EmbeddingManager()
            >>> query = manager.embed("machine learning")
            >>> embeddings = manager.embed_batch(["AI", "ML", "DL"])
            >>> results = manager.find_similar(query, embeddings, top_k=2)
            >>> for idx, score in results:
            ...     print(f"Index {idx}: {score:.2f}")
        """
        try:
            if query_embedding.ndim == 1:
                query_embedding = torch.from_numpy(query_embedding).unsqueeze(0)
            
            embeddings_tensor = torch.from_numpy(embeddings_list)

            # Calculate similarities
            similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]

            # Get top-k
            top_k_scores, top_k_indices = torch.topk(
                similarities, 
                k=min(top_k, len(similarities))
            )

            # Return as list of tuples
            results = [
                (int(idx), float(score))
                for idx, score in zip(top_k_indices, top_k_scores)
            ]

            return results

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {str(e)}")
            return []

    def distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate Euclidean distance between embeddings.
        
        Args:
            embedding1: First vector
            embedding2: Second vector
            
        Returns:
            Distance (0 is identical)
            
        Example:
            >>> manager = EmbeddingManager()
            >>> v1 = manager.embed("Hello")
            >>> v2 = manager.embed("World")
            >>> dist = manager.distance(v1, v2)
            >>> print(f"Distance: {dist:.2f}")
        """
        try:
            return float(np.linalg.norm(embedding1 - embedding2))
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return float('inf')

    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================

    def set_model(self, model_name: str) -> bool:
        """
        Switch to a different embedding model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            bool: True if successful
            
        Example:
            >>> manager = EmbeddingManager()
            >>> manager.set_model("BAAI/bge-base-en-v1.5")
        """
        if model_name not in EMBEDDING_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            return False

        return self._load_model(model_name)

    def get_available_models(self) -> Dict[str, Dict]:
        """
        Get all available embedding models.
        
        Returns:
            Dictionary of models with info
            
        Example:
            >>> manager = EmbeddingManager()
            >>> models = manager.get_available_models()
            >>> for name, info in models.items():
            ...     print(f"{info['name']}: {info['dimensions']}D")
        """
        return EMBEDDING_MODELS.copy()

    def get_current_model_info(self) -> Dict:
        """
        Get info about current model.
        
        Returns:
            Dictionary with model information
            
        Example:
            >>> manager = EmbeddingManager()
            >>> info = manager.get_current_model_info()
            >>> print(f"Dimensions: {info['dimensions']}")
        """
        if self.model_name not in EMBEDDING_MODELS:
            return {
                "name": "Unknown",
                "dimensions": 0,
                "status": "error"
            }

        model_info = EMBEDDING_MODELS[self.model_name].copy()
        model_info["status"] = "loaded"
        model_info["device"] = self.device
        model_info["use_langchain"] = self.use_langchain

        return model_info

    def get_embedding_dimension(self) -> int:
        """Get dimension of current embeddings."""
        if self.current_model is None:
            return 0
        return self.current_model.get_embedding_dimension()

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model_name

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("✅ Embedding cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_items": len(self.embedding_cache),
            "max_cache_size": self.cache_size,
            "cache_full": len(self.embedding_cache) >= self.cache_size,
            "cache_utilization": f"{len(self.embedding_cache) / self.cache_size * 100:.1f}%"
        }


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

class VectorOperations:
    """
    Utility class for vector operations.
    
    Provides common operations on embedding vectors.
    """

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def cosine_similarity_matrix(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            
        Returns:
            Similarity matrix (N x M)
        """
        # Normalize
        e1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        e2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Compute similarities
        similarity = np.dot(e1_norm, e2_norm.T)
        return similarity

    @staticmethod
    def mean_pooling(embeddings: np.ndarray) -> np.ndarray:
        """
        Get mean of multiple embeddings.
        
        Args:
            embeddings: 2D array of embeddings
            
        Returns:
            Mean embedding
        """
        return np.mean(embeddings, axis=0)

    @staticmethod
    def get_centroid(embeddings: np.ndarray) -> np.ndarray:
        """
        Get centroid of embeddings (same as mean pooling).
        
        Args:
            embeddings: 2D array of embeddings
            
        Returns:
            Centroid embedding
        """
        return VectorOperations.mean_pooling(embeddings)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding_model_recommendations() -> Dict[str, str]:
    """
    Get recommended models for different use cases.
    
    Returns:
        Dictionary of recommendations
    """
    return {
        "best_quality": "BAAI/bge-base-en-v1.5",
        "best_balance": "sentence-transformers/all-mpnet-base-v2",
        "fastest": "BAAI/bge-small-en-v1.5",
        "low_memory": "sentence-transformers/all-MiniLM-L6-v2",
    }


def compare_embeddings(
    embedding_manager: EmbeddingManager,
    texts: List[str]
) -> Dict:
    """
    Compare embeddings and their similarities.
    
    Useful for debugging and understanding embeddings.
    
    Args:
        embedding_manager: Initialized manager
        texts: List of texts to compare
        
    Returns:
        Dictionary with comparison results
    """
    try:
        embeddings = embedding_manager.embed_batch(texts)

        comparison = {
            "texts": texts,
            "model": embedding_manager.get_model_name(),
            "dimensions": embedding_manager.get_embedding_dimension(),
            "similarity_matrix": VectorOperations.cosine_similarity_matrix(
                embeddings,
                embeddings
            ).tolist()
        }

        return comparison

    except Exception as e:
        logger.error(f"Error comparing embeddings: {str(e)}")
        return {}


# ============================================================================
# END OF MODULE
# ============================================================================
