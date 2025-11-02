"""
RAG Pipeline Module for Multimodel Chatbot

This module implements a complete Retrieval-Augmented Generation (RAG) pipeline
using LangChain 1.0.3 with LCEL (LangChain Expression Language) composition.

Features:
- Document ingestion and vector storage (FAISS)
- Semantic search with HuggingFace embeddings
- LCEL chain composition for RAG
- GPU detection and device optimization
- Token-aware text splitting (if available)
- Persistence (save/load vectorstore)
- Batch querying support
- Comprehensive logging and error handling

LangChain 1.0.3 Compatible - NO StrOutputParser
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# ============================================================================
# LANGCHAIN 1.0.3 IMPORTS (CORRECT - NO StrOutputParser)
# ============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

# Optional imports
try:
    from langchain_text_splitters import TokenTextSplitter
    TokenTextSplitter_available = True
except ImportError:
    TokenTextSplitter_available = False

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")

# ============================================================================
# SAFE OUTPUT PARSER (ONLY PARSER WE USE)
# ============================================================================

def safe_output_parser(output: Any) -> str:
    """
    Only parser needed - handles all response types safely.
    
    This is the ONLY output parser in the entire file.
    Prevents 'dict' object has no attribute 'replace' errors.
    """
    if isinstance(output, dict):
        for key in ["text", "output", "content", "response", "answer", "result"]:
            if key in output and output[key]:
                return str(output[key]).strip()
        for value in output.values():
            if value:
                return str(value).strip()
        return str(output)
    elif isinstance(output, str):
        return output.strip()
    else:
        return str(output).strip()

# ============================================================================
# RAG PIPELINE CLASS
# ============================================================================

class RAGPipelineLCEL:
    """Retrieval-Augmented Generation Pipeline using LCEL."""

    def __init__(
        self,
        vectorstore_path: str = DEFAULT_VECTORSTORE_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = "gpt-3.5-turbo",
        use_openai: bool = True,
        use_token_splitter: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize RAG Pipeline."""
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.use_openai = use_openai
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_token_splitter = use_token_splitter and TokenTextSplitter_available

        logger.info(
            f"Initializing RAGPipelineLCEL: "
            f"embedding_model={embedding_model}, "
            f"llm_model={llm_model}, "
            f"use_openai={use_openai}"
        )

        self.device = self._detect_device()
        logger.info(f"✅ Using device: {self.device}")

        self.embeddings = self._initialize_embeddings()
        self.llm = self._initialize_llm()

        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

        logger.info("✅ RAGPipelineLCEL initialized")

    def _detect_device(self) -> str:
        """Detect device."""
        if torch_available and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"[Device] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("[Device] Using CPU")
        return device

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings."""
        try:
            logger.info(f"[Embeddings] Loading: {self.embedding_model_name}")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                cache_folder=os.path.join(DEFAULT_DATA_DIR, ".cache"),
            )
            logger.info(f"✅ [Embeddings] Loaded")
            return embeddings
        except Exception as e:
            logger.error(f"❌ [Embeddings] Failed: {str(e)}")
            raise

    def _initialize_llm(self) -> Any:
        """Initialize LLM."""
        try:
            if self.use_openai:
                logger.info(f"[LLM] Loading OpenAI: {self.llm_model}")
                llm = ChatOpenAI(
                    model_name=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            else:
                logger.info(f"[LLM] Loading HuggingFace: {self.llm_model}")
                llm = HuggingFacePipeline(
                    model_id=self.llm_model,
                    model_kwargs={
                        "temperature": self.temperature,
                        "max_length": self.max_tokens,
                    },
                    pipeline_kwargs={
                        "max_new_tokens": self.max_tokens,
                    }
                )
            logger.info(f"✅ [LLM] Loaded")
            return llm
        except Exception as e:
            logger.error(f"❌ [LLM] Failed: {str(e)}")
            raise

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add documents."""
        if not documents:
            logger.warning("add_documents called with empty list")
            return False

        try:
            logger.info(f"[Documents] Adding {len(documents)} documents")
            doc_objects = []
            for i, text in enumerate(documents):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                doc_objects.append(Document(page_content=text, metadata=meta))

            chunks = self._chunk_documents(doc_objects)
            logger.info(f"[Documents] Split into {len(chunks)} chunks")

            if self.vectorstore is None:
                logger.info("[Vectorstore] Creating new FAISS index")
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            else:
                logger.info("[Vectorstore] Adding to existing index")
                self.vectorstore.add_documents(chunks)

            self._build_retriever()
            logger.info(f"✅ [Documents] Added {len(documents)} documents")
            return True

        except Exception as e:
            logger.error(f"❌ [Documents] Error: {str(e)}")
            return False

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents."""
        if self.use_token_splitter and TokenTextSplitter_available:
            logger.info("[Splitter] Using TokenTextSplitter")
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            logger.info("[Splitter] Using RecursiveCharacterTextSplitter")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

        chunks = splitter.split_documents(documents)
        logger.info(f"[Splitter] Created {len(chunks)} chunks")
        return chunks

    def _build_retriever(self):
        """Build retriever."""
        if self.vectorstore is None:
            logger.warning("Cannot build retriever: vectorstore is None")
            self.retriever = None
            return

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        logger.info(f"✅ [Retriever] Built (top_k={self.top_k})")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents."""
        if self.vectorstore is None:
            logger.warning("[Retrieval] Vectorstore is empty")
            return []

        k = top_k if top_k is not None else self.top_k

        try:
            logger.debug(f"[Retrieval] Searching: {query[:50]}...")
            hits = self.vectorstore.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in hits:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })

            logger.info(f"[Retrieval] Found {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"❌ [Retrieval] Error: {str(e)}")
            return []

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search."""
        results = self.retrieve(query, top_k)
        return [r["content"] for r in results]

    def build_rag_chain(self, system_prompt: Optional[str] = None) -> bool:
        """Build RAG chain."""
        if self.retriever is None:
            logger.error("[Chain] Cannot build: no retriever")
            return False

        try:
            logger.info("[Chain] Building RAG chain")

            if system_prompt is None:
                system_prompt = (
                    "You are a helpful assistant. Use the provided context to answer questions.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                )

            prompt = ChatPromptTemplate.from_template(system_prompt)

            def format_docs(docs: List[Document]) -> str:
                if not docs:
                    return "No relevant documents found."
                return "\n\n---\n\n".join(
                    getattr(d, "page_content", str(d)) for d in docs
                )

            # ✅ KEY FIX: Use safe_output_parser ONLY - NO StrOutputParser
            self.rag_chain = (
                {
                    "context": self.retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | RunnableLambda(safe_output_parser)
            )

            logger.info("✅ [Chain] RAG chain built successfully")
            return True

        except Exception as e:
            logger.error(f"❌ [Chain] Error: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def query_with_chain(self, question: str) -> str:
        """Query with chain."""
        if self.rag_chain is None:
            logger.error("[Query] RAG chain not initialized")
            return "Error: RAG chain not initialized"

        try:
            logger.info(f"[Query] Processing: {question[:50]}...")
            response = self.rag_chain.invoke(question)
            
            if not isinstance(response, str):
                response = str(response)
            
            logger.info("[Query] Response generated")
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ [Query] Error: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return f"Error: {str(e)}"

    def batch_query(self, questions: List[str]) -> List[str]:
        """Batch query."""
        if self.rag_chain is None:
            return ["Error: RAG chain not initialized"] * len(questions)

        logger.info(f"[Batch] Processing {len(questions)} questions")
        return [self.query_with_chain(q) for q in questions]

    def save_vectorstore(self) -> bool:
        """Save vectorstore."""
        try:
            if self.vectorstore is None:
                logger.warning("[Save] No vectorstore")
                return False

            Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(self.vectorstore_path)
            logger.info(f"✅ [Save] Saved to {self.vectorstore_path}")
            return True

        except Exception as e:
            logger.error(f"❌ [Save] Error: {str(e)}")
            return False

    def load_vectorstore(self) -> bool:
        """Load vectorstore."""
        try:
            if not os.path.exists(self.vectorstore_path):
                logger.warning(f"[Load] Path not found: {self.vectorstore_path}")
                return False

            logger.info(f"[Load] Loading from {self.vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._build_retriever()
            logger.info("✅ [Load] Loaded")
            return True

        except Exception as e:
            logger.error(f"❌ [Load] Error: {str(e)}")
            return False

    def clear_vectorstore(self) -> bool:
        """Clear vectorstore."""
        try:
            self.vectorstore = None
            self.retriever = None
            self.rag_chain = None

            if os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)
                logger.info(f"✅ [Clear] Cleared")

            return True

        except Exception as e:
            logger.error(f"❌ [Clear] Error: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        try:
            if self.vectorstore is None:
                return {
                    "status": "empty",
                    "num_documents": 0,
                    "embedding_model": self.embedding_model_name,
                    "llm_model": self.llm_model,
                    "rag_chain_ready": False,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            num_docs = getattr(self.vectorstore.index, "ntotal", None)

            return {
                "status": "active",
                "num_documents": int(num_docs) if num_docs is not None else 0,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model,
                "device": self.device,
                "rag_chain_ready": self.rag_chain is not None,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"[Stats] Error: {str(e)}")
            return {"status": "error"}

    def split_text(self, text: str) -> List[str]:
        """Split text."""
        if self.use_token_splitter and TokenTextSplitter_available:
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

        return splitter.split_text(text)


def prepare_context_from_documents(
    documents: List[Dict[str, Any]],
    include_metadata: bool = True,
    include_scores: bool = False,
) -> str:
    """Format documents."""
    if not documents:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        part = f"Document {i}:\n{content}"

        if include_scores:
            score = doc.get("score", 0)
            part += f"\n[Relevance: {score:.2f}]"

        if include_metadata and doc.get("metadata"):
            meta = doc["metadata"]
            source = meta.get("source", "Unknown")
            part += f"\n[Source: {source}]"

        parts.append(part)

    return "\n\n---\n\n".join(parts)

# ============================================================================
# END OF MODULE -
# ============================================================================