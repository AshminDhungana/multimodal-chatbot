"""
LLM Models Module for Multimodel Chatbot (LCEL-based)

This module provides LLM chain wrappers for both OpenAI and Hugging Face models,
with LCEL (LangChain Expression Language) composition and async support.

Features:
- OpenAI and Hugging Face LLM wrappers with LCEL chain composition
- Sync and async invocation (invoke/ainvoke)
- Model switching and optional caching
- ConversationManager with export/load
- CompositeChainBuilder for RAG + memory + LLM composition
- Safe fallback behavior and comprehensive logging
- Full support for .env configuration (MODEL_TYPE, DEFAULT_MODEL, etc.)

LangChain 1.0.3 Compatible
"""

import os
import logging
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime

# ============================================================================
# LANGCHAIN 1.0.3 IMPORTS (COMPATIBLE)
# ============================================================================

# Core LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Optional memory imports (graceful fallback)
try:
    from langchain_core.memory import ConversationBufferMemory
    ConversationBufferMemory_available = True
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
        ConversationBufferMemory_available = True
    except ImportError:
        ConversationBufferMemory = None
        ConversationBufferMemory_available = False

# Optional diskcache for persistent caching
try:
    from diskcache import Cache
    _diskcache_available = True
except ImportError:
    _diskcache_available = False

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Available models
LOCAL_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B (Fast & Good)",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2 7B (Stable)",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2 13B (Better)",
    "tiiuae/falcon-7b-instruct": "Falcon 7B (Efficient)",
    "EleutherAI/gpt-j-6B": "GPT-J 6B (Capable)",
}

OPENAI_MODELS = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast & Cheap)",
    "gpt-4": "GPT-4 (Most Capable)",
    "gpt-4o-mini": "GPT-4o-mini (Fastest)",
}

# Load from .env
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "hybrid")  # local, openai, or hybrid
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# ============================================================================
# BASE LLM CHAIN CLASS
# ============================================================================

class BaseLLMChain(ABC):
    """Abstract base class for LLM chain wrappers."""

    @abstractmethod
    def get_chain(self) -> Any:
        """Return the LCEL runnable chain object."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name."""
        raise NotImplementedError

    @abstractmethod
    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Synchronous invocation."""
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(self, input_data: Dict[str, Any]) -> str:
        """Asynchronous invocation."""
        raise NotImplementedError

    @abstractmethod
    def batch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch synchronous invocation."""
        raise NotImplementedError

    @abstractmethod
    async def abatch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch asynchronous invocation."""
        raise NotImplementedError


# ============================================================================
# OPENAI LLM CHAIN
# ============================================================================

class OpenAILLMChain(BaseLLMChain):
    """
    OpenAI LLM chain wrapper with LCEL composition.
    
    Uses ChatOpenAI from langchain_openai (LangChain 1.0.3 compatible).
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenAI LLM chain.
        
        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            system_prompt: System message for the model
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not set. Provide api_key or set OPENAI_API_KEY environment variable."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or "You are a helpful, accurate, and concise assistant."

        logger.info(f"[OpenAI] Initializing ChatOpenAI: {model_name}")

        try:
            # Initialize ChatOpenAI (LangChain 1.0.3)
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                streaming=False,
            )

            self._build_chain()
            logger.info(f"✅ [OpenAI] ChatOpenAI initialized successfully")

        except Exception as e:
            logger.error(f"❌ [OpenAI] Failed to initialize: {str(e)}")
            raise

    def _build_chain(self):
        """Build LCEL chain: prompt -> llm -> parser."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def get_chain(self) -> Any:
        return self.chain

    def get_model_name(self) -> str:
        return self.model_name

    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke chain synchronously."""
        try:
            result = self.chain.invoke(input_data)
            return str(result).strip()
        except Exception as e:
            logger.error(f"[OpenAI] Error invoking chain: {str(e)}")
            raise

    async def ainvoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke chain asynchronously."""
        try:
            if hasattr(self.chain, "ainvoke"):
                result = await self.chain.ainvoke(input_data)
                return str(result).strip()
            else:
                # Fallback: run sync in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.invoke, input_data)
        except Exception as e:
            logger.error(f"[OpenAI] Error in async invoke: {str(e)}")
            raise

    def batch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch invoke synchronously."""
        try:
            if hasattr(self.chain, "batch"):
                results = self.chain.batch(input_list)
                return [str(r).strip() for r in results]
            else:
                return [self.invoke(inp) for inp in input_list]
        except Exception as e:
            logger.error(f"[OpenAI] Error in batch invoke: {str(e)}")
            raise

    async def abatch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch invoke asynchronously."""
        try:
            if hasattr(self.chain, "abatch"):
                results = await self.chain.abatch(input_list)
                return [str(r).strip() for r in results]
            else:
                # Fallback: run each invoke in executor
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(None, self.invoke, inp) for inp in input_list]
                return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[OpenAI] Error in async batch invoke: {str(e)}")
            raise


# ============================================================================
# HUGGING FACE LLM CHAIN
# ============================================================================

class HuggingFaceLLMChain(BaseLLMChain):
    """
    Hugging Face LLM chain wrapper with LCEL composition.
    
    Supports both local inference (via HuggingFacePipeline) and remote
    inference (via HuggingFaceEndpoint).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
        max_tokens: int = 512,
        hf_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_local: bool = False,
    ):
        """
        Initialize Hugging Face LLM chain.
        
        Args:
            model_name: Model ID from Hugging Face Hub
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            hf_token: Hugging Face API token (for remote inference)
            system_prompt: System message
            use_local: Use local inference (requires GPU/VRAM)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = use_local
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.llm = None
        self.chain = None

        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

        logger.info(f"[HF] Initializing model: {model_name} (local={use_local})")

        try:
            if use_local:
                # Local inference via HuggingFacePipeline
                logger.info("[HF] Using local HuggingFacePipeline")
                self.llm = HuggingFacePipeline(
                    model_id=model_name,
                    model_kwargs={
                        "temperature": temperature,
                        "max_length": max_tokens,
                    },
                    pipeline_kwargs={
                        "max_new_tokens": max_tokens,
                    }
                )
            else:
                # Remote inference via Hugging Face Inference API
                if not hf_token:
                    raise ValueError(
                        "Hugging Face token required for remote inference. "
                        "Set HUGGINGFACE_TOKEN or use use_local=True"
                    )
                logger.info("[HF] Using HuggingFaceEndpoint (remote)")
                self.llm = HuggingFaceEndpoint(
                    repo_id=model_name,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    huggingfacehub_api_token=hf_token,
                    task="text-generation",
                )

            self._build_chain()
            logger.info(f"✅ [HF] Model initialized successfully")

        except Exception as e:
            logger.error(f"❌ [HF] Failed to initialize: {str(e)}")
            self.llm = None
            self.chain = None

    def _build_chain(self):
        """Build LCEL chain with system prompt."""
        if self.llm is None:
            logger.warning("[HF] LLM not initialized; chain not built")
            return

        # Use PromptTemplate with system prompt injection
        self.prompt_template = PromptTemplate.from_template(
            "{system_prompt}\n\nUser: {input}\n\nAssistant:"
        )

        # Compose: inject system_prompt + format + llm + parser
        self.chain = (
            {
                "system_prompt": RunnableLambda(lambda _: self.system_prompt),
                "input": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def get_chain(self) -> Any:
        if self.chain is None:
            raise RuntimeError("Hugging Face chain not initialized")
        return self.chain

    def get_model_name(self) -> str:
        return self.model_name

    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke chain synchronously."""
        if self.chain is None:
            raise RuntimeError("Hugging Face chain not initialized")
        try:
            result = self.chain.invoke(input_data)
            return str(result).strip()
        except Exception as e:
            logger.error(f"[HF] Error invoking chain: {str(e)}")
            raise

    async def ainvoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke chain asynchronously."""
        if self.chain is None:
            raise RuntimeError("Hugging Face chain not initialized")
        try:
            if hasattr(self.chain, "ainvoke"):
                result = await self.chain.ainvoke(input_data)
                return str(result).strip()
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.invoke, input_data)
        except Exception as e:
            logger.error(f"[HF] Error in async invoke: {str(e)}")
            raise

    def batch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch invoke synchronously."""
        if self.chain is None:
            raise RuntimeError("Hugging Face chain not initialized")
        try:
            if hasattr(self.chain, "batch"):
                results = self.chain.batch(input_list)
                return [str(r).strip() for r in results]
            else:
                return [self.invoke(inp) for inp in input_list]
        except Exception as e:
            logger.error(f"[HF] Error in batch invoke: {str(e)}")
            raise

    async def abatch_invoke(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Batch invoke asynchronously."""
        if self.chain is None:
            raise RuntimeError("Hugging Face chain not initialized")
        try:
            if hasattr(self.chain, "abatch"):
                results = await self.chain.abatch(input_list)
                return [str(r).strip() for r in results]
            else:
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(None, self.invoke, inp) for inp in input_list]
                return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[HF] Error in async batch invoke: {str(e)}")
            raise


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """
    Manages conversation history with optional LangChain memory integration.
    
    Features:
    - Store user/assistant message pairs
    - Export/load conversation to/from JSON
    - Get conversation context for RAG
    - Integrate with LangChain memory (if available)
    """

    def __init__(
        self,
        max_history: int = 10,
        use_memory: bool = False,
    ):
        """
        Initialize ConversationManager.
        
        Args:
            max_history: Maximum number of messages to keep
            use_memory: Enable LangChain memory integration
        """
        self.history: List[Tuple[str, str]] = []
        self.max_history = max_history
        self.use_memory = use_memory and ConversationBufferMemory_available

        if self.use_memory and ConversationBufferMemory is not None:
            try:
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    human_prefix="User",
                    ai_prefix="Assistant"
                )
                logger.info("[ConvMgr] LangChain memory integration enabled")
            except Exception as e:
                logger.warning(f"[ConvMgr] Failed to initialize memory: {str(e)}")
                self.memory = None
        else:
            self.memory = None

        logger.info(f"[ConvMgr] Initialized (max_history={max_history})")

    def add_message(self, user_msg: str, assistant_msg: str):
        """Add user and assistant messages to history."""
        self.history.append((user_msg, assistant_msg))

        # Keep only recent messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Add to LangChain memory if available
        if self.memory:
            try:
                self.memory.save_context(
                    {"input": user_msg},
                    {"output": assistant_msg}
                )
            except Exception as e:
                logger.debug(f"[ConvMgr] Failed to save to memory: {str(e)}")

    def get_history(self) -> List[Tuple[str, str]]:
        """Get all messages as list of (user, assistant) tuples."""
        return self.history.copy()

    def get_history_as_messages(self) -> List[Any]:
        """Get history as LangChain Message objects."""
        messages = []
        for user_msg, assistant_msg in self.history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=assistant_msg))
        return messages

    def get_recent_context(self, num_messages: int = 3) -> str:
        """Get recent conversation as formatted text."""
        recent = self.history[-num_messages:]
        lines = []
        for user_msg, asst_msg in recent:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {asst_msg}")
        return "\n".join(lines).strip()

    def clear_history(self):
        """Clear all conversation history."""
        self.history = []
        if self.memory:
            try:
                self.memory.clear()
            except Exception as e:
                logger.debug(f"[ConvMgr] Failed to clear memory: {str(e)}")
        logger.info("[ConvMgr] History cleared")

    def export_to_json(self, path: str):
        """Export conversation to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ [ConvMgr] Exported to {path}")
        except Exception as e:
            logger.error(f"[ConvMgr] Export failed: {str(e)}")
            raise

    def load_from_json(self, path: str):
        """Load conversation from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            logger.info(f"✅ [ConvMgr] Loaded from {path}")
        except Exception as e:
            logger.error(f"[ConvMgr] Load failed: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_user_chars = sum(len(u) for u, _ in self.history)
        total_asst_chars = sum(len(a) for _, a in self.history)
        return {
            "total_exchanges": len(self.history),
            "total_user_characters": total_user_chars,
            "total_assistant_characters": total_asst_chars,
            "average_user_length": total_user_chars / len(self.history) if self.history else 0,
            "average_assistant_length": total_asst_chars / len(self.history) if self.history else 0,
            "max_history": self.max_history,
        }


# ============================================================================
# COMPOSITE CHAIN BUILDER
# ============================================================================

class CompositeChainBuilder:
    """Builder for composing RAG + Memory + LLM chains."""

    @staticmethod
    def build_rag_with_memory_chain(
        llm_chain: Any,
        retriever_chain: Optional[Any] = None,
        conversation_manager: Optional[ConversationManager] = None,
        include_context_label: str = "context"
    ) -> Any:
        """
        Build composite chain: RAG retriever -> inject context -> add memory -> LLM.
        
        Args:
            llm_chain: Base LCEL chain or BaseLLMChain instance
            retriever_chain: Optional LCEL retriever (if None, skip RAG)
            conversation_manager: Optional ConversationManager for memory
            include_context_label: Key name for context in chain
            
        Returns:
            Composed LCEL chain
        """
        # Extract chain if BaseLLMChain
        if isinstance(llm_chain, BaseLLMChain):
            llm_chain = llm_chain.get_chain()

        # Helper to format retrieved documents
        def format_docs(docs):
            if isinstance(docs, list):
                try:
                    return "\n\n---\n\n".join(
                        getattr(d, "page_content", str(d)) for d in docs
                    )
                except Exception:
                    return "\n\n---\n\n".join(str(d) for d in docs)
            return str(docs)

        # Build input with context and memory
        def build_input_with_context(data: Dict[str, Any]) -> Dict[str, Any]:
            query = data.get("query") or data.get("input") or ""
            parts = []

            # Add RAG context
            context = data.get(include_context_label) or data.get("context") or ""
            if context:
                formatted_context = format_docs(context) if isinstance(context, list) else str(context)
                parts.append(f"Context:\n{formatted_context}")

            # Add conversation memory
            if conversation_manager:
                recent = conversation_manager.get_recent_context(3)
                if recent:
                    parts.append(f"Previous Conversation:\n{recent}")

            # Combine and create final input
            combined = "\n\n".join(parts)
            final_input = f"{combined}\n\nQuery: {query}" if combined else query

            return {"input": final_input}

        # Build chain
        if retriever_chain:
            # With retriever: retrieve -> format -> combine -> invoke LLM
            return (
                {
                    include_context_label: retriever_chain,
                    "query": RunnablePassthrough(),
                }
                | RunnableLambda(build_input_with_context)
                | llm_chain
            )
        else:
            # Without retriever: just add memory -> invoke LLM
            return (
                RunnableLambda(build_input_with_context)
                | llm_chain
            )


# ============================================================================
# LLM MANAGER (HIGH-LEVEL API)
# ============================================================================

class LLMManager:
    """
    High-level manager for loading, switching, and invoking LLM models.
    
    Features:
    - Load models from .env configuration
    - Switch between OpenAI and Hugging Face models
    - Caching for model instances
    - Fallback behavior
    - Optional disk caching
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        default_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_diskcache: bool = False,
        diskcache_dir: str = ".model_cache",
    ):
        """
        Initialize LLMManager.
        
        Args:
            model_type: "local", "openai", or "hybrid" (uses .env if None)
            default_model: Model to load initially (uses .env if None)
            temperature: Sampling temperature (uses .env if None)
            max_tokens: Max tokens in response (uses .env if None)
            system_prompt: System message for model
            use_diskcache: Enable persistent caching
            diskcache_dir: Directory for disk cache
        """
        # Use .env values if not provided
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        self.default_model = default_model or DEFAULT_MODEL_NAME
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self.system_prompt = system_prompt

        # Initialize model registry
        self.available_models: Dict[str, str] = {}
        self._initialize_available_models()

        # Cache for loaded model instances
        self.chain_instances: Dict[str, BaseLLMChain] = {}
        self.current_chain_instance: Optional[BaseLLMChain] = None

        # Optional disk cache
        self.use_diskcache = use_diskcache and _diskcache_available
        self.diskcache = Cache(diskcache_dir) if self.use_diskcache else None

        logger.info(
            f"[LLMManager] Initialized (type={self.model_type}, "
            f"default={self.default_model}, temp={self.temperature}, "
            f"diskcache={self.use_diskcache})"
        )

        # Load default model
        self.set_model(self.default_model)

    def _initialize_available_models(self):
        """Populate available models based on model_type."""
        if self.model_type in ["openai", "hybrid"]:
            self.available_models.update(OPENAI_MODELS)
        if self.model_type in ["local", "hybrid"]:
            self.available_models.update(LOCAL_MODELS)

    def set_model(
        self,
        model_name: str,
        use_local: bool = False,
    ) -> bool:
        """
        Load a model and cache it.
        
        Args:
            model_name: Model identifier
            use_local: For HF models, use local inference
            
        Returns:
            True if successful
        """
        logger.info(f"[LLMManager] set_model: {model_name} (use_local={use_local})")

        # Return cached instance if available
        if model_name in self.chain_instances:
            self.current_chain_instance = self.chain_instances[model_name]
            logger.info(f"[LLMManager] Loaded cached instance for {model_name}")
            return True

        try:
            # Determine model type and create instance
            if model_name in OPENAI_MODELS:
                logger.info(f"[LLMManager] Loading OpenAI model: {model_name}")
                instance = OpenAILLMChain(
                    model_name=model_name,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    system_prompt=self.system_prompt,
                )
            elif model_name in LOCAL_MODELS:
                logger.info(f"[LLMManager] Loading Hugging Face model: {model_name}")
                instance = HuggingFaceLLMChain(
                    model_name=model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    hf_token=os.getenv("HUGGINGFACE_TOKEN"),
                    system_prompt=self.system_prompt,
                    use_local=use_local,
                )
                if instance.chain is None:
                    logger.warning(f"[LLMManager] Chain not initialized for {model_name}")
                    return False
            else:
                logger.warning(f"[LLMManager] Unknown model: {model_name}")
                return False

            # Cache and activate
            self.chain_instances[model_name] = instance
            self.current_chain_instance = instance
            logger.info(f"✅ [LLMManager] Model loaded: {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ [LLMManager] Failed to set model {model_name}: {str(e)}")
            return False

    def get_chain(self) -> Any:
        """Get current LCEL chain."""
        if self.current_chain_instance is None:
            raise RuntimeError("No model loaded. Call set_model() first.")
        return self.current_chain_instance.get_chain()

    def invoke(
        self,
        query: str,
        context: str = "",
        conversation: Optional[ConversationManager] = None,
    ) -> str:
        """
        Invoke model synchronously.
        
        Args:
            query: User query
            context: Optional context/RAG results
            conversation: Optional ConversationManager for history
            
        Returns:
            Model response
        """
        if self.current_chain_instance is None:
            raise RuntimeError("No model loaded.")

        input_text = self._build_input(query, context, conversation)

        try:
            return self.current_chain_instance.invoke({"input": input_text})
        except Exception as e:
            logger.warning(f"[LLMManager] Primary model failed: {str(e)}")
            # Try fallback to local model if hybrid
            if self.model_type == "hybrid":
                for model_name, instance in self.chain_instances.items():
                    if model_name in LOCAL_MODELS:
                        try:
                            logger.info(f"[LLMManager] Trying fallback: {model_name}")
                            self.current_chain_instance = instance
                            return instance.invoke({"input": input_text})
                        except Exception:
                            continue
            raise

    async def ainvoke(
        self,
        query: str,
        context: str = "",
        conversation: Optional[ConversationManager] = None,
    ) -> str:
        """Invoke model asynchronously."""
        if self.current_chain_instance is None:
            raise RuntimeError("No model loaded.")

        input_text = self._build_input(query, context, conversation)

        try:
            return await self.current_chain_instance.ainvoke({"input": input_text})
        except Exception as e:
            logger.warning(f"[LLMManager] Primary model failed (async): {str(e)}")
            if self.model_type == "hybrid":
                for model_name, instance in self.chain_instances.items():
                    if model_name in LOCAL_MODELS:
                        try:
                            self.current_chain_instance = instance
                            return await instance.ainvoke({"input": input_text})
                        except Exception:
                            continue
            raise

    def batch_invoke(self, queries: List[str], context: str = "") -> List[str]:
        """Batch invoke synchronously."""
        if self.current_chain_instance is None:
            raise RuntimeError("No model loaded.")

        inputs = [{"input": self._build_input(q, context, None)} for q in queries]
        return self.current_chain_instance.batch_invoke(inputs)

    async def abatch_invoke(self, queries: List[str], context: str = "") -> List[str]:
        """Batch invoke asynchronously."""
        if self.current_chain_instance is None:
            raise RuntimeError("No model loaded.")

        inputs = [{"input": self._build_input(q, context, None)} for q in queries]
        return await self.current_chain_instance.abatch_invoke(inputs)

    def _build_input(
        self,
        query: str,
        context: str = "",
        conversation: Optional[ConversationManager] = None,
    ) -> str:
        """Build input with context and conversation."""
        parts = []

        if context:
            parts.append(f"Context:\n{context}")

        if conversation:
            recent = conversation.get_recent_context(3)
            if recent:
                parts.append(f"Conversation:\n{recent}")

        parts.append(f"Query: {query}")
        return "\n\n".join(parts)

    def get_current_model_name(self) -> str:
        """Get current model name."""
        if self.current_chain_instance is None:
            return "No model loaded"
        return self.current_chain_instance.get_model_name()

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.available_models.keys())

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "model_type": self.model_type,
            "current_model": self.get_current_model_name(),
            "available_models": self.get_available_models(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cached_models": len(self.chain_instances),
            "diskcache_enabled": self.use_diskcache,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# END OF MODULE
# ============================================================================
