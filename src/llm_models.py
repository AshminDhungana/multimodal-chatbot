"""
LLM Models Module for Multimodel Chatbot

This module manages language models and response generation.
It handles:
- Loading and switching between different LLMs
- OpenAI GPT models (GPT-3.5, GPT-4)
- Local Hugging Face models (Mistral, Llama-2, etc.)
- Conversation memory and context management
- Response generation with RAG context

The module supports both cloud-based (OpenAI) and local models,
allowing users to switch between them seamlessly.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

# Third-party imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Available local models (from Hugging Face)
LOCAL_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B (Fast & Good)",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2 7B (Stable)",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2 13B (Better)",
    "tiiuae/falcon-7b-instruct": "Falcon 7B (Efficient)",
    "EleutherAI/gpt-j-6B": "GPT-J 6B (Capable)",
}

# OpenAI models
OPENAI_MODELS = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast & Cheap)",
    "gpt-4": "GPT-4 (Most Capable)",
    "gpt-4-turbo": "GPT-4 Turbo (Better)",
}


# ============================================================================
# BASE LLM CLASS
# ============================================================================

class BaseLLM(ABC):
    """
    Abstract base class for language models.
    
    Defines the interface that all models must implement.
    """
    
    @abstractmethod
    def generate_response(
        self,
        query: str,
        context: str = "",
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """Generate a response based on query and context."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the current model."""
        pass


# ============================================================================
# OPENAI LLM CLASS
# ============================================================================

class OpenAILLM(BaseLLM):
    """
    OpenAI Language Model wrapper.
    
    Handles GPT-3.5 and GPT-4 models from OpenAI API.
    
    Attributes:
        model_name (str): Model identifier (e.g., "gpt-3.5-turbo")
        temperature (float): Creativity level (0.0-1.0)
        max_tokens (int): Max response length
        llm: LangChain ChatOpenAI instance
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key
            temperature: Creativity level
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        
        logger.info(f"Initializing OpenAI LLM: {model_name}")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
    
    def generate_response(
        self,
        query: str,
        context: str = "",
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            query: User's question
            context: Retrieved document context from RAG
            conversation_history: Previous messages in conversation
            
        Returns:
            Generated response text
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            # Get response from OpenAI
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                text = response.content
            else:
                text = str(response)
            
            logger.info(f"Generated response ({len(text)} chars)")
            return text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Build a comprehensive prompt for the LLM.
        
        Combines context, history, and query into a well-structured prompt.
        """
        prompt = ""
        
        # Add system message
        prompt += "You are a helpful AI assistant.\n\n"
        
        # Add context if available (from RAG)
        if context:
            prompt += "Use the following information to answer the question:\n"
            prompt += f"---\n{context}\n---\n\n"
        
        # Add conversation history
        if conversation_history:
            prompt += "Previous conversation:\n"
            for user_msg, assistant_msg in conversation_history[-5:]:  # Last 5 messages
                prompt += f"User: {user_msg}\n"
                prompt += f"Assistant: {assistant_msg}\n\n"
        
        # Add current question
        prompt += f"User: {query}\n"
        prompt += "Assistant:"
        
        return prompt
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


# ============================================================================
# HUGGING FACE LLM CLASS
# ============================================================================

class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face Language Model wrapper.
    
    Handles open-source models like Mistral, Llama-2, Falcon.
    Requires Hugging Face API token for access to gated models.
    
    Attributes:
        model_name (str): Model identifier
        temperature (float): Creativity level
        max_tokens (int): Max response length
        llm: LangChain HuggingFaceEndpoint instance
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        hf_token: str = None
    ):
        """
        Initialize Hugging Face LLM.
        
        Args:
            model_name: HF model identifier
            temperature: Creativity level
            max_tokens: Maximum tokens
            hf_token: Hugging Face API token
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not hf_token:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        logger.info(f"Initializing Hugging Face LLM: {model_name}")
        
        try:
            self.llm = HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=temperature,
                max_new_tokens=max_tokens,
                huggingfacehub_api_token=hf_token,
                task="text-generation"
            )
        except Exception as e:
            logger.warning(f"Could not initialize HF model via API: {str(e)}")
            logger.info("Falling back to local model loading")
            self.llm = None
    
    def generate_response(
        self,
        query: str,
        context: str = "",
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Generate response using Hugging Face model.
        
        Args:
            query: User's question
            context: Retrieved document context
            conversation_history: Previous messages
            
        Returns:
            Generated response text
        """
        try:
            if self.llm is None:
                raise ValueError("Hugging Face model not initialized")
            
            # Build prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Extract and clean response
            text = str(response).strip()
            
            logger.info(f"Generated response ({len(text)} chars)")
            return text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """Build prompt for open-source model."""
        prompt = ""
        
        # Add context
        if context:
            prompt += f"[CONTEXT]\n{context}\n\n"
        
        # Add history
        if conversation_history:
            for user_msg, assistant_msg in conversation_history[-3:]:
                prompt += f"[USER] {user_msg}\n[ASSISTANT] {assistant_msg}\n"
        
        # Add query
        prompt += f"[USER] {query}\n[ASSISTANT]"
        
        return prompt
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


# ============================================================================
# LLM MANAGER CLASS
# ============================================================================

class LLMManager:
    """
    Manages multiple language models and switching between them.
    
    This is the main interface for working with LLMs.
    It handles:
    - Loading different models
    - Switching between OpenAI and local models
    - Generating responses
    - Managing model configuration
    
    Attributes:
        model_type (str): Type of model (openai, local, or hybrid)
        current_model: Currently active LLM instance
        available_models: Dictionary of available models
    """
    
    def __init__(
        self,
        model_type: str = "hybrid",
        default_model: str = "gpt-3.5-turbo",
        openai_api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize LLM Manager.
        
        Args:
            model_type: "openai", "local", or "hybrid"
            default_model: Model to load initially
            openai_api_key: OpenAI API key
            temperature: Creativity level for all models
            max_tokens: Max tokens for all models
        """
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self.current_model: Optional[BaseLLM] = None
        self.available_models = {}
        
        # Build available models based on type
        self._initialize_available_models()
        
        # Load default model
        self.set_model(default_model)
        
        logger.info(f"LLM Manager initialized with type: {model_type}")
    
    def _initialize_available_models(self):
        """
        Build dictionary of available models based on type.
        """
        if self.model_type in ["openai", "hybrid"]:
            self.available_models.update(OPENAI_MODELS)
        
        if self.model_type in ["local", "hybrid"]:
            self.available_models.update(LOCAL_MODELS)
    
    def set_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name/ID of the model to switch to
            
        Returns:
            bool: True if successful
            
        Example:
            >>> manager.set_model("gpt-4")
            >>> manager.set_model("mistralai/Mistral-7B-Instruct-v0.2")
        """
        try:
            logger.info(f"Switching to model: {model_name}")
            
            # Check if model is available
            if model_name not in self.available_models:
                logger.warning(f"Model {model_name} not in available models")
                return False
            
            # Load OpenAI model
            if model_name in OPENAI_MODELS:
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key not set")
                
                self.current_model = OpenAILLM(
                    model_name=model_name,
                    api_key=self.openai_api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # Load local model
            elif model_name in LOCAL_MODELS:
                self.current_model = HuggingFaceLLM(
                    model_name=model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    hf_token=os.getenv("HUGGINGFACE_TOKEN")
                )
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting model: {str(e)}")
            return False
    
    def generate_response(
        self,
        query: str,
        context: str = "",
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Generate a response using the current model.
        
        Args:
            query: User's question
            context: Retrieved document context from RAG
            conversation_history: Previous messages in conversation
            
        Returns:
            Generated response text
            
        Example:
            >>> response = manager.generate_response(
            ...     query="What is AI?",
            ...     context="AI is artificial intelligence...",
            ...     conversation_history=[]
            ... )
            >>> print(response)
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Call set_model() first.")
        
        try:
            response = self.current_model.generate_response(
                query=query,
                context=context,
                conversation_history=conversation_history or []
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def get_current_model_name(self) -> str:
        """
        Get the name of the currently active model.
        
        Returns:
            str: Model name
        """
        if self.current_model is None:
            return "No model loaded"
        return self.current_model.get_model_name()
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names/IDs
            
        Example:
            >>> models = manager.get_available_models()
            >>> for model in models:
            ...     print(model)
        """
        return list(self.available_models.keys())
    
    def get_model_descriptions(self) -> Dict[str, str]:
        """
        Get model names with descriptions.
        
        Returns:
            Dictionary with model descriptions
            
        Example:
            >>> descriptions = manager.get_model_descriptions()
            >>> for name, desc in descriptions.items():
            ...     print(f"{name}: {desc}")
        """
        return self.available_models.copy()
    
    def get_system_info(self) -> Dict:
        """
        Get current system information.
        
        Returns:
            Dictionary with system info
        """
        return {
            "model_type": self.model_type,
            "current_model": self.get_current_model_name(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available_models_count": len(self.available_models),
            "has_openai_key": bool(self.openai_api_key)
        }


# ============================================================================
# CONVERSATION MEMORY CLASS
# ============================================================================

class ConversationManager:
    """
    Manages conversation history and context.
    
    Keeps track of previous messages to maintain context
    across multiple turns of conversation.
    
    Attributes:
        history: List of (user_message, assistant_response) tuples
        max_history: Maximum number of messages to keep
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize Conversation Manager.
        
        Args:
            max_history: Maximum messages to remember
        """
        self.history: List[Tuple[str, str]] = []
        self.max_history = max_history
        logger.info(f"Conversation Manager initialized (max_history={max_history})")
    
    def add_message(self, user_msg: str, assistant_msg: str):
        """
        Add a message pair to history.
        
        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
        """
        self.history.append((user_msg, assistant_msg))
        
        # Keep only last N messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"Added message to history (total: {len(self.history)})")
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get conversation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear all conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_context_string(self) -> str:
        """
        Get formatted conversation history as string.
        
        Returns:
            Formatted conversation for LLM context
        """
        if not self.history:
            return ""
        
        context = "Previous conversation:\n"
        for user_msg, assistant_msg in self.history:
            context += f"User: {user_msg}\n"
            context += f"Assistant: {assistant_msg}\n"
        
        return context
# ============================================================================
# Hi there! The End
# ============================================================================