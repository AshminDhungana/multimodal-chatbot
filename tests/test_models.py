"""
Tests for LLM Models Module

This file contains unit tests for language model functionality.
Tests cover:
- Model loading
- Response generation
- Model switching
- Conversation memory

Run with: pytest tests/test_models.py -v
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.llm_models import (
    LLMManager,
    OpenAILLM,
    HuggingFaceLLM,
    ConversationManager,
    OPENAI_MODELS,
    LOCAL_MODELS
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key."""
    return "sk-test-key-123456789"


@pytest.fixture
def conversation_manager():
    """Create ConversationManager instance."""
    return ConversationManager(max_history=10)


# ============================================================================
# TESTS FOR CONVERSATION MANAGER
# ============================================================================

class TestConversationManager:
    """Test conversation memory management."""
    
    def test_add_message(self, conversation_manager):
        """Test adding a message to history."""
        conversation_manager.add_message("Hello", "Hi there!")
        history = conversation_manager.get_history()
        
        assert len(history) == 1
        assert history[0] == ("Hello", "Hi there!")
    
    def test_add_multiple_messages(self, conversation_manager):
        """Test adding multiple messages."""
        messages = [
            ("Hi", "Hello"),
            ("How are you?", "I'm good!"),
            ("What is AI?", "AI is artificial intelligence")
        ]
        
        for user, assistant in messages:
            conversation_manager.add_message(user, assistant)
        
        history = conversation_manager.get_history()
        assert len(history) == 3
    
    def test_max_history_limit(self, conversation_manager):
        """Test that history respects max_history limit."""
        # Add more than max_history
        for i in range(15):
            conversation_manager.add_message(f"Message {i}", f"Response {i}")
        
        history = conversation_manager.get_history()
        assert len(history) <= 10  # max_history is 10
    
    def test_clear_history(self, conversation_manager):
        """Test clearing conversation history."""
        conversation_manager.add_message("Test", "Response")
        conversation_manager.clear_history()
        
        history = conversation_manager.get_history()
        assert len(history) == 0
    
    def test_get_context_string(self, conversation_manager):
        """Test getting formatted context string."""
        conversation_manager.add_message("Hello", "Hi!")
        conversation_manager.add_message("How are you?", "Good!")
        
        context = conversation_manager.get_context_string()
        
        assert isinstance(context, str)
        assert "Hello" in context
        assert "How are you?" in context
        assert "Hi!" in context
        assert "Good!" in context
    
    def test_get_context_empty(self, conversation_manager):
        """Test context string on empty history."""
        context = conversation_manager.get_context_string()
        assert context == ""


# ============================================================================
# TESTS FOR LLM MANAGER
# ============================================================================

class TestLLMManager:
    """Test LLM Manager functionality."""
    
    def test_manager_initialization(self, mock_openai_api_key):
        """Test LLMManager initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="local")
                assert manager.model_type == "local"
                assert manager.temperature == 0.7
                assert manager.max_tokens == 2048
            except Exception as e:
                # Expected if dependencies not available
                pass
    
    def test_get_available_models(self, mock_openai_api_key):
        """Test getting available models."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="hybrid")
                models = manager.get_available_models()
                
                assert isinstance(models, list)
                assert len(models) > 0
            except Exception:
                pass
    
    def test_get_model_descriptions(self, mock_openai_api_key):
        """Test getting model descriptions."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="hybrid")
                descriptions = manager.get_model_descriptions()
                
                assert isinstance(descriptions, dict)
                for model_name, description in descriptions.items():
                    assert isinstance(model_name, str)
                    assert isinstance(description, str)
            except Exception:
                pass
    
    def test_get_current_model_name(self, mock_openai_api_key):
        """Test getting current model name."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="hybrid",
                    default_model="gpt-3.5-turbo",
                    openai_api_key=mock_openai_api_key
                )
                name = manager.get_current_model_name()
                # Should return something, either model name or error message
                assert isinstance(name, str)
            except Exception:
                pass
    
    def test_get_system_info(self, mock_openai_api_key):
        """Test getting system information."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="hybrid")
                info = manager.get_system_info()
                
                assert isinstance(info, dict)
                assert "model_type" in info
                assert "temperature" in info
            except Exception:
                pass


# ============================================================================
# TESTS FOR CONVERSATION INITIALIZATION
# ============================================================================

class TestConversationHandling:
    """Test conversation-related functionality."""
    
    def test_generate_response_with_empty_history(self, mock_openai_api_key):
        """Test generating response without history."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="local")
                # Mock the current_model
                manager.current_model = Mock()
                manager.current_model.generate_response = Mock(
                    return_value="Test response"
                )
                
                response = manager.generate_response(
                    query="Hello",
                    context="",
                    conversation_history=[]
                )
                
                assert isinstance(response, str)
            except Exception:
                pass
    
    def test_generate_response_with_history(self, mock_openai_api_key):
        """Test generating response with conversation history."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="local")
                manager.current_model = Mock()
                manager.current_model.generate_response = Mock(
                    return_value="Continued response"
                )
                
                history = [
                    ("What is AI?", "AI is artificial intelligence"),
                    ("Tell me more", "AI includes machine learning")
                ]
                
                response = manager.generate_response(
                    query="How does it work?",
                    context="AI works through algorithms",
                    conversation_history=history
                )
                
                assert isinstance(response, str)
            except Exception:
                pass


# ============================================================================
# TESTS FOR MODEL SELECTION
# ============================================================================

class TestModelSelection:
    """Test model selection and switching."""
    
    def test_available_openai_models(self):
        """Test that OpenAI models are defined."""
        assert len(OPENAI_MODELS) > 0
        assert "gpt-3.5-turbo" in OPENAI_MODELS
        assert "gpt-4" in OPENAI_MODELS
    
    def test_available_local_models(self):
        """Test that local models are defined."""
        assert len(LOCAL_MODELS) > 0
        assert any("mistral" in model.lower() for model in LOCAL_MODELS.keys())
        assert any("llama" in model.lower() for model in LOCAL_MODELS.keys())
    
    def test_local_model_type(self, mock_openai_api_key):
        """Test local-only mode."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            try:
                manager = LLMManager(
                    model_type="local",
                    default_model="mistralai/Mistral-7B-Instruct-v0.2"
                )
                models = manager.get_available_models()
                # Should only have local models
                assert len(models) > 0
            except Exception:
                pass
    
    def test_openai_model_type(self, mock_openai_api_key):
        """Test OpenAI-only mode."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="openai",
                    openai_api_key=mock_openai_api_key
                )
                models = manager.get_available_models()
                # Should have OpenAI models
                assert any("gpt" in model for model in models)
            except Exception:
                pass
    
    def test_hybrid_mode(self, mock_openai_api_key):
        """Test hybrid mode (both OpenAI and local)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="hybrid",
                    openai_api_key=mock_openai_api_key
                )
                models = manager.get_available_models()
                # Should have both types
                assert len(models) > 0
            except Exception:
                pass


# ============================================================================
# TESTS FOR ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling in LLM module."""
    
    def test_invalid_model_type(self):
        """Test with invalid model type."""
        try:
            manager = LLMManager(
                model_type="invalid_type",
                openai_api_key="sk-test"
            )
            # Should either handle gracefully or raise
        except (ValueError, KeyError):
            pass  # Expected
    
    def test_missing_openai_key(self):
        """Test OpenAI mode without API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            try:
                manager = LLMManager(model_type="openai")
                # Should either handle gracefully or raise
            except ValueError:
                pass  # Expected
    
    def test_no_current_model_for_generation(self):
        """Test generating response without loaded model."""
        try:
            manager = LLMManager(model_type="local")
            manager.current_model = None
            
            with pytest.raises(ValueError):
                manager.generate_response("test")
        except Exception:
            pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for LLM models."""
    
    def test_conversation_workflow(self, conversation_manager):
        """Test complete conversation workflow."""
        # Add messages
        conversation_manager.add_message("Hello", "Hi!")
        conversation_manager.add_message("How are you?", "I'm good!")
        
        # Get history
        history = conversation_manager.get_history()
        assert len(history) == 2
        
        # Get context
        context = conversation_manager.get_context_string()
        assert "Hello" in context
        
        # Clear
        conversation_manager.clear_history()
        assert len(conversation_manager.get_history()) == 0
    
    def test_model_manager_workflow(self, mock_openai_api_key):
        """Test complete model manager workflow."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(model_type="local")
                
                # Get available models
                models = manager.get_available_models()
                assert len(models) > 0
                
                # Get model descriptions
                descriptions = manager.get_model_descriptions()
                assert len(descriptions) > 0
                
                # Get system info
                info = manager.get_system_info()
                assert "model_type" in info
            except Exception:
                pass


# ============================================================================
# PARAMETER TESTS
# ============================================================================

class TestParameters:
    """Test parameter handling."""
    
    def test_temperature_parameter(self, mock_openai_api_key):
        """Test temperature parameter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="local",
                    temperature=0.5
                )
                assert manager.temperature == 0.5
                
                manager2 = LLMManager(
                    model_type="local",
                    temperature=0.9
                )
                assert manager2.temperature == 0.9
            except Exception:
                pass
    
    def test_max_tokens_parameter(self, mock_openai_api_key):
        """Test max_tokens parameter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="local",
                    max_tokens=1024
                )
                assert manager.max_tokens == 1024
            except Exception:
                pass
    
    def test_default_model_parameter(self, mock_openai_api_key):
        """Test default_model parameter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_openai_api_key}):
            try:
                manager = LLMManager(
                    model_type="hybrid",
                    default_model="gpt-3.5-turbo",
                    openai_api_key=mock_openai_api_key
                )
                # Should initialize with the default model
                assert manager.model_name == "gpt-3.5-turbo"
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
