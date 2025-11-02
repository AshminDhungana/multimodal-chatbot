"""
Multimodal Chatbot with RAG System - Main Application

This application provides a complete Retrieval-Augmented Generation (RAG)
chatbot with support for both local and OpenAI models, document ingestion,
and conversation management.

Features:
- Document ingestion from PDF, TXT, DOCX, Markdown
- Semantic search with HuggingFace embeddings
- RAG-powered responses with context
- Multi-turn conversation support
- Gradio web interface
- Full .env configuration support
- LangChain 1.0.3 compatible

Author: Ashmin Dhungana
License: MIT
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load .env configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ .env file loaded")
except ImportError:
    logger.warning("⚠️  python-dotenv not installed, using environment variables only")

# Read configuration from .env
MODEL_TYPE = os.getenv("MODEL_TYPE", "hybrid")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger.info(
    f"📋 Configuration loaded:\n"
    f"   MODEL_TYPE={MODEL_TYPE}\n"
    f"   DEFAULT_MODEL={DEFAULT_MODEL}\n"
    f"   TEMPERATURE={TEMPERATURE}\n"
    f"   TOP_K={TOP_K}"
)

# ============================================================================
# COMPONENT IMPORTS
# ============================================================================

def safe_import(module_path: str, class_name: str):
    """Safely import a class with error handling."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        logger.info(f"✅ Loaded: {class_name}")
        return cls
    except (ImportError, AttributeError) as e:
        logger.warning(f"⚠️  Failed to load {class_name}: {str(e)}")
        return None


# Import components
logger.info("📦 Importing components...")

RAGPipelineLCEL = safe_import("src.rag_pipeline", "RAGPipelineLCEL")
LLMManager = safe_import("src.llm_models", "LLMManager")
ConversationManager = safe_import("src.llm_models", "ConversationManager")
DocumentLoader = safe_import("src.document_loader", "DocumentLoader")

# ============================================================================
# OPTIONAL COMPONENTS
# ============================================================================

# Try to import Gradio
try:
    import gradio as gr
    gradio_available = True
    logger.info("✅ Gradio available")
except ImportError:
    gradio_available = False
    logger.warning("⚠️  Gradio not installed. Install via: pip install gradio")

# Try to import embeddings for diagnostics
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"✅ Embeddings initialized: {EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
    embeddings = None

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_pipeline() -> Optional[object]:
    """Initialize RAG pipeline."""
    if not RAGPipelineLCEL:
        logger.error("❌ RAG Pipeline class not available")
        return None

    try:
        logger.info("🔄 Initializing RAG Pipeline...")
        pipeline = RAGPipelineLCEL(
            vectorstore_path=VECTORSTORE_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K,
            embedding_model=EMBEDDING_MODEL,
            llm_model=DEFAULT_MODEL,
            use_openai=(MODEL_TYPE == "openai" or MODEL_TYPE == "hybrid"),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        logger.info("✅ RAG Pipeline initialized")
        return pipeline
    except Exception as e:
        logger.error(f"❌ RAG Pipeline initialization failed: {str(e)}")
        return None


def initialize_llm_manager() -> Optional[object]:
    """Initialize LLM manager."""
    if not LLMManager:
        logger.error("❌ LLM Manager class not available")
        return None

    try:
        logger.info("🔄 Initializing LLM Manager...")
        manager = LLMManager(
            model_type=MODEL_TYPE,
            default_model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        logger.info(f"✅ LLM Manager initialized (model: {DEFAULT_MODEL})")
        return manager
    except Exception as e:
        logger.error(f"❌ LLM Manager initialization failed: {str(e)}")
        return None


def initialize_conversation_manager() -> Optional[object]:
    """Initialize conversation manager."""
    if not ConversationManager:
        logger.error("❌ Conversation Manager class not available")
        return None

    try:
        logger.info("🔄 Initializing Conversation Manager...")
        manager = ConversationManager(max_history=10, use_memory=True)
        logger.info("✅ Conversation Manager initialized")
        return manager
    except Exception as e:
        logger.error(f"❌ Conversation Manager initialization failed: {str(e)}")
        return None


def initialize_document_loader() -> Optional[object]:
    """Initialize document loader."""
    if not DocumentLoader:
        logger.error("❌ Document Loader class not available")
        return None

    try:
        logger.info("🔄 Initializing Document Loader...")
        loader = DocumentLoader(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("✅ Document Loader initialized")
        return loader
    except Exception as e:
        logger.error(f"❌ Document Loader initialization failed: {str(e)}")
        return None

# ============================================================================
# CHATBOT CLASS
# ============================================================================

class MultimodalChatbot:
    """
    Complete multimodal chatbot with RAG, LLM, and conversation management.
    
    Features:
    - Document ingestion and management
    - Semantic search with RAG
    - Multi-turn conversations
    - Fallback modes
    """

    def __init__(self):
        """Initialize all components."""
        logger.info("=" * 70)
        logger.info("🤖 MULTIMODAL CHATBOT WITH RAG")
        logger.info("=" * 70)

        # Initialize components
        self.rag_pipeline = initialize_pipeline()
        self.llm_manager = initialize_llm_manager()
        self.conversation_manager = initialize_conversation_manager()
        self.document_loader = initialize_document_loader()

        # Check readiness
        self.is_ready = bool(self.llm_manager)  # LLM is essential
        self.rag_ready = bool(self.rag_pipeline)

        if self.is_ready:
            logger.info("✅ Chatbot is ready!")
        else:
            logger.error("❌ Chatbot failed to initialize")

        # Statistics
        self.stats = {
            "messages_processed": 0,
            "documents_added": 0,
            "start_time": datetime.now(),
        }

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Add documents to RAG pipeline.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            True if successful
        """
        if not self.rag_pipeline:
            logger.error("❌ RAG Pipeline not available")
            return False

        try:
            logger.info(f"📄 Adding {len(documents)} documents...")
            success = self.rag_pipeline.add_documents(documents, metadata)

            if success:
                # Build chain if not already built
                if self.rag_pipeline.rag_chain is None:
                    self.rag_pipeline.build_rag_chain()

                self.stats["documents_added"] += len(documents)
                logger.info(f"✅ Added {len(documents)} documents")
                return True
            else:
                logger.error("❌ Failed to add documents")
                return False

        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return False

    def add_file(self, file_path: str) -> bool:
        """
        Add documents from file.
        
        Args:
            file_path: Path to file (PDF, TXT, DOCX, Markdown)
            
        Returns:
            True if successful
        """
        if not self.document_loader:
            logger.error("❌ Document Loader not available")
            return False

        try:
            logger.info(f"📂 Loading file: {file_path}")
            docs = self.document_loader.load_file(file_path, with_metadata=True)

            if docs:
                metadata = [d.metadata for d in docs] if hasattr(docs[0], 'metadata') else None
                content = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]
                return self.add_documents(content, metadata)
            else:
                logger.warning(f"⚠️  No documents extracted from {file_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Error loading file: {str(e)}")
            return False

    def chat(
        self,
        user_query: str,
        use_rag: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Process user query and generate response.
        
        Args:
            user_query: User input
            use_rag: Use RAG context
            verbose: Print debug info
            
        Returns:
            Chatbot response
        """
        if not self.is_ready:
            return "⚠️  Chatbot not fully initialized. Please check the logs."

        try:
            self.stats["messages_processed"] += 1

            if verbose:
                logger.info(f"💬 Query: {user_query[:50]}...")

            # Get RAG context if available and requested
            context = ""
            if use_rag and self.rag_ready and self.rag_pipeline.rag_chain:
                if verbose:
                    logger.info("🔍 Searching RAG context...")
                try:
                    response = self.rag_pipeline.query_with_chain(user_query)
                    self.conversation_manager.add_message(user_query, response)
                    return response
                except Exception as e:
                    logger.warning(f"⚠️  RAG query failed, falling back: {str(e)}")

            # Fallback: use LLM directly with conversation context
            if verbose:
                logger.info("🤖 Invoking LLM...")

            response = self.llm_manager.invoke(
                user_query,
                context=context,
                conversation=self.conversation_manager,
            )

            # Add to conversation history
            if self.conversation_manager:
                self.conversation_manager.add_message(user_query, response)

            return response

        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            return f"❌ Error generating response: {str(e)}"

    def get_status(self) -> Dict:
        """Get chatbot status."""
        return {
            "is_ready": self.is_ready,
            "rag_ready": self.rag_ready,
            "model": self.llm_manager.get_current_model_name() if self.llm_manager else "N/A",
            "messages_processed": self.stats["messages_processed"],
            "documents_added": self.stats["documents_added"],
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
        }

    def get_rag_stats(self) -> Optional[Dict]:
        """Get RAG pipeline statistics."""
        if not self.rag_pipeline:
            return None
        return self.rag_pipeline.get_stats()

    def reset_conversation(self) -> bool:
        """Clear conversation history."""
        if self.conversation_manager:
            self.conversation_manager.clear_history()
            logger.info("✅ Conversation history cleared")
            return True
        return False

    def save_conversation(self, path: str) -> bool:
        """Save conversation to JSON."""
        if self.conversation_manager:
            try:
                self.conversation_manager.export_to_json(path)
                logger.info(f"✅ Conversation saved to {path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to save conversation: {str(e)}")
                return False
        return False

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def launch_gradio(chatbot: MultimodalChatbot):
    """Launch Gradio web interface."""
    if not gradio_available:
        logger.error("❌ Gradio not available")
        return

    logger.info(f"🌐 Launching Gradio interface on port {GRADIO_PORT}...")

    # Custom CSS
    custom_css = """
    .chatbot-header { 
        text-align: center; 
        color: #4CAF50; 
        font-size: 24px; 
        margin: 10px 0;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f0f0;
        margin: 10px 0;
    }
    """

    def chat_fn(message: str) -> str:
        """Gradio chat function."""
        return chatbot.chat(message, use_rag=True)

    def status_fn() -> str:
        """Get status."""
        status = chatbot.get_status()
        return f"""
        **Status:** {'✅ Ready' if status['is_ready'] else '❌ Not Ready'}
        **Model:** {status['model']}
        **Messages:** {status['messages_processed']}
        **Documents:** {status['documents_added']}
        **RAG:** {'✅ Active' if status['rag_ready'] else '⚠️ Inactive'}
        """

    def rag_stats_fn() -> str:
        """Get RAG statistics."""
        stats = chatbot.get_rag_stats()
        if not stats:
            return "RAG Pipeline not available"
        return f"""
        **Status:** {stats.get('status', 'N/A')}
        **Documents:** {stats.get('num_documents', 0)}
        **Model:** {stats.get('embedding_model', 'N/A')}
        **Device:** {stats.get('device', 'N/A')}
        """

    # Create interface
    with gr.Blocks(css=custom_css, title="🤖 Multimodal Chatbot RAG") as demo:
        
        # Header
        gr.Markdown("# 🤖 Multimodal Chatbot with RAG")
        gr.Markdown(
            "Ask questions and get intelligent answers powered by "
            "Retrieval-Augmented Generation and large language models."
        )

        # Main chat section
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat")
                chatbot_interface = gr.Chatbot(height=500)
                message_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask me anything...",
                    lines=2,
                )
                submit_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 Status")
                status_output = gr.Markdown()
                refresh_status_btn = gr.Button("Refresh Status")

                gr.Markdown("## 📈 RAG Stats")
                rag_stats_output = gr.Markdown()
                refresh_rag_btn = gr.Button("Refresh RAG Stats")

                gr.Markdown("## 🎯 Actions")
                reset_btn = gr.Button("Clear History", variant="secondary")
                save_btn = gr.Button("Save Conversation")

        # Connect functions
        def chat_with_history(message, history):
            history = history or []
            response = chat_fn(message)
            history.append([message, response])
            return history, ""

        submit_btn.click(
            chat_with_history,
            inputs=[message_input, chatbot_interface],
            outputs=[chatbot_interface, message_input],
        )

        message_input.submit(
            chat_with_history,
            inputs=[message_input, chatbot_interface],
            outputs=[chatbot_interface, message_input],
        )

        refresh_status_btn.click(status_fn, outputs=status_output)
        refresh_rag_btn.click(rag_stats_fn, outputs=rag_stats_output)
        reset_btn.click(lambda: (chatbot.reset_conversation(), ""), outputs=[gr.Textbox(visible=False), message_input])
        save_btn.click(
            lambda: (chatbot.save_conversation("conversation.json"), "Saved!"),
            outputs=[gr.Textbox(visible=False), gr.Textbox(visible=False)]
        )

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    try:
        logger.info("🚀 Starting Multimodal Chatbot...")

        # Initialize chatbot
        chatbot = MultimodalChatbot()

        # Add sample documents if RAG is available
        if chatbot.rag_ready:
            logger.info("📚 Adding sample documents...")
            sample_docs = [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
                "Deep learning uses neural networks with multiple layers (hence 'deep') to process complex patterns in data.",
                "Retrieval-Augmented Generation (RAG) combines information retrieval with generation to produce more accurate and contextual responses.",
                "LangChain is a framework for developing applications powered by language models, enabling composition and chaining of LLM interactions.",
                "Natural language processing (NLP) focuses on enabling computers to understand and process human language in a meaningful way.",
            ]
            chatbot.add_documents(sample_docs)

        # Display status
        status = chatbot.get_status()
        logger.info(
            f"\n{'=' * 70}\n"
            f"Chatbot Status:\n"
            f"  Ready: {status['is_ready']}\n"
            f"  Model: {status['model']}\n"
            f"  RAG: {status['rag_ready']}\n"
            f"{'=' * 70}\n"
        )

        # Launch Gradio if available, otherwise provide CLI
        if gradio_available:
            launch_gradio(chatbot)
        else:
            logger.warning("⚠️  Gradio not available. Running in CLI mode...")
            logger.info("Install Gradio with: pip install gradio")

            # Simple CLI interface
            while True:
                try:
                    user_input = input("\n🤖 You: ").strip()
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        logger.info("👋 Goodbye!")
                        break
                    if not user_input:
                        continue

                    response = chatbot.chat(user_input)
                    print(f"\n💬 Chatbot: {response}")

                except KeyboardInterrupt:
                    logger.info("👋 Interrupted by user")
                    break

    except Exception as e:
        logger.critical(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


# ============================================================================
# EXECUTION GUARD
# ============================================================================

if __name__ == "__main__":
    main()
