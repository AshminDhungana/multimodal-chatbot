"""
Multimodal Chatbot with RAG System - Main Application
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ .env file loaded")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed")
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
logger.info(f"📋 Config: MODEL_TYPE={MODEL_TYPE}, DEFAULT_MODEL={DEFAULT_MODEL}")
# ============================================================================
# MODEL DEFINITIONS - FIXED
# ============================================================================
# ✅ KEY FIX: Use actual model IDs in dropdown, not display names
OPENAI_MODELS = [
    ("gpt-3.5-turbo", "🚀 GPT-3.5 Turbo (Fast & Cheap)"),
    ("gpt-4", "💎 GPT-4 (Most Capable)"),
    ("gpt-4o-mini", "⚡ GPT-4o-mini (Fastest)"),
]
LOCAL_MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.2", "⚡ Mistral 7B"),
    ("meta-llama/Llama-2-7b-chat-hf", "🦙 Llama-2 7B"),
    ("meta-llama/Llama-2-13b-chat-hf", "🦙 Llama-2 13B"),
    ("tiiuae/falcon-7b-instruct", "🦅 Falcon 7B"),
    ("EleutherAI/gpt-j-6B", "🔥 GPT-J 6B"),
]
# ============================================================================
# COMPONENT IMPORTS
# ============================================================================
def safe_import(module_path: str, class_name: str):
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        logger.info(f"✅ Loaded: {class_name}")
        return cls
    except (ImportError, AttributeError) as e:
        logger.warning(f"⚠️ Failed to load {class_name}: {str(e)}")
        return None
logger.info("📦 Importing components...")
RAGPipelineLCEL = safe_import("src.rag_pipeline", "RAGPipelineLCEL")
LLMManager = safe_import("src.llm_models", "LLMManager")
ConversationManager = safe_import("src.llm_models", "ConversationManager")
DocumentLoader = safe_import("src.document_loader", "DocumentLoader")
try:
    import gradio as gr
    gradio_available = True
    logger.info("✅ Gradio available")
except ImportError:
    gradio_available = False
    logger.warning("⚠️ Gradio not installed")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"✅ Embeddings initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
    embeddings = None
# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================
def initialize_pipeline() -> Optional[object]:
    if not RAGPipelineLCEL:
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
def initialize_llm_manager(model_type: str = MODEL_TYPE, default_model: str = DEFAULT_MODEL) -> Optional[object]:
    if not LLMManager:
        return None
    try:
        logger.info("🔄 Initializing LLM Manager...")
        manager = LLMManager(
            model_type=model_type,
            default_model=default_model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        logger.info(f"✅ LLM Manager initialized")
        return manager
    except Exception as e:
        logger.error(f"❌ LLM Manager initialization failed: {str(e)}")
        return None
def initialize_conversation_manager() -> Optional[object]:
    if not ConversationManager:
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
    if not DocumentLoader:
        return None
    try:
        logger.info("🔄 Initializing Document Loader...")
        loader = DocumentLoader(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logger.info("✅ Document Loader initialized")
        return loader
    except Exception as e:
        logger.error(f"❌ Document Loader initialization failed: {str(e)}")
        return None
# ============================================================================
# CHATBOT CLASS
# ============================================================================
class MultimodalChatbot:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🤖 MULTIMODAL CHATBOT WITH RAG")
        logger.info("=" * 70)
        self.rag_pipeline = initialize_pipeline()
        self.llm_manager = initialize_llm_manager()
        self.conversation_manager = initialize_conversation_manager()
        self.document_loader = initialize_document_loader()
        self.current_model_type = MODEL_TYPE
        self.is_ready = bool(self.llm_manager)
        self.rag_ready = bool(self.rag_pipeline)
        if self.is_ready:
            logger.info("✅ Chatbot is ready!")
        else:
            logger.error("❌ Chatbot failed to initialize")
        self.stats = {
            "messages_processed": 0,
            "documents_added": 0,
            "start_time": datetime.now(),
        }
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        if not self.rag_pipeline:
            return False
        try:
            logger.info(f"📄 Adding {len(documents)} documents...")
            success = self.rag_pipeline.add_documents(documents, metadata)
            if success:
                if self.rag_pipeline.rag_chain is None:
                    self.rag_pipeline.build_rag_chain()
                self.stats["documents_added"] += len(documents)
                logger.info(f"✅ Added {len(documents)} documents")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return False
    def chat(self, user_query: str, use_rag: bool = True) -> str:
        if not isinstance(user_query, str):
            return f"Error: Query must be string"
        if not self.is_ready:
            return "⚠️ Chatbot not fully initialized."
        try:
            self.stats["messages_processed"] += 1
            if use_rag and self.rag_ready and self.rag_pipeline.rag_chain:
                try:
                    response = self.rag_pipeline.query_with_chain(user_query)
                    if self.conversation_manager:
                        self.conversation_manager.add_message(user_query, response)
                    return response
                except Exception as e:
                    logger.warning(f"⚠️ RAG query failed: {str(e)}")
            response = self.llm_manager.invoke(
                user_query,
                context="",
                conversation=self.conversation_manager,
            )
            if self.conversation_manager:
                self.conversation_manager.add_message(user_query, response)
            return response
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            return f"❌ Error: {str(e)}"
    def switch_model(self, model_id: str, model_type: str) -> Tuple[bool, str]:
        """✅ KEY FIX: Expects model ID, not display name; reinitializes RAG pipeline for model switch; reinitializes LLMManager if type changes"""
        if not self.llm_manager:
            return False, "❌ LLM Manager not available"
        try:
            logger.info(f"🔄 Switching model to: {model_id} ({model_type})")
            if model_type != self.current_model_type:
                self.llm_manager = initialize_llm_manager(model_type=model_type, default_model=model_id)
                self.current_model_type = model_type
                success = bool(self.llm_manager)
            else:
                success = self.llm_manager.set_model(model_id)
           
            if success:
                new_model = self.llm_manager.get_current_model_name()
                if self.rag_pipeline:
                    use_openai = (model_type == "OpenAI")
                    self.rag_pipeline = RAGPipelineLCEL(
                        vectorstore_path=VECTORSTORE_PATH,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        top_k=TOP_K,
                        embedding_model=EMBEDDING_MODEL,
                        llm_model=model_id,
                        use_openai=use_openai,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    self.rag_ready = True
                msg = f"✅ Switched to: {new_model}"
                logger.info(msg)
                return True, msg
            else:
                msg = f"❌ Failed to switch to: {model_id}"
                logger.error(msg)
                return False, msg
               
        except Exception as e:
            msg = f"❌ Error switching model: {str(e)}"
            logger.error(msg)
            return False, msg
    def get_status(self) -> Dict:
        return {
            "is_ready": self.is_ready,
            "rag_ready": self.rag_ready,
            "model": self.llm_manager.get_current_model_name() if self.llm_manager else "N/A",
            "messages_processed": self.stats["messages_processed"],
            "documents_added": self.stats["documents_added"],
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
        }
    def get_rag_stats(self) -> Optional[Dict]:
        if not self.rag_pipeline:
            return None
        return self.rag_pipeline.get_stats()
    def reset_conversation(self) -> bool:
        if self.conversation_manager:
            self.conversation_manager.clear_history()
            logger.info("✅ Conversation history cleared")
            return True
        return False
    def save_conversation(self, path: str) -> bool:
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
# GRADIO INTERFACE - BEAUTIFUL
# ============================================================================
def launch_gradio(chatbot: MultimodalChatbot):
    if not gradio_available:
        logger.error("❌ Gradio not available")
        return
    logger.info(f"🌐 Launching on port {GRADIO_PORT}...")
    # ✅ BEAUTIFUL CSS (adjusted for darker background and full width)
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * {
        font-family: 'Inter', sans-serif;
    }
    .gradio-container {
        max-width: none !important;
        width: 100% !important;
        background: #121212 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: #e0e0e0 !important;
    }
    .header-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .header-box h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    .header-box p {
        font-size: 1em;
        opacity: 0.85;
        margin: 8px 0 0 0;
    }
    h2 {
        color: #ffffff !important;
        font-size: 1.4em !important;
        font-weight: 600 !important;
        margin: 20px 0 10px 0 !important;
    }
    /* Model Switcher */
    .model-section {
        background: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .model-section:hover {
        border-color: #5a67d8;
        box-shadow: 0 4px 12px rgba(90, 103, 216, 0.1);
    }
    /* Buttons */
    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 14px !important;
        background: #333333 !important;
        color: #e0e0e0 !important;
    }
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
        background: #444444 !important;
    }
    button:active {
        transform: translateY(0);
    }
    /* Primary Button */
    .gr-button.primary, button[type="submit"],
    [class*="Button"][class*="primary"] {
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%) !important;
        color: white !important;
    }
    /* Secondary Button */
    .gr-button.secondary {
        background: linear-gradient(135deg, #ed64a6 0%, #f56565 100%) !important;
        color: white !important;
    }
    /* Input Fields */
    input, textarea, select,
    .gr-textbox, .gr-dropdown {
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        background: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    input:focus, textarea:focus, select:focus,
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: #5a67d8 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(90, 103, 216, 0.2) !important;
    }
    /* Chatbot */
    .gr-chatbot {
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        background: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    .message {
        border-radius: 10px !important;
        background: #252525 !important;
        color: #e0e0e0 !important;
    }
    .message.user {
        background: #2c3e50 !important;
    }
    /* Status Box */
    .status-box {
        background: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 14px;
        margin: 10px 0;
        color: #e0e0e0 !important;
    }
    /* Group */
    .group {
        background: #1e1e1e !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    /* Radio */
    input[type="radio"] {
        cursor: pointer;
        margin-right: 6px;
    }
    label {
        cursor: pointer;
        font-weight: 500;
        color: #e0e0e0 !important;
    }
    """
    def status_fn() -> str:
        status = chatbot.get_status()
        uptime = status["uptime_seconds"]
        uptime_str = f"{int(uptime // 60)}m {int(uptime % 60)}s"
       
        status_icon = "✅" if status['is_ready'] else "❌"
        rag_icon = "✅" if status['rag_ready'] else "⚠️"
       
        return f"""
        **{status_icon} Status:** {'Ready' if status['is_ready'] else 'Not Ready'}
       
        **🤖 Model:** {status['model']}
       
        **💬 Messages:** {status['messages_processed']}
       
        **📄 Documents:** {status['documents_added']}
       
        **{rag_icon} RAG:** {'Active' if status['rag_ready'] else 'Inactive'}
       
        **⏱️ Uptime:** {uptime_str}
        """
    def rag_stats_fn() -> str:
        stats = chatbot.get_rag_stats()
        if not stats:
            return "⚠️ RAG not available"
       
        status = stats.get('status', 'N/A')
        status_icon = "✅" if status == 'active' else "⚠️"
       
        return f"""
        **{status_icon} Status:** {status}
       
        **📚 Documents:** {stats.get('num_documents', 0)}
       
        **🧠 Embedding:** {stats.get('embedding_model', 'N/A').split('/')[-1]}
       
        **💻 Device:** {stats.get('device', 'N/A')}
        """
    def handle_user_input(message: str, chat_history):
        if not message or not message.strip():
            return chat_history, ""
        try:
            user_input = message.strip()
            response = chatbot.chat(user_input, use_rag=True)
           
            if chat_history is None:
                chat_history = []
            chat_history.append([user_input, response])
           
            return chat_history, ""
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            if chat_history is None:
                chat_history = []
            chat_history.append([message, f"❌ Error: {str(e)}"])
            return chat_history, ""
    # ✅ KEY FIX: Pass model ID, not display name
    def switch_model_fn(model_type, selected_model):
        """Switch model - model_id passed directly"""
        try:
            if model_type == "OpenAI":
                logger.info(f"[Model] OpenAI selected: {selected_model}")
            else:
                logger.info(f"[Model] Local selected: {selected_model}")
           
            # ✅ Direct model ID - no conversion needed
            success, message = chatbot.switch_model(selected_model, model_type)
            return message
           
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    # ============================================================================
    # CREATE INTERFACE
    # ============================================================================
    with gr.Blocks(css=custom_css, title="🤖 AI Chatbot with RAG", theme=gr.themes.Soft()) as demo:
       
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1>🤖 AI Chatbot with RAG</h1>
            <p>Powered by Retrieval-Augmented Generation & Advanced LLMs</p>
        </div>
        """)
        # ============================================================================
        # MODEL SWITCHER
        # ============================================================================
        with gr.Group(elem_classes="model-section"):
            gr.Markdown("## 🎛️ Model Selection")
           
            with gr.Row():
                model_type = gr.Radio(
                    choices=["OpenAI", "Local"],
                    value="OpenAI",
                    label="📌 Model Type",
                    scale=1
                )
               
                # ✅ KEY FIX: Store model IDs in value
                openai_model = gr.Dropdown(
                    choices=[display for model_id, display in OPENAI_MODELS],
                    value=OPENAI_MODELS[0][1],
                    label="🔗 OpenAI Model",
                    scale=2,
                    visible=True
                )
               
                local_model = gr.Dropdown(
                    choices=[display for model_id, display in LOCAL_MODELS],
                    value=LOCAL_MODELS[0][1],
                    label="🔗 Local Model",
                    scale=2,
                    visible=False
                )
               
                apply_btn = gr.Button("🔄 Switch", variant="primary", scale=1)
           
            switch_status = gr.Textbox(
                label="Status",
                interactive=False,
                value="✅ Ready to switch",
                lines=1
            )
        # ============================================================================
        # MAIN CHAT
        # ============================================================================
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat")
               
                chat_display = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    show_copy_button=True
                )
               
                with gr.Row():
                    message_input = gr.Textbox(
                        label="Message",
                        placeholder="Type here...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("📤 Send", variant="primary", scale=1)
            # ============================================================================
            # SIDEBAR
            # ============================================================================
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Status")
                status_output = gr.Markdown()
                refresh_status_btn = gr.Button("🔄 Refresh", scale=1)
                gr.Markdown("---")
                gr.Markdown("## 📈 RAG Stats")
                rag_stats_output = gr.Markdown()
                refresh_rag_btn = gr.Button("🔄 Refresh", scale=1)
                gr.Markdown("---")
                gr.Markdown("## 🎯 Actions")
                reset_btn = gr.Button("🗑️ Clear", scale=1)
                save_btn = gr.Button("💾 Save", scale=1)
        # ============================================================================
        # EVENTS
        # ============================================================================
        submit_btn.click(
            handle_user_input,
            inputs=[message_input, chat_display],
            outputs=[chat_display, message_input],
            queue=True
        )
        message_input.submit(
            handle_user_input,
            inputs=[message_input, chat_display],
            outputs=[chat_display, message_input],
            queue=True
        )
        # Model type toggle
        def update_visibility(model_type_value):
            if model_type_value == "OpenAI":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
       
        model_type.change(
            update_visibility,
            inputs=model_type,
            outputs=[openai_model, local_model]
        )
        # ✅ KEY FIX: Pass model_id directly from dropdown value
        def switch_with_id(model_type, openai_display, local_display):
            # Get model ID from display text
            if model_type == "OpenAI":
                model_id = next(mid for mid, display in OPENAI_MODELS if display == openai_display)
            else:
                model_id = next(mid for mid, display in LOCAL_MODELS if display == local_display)
           
            return switch_model_fn(model_type, model_id)
        apply_btn.click(
            switch_with_id,
            inputs=[model_type, openai_model, local_model],
            outputs=switch_status,
            queue=False
        )
        refresh_status_btn.click(status_fn, outputs=status_output, queue=False)
        refresh_rag_btn.click(rag_stats_fn, outputs=rag_stats_output, queue=False)
       
        def clear_history_fn():
            chatbot.reset_conversation()
            return [], ""
       
        reset_btn.click(clear_history_fn, outputs=[chat_display, message_input], queue=False)
       
        def save_conv_fn():
            success = chatbot.save_conversation("conversation.json")
            return "✅ Saved!" if success else "❌ Failed"
       
        save_btn.click(save_conv_fn, queue=False)
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False, show_error=True)
# ============================================================================
# MAIN
# ============================================================================
def main():
    try:
        logger.info("🚀 Starting Chatbot...")
        chatbot = MultimodalChatbot()
        if chatbot.rag_ready:
            logger.info("📚 Adding sample documents...")
            sample_docs = [
                "Machine learning is a subset of AI that learns from data.",
                "Deep learning uses neural networks with multiple layers.",
                "RAG combines retrieval with generation for better responses.",
                "LangChain is a framework for LLM applications.",
                "NLP focuses on understanding human language.",
            ]
            chatbot.add_documents(sample_docs)
        status = chatbot.get_status()
        logger.info(f"\n{'=' * 70}\nChatbot Ready!\nModel: {status['model']}\nRAG: {status['rag_ready']}\n{'=' * 70}\n")
        if gradio_available:
            launch_gradio(chatbot)
        else:
            logger.warning("⚠️ Gradio not available")
    except Exception as e:
        logger.critical(f"❌ Fatal error: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
if __name__ == "__main__":
    main()