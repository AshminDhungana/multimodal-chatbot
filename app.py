"""
Main Application Entry Point for Multimodel Chatbot

This is the primary application file that creates and runs the Gradio web interface
for the multimodel chatbot with RAG capabilities.

The application integrates:
- RAG pipeline for document retrieval and context
- Multiple language models (OpenAI GPT and Hugging Face models)
- Conversation memory for multi-turn conversations
- Document upload and processing
- Real-time chat interface

Author: Ashmin Dhungana
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import logging
from typing import List, Tuple
from src.rag_pipeline import RAGPipeline
from src.llm_models import LLMManager
from src.document_loader import DocumentLoader

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
MODEL_TYPE = os.getenv("MODEL_TYPE", "hybrid")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")

# Create necessary directories
Path(DATA_DIR).mkdir(exist_ok=True)
Path(VECTORSTORE_PATH).mkdir(exist_ok=True)

# ============================================================================
# GLOBAL VARIABLES (Initialized on startup)
# ============================================================================

# These will store the RAG pipeline, LLM manager, and conversation history
rag_pipeline: RAGPipeline = None
llm_manager: LLMManager = None
conversation_history: List[Tuple[str, str]] = []
uploaded_documents: List[str] = []

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_system():
    """
    Initialize all system components on application startup.
    
    This function:
    1. Initializes the LLM Manager with configured models
    2. Initializes the RAG Pipeline with vector store
    3. Loads any previously uploaded documents
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global rag_pipeline, llm_manager
    
    try:
        logger.info("Initializing Multimodel Chatbot system...")
        
        # Initialize LLM Manager
        logger.info(f"Loading LLM Manager with model type: {MODEL_TYPE}")
        llm_manager = LLMManager(
            model_type=MODEL_TYPE,
            default_model=DEFAULT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Initialize RAG Pipeline
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            vectorstore_path=VECTORSTORE_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        
        # Load existing documents if available
        if os.path.exists(VECTORSTORE_PATH):
            logger.info("Loading existing vector store...")
            rag_pipeline.load_vectorstore()
        
        logger.info("System initialization successful!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return False

# ============================================================================
# DOCUMENT MANAGEMENT FUNCTIONS
# ============================================================================

def process_uploaded_file(file):
    """
    Process and ingest an uploaded document into the RAG system.
    
    Args:
        file: Gradio File object containing the uploaded document
        
    Returns:
        str: Status message about the upload and processing
    """
    global rag_pipeline, uploaded_documents
    
    if file is None:
        return "❌ No file uploaded. Please select a file."
    
    try:
        logger.info(f"Processing uploaded file: {file.name}")
        
        # Save uploaded file temporarily
        temp_path = os.path.join(DATA_DIR, file.name)
        
        # The file object from Gradio contains the path
        if hasattr(file, 'name'):
            temp_path = file.name
        
        # Load and process document
        loader = DocumentLoader(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = loader.load_file(temp_path)
        
        if not chunks:
            return f"❌ Could not extract text from {file.name}"
        
        # Add to RAG pipeline
        rag_pipeline.add_documents(chunks, metadata={"source": file.name})
        
        # Save vector store
        rag_pipeline.save_vectorstore()
        
        uploaded_documents.append(file.name)
        
        logger.info(f"Successfully processed {len(chunks)} chunks from {file.name}")
        return f"✅ Successfully processed '{file.name}'\n📊 Extracted {len(chunks)} text chunks\n💾 Added to knowledge base"
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return f"❌ Error processing file: {str(e)}"

def get_uploaded_documents_list() -> str:
    """
    Get a formatted list of uploaded documents.
    
    Returns:
        str: Formatted list of documents or message if none uploaded
    """
    if not uploaded_documents:
        return "No documents uploaded yet. Upload documents to get started!"
    
    doc_list = "\n".join([f"• {doc}" for doc in uploaded_documents])
    return f"📚 Uploaded Documents ({len(uploaded_documents)}):\n{doc_list}"

def clear_knowledge_base():
    """
    Clear all documents from the knowledge base.
    
    Returns:
        str: Confirmation message
    """
    global rag_pipeline, uploaded_documents, conversation_history
    
    try:
        rag_pipeline.clear_vectorstore()
        uploaded_documents = []
        conversation_history = []
        
        logger.info("Knowledge base cleared")
        return "✅ Knowledge base cleared successfully!"
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        return f"❌ Error clearing knowledge base: {str(e)}"

# ============================================================================
# CHAT AND RESPONSE GENERATION FUNCTIONS
# ============================================================================

def process_message(message: str, history: List[List]) -> str:
    """
    Process user message and generate AI response using RAG pipeline.
    
    This function:
    1. Retrieves relevant documents from the knowledge base
    2. Prepares context from retrieved documents
    3. Generates response using selected LLM
    4. Updates conversation history
    
    Args:
        message: User's input message
        history: Conversation history from Gradio ChatInterface
        
    Returns:
        str: AI-generated response
    """
    global rag_pipeline, llm_manager, conversation_history
    
    if not message.strip():
        return "Please enter a message."
    
    try:
        logger.info(f"Processing user message: {message[:50]}...")
        
        # Retrieve relevant context from documents
        if uploaded_documents and len(uploaded_documents) > 0:
            retrieved_docs = rag_pipeline.retrieve(message, top_k=TOP_K)
            context = "\n".join([doc['content'] for doc in retrieved_docs])
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        else:
            context = ""
            logger.info("No documents in knowledge base. Using general knowledge.")
        
        # Generate response using LLM
        response = llm_manager.generate_response(
            query=message,
            context=context,
            conversation_history=conversation_history
        )
        
        # Update conversation history
        conversation_history.append((message, response))
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return f"⚠️ Error generating response: {str(e)}\n\nPlease check your configuration and try again."

def clear_conversation():
    """
    Clear conversation history.
    
    Returns:
        str: Confirmation message
    """
    global conversation_history
    conversation_history = []
    return "✅ Conversation cleared!"

def get_system_info() -> str:
    """
    Get current system configuration information.
    
    Returns:
        str: Formatted system information
    """
    info = f"""
    🤖 **System Information**
    
    **Configuration:**
    • Model Type: {MODEL_TYPE}
    • Default Model: {DEFAULT_MODEL}
    • Temperature: {TEMPERATURE}
    • Max Tokens: {MAX_TOKENS}
    
    **RAG Settings:**
    • Chunk Size: {CHUNK_SIZE}
    • Chunk Overlap: {CHUNK_OVERLAP}
    • Top K Results: {TOP_K}
    
    **Storage:**
    • Data Directory: {DATA_DIR}
    • Vector Store: {VECTORSTORE_PATH}
    • Documents Uploaded: {len(uploaded_documents)}
    """
    return info

# ============================================================================
# MODEL MANAGEMENT FUNCTIONS
# ============================================================================

def switch_model(model_name: str) -> str:
    """
    Switch to a different language model.
    
    Args:
        model_name: Name of the model to switch to
        
    Returns:
        str: Confirmation message
    """
    global llm_manager
    
    try:
        logger.info(f"Switching to model: {model_name}")
        llm_manager.set_model(model_name)
        return f"✅ Successfully switched to {model_name}"
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return f"❌ Error switching model: {str(e)}"

def get_available_models() -> str:
    """
    Get list of available models.
    
    Returns:
        str: Formatted list of available models
    """
    try:
        models = llm_manager.get_available_models()
        model_list = "\n".join([f"• {model}" for model in models])
        return f"**Available Models:**\n{model_list}"
    except Exception as e:
        return f"❌ Error retrieving models: {str(e)}"

# ============================================================================
# GRADIO INTERFACE CREATION
# ============================================================================

def create_gradio_interface():
    """
    Create and configure the Gradio web interface.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    
    with gr.Blocks(title="Multimodel Chatbot with RAG", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("# 🤖 Multimodel Chatbot with RAG")
        gr.Markdown("An intelligent chatbot that combines multiple language models with retrieval-augmented generation")
        
        with gr.Tabs():
            
            # ===== TAB 1: CHAT INTERFACE =====
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Chat with the AI Assistant")
                
                chat_interface = gr.ChatInterface(
                    fn=process_message,
                    examples=[
                        "What is this document about?",
                        "Can you summarize the key points?",
                        "Answer my question based on the uploaded documents"
                    ],
                    title="Chat with Multimodel Chatbot",
                    description="Upload documents, then ask questions about them!"
                )
            
            # ===== TAB 2: DOCUMENT MANAGEMENT =====
            with gr.Tab("📄 Documents"):
                gr.Markdown("### Document Management")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Upload Documents")
                        file_input = gr.File(
                            label="Select a document (PDF, TXT)",
                            file_count="single"
                        )
                        upload_btn = gr.Button("📤 Upload and Process", variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False,
                            lines=3
                        )
                        
                        upload_btn.click(
                            fn=process_uploaded_file,
                            inputs=file_input,
                            outputs=upload_status
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### Knowledge Base")
                        doc_list = gr.Textbox(
                            label="Uploaded Documents",
                            interactive=False,
                            lines=6
                        )
                        refresh_btn = gr.Button("🔄 Refresh")
                        clear_btn = gr.Button("🗑️ Clear All Documents", variant="stop")
                        
                        refresh_btn.click(
                            fn=get_uploaded_documents_list,
                            outputs=doc_list
                        )
                        clear_btn.click(
                            fn=clear_knowledge_base,
                            outputs=upload_status
                        )
            
            # ===== TAB 3: MODEL SELECTION =====
            with gr.Tab("🎛️ Settings"):
                gr.Markdown("### Model & Configuration Settings")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model Selection")
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "gpt-3.5-turbo",
                                "gpt-4",
                                "mistralai/Mistral-7B-Instruct-v0.2",
                                "meta-llama/Llama-2-7b-chat-hf"
                            ],
                            value=DEFAULT_MODEL,
                            label="Select Model"
                        )
                        switch_btn = gr.Button("🔄 Switch Model", variant="primary")
                        model_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        switch_btn.click(
                            fn=switch_model,
                            inputs=model_dropdown,
                            outputs=model_status
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### Available Models")
                        models_display = gr.Textbox(
                            label="Models",
                            interactive=False,
                            lines=6
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh")
                        
                        refresh_models_btn.click(
                            fn=get_available_models,
                            outputs=models_display
                        )
                
                gr.Markdown("#### System Information")
                info_display = gr.Textbox(
                    label="System Config",
                    interactive=False,
                    lines=8
                )
                info_btn = gr.Button("📊 Show System Info")
                info_btn.click(
                    fn=get_system_info,
                    outputs=info_display
                )
            
            # ===== TAB 4: HELP & INFO =====
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                    ### 📚 How to Use Multimodel Chatbot
                    
                    **1. Upload Documents**
                    - Go to the **Documents** tab
                    - Upload PDF or TXT files
                    - The system will automatically process and index them
                    
                    **2. Ask Questions**
                    - Go to the **Chat** tab
                    - Type your question or prompt
                    - The AI will search through your documents and provide answers
                    
                    **3. Switch Models**
                    - Go to the **Settings** tab
                    - Select a different model from the dropdown
                    - Click "Switch Model" to use it
                    
                    **4. Clear Data**
                    - In the **Documents** tab, click "Clear All Documents" to reset
                    - In the **Chat** tab, conversations are auto-cleared
                    
                    ### 🔧 Features
                    - **Multiple Models**: Switch between OpenAI GPT and open-source models
                    - **RAG Technology**: Retrieval-Augmented Generation for accurate answers
                    - **Document Processing**: Handle PDFs and text files
                    - **Conversation Memory**: Context-aware multi-turn conversations
                    - **Privacy**: Option to run completely locally
                    
                    ### ⚙️ Configuration
                    - Edit `.env` file to change:
                      - Model type (hybrid/openai/local)
                      - API keys
                      - Chunk size and retrieval settings
                      - Temperature and token limits
                    
                    ### 📝 Tips
                    - Upload relevant documents for better answers
                    - Ask specific questions for accurate results
                    - Use clear language in your prompts
                    - Check the System Info to verify your configuration
                """)
        
        # Footer
        gr.Markdown(
            """
            ---
            **Multimodel Chatbot v1.0.0** | Built with ❤️ using Gradio, LangChain, and OpenAI
            
            [GitHub](https://github.com/) | [Documentation](docs/) | [Issues](https://github.com/issues)
            """
        )
    
    return demo

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point.
    
    Initializes the system and launches the Gradio web interface.
    """
    
    # Print startup message
    print("\n" + "="*70)
    print("🚀 Starting Multimodel Chatbot with RAG")
    print("="*70)
    
    # Initialize system components
    if not initialize_system():
        logger.error("Failed to initialize system. Exiting.")
        print("❌ System initialization failed. Please check your configuration.")
        return
    
    # Create Gradio interface
    logger.info("Creating Gradio interface...")
    demo = create_gradio_interface()
    
    # Launch the application
    print("\n" + "="*70)
    print("✅ Application initialized successfully!")
    print("="*70)
    print("\n🌐 Opening web interface...")
    print("📍 Access at: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Launch Gradio app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.getenv("GRADIO_SHARE", "False").lower() == "true"
    )

if __name__ == "__main__":
    main()
