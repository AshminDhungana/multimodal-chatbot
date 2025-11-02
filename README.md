# Multimodel Chatbot

A powerful conversational AI system featuring Retrieval-Augmented Generation (RAG) with support for multiple language models including OpenAI GPT and local Hugging Face models.

## 🌟 Features

- 🤖 **Multiple Model Support** — Switch between OpenAI GPT and local Hugging Face models
- 🔍 **Retrieval-Augmented Generation (RAG)** — Context-aware responses using your documents
- 💬 **Conversation Memory** — Multi-turn conversations with context retention
- 📄 **Document Processing** — Support for PDF, TXT, and other document formats
- 🎨 **Modern Web Interface** — Built with Gradio for intuitive user experience
- 🔐 **Privacy First** — Option to run completely locally with open-source models
- ⚡ **Fast & Efficient** — Optimized vector search with FAISS

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended for local models)
- OpenAI API key (optional, for GPT models)

### Installation

```bash
# Clone the repository
git clone https://github.com/AshminDhungana/multimodel-chatbot.git
cd multimodel-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your configuration
```

### Configuration

Edit the `.env` file with your settings:

```env
# Model Configuration
OPENAI_API_KEY=your_api_key_here  # Optional: Leave empty for local-only mode
MODEL_TYPE=hybrid                  # Options: hybrid, openai, local
DEFAULT_MODEL=gpt-3.5-turbo       # For OpenAI models

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Generation Parameters
TEMPERATURE=0.7
MAX_TOKENS=2048
TOP_K=5                           # Number of retrieved documents
```

### Usage

```bash
# Start the application
python app.py
```

Then open your browser and navigate to `http://localhost:7860`

## 📁 Project Structure

```
multimodel-chatbot/
├── app.py                      # Main Gradio application
├── src/
│   ├── rag_pipeline.py        # RAG logic and orchestration
│   ├── embeddings.py          # Embedding models manager
│   ├── llm_models.py          # LLM interface (OpenAI + HF)
│   ├── document_loader.py     # Document processing
│   └── vectorstore.py         # Vector database operations
├── data/
│   └── sample_documents/      # Sample files for testing
├── tests/
│   ├── test_rag.py           # RAG pipeline tests
│   ├── test_models.py        # Model integration tests
│   └── test_embeddings.py    # Embedding tests
├── docs/
│   ├── QUICKSTART.md
│   ├── SETUP_GUIDE.md
│   
├── requirements.txt          #  dependencies
├── .env                     # Environment template
├── .gitignore
├── LICENSE
├── README.md
```

## 💻 Technology Stack

- **[LangChain](https://langchain.com/)** — LLM orchestration and chains
- **[Gradio](https://gradio.app/)** — Web UI framework
- **[Hugging Face](https://huggingface.co/)** — Embeddings and local models
- **[OpenAI](https://openai.com/)** — GPT language models
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[Sentence Transformers](https://www.sbert.net/)** — Text embeddings

## 🎯 Use Cases

- **Document Q&A** — Ask questions about your documents
- **Knowledge Base Search** — Retrieve relevant information from large corpora
- **Research Assistant** — Get AI-powered insights from your research papers
- **Customer Support** — Build context-aware chatbots for support
- **Educational Tool** — Learn and explore topics with AI assistance

## 🔧 Advanced Configuration

### Using Local Models Only

Set `MODEL_TYPE=local` in your `.env` file to run completely offline:

```env
MODEL_TYPE=local
LOCAL_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

### Custom Document Sources

Add your documents to the `data/` directory or configure custom paths:

```python
from src.document_loader import DocumentLoader

loader = DocumentLoader()
documents = loader.load_directory("path/to/your/documents")
```

### Model Selection

Switch between models in the UI or programmatically:

```python
from src.llm_models import LLMManager

llm = LLMManager()
llm.set_model("gpt-4")  # or "mistral-7b", "llama-2-7b", etc.
```

## 📊 Performance

- **Response Time:** ~2-5 seconds (with OpenAI)
- **Local Mode:** ~5-15 seconds (depends on hardware)
- **Document Processing:** ~1-2 seconds per MB
- **Memory Usage:** 2-4GB (local models require 8-16GB)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Hugging Face for open-source models and infrastructure
- LangChain community for excellent documentation
- All contributors and supporters

## 📧 Contact

- **GitHub Issues:** [Report bugs or request features](https://github.com/AshminDhungana/multimodel-chatbot/issues)
- **Discussions:** [Join the conversation](https://github.com/AshminDhungana/multimodel-chatbot/discussions)


**Built with ❤️ for the AI community**