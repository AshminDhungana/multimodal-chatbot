# 🚀 Quick Start Guide - Multimodel Chatbot

## 5-Minute Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) OpenAI API key

### Installation

```bash
# 1. Navigate to project
cd multimodel-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your OpenAI key (if using OpenAI models)

# 5. Run application
python app.py
```

Then open: **http://localhost:7860**

---

## 📖 First Time Usage

### 1. Upload Documents
- Go to **📄 Documents** tab
- Click "Select a document"
- Choose PDF, TXT, DOCX, or MD file
- Click "📤 Upload and Process"
- Wait for confirmation

### 2. Ask Questions
- Go to **💬 Chat** tab
- Type your question
- Press Enter or click Send
- AI will search documents and answer

### 3. Switch Models
- Go to **🎛️ Settings** tab
- Select a different model
- Click "🔄 Switch Model"
- Next message uses new model

### 4. Clear Data
- In **📄 Documents** tab: Click "🗑️ Clear All Documents"
- In **💬 Chat** tab: History auto-clears

---

## 🔧 Configuration

### Model Types

**Hybrid (Recommended)** - Use both OpenAI and local models
```env
MODEL_TYPE=hybrid
OPENAI_API_KEY=sk-your-key-here
```

**OpenAI Only** - Use only GPT models (requires API key)
```env
MODEL_TYPE=openai
OPENAI_API_KEY=sk-your-key-here
```

**Local Only** - Use only open-source models (free, offline)
```env
MODEL_TYPE=local
```

### Model Selection

**Fast & Free:**
```env
DEFAULT_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**High Quality:**
```env
DEFAULT_MODEL=gpt-4
OPENAI_API_KEY=sk-your-key-here
```

**Balanced:**
```env
DEFAULT_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-your-key-here
```

### RAG Settings

**For Large Documents:**
```env
CHUNK_SIZE=1000      # Larger chunks
CHUNK_OVERLAP=200    # More overlap
TOP_K=10             # More results
```

**For Fast Response:**
```env
CHUNK_SIZE=300       # Smaller chunks
CHUNK_OVERLAP=50     # Less overlap
TOP_K=3              # Fewer results
```

---

## 🎯 Common Tasks

### Upload Multiple Documents
```bash
# Copy all files to data/ folder
cp *.pdf ./data/

# Then upload from UI or code:
from src.document_loader import DocumentLoader
loader = DocumentLoader()
chunks = loader.load_directory("./data/")
```

### Change Embedding Model
Edit `.env`:
```env
# Default (fast)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Better quality
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### Use GPU
```env
# Enable GPU (if available)
USE_GPU=True
```

### Change Port
```env
# Instead of 7860
GRADIO_PORT=8080
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check you're in venv
which python  # Should show venv path
```

### "OpenAI key not found"
```bash
# Check .env exists
ls -la .env

# Reload environment
export $(cat .env | xargs)

# Or restart terminal
```

### "Out of memory"
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Reduce batch size in .env
BATCH_SIZE=8
```

### "Port already in use"
```bash
# Use different port in .env
GRADIO_PORT=8080

# Or kill process on port 7860
lsof -ti :7860 | xargs kill -9
```

---

## 💡 Tips & Tricks

### 1. Smart Document Uploading
- Upload similar documents together
- Use descriptive filenames
- Split huge files (>50MB) before uploading

### 2. Better Prompts
```
Good: "Summarize the key points about machine learning"
Bad: "Tell me about the document"

Good: "What are the benefits of neural networks?"
Bad: "What is in the document?"
```

### 3. Use Local Models for Privacy
```env
MODEL_TYPE=local
DEFAULT_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```
- No data sent to servers
- Free (no API costs)
- Slower on CPU

### 4. Export Conversations
```python
# Save chat history
conversation_history = [
    ("User question", "AI response"),
    ("Another question", "Another response")
]

import json
with open("conversation.json", "w") as f:
    json.dump(conversation_history, f)
```

### 5. Batch Process Documents
```python
from src.document_loader import DocumentLoader
from src.rag_pipeline import RAGPipeline

loader = DocumentLoader()
rag = RAGPipeline()

# Load multiple files
files = ["doc1.pdf", "doc2.txt", "doc3.docx"]
chunks = loader.load_files(files)

# Add all at once
rag.add_documents(chunks)
rag.save_vectorstore()
```

---

## 📊 Performance Expectations

| Task | Time | Notes |
|------|------|-------|
| Upload 10-page PDF | 2-5 seconds | Depends on file size |
| First question | 3-5 seconds | Model loads once |
| Follow-up question | 1-3 seconds | Using cached model |
| Search similar docs | 100-500ms | Very fast |
| Rebuild index | 5-10 seconds | After many operations |

---

## 🔗 Integration Examples

### Use in Python Code
```python
from src.rag_pipeline import RAGPipeline
from src.llm_models import LLMManager

# Initialize
rag = RAGPipeline()
llm = LLMManager()

# Add documents
rag.add_documents(["Your document text here"])

# Search and answer
results = rag.retrieve("What is AI?", top_k=3)
response = llm.generate_response(
    query="What is AI?",
    context="\n".join([r["content"] for r in results])
)
print(response)
```

### Use as Web API
```python
# Create fastapi_app.py
from fastapi import FastAPI
from src.rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

@app.post("/search")
def search(query: str):
    results = rag.retrieve(query, top_k=5)
    return results

# Run: uvicorn fastapi_app:app --reload
```

---

### Code Comments
Each file has detailed docstrings:
```python
# Every class has documentation
class MyClass:
    """Detailed explanation of what this class does."""
    
    def my_method(self):
        """Detailed explanation of what this method does."""
        pass
```

### GitHub
- Open an issue for bugs
- Discuss in discussions tab
- Check existing issues first

---

## ✨ Features Quick Reference

| Feature | Command | Details |
|---------|---------|---------|
| Run app | `python app.py` | Starts on http://localhost:7860 |
| Upload docs | UI button | Supports PDF, TXT, DOCX, MD |
| Change model | Settings tab | Switch between models instantly |
| Clear data | Documents tab | Delete all uploaded files |
| View logs | Terminal | See real-time activity |
| Save conversations | Code | Export as JSON |
| Run tests | `pytest` | Unit tests for all modules |
| Install dev tools | `pip install -r requirements-dev.txt` | Development dependencies |

---

## 🎯 Project Layout

```
Your local copy:
multimodel-chatbot/
├── app.py                 ← Start here
├── .env                   ← Your config (create from .env.example)
├── README.md              ← Main documentation
├── src/                   ← Core code (RAG, LLM, etc)
├── data/                  ← Your uploaded documents
├── vectorstore/           ← AI search database (created auto)
└── requirements.txt       ← All dependencies
```

---

## 🚀 You're Ready!

Run this command and you're live:

```bash
python app.py
```

Then visit: **http://localhost:7860**

**Enjoy exploring AI! 🎉**
