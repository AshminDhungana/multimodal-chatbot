# Complete Setup & Deployment Guide for Multimodel Chatbot

## 📋 Project Structure

```
multimodel-chatbot/
├── app.py                          # Main application entry point
├── requirements.txt                # Production dependencies
├── .env                             # Environment template
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
│
├── src/                            # Source code package
│   ├── __init__.py                # Package initialization
│   ├── rag_pipeline.py            # RAG pipeline logic
│   ├── llm_models.py              # Language model management
│   ├── document_loader.py         # Document processing
│   ├── embeddings.py              # Text embeddings
│   └── vectorstore.py             # Vector database (FAISS)
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_models.py
│   ├── test_embeddings.py
│   └── test_document_loader.py
│
├── docs/                           # Documentation
│   ├── QUICKSTART.md
│   ├── SETUP_GUIDE.md
│   
│
├── data/                           # Sample data
│   └── sample_documents/
│
└── vectorstore/                    # Vector database (created at runtime)
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/multimodel-chatbot.git
cd multimodel-chatbot
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit .env with your settings
# IMPORTANT: Add your OpenAI API key if using OpenAI models
OPENAI_API_KEY=sk-your-key-here
```

### Step 5: Run Application
```bash
python app.py
```

Then open: **http://localhost:7860**

---

## 🔧 Detailed Configuration

### Environment Variables (.env)

**Model Configuration:**
```env
# Choose model type: hybrid, openai, or local
MODEL_TYPE=hybrid

# Default model to load
DEFAULT_MODEL=gpt-3.5-turbo

# OpenAI API Key (required for OpenAI models)
OPENAI_API_KEY=sk-your-key-here

# Local model (for local/hybrid mode)
LOCAL_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**RAG Settings:**
```env
# How to split documents
CHUNK_SIZE=500              # Characters per chunk
CHUNK_OVERLAP=100           # Overlap between chunks
TOP_K=5                     # Results to retrieve

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Generation Settings:**
```env
TEMPERATURE=0.7             # 0=deterministic, 1=creative
MAX_TOKENS=2048             # Max response length
TOP_P=0.9                   # Nucleus sampling
```

**Paths:**
```env
DATA_DIR=./data             # Where uploaded documents go
VECTORSTORE_PATH=./vectorstore  # Where vector DB is stored
```

**Application:**
```env
GRADIO_PORT=7860            # Web interface port
GRADIO_SHARE=False          # Create public link?
LOG_LEVEL=INFO              # Logging level
```

---

## 💻 Running Locally

### Development Mode (with auto-reload)
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with debug logging
python app.py
```

### Production Mode
```bash
# Use gunicorn (WSGI server)
pip install gunicorn
gunicorn --bind 0.0.0.0:7860 app:demo
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_rag.py -v
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test
```bash
pytest tests/test_rag.py::test_document_chunking -v
```

---

## 📦 Docker Deployment

### Build Docker Image
```bash
# Create Dockerfile (if not present)
docker build -t multimodel-chatbot .

# Run container
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/vectorstore:/app/vectorstore \
  multimodel-chatbot
```

### Docker Compose
```bash
docker-compose up
```

---

## ☁️ Cloud Deployment

### Hugging Face Spaces (FREE)
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Docker" runtime
4. Upload your files
5. Add `OPENAI_API_KEY` as a secret
6. It deploys automatically!

### Streamlit Cloud (FREE)
```bash
# Create streamlit_app.py wrapper
streamlit run streamlit_app.py
```

Deploy from GitHub:
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect GitHub account
4. Select repository
5. Set secrets in dashboard

### Heroku (PAID)
```bash
# Install Heroku CLI
heroku login

# Create app
heroku create multimodel-chatbot

# Set environment variables
heroku config:set OPENAI_API_KEY=sk-your-key

# Deploy
git push heroku main
```

### AWS (PAID)
```bash
# Using AWS Lambda + API Gateway
# Or EC2 instance with Gunicorn/Nginx
```

---

## 🔐 Security Best Practices

### 1. Environment Variables
```bash
# ✅ DO: Use environment variables
OPENAI_API_KEY=$(cat /secret/key.txt)

# ❌ DON'T: Hardcode secrets
OPENAI_API_KEY="sk-xxx"  # Never!
```

### 2. Git Security
```bash
# ✅ DO: Use .gitignore
echo ".env" >> .gitignore

# ❌ DON'T: Commit secrets
git add .env  # Never!
```

### 3. File Permissions
```bash
# Make .env readable only by owner
chmod 600 .env
```

### 4. API Rate Limiting
```python
# In app.py, add rate limiting
from functools import wraps
import time

def rate_limit(calls_per_minute=60):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Ensure you're in project root
pwd  # Should show: .../multimodel-chatbot

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in .env
BATCH_SIZE=8  # Instead of 32
```

### Issue: "OpenAI API key not found"

**Solution:**
```bash
# Check .env is in root directory
ls -la .env

# Check it has the key
grep OPENAI_API_KEY .env

# Reload environment
export $(cat .env | xargs)
```

### Issue: "Vector store corrupted"

**Solution:**
```bash
# Delete and rebuild
rm -rf ./vectorstore

# Run app again - will create new index
python app.py
```

### Issue: "Model download fails"

**Solution:**
```bash
# Download manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Or set cache
export HF_HOME="/path/to/cache"
```

---

## 📊 Performance Optimization

### 1. Use Smaller Embedding Model
```env
# Faster (384D)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Better quality (768D, slower)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### 2. Use GPU if Available
```bash
# Check GPU
nvidia-smi

# Use GPU in .env
USE_GPU=True
```

### 3. Batch Processing
```env
# Process multiple documents at once
BATCH_SIZE=32
```

### 4. Caching
```env
# Enable caching
ENABLE_CACHE=True
CACHE_DIR=./cache
```

### 5. Index Optimization
```python
# In src/vectorstore.py
store.rebuild_index()  # After many operations
```

---

## 📈 Monitoring & Logging

### Set Log Level
```env
LOG_LEVEL=DEBUG    # Verbose logging
LOG_LEVEL=INFO     # Normal (default)
LOG_LEVEL=WARNING  # Only warnings
LOG_LEVEL=ERROR    # Only errors
```

### View Logs
```bash
# Print to console
tail -f app.log

# Search logs
grep "ERROR" app.log

# Filter by module
grep "rag_pipeline" app.log
```

---

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
# Check for updates
pip list --outdated

# Update all
pip install -U -r requirements.txt

# Update specific package
pip install -U langchain
```

### Version Management
```bash
# Tag release
git tag -a v1.1.0 -m "Version 1.1.0"

# Push tags
git push origin --tags
```

---

## 📚 API Integration

### Use as Python Package
```python
from src.rag_pipeline import RAGPipeline
from src.llm_models import LLMManager
from src.document_loader import DocumentLoader

# Initialize
loader = DocumentLoader()
llm = LLMManager()
rag = RAGPipeline()

# Use
chunks = loader.load_file("document.pdf")
rag.add_documents(chunks)
response = llm.generate_response("Query", context="...")
```

### Use as Web Service
```bash
# Deploy with FastAPI
pip install fastapi uvicorn

# Create api.py with FastAPI routes
# Deploy to cloud
```

---

## 🎯 Next Steps

1. **Customize UI**: Modify Gradio interface in `app.py`
2. **Add Models**: Support more LLMs in `llm_models.py`
3. **Improve RAG**: Add advanced techniques (HyDE, multi-query)
4. **Add API**: Create REST API with FastAPI
5. **Scale**: Deploy to production cloud platform

---


**You're ready! Good luck! 🚀**
