# Quick Start: RAG-Anything with Ollama + Streamlit

Get up and running with RAG-Anything using local Ollama models in under 5 minutes!

## Prerequisites

- Python 3.8+
- Ollama installed ([ollama.ai](https://ollama.ai))
- 8GB+ RAM (16GB recommended)

## Installation (3 steps)

### 1. Install Dependencies

```bash
# Install Ollama dependencies
uv pip install -r requirements_ollama.txt

# Or using pip
pip install -r requirements_ollama.txt
```

### 2. Start Ollama & Pull Models

```bash
# Start Ollama server
ollama serve

# In another terminal, pull models
ollama pull gpt-oss:20b           # LLM (or use llama3.2:3b for faster)
ollama pull nomic-embed-text      # Embeddings
```

### 3. Verify Setup

```bash
python3 test_ollama_connection.py
```

You should see:
```
✅ All tests passed! You're ready to use the Streamlit app.
```

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Initialize**: Click "🚀 Initialize RAG System" in sidebar
2. **Upload**: Upload PDF documents
3. **Process**: Click "📤 Process Documents"
4. **Chat**: Ask questions about your documents!

## Quick Example

```python
# rag_ollama_setup.py is ready to use
python3 rag_ollama_setup.py
```

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| ❌ Connection refused | Run `ollama serve` |
| ❌ Module not found | Run `uv pip install ollama` |
| ❌ Model not found | Run `ollama pull <model>` |
| ❌ Out of memory | Use smaller model: `llama3.2:3b` |

**For detailed troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Files Created

- **`streamlit_app.py`**: Web interface for document chat
- **`rag_ollama_setup.py`**: Command-line example
- **`test_ollama_connection.py`**: Setup verification script
- **`STREAMLIT_GUIDE.md`**: Detailed usage guide
- **`TROUBLESHOOTING.md`**: Complete error reference

## Alternative Models

### Faster (Lower RAM)
```bash
ollama pull llama3.2:3b          # 3GB model
```
Update in Streamlit sidebar: LLM Model → `llama3.2:3b`

### Higher Quality (More RAM)
```bash
ollama pull llama3.1:70b         # 70GB model
```
Update in Streamlit sidebar: LLM Model → `llama3.1:70b`

## What Was Fixed

Your original error `ModuleNotFoundError: No module named 'ollama'` was fixed by:

1. ✅ Installing ollama package: `uv pip install ollama`
2. ✅ Using correct RAGAnything parameters (lightrag_kwargs)
3. ✅ Using correct method (insert_content_list instead of insert)
4. ✅ Using correct parameter names (host instead of base_url)

## Next Steps

- 📖 Read [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed features
- 🔧 Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter errors
- 💬 Start chatting with your documents!

---

**Need help?** Run `python3 test_ollama_connection.py` first to diagnose issues!
