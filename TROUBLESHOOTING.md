# RAG-Anything with Ollama - Troubleshooting Guide

This guide helps you debug and fix common issues when using RAG-Anything with Ollama.

## Table of Contents
1. [Original Error: ModuleNotFoundError: No module named 'ollama'](#error-1-modulenotfounderror)
2. [TypeError: unexpected keyword argument 'llm_model_name'](#error-2-typeerror-unexpected-keyword)
3. [AttributeError: 'RAGAnything' object has no attribute 'insert'](#error-3-attributeerror-no-insert)
4. [Ollama Connection Errors](#error-4-ollama-connection)
5. [General Debugging Steps](#general-debugging-steps)

---

## Error 1: ModuleNotFoundError

### Error Message
```
ModuleNotFoundError: No module named 'ollama'
```

### Cause
The `ollama` Python package is not installed in your environment.

### Solution

**Option 1: Using uv (recommended for this project)**
```bash
uv pip install ollama
```

**Option 2: Using pip**
```bash
pip install ollama
```

**Option 3: Install from requirements file**
```bash
# Make sure ollama is in requirements_ollama.txt
echo "ollama>=0.6.0" >> requirements_ollama.txt

# Then install
uv pip install -r requirements_ollama.txt
# or
pip install -r requirements_ollama.txt
```

### How to Prevent This in the Future
Always install dependencies before running scripts:
```bash
# Check requirements file first
cat requirements_ollama.txt

# Install all dependencies
uv pip install -r requirements_ollama.txt
```

---

## Error 2: TypeError: unexpected keyword argument

### Error Message
```
TypeError: RAGAnything.__init__() got an unexpected keyword argument 'llm_model_name'.
Did you mean 'llm_model_func'?
```

### Cause
RAG-Anything doesn't accept `llm_model_name` as a direct parameter. Instead, it should be passed through `lightrag_kwargs`.

### Wrong Code
```python
rag = RAGAnything(
    config=config,
    llm_model_func=ollama_model_complete,
    llm_model_name="gpt-oss:20b",  # ❌ Wrong!
    llm_model_kwargs={...}
)
```

### Correct Code
```python
rag = RAGAnything(
    config=config,
    llm_model_func=ollama_model_complete,
    embedding_func=embedding_func,
    lightrag_kwargs={  # ✅ Correct!
        "llm_model_name": "gpt-oss:20b",
        "llm_model_kwargs": {
            "options": {
                "num_ctx": 32768,
                "temperature": 0.1
            },
            "host": "http://localhost:11434"
        }
    }
)
```

### Key Points
1. Use `lightrag_kwargs` dictionary for LightRAG-specific parameters
2. Pass `llm_model_name` inside `lightrag_kwargs`
3. Pass `llm_model_kwargs` inside `lightrag_kwargs`
4. Use `host` parameter (not `base_url`) for Ollama

---

## Error 3: AttributeError: No 'insert' method

### Error Message
```
AttributeError: 'RAGAnything' object has no attribute 'insert'
```

### Cause
RAGAnything doesn't have a simple `insert()` method. You need to use `insert_content_list()`.

### Wrong Code
```python
sample_text = "Some text..."
await rag.insert(sample_text)  # ❌ Wrong!
```

### Correct Code
```python
# Format text as content list
sample_content = [
    {
        "type": "text",
        "text": "Your text content here",
        "page_idx": 0
    }
]

# Insert using correct method
await rag.insert_content_list(
    sample_content,
    file_path="sample_document"
)
```

### Content List Format

**Text content:**
```python
{
    "type": "text",
    "text": "Your text here",
    "page_idx": 0
}
```

**Image content:**
```python
{
    "type": "image",
    "img_path": "/absolute/path/to/image.jpg",
    "image_caption": ["Caption text"],
    "image_footnote": ["Footnote"],
    "page_idx": 1
}
```

**Table content:**
```python
{
    "type": "table",
    "table_body": "| Col1 | Col2 |\n|------|------|\n| A | B |",
    "table_caption": ["Table caption"],
    "table_footnote": ["Note"],
    "page_idx": 2
}
```

**Equation content:**
```python
{
    "type": "equation",
    "latex": "E = mc^2",
    "text": "Einstein's equation",
    "page_idx": 3
}
```

---

## Error 4: Ollama Connection

### Error Message
```
❌ Ollama is not running. Please start with: ollama serve
```

### Cause
Ollama server is not running or not accessible.

### Solution

**Step 1: Start Ollama**
```bash
ollama serve
```

**Step 2: Verify it's running**
```bash
# Check version
curl http://localhost:11434/api/version

# Or use the test script
python3 test_ollama_connection.py
```

**Step 3: Check if models are available**
```bash
ollama list
```

**Step 4: Pull required models if missing**
```bash
# LLM model (choose one)
ollama pull gpt-oss:20b       # Large, high quality
ollama pull llama3.2:3b        # Small, fast

# Embedding model
ollama pull nomic-embed-text
```

### Common Connection Issues

**Port Already in Use**
```bash
# Check what's using port 11434
lsof -i :11434

# Kill the process if needed
kill -9 <PID>

# Start Ollama again
ollama serve
```

**Firewall Blocking**
- Check firewall settings
- Allow port 11434
- Try accessing http://localhost:11434 in browser

**Wrong Host Configuration**
Make sure you're using:
- `host="http://localhost:11434"` (not `base_url`)
- Include `http://` prefix
- Check port number is correct

---

## General Debugging Steps

### 1. Check Your Environment

```bash
# Verify Python version (3.8+ required)
python3 --version

# Check if uv is being used
which python3
ls -la .venv/bin/

# List installed packages
uv pip list | grep ollama
uv pip list | grep streamlit
```

### 2. Verify Ollama Setup

```bash
# Run comprehensive test
python3 test_ollama_connection.py

# Manual checks
ollama list                    # List models
ollama serve                   # Start server
curl http://localhost:11434/api/version  # Test connection
```

### 3. Check File Permissions

```bash
# Make scripts executable
chmod +x rag_ollama_setup.py
chmod +x streamlit_app.py
chmod +x test_ollama_connection.py

# Check working directories
ls -la ./rag_storage
ls -la ./output
```

### 4. Enable Verbose Logging

Add to your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export VERBOSE=true
python3 your_script.py
```

### 5. Test Components Individually

**Test Ollama connection:**
```python
import requests
response = requests.get("http://localhost:11434/api/version")
print(response.json())
```

**Test embedding:**
```python
import ollama
client = ollama.Client(host="http://localhost:11434")
result = client.embed(model="nomic-embed-text", input=["test"])
print(len(result["embeddings"][0]))  # Should print 768
```

**Test LLM:**
```python
import asyncio
from lightrag.llm.ollama import ollama_model_complete

async def test():
    # Note: This needs proper initialization with hashing_kv
    # Better to test through RAGAnything
    pass
```

---

## Quick Reference: Correct Script Structure

### Minimal Working Example

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # 1. Create config
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
    )

    # 2. Initialize RAG with Ollama
    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            ),
        ),
        lightrag_kwargs={
            "llm_model_name": "gpt-oss:20b",
            "llm_model_kwargs": {
                "options": {"num_ctx": 32768, "temperature": 0.1},
                "host": "http://localhost:11434"
            }
        }
    )

    # 3. Insert content
    content = [{"type": "text", "text": "Test", "page_idx": 0}]
    await rag.insert_content_list(content, file_path="test")

    # 4. Query
    result = await rag.aquery("What is this about?", mode="hybrid")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Getting More Help

### Before Asking for Help

1. Run the test script: `python3 test_ollama_connection.py`
2. Check Ollama logs: `ollama logs` (if available)
3. Verify all dependencies: `uv pip list`
4. Try the minimal example above
5. Check this troubleshooting guide

### What to Include in Bug Reports

1. **Error message** (full traceback)
2. **Python version**: `python3 --version`
3. **Ollama version**: `ollama --version`
4. **Test results**: Output from `test_ollama_connection.py`
5. **Code snippet** that causes the error
6. **Steps to reproduce**

### Useful Commands for Debugging

```bash
# System info
python3 --version
ollama --version
uv --version

# Package info
uv pip list | grep -E "(ollama|streamlit|lightrag|raganything)"

# Check processes
ps aux | grep ollama
lsof -i :11434

# Test connection
curl -v http://localhost:11434/api/version

# Run test script
python3 test_ollama_connection.py

# Check logs
tail -f raganything_example.log
```

---

## Common Issues Summary

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: ollama` | Package not installed | `uv pip install ollama` |
| `TypeError: llm_model_name` | Wrong parameter location | Use `lightrag_kwargs` |
| `AttributeError: insert` | Wrong method name | Use `insert_content_list()` |
| `Connection refused` | Ollama not running | `ollama serve` |
| `Model not found` | Model not pulled | `ollama pull <model>` |
| `Out of memory` | Model too large | Use smaller model |
| `Timeout` | Slow response | Increase timeout or use faster model |

---

**Remember**: Always start with `python3 test_ollama_connection.py` to verify your setup! 🎯
