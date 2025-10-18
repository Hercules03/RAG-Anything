# RAG-Anything Streamlit Chatbot Guide

A simple web interface for uploading PDF documents and chatting with them using Ollama.

## Prerequisites

1. **Ollama installed and running**
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.ai/download

   # Start Ollama server
   ollama serve
   ```

2. **Pull required Ollama models**
   ```bash
   # LLM model for text generation
   ollama pull gpt-oss:20b
   # Or use a smaller model like:
   # ollama pull llama3.2:3b

   # Embedding model
   ollama pull nomic-embed-text
   ```

## Installation

1. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv pip install -r requirements_ollama.txt

   # Or using pip
   pip install -r requirements_ollama.txt
   ```

2. **Install RAG-Anything**
   ```bash
   pip install -e .
   ```

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### 1. Initialize RAG System
- In the sidebar, configure your settings:
  - **Ollama Host**: Default `http://localhost:11434`
  - **LLM Model**: Your Ollama model (e.g., `gpt-oss:20b`, `llama3.2:3b`)
  - **Embedding Model**: Default `nomic-embed-text`
  - **Working Directory**: Where RAG data is stored
  - **Output Directory**: Where processed documents are saved

- Click **"🚀 Initialize RAG System"**
- Wait for confirmation that Ollama is connected and RAG is initialized

### 2. Upload Documents
- Click **"Browse files"** to select PDF documents
- Select one or more PDF files
- Click **"📤 Process Documents"**
- Wait for processing to complete (may take a few minutes per document)

### 3. Chat with Your Documents
- Type your question in the chat input at the bottom
- Press Enter or click send
- The AI will analyze your documents and provide an answer
- Continue the conversation with follow-up questions

## Features

- ✅ **Multi-document support**: Upload and process multiple PDFs
- ✅ **Multimodal processing**: Handles text, images, tables, and equations
- ✅ **Local & Private**: Runs entirely on your machine with Ollama
- ✅ **Chat history**: Maintains conversation context
- ✅ **Document tracking**: Shows which documents have been processed
- ✅ **Easy reset**: Clear all documents and start fresh

## Troubleshooting

### Ollama Connection Error
**Error**: `❌ Error: Connection refused`

**Solution**:
```bash
# Make sure Ollama is running
ollama serve
```

### Model Not Found
**Error**: `model 'gpt-oss:20b' not found`

**Solution**:
```bash
# Pull the model first
ollama pull gpt-oss:20b

# Or use a different model that you have
ollama list  # See available models
```

### Slow Processing
If document processing is slow:
- Use a smaller LLM model (e.g., `llama3.2:3b` instead of `gpt-oss:20b`)
- Process documents one at a time
- Ensure you have enough RAM (8GB+ recommended)

### Memory Issues
**Error**: Out of memory

**Solution**:
- Close other applications
- Use a smaller model
- Process fewer documents at once
- Increase swap space on your system

### Parser Installation
If you get parser errors:
```bash
# For MinerU (default)
pip install magic-pdf[full]==0.7.1b1 --extra-index-url https://wheels.myhloli.com

# For Docling (alternative)
pip install docling
```

## Configuration Options

### Using Different Models

Edit the default values in the sidebar:

**For smaller/faster responses**:
- LLM Model: `llama3.2:3b` or `llama3.2:1b`
- Embedding Model: `nomic-embed-text`

**For better quality**:
- LLM Model: `gpt-oss:20b` or `llama3.1:70b`
- Embedding Model: `nomic-embed-text`

### Adjusting Context Size

The default context size is 32,768 tokens. To modify it, edit `streamlit_app.py`:

```python
"options": {
    "num_ctx": 32768,  # Change this value
    "temperature": 0.1
}
```

## Tips for Best Results

1. **Document Quality**: Use clear, text-based PDFs (not scanned images)
2. **Question Specificity**: Ask specific questions for better answers
3. **Context**: Reference specific sections or topics from your documents
4. **Model Selection**: Larger models give better answers but are slower
5. **Processing Time**: First query after processing may be slower

## Example Questions

After uploading a research paper:
- "What is the main contribution of this paper?"
- "Summarize the methodology used in this research"
- "What are the key findings?"
- "Can you explain the results in the performance table?"

After uploading a manual:
- "How do I configure the API settings?"
- "What are the system requirements?"
- "Explain the installation process step by step"

## Advanced Usage

### Custom Working Directory
Store RAG data in a specific location:
```
Working Directory: /path/to/your/rag_data
```

### Multiple Document Sets
- Use different working directories for different projects
- Clear documents between different topics

### Export Chat History
Currently, chat history is session-based. To export:
- Copy-paste from the chat interface
- Take screenshots of important conversations

## Performance Benchmarks

Typical processing times (on M1 Mac):
- **Initialization**: 5-10 seconds
- **Document Processing**: 1-3 minutes per 10-page PDF
- **Query Response**: 5-30 seconds depending on model

## Known Limitations

1. **PDF-only**: Currently only supports PDF files (not Word, Excel, etc.)
2. **Session-based**: Chat history clears when you close the browser
3. **No authentication**: Not suitable for multi-user deployment as-is
4. **Memory usage**: Large documents may require significant RAM

## Future Enhancements

Planned features:
- [ ] Support for more file formats (DOCX, PPTX, etc.)
- [ ] Persistent chat history
- [ ] Document export/import
- [ ] Multi-user support
- [ ] Advanced query modes
- [ ] Source citation in answers

## Getting Help

If you encounter issues:

1. Check the error message in the app
2. Review this troubleshooting guide
3. Check Ollama logs: `ollama logs`
4. Verify models are installed: `ollama list`
5. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information (OS, RAM, etc.)

## Resources

- [RAG-Anything Documentation](https://github.com/Your-Repo/RAG-Anything)
- [Ollama Documentation](https://ollama.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io)

---

**Happy chatting with your documents! 📚🤖**
