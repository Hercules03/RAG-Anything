# Testing Guide - RAG-Anything Streamlit App

## Quick Start

### 1. Start Ollama Server

```bash
ollama serve
```

Keep this terminal running.

### 2. Start Streamlit App

In a new terminal:

```bash
cd /Users/GitHub/RAG-Anything
streamlit run streamlit_app.py
```

The app should open in your browser at `http://localhost:8501`

### 3. Initialize RAG System

1. In the **sidebar**, verify settings:
   - Ollama Host: `http://localhost:11434`
   - LLM Model: `gpt-oss:20b`
   - Embedding Model: `nomic-embed-text`
   - Working Directory: `./rag_storage`
   - Output Directory: `./output`

2. Click **"🚀 Initialize RAG System"**

3. Wait for success messages:
   - ✅ Ollama connected
   - ✅ RAG system initialized

## Testing Workflow

### Test 1: Upload Single Document

1. Go to **"📄 Upload"** tab

2. Select **"Single Document"** mode

3. Configure chunking:
   - Strategy: `default` (start simple)
   - Chunk Size: `1200` tokens

4. Upload a PDF file

5. Click **"📤 Process Document"**

6. **Expected Result**:
   - ✅ Processing spinner appears
   - ✅ "Processed: filename.pdf" message
   - ✅ Regulation ID displayed (e.g., "APP-1")
   - ❌ NO event loop errors

### Test 2: View Database

1. Go to **"🗄️ Database"** tab

2. **Expected Result**:
   - ✅ Statistics shown (Total Documents, Pages, Chunks, Size)
   - ✅ Uploaded document appears in list
   - ✅ Document metadata visible (Doc ID, Status, Chunking strategy)

3. Try **Search**:
   - Type part of filename or regulation ID
   - ✅ Filtering works

### Test 3: View Chunks

1. Go to **"🔍 Chunks"** tab

2. Select your document from dropdown

3. **Expected Result**:
   - ✅ Chunk statistics displayed (Total, Avg/Min/Max tokens)
   - ✅ Validation shows "All chunks are valid" (green)
   - ✅ All chunks listed below

4. Try **Search in Chunks**:
   - Enter a word from your document
   - ✅ Matching chunks highlighted

### Test 4: Chat with Document

1. Go to **"💬 Chat"** tab

2. Ask a question about your document:
   - Example: "What are the main requirements?"
   - Example: "Summarize this regulation"

3. **Expected Result**:
   - ✅ "Thinking..." spinner
   - ✅ Response appears
   - ✅ Referenced regulations listed at bottom
   - ❌ NO event loop errors

### Test 5: Batch Processing (Optional)

1. Create a folder with multiple PDFs:
   ```bash
   mkdir -p ./documents
   # Copy your PDFs to ./documents/
   ```

2. Go to **"📄 Upload"** tab

3. Select **"Batch Processing (Folder)"** mode

4. Enter folder path: `./documents`

5. Choose chunking strategy

6. Click **"📦 Process All PDFs in Folder"**

7. **Expected Result**:
   - ✅ Progress bar shows processing
   - ✅ All PDFs processed
   - ✅ "Batch processing complete!" message

### Test 6: Advanced Operations

**Re-process Document**:
1. Go to **"🗄️ Database"** tab
2. Expand a document
3. Click **"🔄 Re-process"**
4. ✅ Document re-chunked with default strategy

**Delete Document**:
1. Expand a document
2. Click **"🗑️ Delete"**
3. ✅ Document removed from database

**Validate Database**:
1. Click **"📊 Validate Database"**
2. ✅ Shows validation results (issues, warnings, recommendations)

**Sync with LightRAG**:
1. Click **"🔄 Sync with LightRAG"**
2. ✅ Shows sync results

## Common Issues

### Issue: "Ollama not connected"
**Solution**: Make sure `ollama serve` is running

### Issue: "No module named 'utils'"
**Solution**: Run from project root directory

### Issue: Slow processing
**Expected**: Document parsing takes time (30s - 2min per PDF)
- MinerU parser is thorough but slow
- Normal for complex PDFs with images/tables

### Issue: High memory usage
**Expected**: LightRAG + Ollama are memory-intensive
- Monitor with `htop` or Activity Monitor
- Ensure sufficient RAM (8GB+ recommended)

### Issue: Event loop errors still appear
**Solution**:
1. Restart Streamlit completely (Ctrl+C, then restart)
2. Clear cache: `streamlit cache clear`
3. Verify `utils/async_helpers.py` exists
4. Check import: `from utils import run_async`

## Success Indicators

✅ **All Good**:
- No red error messages in terminal
- Documents process successfully
- Chunks appear in viewer
- Chat responds to queries
- Database operations work

❌ **Problems**:
- Event loop errors (PriorityQueue messages)
- "Module not found" errors
- Processing hangs indefinitely
- No chunks appear after processing

## Performance Expectations

### Document Processing Time:
- **Small PDF** (1-5 pages): 30-60 seconds
- **Medium PDF** (10-20 pages): 1-3 minutes
- **Large PDF** (50+ pages): 5-10 minutes

### Factors Affecting Speed:
- PDF complexity (images, tables, equations)
- Ollama model size (gpt-oss:20b is large)
- System CPU/RAM
- Chunking strategy (custom strategies slower than default)

## Sample Test Document

If you don't have building regulation PDFs yet, you can test with any PDF:

```bash
# Download a sample PDF
curl -o sample.pdf https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf

# Or use any PDF file you have
```

## Expected Terminal Output

**Successful Run**:
```
$ streamlit run streamlit_app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**During Document Processing** (normal):
```
INFO: Processing document: sample.pdf
INFO: Parsing with MinerU...
INFO: Extracting text and images...
INFO: Creating chunks...
INFO: Inserting into RAG...
```

**Event Loop Errors** (should NOT appear):
```
ERROR: LLM func: Critical error in worker: <PriorityQueue...>
ERROR: Embedding func: Critical error in worker: <PriorityQueue...>
```

If you see event loop errors, the fix didn't apply correctly.

## Next Steps

Once testing is successful:

1. **Upload your 200 building regulation PDFs**:
   - Use batch processing mode
   - Process in batches of 10-20 at a time
   - Monitor memory usage

2. **Validate chunk quality**:
   - Check chunk statistics
   - Ensure regulation IDs extracted correctly
   - Verify chunks are meaningful

3. **Test chat functionality**:
   - Ask domain-specific questions
   - Verify regulation citations appear
   - Check answer accuracy

4. **Experiment with chunking strategies**:
   - Try `fixed_token` with different sizes
   - Test `paragraph` chunking
   - Compare results in Chunk Viewer

## Support

If you encounter issues:

1. Check `ASYNCIO_FIX_SUMMARY.md` for event loop troubleshooting
2. Check `IMPLEMENTATION_SUMMARY.md` for feature documentation
3. Review terminal output for specific error messages
4. Verify all dependencies installed: `uv pip list`

Happy testing! 🚀
