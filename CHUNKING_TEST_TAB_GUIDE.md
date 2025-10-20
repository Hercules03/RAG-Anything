# Chunking Test Tab Guide

**Status**: ✅ Implemented
**Date**: 2025-01-19
**Location**: streamlit_app.py - Tab 5

---

## Overview

The **Chunking Test Tab** provides an interactive interface for testing different chunking strategies on PDF documents without inserting them into the RAG system. This allows users to experiment with chunking parameters and see real-time results.

---

## Features

### 1️⃣ PDF Upload
- Upload any PDF document for testing
- No storage in RAG system (temporary processing only)
- Supports all document types (building regulations, manuals, papers, etc.)

### 2️⃣ Strategy Selection
Choose from 4 chunking strategies:
- **Fixed Token**: Fixed-size chunks with overlap
- **Paragraph**: Paragraph-based chunking
- **Hybrid**: Combination of strategies
- **Hierarchical**: Parent-child relationships (best for structured docs)

### 3️⃣ Parameter Configuration

#### Fixed Token Strategy
- **Chunk Size**: 100-4000 tokens (default: 1200)
- **Overlap**: 0-500 tokens (default: 100)
- **Split on Sentences**: Enable/disable sentence boundary splitting

#### Paragraph Strategy
- **Chunk Size**: Base size for paragraphs
- **Min Tokens**: Minimum tokens per chunk (default: 600)
- **Max Tokens**: Maximum tokens per chunk (default: 1800)

#### Hybrid Strategy
- **Chunk Size**: Target chunk size
- **Hybrid Strategy**: balanced | paragraph-first | semantic-first

#### Hierarchical Strategy
- **Parent Chunk Size**: 500-4000 tokens (default: 1500)
- **Child Chunk Size**: Main chunk size setting
- **Child Overlap**: 0-200 tokens (default: 100)
- **Extract Sections**: Detect document structure (recommended)
- **3-Level Hierarchy**: Enable intermediate level (optional)

---

## Usage Workflow

### Step 1: Upload PDF
1. Click "Choose a PDF file to test chunking"
2. Select your PDF document
3. Wait for upload confirmation

### Step 2: Configure Strategy
1. Select chunking strategy from dropdown
2. Set chunk size (applies to all strategies)
3. Configure strategy-specific parameters

### Step 3: Chunk Document
1. Click "🔬 Chunk Document" button
2. Wait for PDF parsing (progress shown)
3. Wait for chunking process
4. View results automatically

### Step 4: Analyze Results
- View chunking statistics (total chunks, avg tokens, etc.)
- Browse individual chunks with different view modes
- Filter hierarchical chunks by level (parents/children)

---

## Results Display

### Statistics Dashboard

**Standard Metrics** (all strategies):
- Total Chunks created
- Average Tokens per Chunk
- Total Tokens processed
- Source Characters (original text length)

**Hierarchical Metrics** (hierarchical strategy only):
- Parent Chunks count
- Child Chunks count
- Average Children per Parent ratio

### View Modes

#### Compact View
- Shows first 300 characters of each chunk
- Quick overview of content distribution
- Hierarchical metadata caption (if applicable)

#### Detailed View
- Full chunk content in text area
- Complete metadata display:
  - Token count
  - Chunk index
  - Level (hierarchical)
  - Parent ID (hierarchical)
  - Children count (hierarchical)
  - Section title (hierarchical)

#### Metadata Only
- JSON display of all chunk metadata
- No content preview
- Useful for analyzing structure

### Filtering Options

**For Hierarchical Strategy**:
- **All**: Show all chunks (parents + children)
- **Parents Only**: Show only parent chunks (level 0)
- **Children Only**: Show only child chunks (level 1 or 2)

---

## Example Use Cases

### Use Case 1: Testing Optimal Chunk Size
**Scenario**: Find the best chunk size for building regulations

**Steps**:
1. Upload sample building regulation PDF
2. Test with different chunk sizes (300, 600, 1200, 1800)
3. Compare avg tokens and total chunks
4. Evaluate content completeness in detailed view
5. Select optimal size based on results

**Example Results**:
```
Chunk Size 300:  150 chunks, 290 avg tokens → Very fragmented
Chunk Size 600:  75 chunks, 580 avg tokens → Good balance
Chunk Size 1200: 40 chunks, 1150 avg tokens → Larger context
Chunk Size 1800: 25 chunks, 1750 avg tokens → Too large, loses precision
```

### Use Case 2: Comparing Strategies
**Scenario**: Compare fixed_token vs hierarchical for technical manual

**Steps**:
1. Upload technical manual PDF
2. Test with `fixed_token` (chunk_size=1200, overlap=100)
3. Note total chunks and browse results
4. Test with `hierarchical` (parent=1500, child=300)
5. Compare parent-child structure vs flat structure
6. Evaluate which preserves context better

**Example Comparison**:
```
Fixed Token:
- 50 chunks
- Flat structure
- 1150 avg tokens
- No section awareness

Hierarchical:
- 60 total chunks (25 parents, 35 children)
- Parent-child relationships
- 1450 avg tokens (parents), 280 avg tokens (children)
- Section titles extracted
- Better for retrieval (10-15% improvement expected)
```

### Use Case 3: Validating Section Detection
**Scenario**: Verify hierarchical chunking detects document structure

**Steps**:
1. Upload PDF with clear headings
2. Select `hierarchical` strategy
3. Enable "Extract Sections"
4. Chunk document
5. Browse chunks and check section_title metadata
6. Filter "Parents Only" to see detected sections

**Expected Results**:
- Each parent chunk has section_title
- Section titles match PDF headings
- Proper parent-child linking
- Logical content grouping

---

## Tips and Best Practices

### General Tips
1. **Start with Default Settings**: Use .env defaults as starting point
2. **Compare Strategies**: Test multiple strategies on same document
3. **Check Statistics First**: Review metrics before browsing chunks
4. **Use Compact View First**: Quick overview before detailed analysis
5. **Test Representative Documents**: Use typical documents from your use case

### Optimization Tips
1. **Fixed Token**:
   - Enable "Split on Sentences" for better readability
   - Increase overlap for better context continuity
   - Reduce chunk size if too much information per chunk

2. **Paragraph**:
   - Increase max_tokens if paragraphs getting split unnecessarily
   - Decrease min_tokens if too much merging happening
   - Works best with well-structured documents

3. **Hierarchical**:
   - Always enable "Extract Sections" for structured docs
   - Set parent size 3-5x larger than child size
   - Disable "Extract Sections" for unstructured narrative text
   - Use "Children Only" filter to see what gets embedded in vector DB

### Performance Tips
1. **Large PDFs**: Start with smaller chunk sizes to reduce processing time
2. **Testing Multiple Configurations**: Use same PDF repeatedly to compare
3. **Session State**: Results persist until new chunking performed
4. **Browser Performance**: For 100+ chunks, use "Metadata Only" view

---

## Troubleshooting

### Issue: No text extracted from PDF
**Cause**: PDF is image-based (scanned) or corrupt
**Solution**:
- Ensure PDF has selectable text
- Try different PDF
- Check console for parsing errors

### Issue: Too many small chunks
**Cause**: Chunk size too small or document highly fragmented
**Solution**:
- Increase chunk size
- For hierarchical: increase parent_chunk_size
- Try paragraph strategy instead

### Issue: Chunks too large
**Cause**: Chunk size too large or paragraphs being merged
**Solution**:
- Decrease chunk size
- For paragraph: decrease max_tokens
- Enable "Split on Sentences" for fixed_token

### Issue: Section titles not detected (hierarchical)
**Cause**: Document doesn't have recognizable heading patterns
**Solution**:
- Verify PDF has actual headings (# Header, numbered sections, etc.)
- Try disabling "Extract Sections" to treat as single section
- Use different chunking strategy for unstructured docs

### Issue: Slow processing
**Cause**: Large PDF or complex chunking strategy
**Solution**:
- Wait for parsing to complete (mineru can take time)
- Try smaller test PDF first
- Reduce chunk size to decrease total chunks

---

## Technical Details

### PDF Parsing
- Uses RAGAnything parser (mineru or docling from .env)
- Temporary working directory created automatically
- Extracted text cached in session state
- Temp files cleaned up automatically

### Chunking Process
1. PDF uploaded → temp file created
2. Parser extracts text content
3. ChunkingStrategies.apply_strategy() called
4. Chunks stored in `st.session_state.test_chunks`
5. Results displayed with statistics

### Session State Variables
```python
st.session_state.test_chunks        # List[Dict[str, Any]] - chunked results
st.session_state.test_strategy      # str - strategy used
st.session_state.test_text_length   # int - original text length
```

### Memory Management
- Chunks persist in session state until new chunking
- Temp files deleted after parsing
- No storage in RAG system or database
- Browser memory used for chunk display

---

## Integration with Main Workflow

### Testing Before Upload
1. Use Chunking Test tab to find optimal strategy
2. Note the best parameters (chunk_size, overlap, etc.)
3. Go to "Upload" tab
4. Configure with tested parameters
5. Upload documents with confidence

### Comparing with Stored Chunks
1. Upload document in "Upload" tab with strategy A
2. View stored chunks in "Chunks" tab
3. Re-test same document in "Chunking Test" with strategy B
4. Compare results to evaluate improvement
5. Re-upload with better strategy if needed

---

## UI Components

### Layout Structure
```
📚 RAG-Anything Document Manager
└── Tab 5: 🧪 Chunking Test
    ├── 1️⃣ Upload Test Document
    │   └── PDF file uploader
    ├── 2️⃣ Select Chunking Strategy
    │   ├── Strategy dropdown
    │   └── Chunk size input
    ├── 3️⃣ Configure Parameters
    │   └── Strategy-specific controls
    ├── 🔬 Chunk Document Button
    ├── 📊 Chunking Results
    │   ├── Statistics metrics
    │   ├── Hierarchical structure (if applicable)
    │   └── 🔍 Browse Chunks
    │       ├── View mode selector
    │       ├── Level filter (hierarchical)
    │       └── Expandable chunk list
```

### Interactive Elements
- File uploader (PDF only)
- Strategy selectbox (4 options)
- Dynamic parameter controls (based on strategy)
- Primary action button (chunk document)
- View mode radio (3 options)
- Level filter selectbox (hierarchical only)
- Expandable chunk expanders

---

## Future Enhancements

Potential improvements for future versions:

1. **Comparison Mode**:
   - Side-by-side strategy comparison
   - Diff viewer for chunk boundaries
   - Metrics comparison table

2. **Export Features**:
   - Download chunks as JSON
   - Export statistics as CSV
   - Save configuration presets

3. **Visualization**:
   - Chunk size distribution histogram
   - Hierarchical tree visualization
   - Token usage heatmap

4. **Advanced Analytics**:
   - Semantic similarity between chunks
   - Overlap analysis (redundancy detection)
   - Retrieval simulation

5. **Batch Testing**:
   - Test multiple PDFs at once
   - Aggregate statistics across documents
   - Best strategy recommendation

---

## Summary

The Chunking Test Tab provides a powerful, user-friendly interface for:
- ✅ Testing chunking strategies without affecting RAG system
- ✅ Comparing different parameters and configurations
- ✅ Visualizing chunk structure and metadata
- ✅ Optimizing chunking before production use
- ✅ Understanding how different strategies work

**Key Benefits**:
- No side effects (temporary processing only)
- Instant feedback on chunking results
- Interactive parameter tuning
- Comprehensive statistics and metadata
- Support for all chunking strategies including hierarchical

**Perfect For**:
- Finding optimal chunk size for your documents
- Comparing chunking strategies
- Understanding hierarchical structure
- Debugging chunking issues
- Training and demonstration
