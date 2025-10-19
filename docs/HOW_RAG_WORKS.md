# How RAG-Anything Works in Streamlit App

A complete explanation of the RAG pipeline from document upload to answering queries.

## 🔄 Traditional RAG vs RAG-Anything

### Traditional RAG Pipeline

```
Document → Parsing → Chunking → Vectorization → VectorDB → Query → Answer
           ↓          ↓           ↓               ↓
         Text only  Fixed size  Text embeddings  Similarity search
```

**Limitations:**
- ❌ Only processes text, loses images/tables/equations
- ❌ Fixed chunk sizes break context
- ❌ Simple similarity search misses relationships
- ❌ No understanding of document structure

### RAG-Anything Pipeline (Graph-Based + Multimodal)

```
Document → Multi-Parser → Content Separation → Dual Processing Path
           ↓              ↓                     ↓
         MinerU/        Text +                Text Path | Multimodal Path
         Docling        Multimodal             ↓             ↓
                                           LightRAG      Modal Processors
                                              ↓               ↓
                                         Knowledge      Visual/Table
                                          Graph         Embeddings
                                              ↓               ↓
                                         Entity/        Caption +
                                         Relations      Context
                                              ↓               ↓
                                              └─── Hybrid Query ───┘
                                                       ↓
                                                   Rich Answer
```

**Advantages:**
- ✅ Processes ALL content types (text, images, tables, equations)
- ✅ Preserves document structure and context
- ✅ Knowledge graph for entity relationships
- ✅ Hybrid retrieval: vector + graph + multimodal

---

## 📊 Detailed Pipeline Breakdown

### Phase 1: Initialization (When you click "Initialize RAG System")

```python
# streamlit_app.py: initialize_rag()

rag = RAGAnything(
    config=config,                    # Document processing settings
    llm_model_func=ollama_model_complete,  # LLM for text analysis
    embedding_func=EmbeddingFunc(...),     # Embedding function
    lightrag_kwargs={                      # LightRAG config
        "llm_model_name": "gpt-oss:20b",
        "llm_model_kwargs": {...}
    }
)
```

**What happens internally:**

1. **RAGAnything Initialization** (`raganything/raganything.py:__post_init__`)
   - Creates working directory (`./rag_storage`)
   - Sets up parser (MinerU or Docling)
   - Initializes configuration

2. **LightRAG Initialization** (when first document is processed)
   - Creates **Knowledge Graph Storage** (GraphML format)
   - Sets up **Vector Database** (using nano-vectordb by default)
   - Initializes **Key-Value Stores**:
     - `full_docs`: Complete documents
     - `text_chunks`: Text chunks with metadata
     - `full_entities`: Extracted entities
     - `full_relations`: Entity relationships
     - `llm_response_cache`: Cached LLM responses
     - `doc_status`: Document processing status
     - `parse_cache`: Cached parsing results

3. **Modal Processors Setup**
   - **ImageModalProcessor**: For images/diagrams
   - **TableModalProcessor**: For structured tables
   - **EquationModalProcessor**: For math formulas
   - **GenericModalProcessor**: Fallback for other types

---

### Phase 2: Document Upload & Processing

#### Step 2.1: Upload PDF in Streamlit

```python
# User uploads PDF → Saved to temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_path = tmp_file.name
```

#### Step 2.2: Process Document Complete

```python
await rag.process_document_complete(
    file_path=tmp_path,
    output_dir=output_dir,
    parse_method="auto"
)
```

**What happens:**

```
process_document_complete() in raganything/processor.py
    ↓
1. Check parse cache (reuse if document unchanged)
    ↓
2. Parse document using MinerU/Docling
    ↓
3. Extract content list (text, images, tables, equations)
    ↓
4. Store parse result in cache
    ↓
5. Call insert_content_list() to process content
```

---

### Phase 3: Document Parsing (MinerU)

**MinerU Parser** (`raganything/parser.py: MineruParser`)

```python
# MinerU processes PDF and extracts:
content_list = [
    {
        "type": "text",
        "text": "Document text content...",
        "page_idx": 0
    },
    {
        "type": "image",
        "img_path": "/absolute/path/to/extracted_image.jpg",
        "image_caption": ["Figure 1: Architecture"],
        "page_idx": 1
    },
    {
        "type": "table",
        "table_body": "| Header1 | Header2 |\n|---------|---------|...",
        "table_caption": ["Table 1: Results"],
        "page_idx": 2
    },
    {
        "type": "equation",
        "latex": "E = mc^2",
        "text": "Energy-mass equivalence",
        "page_idx": 3
    }
]
```

**MinerU Features:**
- ✅ Layout analysis (detects headers, paragraphs, columns)
- ✅ Image extraction (extracts figures/diagrams)
- ✅ Table recognition (converts to markdown)
- ✅ Equation detection (extracts LaTeX)
- ✅ Text extraction (with OCR fallback)

---

### Phase 4: Content Separation & Dual Processing

**Content Separation** (`raganything/utils.py: separate_content()`)

```python
def separate_content(content_list):
    text_content = []      # Pure text blocks
    multimodal_items = []  # Images, tables, equations

    for item in content_list:
        if item["type"] == "text":
            text_content.append(item["text"])
        else:
            multimodal_items.append(item)

    return text_content, multimodal_items
```

**Result:**
- **Text Content**: `"Combined text from all text blocks..."`
- **Multimodal Items**: List of images, tables, equations with metadata

---

### Phase 5: Text Processing Path (LightRAG)

**LightRAG Processing** (`raganything/utils.py: insert_text_content()`)

```python
await lightrag.ainsert(text_content)
```

**What LightRAG does:**

1. **Text Chunking**
   ```python
   # LightRAG chunks text intelligently (not fixed size)
   chunks = [
       "Chunk 1: Introduction and overview...",
       "Chunk 2: Methodology description...",
       "Chunk 3: Results and discussion..."
   ]
   ```

2. **Entity Extraction** (using LLM)
   ```python
   # LLM extracts entities from each chunk
   entities = [
       {"name": "RAG-Anything", "type": "SYSTEM", "description": "..."},
       {"name": "LightRAG", "type": "FRAMEWORK", "description": "..."},
       {"name": "Knowledge Graph", "type": "CONCEPT", "description": "..."}
   ]
   ```

3. **Relationship Extraction** (using LLM)
   ```python
   # LLM identifies relationships between entities
   relations = [
       {
           "source": "RAG-Anything",
           "target": "LightRAG",
           "type": "BASED_ON",
           "description": "RAG-Anything is built on top of LightRAG"
       },
       {
           "source": "LightRAG",
           "target": "Knowledge Graph",
           "type": "USES",
           "description": "LightRAG uses knowledge graph for retrieval"
       }
   ]
   ```

4. **Knowledge Graph Construction**
   ```
   Graph Structure (GraphML):

   [RAG-Anything] ──BASED_ON──> [LightRAG] ──USES──> [Knowledge Graph]
         │                            │
         │                            └──SUPPORTS──> [Multimodal Processing]
         │
         └──INCLUDES──> [Document Parsing]
   ```

5. **Vector Embeddings**
   ```python
   # Generate embeddings for each chunk
   chunk_embeddings = embedding_func([
       "Chunk 1: Introduction and overview...",
       "Chunk 2: Methodology description...",
       "Chunk 3: Results and discussion..."
   ])
   # Result: numpy array of shape (3, 768)
   ```

6. **Storage**
   - **Vector DB**: Stores chunk embeddings for similarity search
   - **Knowledge Graph**: Stores entities and relations
   - **KV Store**: Stores original chunks and metadata

---

### Phase 6: Multimodal Processing Path

**For Each Multimodal Item** (`raganything/processor.py: _process_multimodal_content()`)

#### 6.1 Image Processing

```python
# ImageModalProcessor processes image
processor = ImageModalProcessor(
    lightrag=lightrag,
    modal_caption_func=vision_model_func,  # VLM for image understanding
    context_extractor=context_extractor
)

# 1. Extract surrounding context
context = context_extractor.extract_context(
    content_list=content_list,
    target_item=image_item,
    page_idx=image_item["page_idx"]
)
# Context includes text before/after image, captions, headers

# 2. Generate image caption using VLM
prompt = f"""
Context: {context}
Please describe this image in detail, considering the surrounding context.
"""
caption = await vision_model_func(
    prompt=prompt,
    image_data=base64_encoded_image
)

# 3. Create multimodal document
multimodal_doc = f"""
[IMAGE: {image_path}]
Context: {context}
Caption: {caption}
Page: {page_idx}
"""

# 4. Insert into LightRAG
await lightrag.ainsert(multimodal_doc)
```

**Result:**
- Image content is converted to text description
- Context provides semantic grounding
- Inserted into knowledge graph with entities/relations

#### 6.2 Table Processing

```python
# TableModalProcessor processes table
processor = TableModalProcessor(
    lightrag=lightrag,
    modal_caption_func=llm_model_func,
    context_extractor=context_extractor
)

# 1. Extract context
context = context_extractor.extract_context(...)

# 2. Analyze table using LLM
prompt = f"""
Context: {context}

Table (Markdown format):
{table_markdown}

Please analyze this table and provide:
1. Summary of what the table shows
2. Key insights and patterns
3. Relationship to the surrounding context
"""
analysis = await llm_model_func(prompt)

# 3. Create multimodal document
multimodal_doc = f"""
[TABLE]
Context: {context}
Table Data:
{table_markdown}
Analysis: {analysis}
Page: {page_idx}
"""

# 4. Insert into LightRAG
await lightrag.ainsert(multimodal_doc)
```

#### 6.3 Equation Processing

```python
# EquationModalProcessor processes equation
processor = EquationModalProcessor(
    lightrag=lightrag,
    modal_caption_func=llm_model_func,
    context_extractor=context_extractor
)

# 1. Extract context
context = context_extractor.extract_context(...)

# 2. Explain equation using LLM
prompt = f"""
Context: {context}

LaTeX Equation: {latex_formula}
Text Description: {equation_text}

Please explain:
1. What this equation represents
2. Its significance in the context
3. How it relates to the surrounding content
"""
explanation = await llm_model_func(prompt)

# 3. Create multimodal document
multimodal_doc = f"""
[EQUATION]
Context: {context}
LaTeX: {latex_formula}
Explanation: {explanation}
Page: {page_idx}
"""

# 4. Insert into LightRAG
await lightrag.ainsert(multimodal_doc)
```

---

### Phase 7: Storage Structure

After processing, your `./rag_storage` directory contains:

```
./rag_storage/
├── graph_chunk_entity_relation.graphml    # Knowledge graph (nodes + edges)
├── vdb_chunks.json                        # Vector database (chunk embeddings)
├── kv_store_full_docs.json                # Complete documents
├── kv_store_text_chunks.json              # Text chunks with metadata
├── kv_store_full_entities.json            # Extracted entities
├── kv_store_full_relations.json           # Entity relationships
├── kv_store_llm_response_cache.json       # Cached LLM responses
├── doc_status.json                        # Document processing status
└── parse_cache.json                       # Cached parsing results
```

**Data Example:**

```json
// graph_chunk_entity_relation.graphml (simplified)
{
  "nodes": [
    {"id": "entity_1", "name": "RAG-Anything", "type": "SYSTEM"},
    {"id": "entity_2", "name": "Knowledge Graph", "type": "CONCEPT"},
    {"id": "chunk_1", "content": "Introduction...", "embedding": [...]}
  ],
  "edges": [
    {"source": "entity_1", "target": "entity_2", "type": "USES"}
  ]
}

// vdb_chunks.json (simplified)
{
  "chunk_1": {
    "content": "RAG-Anything is a multimodal RAG system...",
    "embedding": [0.123, 0.456, ..., 0.789],  // 768 dimensions
    "metadata": {"page": 0, "doc_id": "doc_123"}
  }
}
```

---

### Phase 8: Query Processing

When you ask: **"What is RAG-Anything?"**

```python
result = await rag.aquery(
    "What is RAG-Anything?",
    mode="hybrid"  # Uses both vector search + graph traversal
)
```

**Hybrid Query Process** (`raganything/query.py: aquery()`)

#### 8.1 Vector Search (Similarity)

```python
# 1. Embed query
query_embedding = embedding_func(["What is RAG-Anything?"])
# Result: [0.234, 0.567, ..., 0.890]  (768 dimensions)

# 2. Find similar chunks (cosine similarity)
similar_chunks = vector_db.search(
    query_embedding,
    top_k=10
)
# Returns: [
#   {"chunk": "RAG-Anything is...", "score": 0.95},
#   {"chunk": "The system processes...", "score": 0.87},
#   ...
# ]
```

#### 8.2 Graph Traversal (Relationships)

```python
# 1. Extract entities from query using LLM
query_entities = llm_extract_entities("What is RAG-Anything?")
# Result: ["RAG-Anything"]

# 2. Find entity in knowledge graph
entity_node = graph.find_node("RAG-Anything")

# 3. Get related entities (graph traversal)
related = graph.get_neighbors(entity_node, depth=2)
# Result: [
#   {"entity": "LightRAG", "relation": "BASED_ON"},
#   {"entity": "Multimodal Processing", "relation": "SUPPORTS"},
#   {"entity": "Knowledge Graph", "relation": "USES"}
# ]

# 4. Get chunks related to these entities
related_chunks = graph.get_chunks_for_entities(related)
```

#### 8.3 Multimodal Content Retrieval

```python
# Find multimodal content related to query
multimodal_results = search_multimodal_content(
    query="What is RAG-Anything?",
    types=["image", "table", "equation"]
)
# Returns images/tables/equations that were captioned/analyzed
# with "RAG-Anything" in their descriptions
```

#### 8.4 Answer Generation

```python
# Combine all retrieved context
context = {
    "vector_chunks": similar_chunks,      # Top similar text
    "graph_entities": related,            # Related entities
    "graph_chunks": related_chunks,       # Entity-related chunks
    "multimodal": multimodal_results      # Images/tables/equations
}

# Generate answer using LLM
prompt = f"""
Context from document:
{format_context(context)}

Question: What is RAG-Anything?

Please provide a comprehensive answer based on the context above.
Include information from text, and reference any relevant images, tables, or equations.
"""

answer = await llm_model_func(prompt)
# Returns rich answer combining all modalities
```

---

## 🔍 Key Differences from Traditional RAG

| Aspect | Traditional RAG | RAG-Anything |
|--------|----------------|--------------|
| **Content Types** | Text only | Text + Images + Tables + Equations |
| **Chunking** | Fixed size (512 tokens) | Intelligent chunking by LightRAG |
| **Retrieval** | Vector similarity only | Hybrid (vector + graph + multimodal) |
| **Context** | Isolated chunks | Entity relationships + document structure |
| **Storage** | Vector DB only | Vector DB + Knowledge Graph + KV stores |
| **Understanding** | Embeddings | Entities + Relations + Captions |
| **Parsing** | Simple text extraction | MinerU/Docling with layout analysis |

---

## 💡 Why This Approach is Better

### 1. **Multimodal Understanding**
- Traditional: "The results are shown in the table" → No table data
- RAG-Anything: Analyzes table, extracts insights, includes in answer

### 2. **Entity Relationships**
- Traditional: Chunks mention "RAG-Anything" and "LightRAG" separately
- RAG-Anything: Knows RAG-Anything is BASED_ON LightRAG (from graph)

### 3. **Context Preservation**
- Traditional: Chunk might lose context when split mid-sentence
- RAG-Anything: Preserves document structure, page references, captions

### 4. **Visual Content**
- Traditional: Skips images entirely
- RAG-Anything: Uses VLM to understand images, links to surrounding text

### 5. **Hybrid Retrieval**
- Traditional: Only finds similar text
- RAG-Anything: Finds similar text + related entities + multimodal content

---

## 🎯 Summary Flow

```
1. Upload PDF
   ↓
2. MinerU parses → [Text blocks] + [Images] + [Tables] + [Equations]
   ↓
3. Split into two paths:
   ├─ Text → LightRAG → Entities + Relations + Graph + Vectors
   └─ Multimodal → Modal Processors → Captions + Analysis → LightRAG
   ↓
4. Storage: Knowledge Graph + Vector DB + KV Stores
   ↓
5. Query → Hybrid Search (Vector + Graph + Multimodal)
   ↓
6. LLM generates answer from rich context
   ↓
7. User gets comprehensive answer with all content types
```

---

## 🚀 Performance Characteristics

**Processing Time (typical PDF):**
- Parse: 30-60 seconds (MinerU)
- Entity extraction: 10-20 seconds per page
- Graph construction: 5-10 seconds
- Vector embedding: 2-5 seconds
- Total: 1-3 minutes for 10-page document

**Storage (10-page PDF):**
- Original PDF: 2 MB
- Extracted images: 5 MB
- Knowledge graph: 100 KB
- Vector embeddings: 500 KB
- KV stores: 200 KB
- Total: ~6 MB

**Query Time:**
- Vector search: 10-50ms
- Graph traversal: 50-200ms
- LLM generation: 5-30 seconds
- Total: 5-30 seconds per query

---

## 📚 Further Reading

- [LightRAG Documentation](https://github.com/HKUDS/LightRAG)
- [MinerU Documentation](https://github.com/opendatalab/MinerU)
- [RAG-Anything Paper](https://arxiv.org/abs/2510.12323)
- [Context-Aware Processing](docs/context_aware_processing.md)

---

**This comprehensive pipeline ensures RAG-Anything provides deeper, more accurate answers than traditional RAG systems by leveraging multimodal understanding and knowledge graphs!** 🎉
