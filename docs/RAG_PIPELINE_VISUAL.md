# RAG-Anything Visual Pipeline

A visual representation of how documents flow through RAG-Anything.

## 📊 Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT WEB INTERFACE                          │
│  User uploads PDF → Temporary storage → Process button clicked          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DOCUMENT PARSING (MinerU)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Input: research_paper.pdf                                               │
│                                                                          │
│  MinerU analyzes:                                                        │
│  ├─ Layout detection (headers, paragraphs, columns)                     │
│  ├─ Image extraction (figures, diagrams, charts)                        │
│  ├─ Table recognition (converts to markdown)                            │
│  ├─ Equation detection (extracts LaTeX)                                 │
│  └─ Text extraction (with OCR fallback)                                 │
│                                                                          │
│  Output: content_list = [                                               │
│    {type: "text", text: "Introduction...", page_idx: 0},               │
│    {type: "image", img_path: "/path/fig1.jpg", page_idx: 1},           │
│    {type: "table", table_body: "| A | B |...", page_idx: 2},           │
│    {type: "equation", latex: "E=mc^2", page_idx: 3}                    │
│  ]                                                                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 2: CONTENT SEPARATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│  separate_content(content_list)                                          │
│                                                                          │
│  ┌────────────────────┐              ┌──────────────────────┐           │
│  │   TEXT CONTENT     │              │  MULTIMODAL CONTENT   │           │
│  ├────────────────────┤              ├──────────────────────┤           │
│  │ "Introduction...   │              │ • 5 images            │           │
│  │  Methodology...    │              │ • 3 tables            │           │
│  │  Results...        │              │ • 2 equations         │           │
│  │  Conclusion..."    │              │                       │           │
│  └────────┬───────────┘              └─────────┬────────────┘           │
│           │                                    │                        │
└───────────┼────────────────────────────────────┼────────────────────────┘
            │                                    │
            ▼                                    ▼
┌───────────────────────────┐      ┌────────────────────────────────────┐
│  PHASE 3: TEXT PATH       │      │  PHASE 4: MULTIMODAL PATH          │
│  (LightRAG Processing)    │      │  (Modal Processors)                │
├───────────────────────────┤      ├────────────────────────────────────┤
│                           │      │                                    │
│ Step 1: Chunking          │      │ FOR EACH MULTIMODAL ITEM:          │
│ ┌───────────────────┐     │      │                                    │
│ │ Chunk 1: Intro... │     │      │ Step 1: Extract Context            │
│ │ Chunk 2: Method...│     │      │ ┌────────────────────────────┐     │
│ │ Chunk 3: Results..│     │      │ │ Text before: "Figure 1..." │     │
│ │ Chunk 4: Concl... │     │      │ │ Caption: "Architecture"    │     │
│ └───────────────────┘     │      │ │ Text after: "The system..."│     │
│         │                 │      │ └────────────────────────────┘     │
│         ▼                 │      │         │                          │
│ Step 2: Entity Extract    │      │         ▼                          │
│ ┌───────────────────┐     │      │ Step 2: Process by Type            │
│ │ LLM analyzes:     │     │      │                                    │
│ │ • RAG-Anything    │     │      │ ┌─────────────────────────────┐    │
│ │ • LightRAG        │     │      │ │ IMAGE → VLM Caption         │    │
│ │ • Knowledge Graph │     │      │ │ TABLE → LLM Analysis        │    │
│ │ • MinerU          │     │      │ │ EQUATION → LLM Explanation  │    │
│ └───────────────────┘     │      │ └─────────────────────────────┘    │
│         │                 │      │         │                          │
│         ▼                 │      │         ▼                          │
│ Step 3: Relation Extract  │      │ Step 3: Create Multimodal Doc      │
│ ┌───────────────────┐     │      │ ┌────────────────────────────┐     │
│ │ LLM identifies:   │     │      │ │ [IMAGE: fig1.jpg]          │     │
│ │ RAG-Anything      │     │      │ │ Context: "Section 3..."    │     │
│ │   ↓ BASED_ON      │     │      │ │ Caption: "Shows the..."    │     │
│ │ LightRAG          │     │      │ │ Page: 1                    │     │
│ │   ↓ USES          │     │      │ └────────────────────────────┘     │
│ │ Knowledge Graph   │     │      │         │                          │
│ └───────────────────┘     │      │         ▼                          │
│         │                 │      │ Step 4: Insert into LightRAG       │
│         ▼                 │      │ (Same as text path →)              │
│ Step 4: Vector Embeddings │      │                                    │
│ ┌───────────────────┐     │      └────────────────┬───────────────────┘
│ │ Embed each chunk: │     │                       │
│ │ [0.12, 0.45, ...] │     │                       │
│ │ 768 dimensions    │     │                       │
│ └───────────────────┘     │                       │
│         │                 │                       │
│         ▼                 │                       │
│ Step 5: Build Graph       │                       │
│ ┌───────────────────┐     │                       │
│ │  [Entity Nodes]   │     │                       │
│ │       +           │     │                       │
│ │  [Relations]      │     │                       │
│ │       +           │     │                       │
│ │  [Chunk Nodes]    │     │                       │
│ └───────────────────┘     │                       │
└───────────┬───────────────┘                       │
            │                                       │
            └───────────────┬───────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: STORAGE LAYER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ./rag_storage/                                                          │
│  ├─ graph_chunk_entity_relation.graphml  ← Knowledge Graph              │
│  │    Nodes: [RAG-Anything] [LightRAG] [Chunk1] [Chunk2] ...           │
│  │    Edges: BASED_ON, USES, INCLUDES, RELATES_TO ...                   │
│  │                                                                       │
│  ├─ vdb_chunks.json  ← Vector Database                                  │
│  │    {                                                                  │
│  │      "chunk_1": {"embedding": [0.12, ...], "text": "..."},          │
│  │      "chunk_2": {"embedding": [0.45, ...], "text": "..."}           │
│  │    }                                                                  │
│  │                                                                       │
│  ├─ kv_store_full_entities.json  ← Entity Store                         │
│  │    {                                                                  │
│  │      "RAG-Anything": {"type": "SYSTEM", "desc": "..."},             │
│  │      "LightRAG": {"type": "FRAMEWORK", "desc": "..."}               │
│  │    }                                                                  │
│  │                                                                       │
│  ├─ kv_store_full_relations.json  ← Relation Store                      │
│  │    [                                                                  │
│  │      {"src": "RAG-Anything", "tgt": "LightRAG", "rel": "BASED_ON"}, │
│  │      {"src": "LightRAG", "tgt": "Knowledge Graph", "rel": "USES"}   │
│  │    ]                                                                  │
│  │                                                                       │
│  └─ parse_cache.json  ← Parsing Cache (for reuse)                       │
│                                                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
                    DOCUMENT PROCESSING COMPLETE!
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    USER ASKS QUESTION IN CHAT                            │
│  "What is RAG-Anything and how does it work?"                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 6: HYBRID QUERY PROCESSING                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Step 1: Vector Search (Similarity)                                     │
│  ┌────────────────────────────────────────┐                             │
│  │ 1. Embed query: "What is RAG-Anything" │                             │
│  │    → [0.23, 0.56, ..., 0.89]           │                             │
│  │                                         │                             │
│  │ 2. Find similar chunks (cosine sim):   │                             │
│  │    • Chunk 1: "RAG-Anything is..." 0.95│                             │
│  │    • Chunk 3: "The system uses..." 0.87│                             │
│  │    • Chunk 7: "Features include..." 0.82                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
│  Step 2: Graph Traversal (Relationships)                                │
│  ┌────────────────────────────────────────┐                             │
│  │ 1. Extract entities from query:        │                             │
│  │    → ["RAG-Anything"]                  │                             │
│  │                                         │                             │
│  │ 2. Find in knowledge graph:            │                             │
│  │    [RAG-Anything] found!               │                             │
│  │                                         │                             │
│  │ 3. Get related entities (depth=2):     │                             │
│  │    [RAG-Anything]                      │                             │
│  │       ├─BASED_ON→ [LightRAG]           │                             │
│  │       ├─INCLUDES→ [MinerU]             │                             │
│  │       └─SUPPORTS→ [Multimodal]         │                             │
│  │                                         │                             │
│  │ 4. Get chunks for these entities       │                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
│  Step 3: Multimodal Retrieval                                           │
│  ┌────────────────────────────────────────┐                             │
│  │ Search captions/analyses for keywords: │                             │
│  │ • Figure 1: "RAG-Anything architecture"│                             │
│  │ • Table 2: "RAG-Anything performance"  │                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
│  Step 4: Combine All Context                                            │
│  ┌────────────────────────────────────────┐                             │
│  │ Context = {                            │                             │
│  │   vector_chunks: [Chunk1, Chunk3, ...],│                             │
│  │   graph_entities: [RAG, LightRAG, ...],│                             │
│  │   multimodal: [Fig1, Table2, ...]      │                             │
│  │ }                                      │                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 7: ANSWER GENERATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM Prompt:                                                             │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ You are a helpful assistant. Use the following context to      │     │
│  │ answer the question comprehensively.                           │     │
│  │                                                                 │     │
│  │ CONTEXT FROM DOCUMENT:                                         │     │
│  │                                                                 │     │
│  │ Text chunks:                                                   │     │
│  │ - RAG-Anything is a multimodal RAG system built on LightRAG... │     │
│  │ - The system uses knowledge graphs for entity relationships... │     │
│  │ - Features include document parsing, multimodal processing...  │     │
│  │                                                                 │     │
│  │ Related entities:                                              │     │
│  │ - RAG-Anything (SYSTEM): Multimodal RAG framework              │     │
│  │ - LightRAG (FRAMEWORK): Base framework for graph-based RAG     │     │
│  │ - MinerU (TOOL): Document parser for layout analysis           │     │
│  │                                                                 │     │
│  │ Multimodal content:                                            │     │
│  │ - Figure 1 (page 2): Shows RAG-Anything architecture with...  │     │
│  │ - Table 2 (page 5): Performance comparison showing 95.2%...    │     │
│  │                                                                 │     │
│  │ QUESTION: What is RAG-Anything and how does it work?          │     │
│  │                                                                 │     │
│  │ Please provide a comprehensive answer.                         │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                ▼                                         │
│  LLM (Ollama gpt-oss:20b) generates answer...                           │
│                                ▼                                         │
│  Generated Answer:                                                       │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ RAG-Anything is a comprehensive multimodal RAG system built on │     │
│  │ top of LightRAG framework. It processes documents containing   │     │
│  │ text, images, tables, and equations using MinerU parser for    │     │
│  │ high-fidelity extraction.                                      │     │
│  │                                                                 │     │
│  │ The system works by:                                           │     │
│  │ 1. Parsing documents to extract all content types              │     │
│  │ 2. Building knowledge graphs with entity relationships         │     │
│  │ 3. Processing multimodal content with specialized processors   │     │
│  │ 4. Using hybrid retrieval combining vector and graph search    │     │
│  │                                                                 │     │
│  │ As shown in Figure 1 on page 2, the architecture includes...   │     │
│  │ Performance results in Table 2 demonstrate 95.2% accuracy...   │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANSWER DISPLAYED IN STREAMLIT                         │
│  User sees comprehensive answer with context from all modalities!       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Comparison: Traditional vs RAG-Anything

```
TRADITIONAL RAG:
================
PDF → Simple text extract → Fixed chunks (512 tokens)
                                   ↓
                         Vector embeddings only
                                   ↓
                            Vector DB storage
                                   ↓
                Query → Vector similarity search only
                                   ↓
              LLM with limited text-only context
                                   ↓
                     Basic text-based answer


RAG-ANYTHING:
=============
PDF → MinerU (layout analysis)
         ↓
    [Text] + [Images] + [Tables] + [Equations]
         ↓                           ↓
    Intelligent         Modal Processors (VLM/LLM)
    Chunking                       ↓
         ↓                    Caption + Analysis
    Entity/Relation                ↓
    Extraction          Convert to text descriptions
         ↓                           ↓
    Knowledge Graph ←────────────────┘
         +
    Vector DB
         +
    KV Stores
         ↓
Query → Hybrid Search (Vector + Graph + Multimodal)
         ↓
    Rich context from all modalities
         ↓
    LLM with comprehensive context
         ↓
    Detailed answer with images/tables/equations referenced
```

## 📊 Data Flow Example

```
INPUT DOCUMENT (research_paper.pdf):
┌──────────────────────────────────────┐
│ Page 1: Title & Abstract             │
│ Page 2: Introduction + Figure 1      │
│ Page 3: Methodology + Table 1        │
│ Page 4: Results + Equation 1         │
│ Page 5: Conclusion                   │
└──────────────────────────────────────┘
                ↓
        AFTER PARSING:
┌──────────────────────────────────────┐
│ content_list = [                     │
│   {type: "text", text: "Abstract"}, │
│   {type: "text", text: "Intro"},    │
│   {type: "image", path: "fig1.jpg"}, │
│   {type: "text", text: "Method"},   │
│   {type: "table", body: "|A|B|"},   │
│   {type: "text", text: "Results"},  │
│   {type: "equation", latex: "..."},  │
│   {type: "text", text: "Concl"}     │
│ ]                                    │
└──────────────────────────────────────┘
                ↓
      AFTER PROCESSING:
┌──────────────────────────────────────┐
│ KNOWLEDGE GRAPH:                     │
│  [RAG-Anything] ─BASED_ON→ [LightRAG]│
│       │                               │
│       ├─USES→ [Knowledge Graph]       │
│       │                               │
│       └─INCLUDES→ [Multimodal]        │
│                                       │
│ VECTOR DB:                            │
│  chunk_1: [0.12, ..., 0.67] (768-d)  │
│  chunk_2: [0.45, ..., 0.89] (768-d)  │
│  ...                                  │
│                                       │
│ MULTIMODAL:                           │
│  "Figure 1 shows RAG architecture..." │
│  "Table 1 displays performance..."    │
│  "Equation defines loss function..."  │
└──────────────────────────────────────┘
                ↓
         READY FOR QUERIES!
```

## 🎯 Query Example Flow

```
USER QUERY: "How does RAG-Anything handle images?"
                    ↓
         ┌──────────┴──────────┐
         ▼                     ▼
    VECTOR SEARCH        GRAPH SEARCH
         │                     │
         ▼                     ▼
   Find similar chunks   Find "images" entity
   mentioning "images"   and related nodes
         │                     │
         └──────────┬──────────┘
                    ▼
           MULTIMODAL SEARCH
                    │
                    ▼
         Find image captions
         with relevant content
                    │
                    ▼
            COMBINE CONTEXT
                    │
                    ▼
               LLM PROMPT
                    │
                    ▼
    "RAG-Anything processes images using
     ImageModalProcessor which:
     1. Extracts images during parsing
     2. Uses VLM for caption generation
     3. Links to surrounding context
     4. Stores in knowledge graph

     As shown in Figure 1 (page 2)..."
```

---

**This visual guide shows how RAG-Anything transforms your documents into a rich, queryable knowledge base!** 🚀
