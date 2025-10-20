# Chunking Format Alignment Guide

## Executive Summary

**Issue**: Ensuring output format consistency between LightRAG's built-in chunking and our custom `ChunkingStrategies`.

**Status**: ⚠️ **MISALIGNMENT DETECTED** - Our custom chunking returns `List[str]`, but LightRAG expects `List[Dict[str, Any]]`

**Action Required**: Align `ChunkingStrategies` output format with LightRAG's `TextChunkSchema`

---

## LightRAG's Expected Format

### TextChunkSchema Definition

From `lightrag/base.py:75-79`:

```python
class TextChunkSchema(TypedDict):
    tokens: int               # Number of tokens in the chunk
    content: str              # The text content of the chunk
    full_doc_id: str          # Document ID this chunk belongs to
    chunk_order_index: int    # Sequential index (0, 1, 2, ...)
```

### LightRAG's Chunking Function Output

From `lightrag/operate.py:66-116` (`chunking_by_token_size`):

```python
def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    # ... chunking logic ...

    # Returns format:
    results.append({
        "tokens": _len,              # int: Token count
        "content": chunk.strip(),    # str: Text content
        "chunk_order_index": index,  # int: Sequential index
    })
    # Note: full_doc_id is added later by LightRAG
```

**Key Points**:
- ✅ Returns `List[Dict[str, Any]]`
- ✅ Each dict has `tokens`, `content`, `chunk_order_index`
- ✅ `full_doc_id` is added automatically by LightRAG during insertion
- ✅ `content` is stripped (no leading/trailing whitespace)
- ✅ `chunk_order_index` starts at 0 and increments sequentially

---

## Our Current Implementation

### ChunkingStrategies Output Format

From `utils/chunking_strategies.py`:

```python
class ChunkingStrategies:

    def fixed_token_chunking(...) -> List[str]:
        # Returns: ["chunk 1 text", "chunk 2 text", ...]
        return chunks  # List[str]

    def paragraph_chunking(...) -> List[str]:
        # Returns: ["chunk 1 text", "chunk 2 text", ...]
        return chunks  # List[str]

    def hybrid_chunking(...) -> List[str]:
        # Returns: ["chunk 1 text", "chunk 2 text", ...]
        return chunks  # List[str]

    def contextual_chunking(...) -> List[Dict[str, str]]:
        # Returns: [{"content": "...", "context": "...", "full_text": "..."}, ...]
        # Different format for contextual chunking
        return contextual_chunks  # List[Dict[str, str]]
```

**Problem**:
- ❌ Returns `List[str]` (plain text chunks)
- ❌ Missing `tokens` field
- ❌ Missing `chunk_order_index` field
- ❌ Not compatible with LightRAG's insertion methods

---

## How streamlit_app.py Currently Uses Custom Chunking

### Current Code (BROKEN)

From `streamlit_app.py:156-176`:

```python
# Apply custom chunking
chunker = LateChunkingStrategies()  # ← OLD CLASS NAME (needs fix)
chunks = chunker.apply_strategy(
    text_content,
    strategy=chunking_strategy,
    chunk_size=chunk_size,
    **chunking_kwargs
)
# chunks = ["text1", "text2", "text3"]  ← Wrong format!

# Insert custom chunks
await rag.lightrag.ainsert_custom_chunks(
    full_text=text_content,
    text_chunks=chunks,  # ← Expects List[Dict], gets List[str]
    doc_id=doc_id
)
```

**Problem**: `ainsert_custom_chunks` doesn't exist in LightRAG!

### What Actually Works

Looking at LightRAG's API, there's no `ainsert_custom_chunks` method. The standard flow is:

1. **LightRAG.ainsert()** - Inserts full text, automatically chunks it
2. **LightRAG's internal chunking** - Uses `chunking_func` parameter
3. **Custom chunking_func** - Can be passed during LightRAG initialization

---

## Solution: Three Approaches

### Approach 1: Align Output Format (RECOMMENDED)

**Modify `ChunkingStrategies` to return LightRAG-compatible format**:

```python
def fixed_token_chunking(
    self,
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    split_sentences: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries matching TextChunkSchema
        [
            {
                "tokens": 1150,
                "content": "chunk text...",
                "chunk_order_index": 0
            },
            ...
        ]
    """
    # ... chunking logic (same as before) ...

    # NEW: Format output to match LightRAG
    formatted_chunks = []
    for index, chunk_text in enumerate(chunks):
        formatted_chunks.append({
            "tokens": self.count_tokens(chunk_text),
            "content": chunk_text.strip(),
            "chunk_order_index": index,
        })

    return formatted_chunks
```

**Benefits**:
- ✅ Direct compatibility with LightRAG
- ✅ Consistent format across entire system
- ✅ No wrapper functions needed
- ✅ Proper token counting included

**Migration**:
```python
# Before
chunks = chunker.apply_strategy(text, strategy="fixed_token")
# chunks = ["text1", "text2"]

# After
chunks = chunker.apply_strategy(text, strategy="fixed_token")
# chunks = [
#     {"tokens": 1150, "content": "text1", "chunk_order_index": 0},
#     {"tokens": 1200, "content": "text2", "chunk_order_index": 1}
# ]

# Access content
for chunk in chunks:
    print(chunk["content"])  # Access via dict key
```

### Approach 2: Add Wrapper Function

**Keep current format, add conversion function**:

```python
# In ChunkingStrategies class
def to_lightrag_format(
    self,
    chunks: List[str]
) -> List[Dict[str, Any]]:
    """Convert plain text chunks to LightRAG format"""
    formatted = []
    for index, chunk_text in enumerate(chunks):
        formatted.append({
            "tokens": self.count_tokens(chunk_text),
            "content": chunk_text.strip(),
            "chunk_order_index": index,
        })
    return formatted
```

**Usage**:
```python
# Get plain text chunks
text_chunks = chunker.apply_strategy(text, strategy="fixed_token")

# Convert to LightRAG format
lightrag_chunks = chunker.to_lightrag_format(text_chunks)

# Insert into LightRAG
await rag.lightrag.text_chunks.insert_many(lightrag_chunks)
```

**Drawbacks**:
- ❌ Extra step needed
- ❌ Two different formats to maintain
- ❌ Easy to forget conversion

### Approach 3: Use LightRAG's Custom chunking_func

**Set custom chunking function during LightRAG initialization**:

```python
from utils import ChunkingStrategies

def custom_chunking_func(tokenizer, content, split_by_character,
                         split_by_character_only, chunk_token_size,
                         chunk_overlap_token_size):
    """Custom chunking function for LightRAG"""
    chunker = ChunkingStrategies()

    # Get plain text chunks
    text_chunks = chunker.apply_strategy(
        content,
        strategy="paragraph",  # Or any strategy
        chunk_size=chunk_token_size
    )

    # Convert to LightRAG format
    formatted = []
    for index, chunk_text in enumerate(text_chunks):
        formatted.append({
            "tokens": chunker.count_tokens(chunk_text),
            "content": chunk_text.strip(),
            "chunk_order_index": index,
        })
    return formatted

# Initialize RAGAnything with custom chunking
rag = RAGAnything(
    config=config,
    lightrag_kwargs={
        "chunking_func": custom_chunking_func
    }
)
```

**Benefits**:
- ✅ Integrates at initialization level
- ✅ All documents use custom chunking automatically
- ✅ No manual chunking in streamlit_app.py

**Drawbacks**:
- ❌ Can't change strategy per-document
- ❌ Less flexible than per-document chunking

---

## Recommended Implementation

### Step 1: Update ChunkingStrategies Output Format

**Modify all methods to return `List[Dict[str, Any]]`**:

```python
# utils/chunking_strategies.py

from typing import List, Dict, Any

class ChunkingStrategies:

    def _format_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Convert text chunks to LightRAG format"""
        formatted = []
        for index, chunk_text in enumerate(chunks):
            formatted.append({
                "tokens": self.count_tokens(chunk_text),
                "content": chunk_text.strip(),
                "chunk_order_index": index,
            })
        return formatted

    def fixed_token_chunking(...) -> List[Dict[str, Any]]:
        # ... existing chunking logic ...
        chunks = [...]  # List[str]

        # Format and return
        return self._format_chunks(chunks)

    def paragraph_chunking(...) -> List[Dict[str, Any]]:
        # ... existing chunking logic ...
        chunks = [...]  # List[str]

        # Format and return
        return self._format_chunks(chunks)

    def hybrid_chunking(...) -> List[Dict[str, Any]]:
        # ... existing chunking logic ...
        chunks = [...]  # List[str]

        # Format and return
        return self._format_chunks(chunks)

    def contextual_chunking(...) -> List[Dict[str, Any]]:
        # Special case: already returns dict format
        # Update to match TextChunkSchema
        contextual_chunks = []
        for i, chunk in enumerate(base_chunks):
            contextual_chunks.append({
                "tokens": self.count_tokens(chunk),
                "content": chunk.strip(),
                "chunk_order_index": i,
                # Keep contextual info as extra fields if needed
                "context": f"Chunk {i+1} of {len(base_chunks)}",
            })
        return contextual_chunks
```

### Step 2: Fix streamlit_app.py

**Update to use correct class name and new format**:

```python
# Line 156: Fix class name
chunker = ChunkingStrategies()  # ← FIXED from LateChunkingStrategies

# Apply custom chunking (now returns List[Dict])
chunks = chunker.apply_strategy(
    text_content,
    strategy=chunking_strategy,
    chunk_size=chunk_size,
    **chunking_kwargs
)
# chunks = [
#     {"tokens": 1150, "content": "...", "chunk_order_index": 0},
#     {"tokens": 1200, "content": "...", "chunk_order_index": 1},
# ]

# Insert chunks directly into LightRAG storage
for chunk in chunks:
    chunk["full_doc_id"] = doc_id  # Add document ID
    chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
    await rag.lightrag.text_chunks.upsert(chunk_id, chunk)
```

### Step 3: Update Documentation

**Update docstrings to reflect new return type**:

```python
def apply_strategy(
    self,
    text: str,
    strategy: str = "fixed_token",
    chunk_size: int = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Apply a chunking strategy to text

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries with format:
        [
            {
                "tokens": int,              # Token count
                "content": str,             # Chunk text (stripped)
                "chunk_order_index": int,   # Sequential index
            },
            ...
        ]

        Note: `full_doc_id` will be added by LightRAG during insertion
    """
```

---

## Testing the Alignment

### Test 1: Verify Output Format

```python
from utils import ChunkingStrategies

chunker = ChunkingStrategies()
chunks = chunker.apply_strategy(
    "This is a test document. It has multiple sentences. We will chunk it.",
    strategy="fixed_token",
    chunk_size=10
)

# Expected format
assert isinstance(chunks, list)
assert all(isinstance(chunk, dict) for chunk in chunks)

for chunk in chunks:
    assert "tokens" in chunk
    assert "content" in chunk
    assert "chunk_order_index" in chunk
    assert isinstance(chunk["tokens"], int)
    assert isinstance(chunk["content"], str)
    assert isinstance(chunk["chunk_order_index"], int)
    assert chunk["content"] == chunk["content"].strip()  # No whitespace

print("✅ Format matches LightRAG's TextChunkSchema!")
```

### Test 2: Verify Compatibility with LightRAG

```python
import asyncio
from lightrag import LightRAG
from utils import ChunkingStrategies

async def test_lightrag_insertion():
    # Initialize LightRAG
    rag = LightRAG(working_dir="./test_storage")

    # Create chunks
    chunker = ChunkingStrategies()
    chunks = chunker.apply_strategy(
        "Test document content",
        strategy="fixed_token"
    )

    # Add full_doc_id
    doc_id = "test-doc-1"
    for chunk in chunks:
        chunk["full_doc_id"] = doc_id

    # Insert into LightRAG
    for chunk in chunks:
        chunk_id = f"chunk-{chunk['chunk_order_index']}"
        await rag.text_chunks.upsert(chunk_id, chunk)

    # Verify
    stored_chunk = await rag.text_chunks.get_by_id("chunk-0")
    assert stored_chunk["tokens"] == chunks[0]["tokens"]
    assert stored_chunk["content"] == chunks[0]["content"]

    print("✅ LightRAG insertion successful!")

asyncio.run(test_lightrag_insertion())
```

---

## Migration Checklist

- [ ] Update `ChunkingStrategies._format_chunks()` helper method
- [ ] Update `fixed_token_chunking()` return type and implementation
- [ ] Update `paragraph_chunking()` return type and implementation
- [ ] Update `hybrid_chunking()` return type and implementation
- [ ] Update `contextual_chunking()` return type and implementation
- [ ] Update `apply_strategy()` return type annotation
- [ ] Update `get_available_strategies()` documentation
- [ ] Fix `streamlit_app.py` line 156 class name
- [ ] Update `streamlit_app.py` chunk insertion logic
- [ ] Update all docstrings to reflect new format
- [ ] Add tests for format compliance
- [ ] Update `ENV_CONFIGURATION_GUIDE.md` examples

---

## Summary

**Current State**: ❌ **Incompatible**
- ChunkingStrategies returns `List[str]`
- LightRAG expects `List[Dict[str, Any]]` with `TextChunkSchema` format

**Target State**: ✅ **Aligned**
- ChunkingStrategies returns `List[Dict[str, Any]]`
- Format matches: `{"tokens": int, "content": str, "chunk_order_index": int}`
- Direct compatibility with LightRAG storage

**Next Steps**:
1. Implement `_format_chunks()` helper method
2. Update all chunking methods to return formatted dicts
3. Fix streamlit_app.py class name and insertion logic
4. Test format compliance
