# Environment Configuration Guide - Chunking Strategies

## Overview

The `ChunkingStrategies` class now supports `.env` configuration, matching the same pattern used by LightRAG. This provides centralized configuration management for chunking parameters across the entire RAG pipeline.

## Configuration Pattern (Matches LightRAG)

**Priority Order**:
1. **Explicit parameters** passed to functions/methods
2. **Environment variables** from `.env` file
3. **Default values** (hardcoded fallbacks)

This allows you to:
- Set project-wide defaults in `.env`
- Override per-instance when needed
- Have sensible fallbacks if `.env` is missing

## Environment Variables

### Chunking Parameters

```bash
# .env file

# Maximum number of tokens per chunk
CHUNK_SIZE=1200

# Number of overlapping tokens between consecutive chunks
CHUNK_OVERLAP_SIZE=100

# Tiktoken encoding model
# Options: cl100k_base (GPT-4), p50k_base (GPT-3), r50k_base (Codex)
TIKTOKEN_ENCODING=cl100k_base
```

### How It Works

**LightRAG Integration**:
- LightRAG already uses `CHUNK_SIZE` and `CHUNK_OVERLAP_SIZE` for its built-in chunking
- Our `ChunkingStrategies` now uses the **same** environment variables
- This ensures **consistent chunking behavior** across both systems

**Example Flow**:

```python
# 1. Load .env (automatically done in both LightRAG and ChunkingStrategies)
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# 2. Initialize with .env defaults
from utils import ChunkingStrategies
chunker = ChunkingStrategies()  # Uses CHUNK_SIZE=1200, CHUNK_OVERLAP_SIZE=100

# 3. Override defaults if needed
chunker_custom = ChunkingStrategies(chunk_size=800, overlap=50)

# 4. Methods respect instance defaults
chunks = chunker.fixed_token_chunking(text)  # Uses 1200/100 from .env
chunks = chunker.fixed_token_chunking(text, chunk_size=1500)  # Overrides to 1500
```

## Usage Examples

### Basic Usage (Default from .env)

```python
from utils import ChunkingStrategies

# Loads defaults from .env
chunker = ChunkingStrategies()

# Uses CHUNK_SIZE=1200, CHUNK_OVERLAP_SIZE=100
chunks = chunker.apply_strategy(text, strategy="fixed_token")
```

### Override Instance Defaults

```python
# Set custom defaults for this instance
chunker = ChunkingStrategies(chunk_size=800, overlap=50)

# All methods use 800/50 unless explicitly overridden
chunks = chunker.fixed_token_chunking(text)  # 800/50
chunks = chunker.paragraph_chunking(text)    # min=400, max=1200 (derived from 800)
```

### Override Per-Method Call

```python
chunker = ChunkingStrategies()  # Uses .env defaults

# Override just for this call
chunks = chunker.fixed_token_chunking(
    text,
    chunk_size=1500,  # Override
    overlap=200       # Override
)
```

### Streamlit Integration

```python
# streamlit_app.py automatically loads .env defaults

import os
default_chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))

chunk_size = st.number_input(
    "Chunk Size (tokens)",
    value=default_chunk_size,  # Pre-filled from .env
    help=f"Default from .env: {default_chunk_size}"
)
```

## Configuration Scenarios

### Scenario 1: Building Regulations (Current Project)

```bash
# .env
CHUNK_SIZE=1200  # Good for structured documents
CHUNK_OVERLAP_SIZE=100  # Preserve context across chunks
TIKTOKEN_ENCODING=cl100k_base  # GPT-4 encoding
```

**Rationale**:
- 1200 tokens ≈ 900 words, good for regulation sections
- 100 token overlap ensures continuity
- cl100k_base matches modern LLMs

### Scenario 2: Short Documents (FAQ, Knowledge Base)

```bash
# .env
CHUNK_SIZE=500  # Smaller chunks for precise retrieval
CHUNK_OVERLAP_SIZE=50  # Less overlap needed
TIKTOKEN_ENCODING=cl100k_base
```

### Scenario 3: Long Technical Documents (Research Papers)

```bash
# .env
CHUNK_SIZE=2000  # Larger chunks to preserve complex context
CHUNK_OVERLAP_SIZE=200  # More overlap for continuity
TIKTOKEN_ENCODING=cl100k_base
```

### Scenario 4: Code Documentation

```bash
# .env
CHUNK_SIZE=800  # Medium chunks for code blocks
CHUNK_OVERLAP_SIZE=50  # Minimal overlap (code has clear boundaries)
TIKTOKEN_ENCODING=p50k_base  # Codex encoding for code
```

## Method-Specific Defaults

### `fixed_token_chunking()`

```python
def fixed_token_chunking(self, text, chunk_size=None, overlap=None):
    # Uses: .env CHUNK_SIZE if chunk_size is None
    # Uses: .env CHUNK_OVERLAP_SIZE if overlap is None
```

**Default Behavior**:
- `chunk_size`: Falls back to `CHUNK_SIZE` from .env (default: 1200)
- `overlap`: Falls back to `CHUNK_OVERLAP_SIZE` from .env (default: 100)

### `paragraph_chunking()`

```python
def paragraph_chunking(self, text, min_tokens=None, max_tokens=None):
    # Uses: CHUNK_SIZE / 2 if min_tokens is None
    # Uses: CHUNK_SIZE * 1.5 if max_tokens is None
```

**Default Behavior**:
- `min_tokens`: `CHUNK_SIZE / 2` (e.g., 600 if CHUNK_SIZE=1200)
- `max_tokens`: `CHUNK_SIZE * 1.5` (e.g., 1800 if CHUNK_SIZE=1200)

### `hybrid_chunking()`

```python
def hybrid_chunking(self, text, chunk_size=None, strategy="balanced"):
    # Uses: .env CHUNK_SIZE if chunk_size is None
```

### `contextual_chunking()`

```python
def contextual_chunking(self, text, context_sentences=2, chunk_size=None):
    # Uses: .env CHUNK_SIZE if chunk_size is None
```

## Testing Configuration

### Test 1: Verify .env Loading

```bash
python3 -c "
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env', override=False)

print('CHUNK_SIZE:', os.getenv('CHUNK_SIZE'))
print('CHUNK_OVERLAP_SIZE:', os.getenv('CHUNK_OVERLAP_SIZE'))
print('TIKTOKEN_ENCODING:', os.getenv('TIKTOKEN_ENCODING'))
"
```

**Expected Output**:
```
CHUNK_SIZE: 1200
CHUNK_OVERLAP_SIZE: 100
TIKTOKEN_ENCODING: cl100k_base
```

### Test 2: Verify ChunkingStrategies Initialization

```python
from utils import ChunkingStrategies

# Test default loading
chunker = ChunkingStrategies()
assert chunker.default_chunk_size == 1200
assert chunker.default_overlap == 100
assert chunker.encoding_name == "cl100k_base"

# Test explicit override
chunker_custom = ChunkingStrategies(chunk_size=800, overlap=50)
assert chunker_custom.default_chunk_size == 800
assert chunker_custom.default_overlap == 50
```

### Test 3: Verify Streamlit Integration

1. Start Streamlit: `streamlit run streamlit_app.py`
2. Go to "📄 Upload" tab
3. Check "Chunk Size" input field
4. **Expected**: Default value should be 1200 (from .env)
5. Tooltip should show: "default from .env: 1200"

## Comparison: LightRAG vs ChunkingStrategies

| Aspect | LightRAG (Built-in) | ChunkingStrategies (Custom) |
|--------|---------------------|----------------------------|
| **Environment Variables** | `CHUNK_SIZE`, `CHUNK_OVERLAP_SIZE` | Same variables |
| **Default Chunk Size** | 1200 tokens | 1200 tokens (from .env) |
| **Default Overlap** | 100 tokens | 100 tokens (from .env) |
| **Encoding** | `tiktoken_model_name` (default: gpt-4o-mini) | `TIKTOKEN_ENCODING` (default: cl100k_base) |
| **Chunking Type** | Token-based, character-based | Token, paragraph, semantic, hybrid |
| **Customization** | `chunking_func` parameter | Multiple strategies via `apply_strategy()` |
| **Configuration Priority** | Explicit > .env > defaults | Same |

**Key Alignment**:
- ✅ Both use `CHUNK_SIZE` from .env
- ✅ Both use `CHUNK_OVERLAP_SIZE` from .env
- ✅ Both follow same priority: explicit > .env > defaults
- ✅ Consistent behavior across entire RAG pipeline

## Best Practices

### 1. Set Project-Wide Defaults in .env

```bash
# Good: Centralized configuration
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
```

### 2. Override Only When Needed

```python
# Good: Use defaults for most cases
chunker = ChunkingStrategies()
chunks = chunker.apply_strategy(text, strategy="fixed_token")

# Override for special cases
chunks_large = chunker.apply_strategy(
    text,
    strategy="fixed_token",
    chunk_size=2000  # Exception for very long documents
)
```

### 3. Document Custom Values

```python
# Good: Explain why you're overriding
# Use larger chunks for legal documents to preserve clause context
chunker = ChunkingStrategies(chunk_size=2000, overlap=200)
```

### 4. Consistency with LightRAG

```python
# Good: Ensure both systems use same values
# .env: CHUNK_SIZE=1200, CHUNK_OVERLAP_SIZE=100

# LightRAG will use these automatically
rag = RAGAnything(...)

# ChunkingStrategies will use same values
chunker = ChunkingStrategies()
```

## Migration from Hardcoded Values

**Before** (Hardcoded):
```python
# Old code with hardcoded values
chunker = ChunkingStrategies()
chunks = chunker.fixed_token_chunking(text, chunk_size=1200, overlap=100)
```

**After** (Environment-based):
```python
# New code using .env defaults
chunker = ChunkingStrategies()
chunks = chunker.fixed_token_chunking(text)  # Uses .env values
```

**Benefits**:
1. ✅ Change config once in .env, applies everywhere
2. ✅ Consistent with LightRAG
3. ✅ Easy to tune for different projects
4. ✅ No code changes needed to adjust chunking

## Troubleshooting

### Issue: Defaults not loading from .env

**Symptom**: `ChunkingStrategies()` still uses hardcoded 1200/100 instead of .env values

**Solution**:
1. Ensure `.env` file exists in project root
2. Check `.env` syntax: `CHUNK_SIZE=1200` (no spaces around `=`)
3. Verify dotenv is installed: `pip install python-dotenv`
4. Add debug logging:
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv(dotenv_path=".env", override=False)
   print("CHUNK_SIZE:", os.getenv("CHUNK_SIZE"))
   ```

### Issue: Streamlit not showing .env defaults

**Symptom**: Streamlit UI shows 1200 but .env has different value

**Solution**:
1. Restart Streamlit completely (Ctrl+C, then `streamlit run streamlit_app.py`)
2. Clear Streamlit cache: `streamlit cache clear`
3. Check that `.env` is in same directory as `streamlit_app.py`

### Issue: Different values in LightRAG vs ChunkingStrategies

**Symptom**: LightRAG chunks differently than custom chunking

**Solution**:
1. Ensure both use same `.env` file
2. Check if LightRAG was initialized with explicit `chunk_token_size` parameter
3. Verify encoding matches: LightRAG uses `tiktoken_model_name`, ChunkingStrategies uses `TIKTOKEN_ENCODING`

## Summary

**What Changed**:
1. ✅ Added `.env` support to `ChunkingStrategies` class
2. ✅ Updated `.env` with chunking configuration
3. ✅ Modified Streamlit app to respect `.env` defaults
4. ✅ Aligned with LightRAG's configuration pattern

**Benefits**:
- **Consistency**: Same chunking config across LightRAG and custom strategies
- **Flexibility**: Override defaults when needed, per-instance or per-method
- **Maintainability**: Change config once in `.env`, applies everywhere
- **Best Practices**: Follows LightRAG's proven configuration pattern

**Next Steps**:
1. Test with your building regulation PDFs
2. Tune `CHUNK_SIZE` and `CHUNK_OVERLAP_SIZE` based on document structure
3. Experiment with different strategies using `.env` defaults
