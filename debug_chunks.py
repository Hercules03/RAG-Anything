#!/usr/bin/env python3
"""
Debug script to check if chunks exist in LightRAG storage
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc


async def check_chunks():
    """Check what chunks exist in LightRAG storage"""

    # Initialize RAG (read-only, won't modify anything)
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
    )

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
    )

    await rag._ensure_lightrag_initialized()

    print("=" * 70)
    print("DIAGNOSTIC: Checking LightRAG Storage")
    print("=" * 70)

    # Get all chunk keys
    print("\n1. Checking text_chunks storage...")
    try:
        all_chunk_keys = await rag.lightrag.text_chunks.get_all_keys()
        print(f"   ✅ Found {len(all_chunk_keys)} total chunk keys")

        if len(all_chunk_keys) > 0:
            print(f"\n   First 5 chunk keys:")
            for key in list(all_chunk_keys)[:5]:
                print(f"   - {key}")

            # Get sample chunk data
            print(f"\n2. Checking sample chunk data...")
            sample_key = all_chunk_keys[0]
            chunk_data = await rag.lightrag.text_chunks.get_by_id(sample_key)

            print(f"\n   Sample chunk data structure:")
            print(f"   - Keys: {list(chunk_data.keys())}")
            print(f"   - full_doc_id: {chunk_data.get('full_doc_id', 'NOT FOUND')}")
            print(f"   - tokens: {chunk_data.get('tokens', 'NOT FOUND')}")
            print(f"   - chunk_order_index: {chunk_data.get('chunk_order_index', 'NOT FOUND')}")
            print(f"   - content preview: {chunk_data.get('content', '')[:100]}...")

            # Group chunks by document
            print(f"\n3. Grouping chunks by document...")
            doc_chunks = {}
            for key in all_chunk_keys:
                chunk_data = await rag.lightrag.text_chunks.get_by_id(key)
                if chunk_data:
                    doc_id = chunk_data.get("full_doc_id", "UNKNOWN")
                    if doc_id not in doc_chunks:
                        doc_chunks[doc_id] = []
                    doc_chunks[doc_id].append(key)

            print(f"\n   Documents with chunks:")
            for doc_id, chunks in doc_chunks.items():
                print(f"   - {doc_id}: {len(chunks)} chunks")
        else:
            print(f"   ❌ NO CHUNKS FOUND in storage!")
            print(f"   This means documents were not properly processed.")

    except Exception as e:
        print(f"   ❌ Error accessing text_chunks: {e}")

    # Check full_docs
    print(f"\n4. Checking full_docs storage...")
    try:
        all_doc_keys = await rag.lightrag.full_docs.get_all_keys()
        print(f"   ✅ Found {len(all_doc_keys)} documents in full_docs")

        if len(all_doc_keys) > 0:
            print(f"\n   Document IDs:")
            for key in list(all_doc_keys)[:5]:
                print(f"   - {key}")
    except Exception as e:
        print(f"   ❌ Error accessing full_docs: {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(check_chunks())
