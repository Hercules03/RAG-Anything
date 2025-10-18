import asyncio
import os
import platform
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

async def setup_rag_anything_with_ollama():
    # Configuration for Ollama
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # Use MinerU parser
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Initialize RAG-Anything with Ollama
    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # nomic-embed-text dimension
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            ),
        ),
        lightrag_kwargs={
            "llm_model_name": "gpt-oss:20b",  # Your Ollama model
            "llm_model_kwargs": {
                "options": {
                    "num_ctx": 32768,  # Increase context size
                    "temperature": 0.1  # Lower temperature for factual responses
                },
                "host": "http://localhost:11434"
            }
        }
    )

    return rag

async def main():
    print("🚀 Setting up RAG-Anything with Ollama...")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        print(f"✅ Ollama is running: {response.json()}")
    except Exception as e:
        print("❌ Ollama is not running. Please start with: ollama serve")
        print(f"Error: {e}")
        return
    
    # Setup RAG-Anything with Ollama
    rag = await setup_rag_anything_with_ollama()
    
    # Test with sample text first (no document processing)
    print("📝 Testing with sample text...")
    sample_content = [
        {
            "type": "text",
            "text": "RAG-Anything is a revolutionary multimodal RAG framework developed by HKU.",
            "page_idx": 0
        },
        {
            "type": "text",
            "text": "It can process text, images, tables, and equations from various document formats including PDFs, Office documents, and more.",
            "page_idx": 0
        },
        {
            "type": "text",
            "text": "It uses advanced dual-graph construction for cross-modal relationships and hybrid retrieval methods.",
            "page_idx": 0
        }
    ]

    await rag.insert_content_list(sample_content, file_path="sample_intro")
    
    # Test queries
    test_queries = [
        "What is RAG-Anything?",
        "Which university developed it?",
        "What types of content can it process?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"🤖 Answer: {result}")
        except Exception as e:
            print(f"❌ Query failed: {e}")
        print("-" * 50)
    
    print("\n🎉 RAG-Anything with Ollama setup complete!")
    
    # Uncomment below to test document processing
    # Replace 'path/to/your/document.pdf' with actual file path
    """
    print("📄 Processing document...")
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        device="mps" if "arm64" in platform.machine() else "cpu"
    )
    
    # Query the processed document
    result = await rag.aquery(
        "What are the main findings in this document?",
        mode="hybrid"
    )
    print("Document Result:", result)
    """

if __name__ == "__main__":
    asyncio.run(main())
