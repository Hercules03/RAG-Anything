#!/usr/bin/env python
"""
Streamlit RAG-Anything Chatbot with Ollama
A simple web interface for uploading PDFs and chatting with your documents
"""

import streamlit as st
import asyncio
import os
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Page configuration
st.set_page_config(
    page_title="RAG-Anything Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = []


async def initialize_rag(
    ollama_model: str,
    embedding_model: str,
    ollama_host: str,
    working_dir: str
):
    """Initialize RAG-Anything with Ollama"""
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # nomic-embed-text dimension
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embedding_model,
                host=ollama_host
            ),
        ),
        lightrag_kwargs={
            "llm_model_name": ollama_model,
            "llm_model_kwargs": {
                "options": {
                    "num_ctx": 32768,
                    "temperature": 0.1
                },
                "host": ollama_host
            }
        }
    )

    return rag


async def process_document(rag, file_path: str, output_dir: str):
    """Process uploaded document"""
    await rag.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        parse_method="auto"
    )


async def query_rag(rag, question: str, mode: str = "hybrid"):
    """Query the RAG system"""
    result = await rag.aquery(question, mode=mode)
    return result


def main():
    st.title("📚 RAG-Anything Chatbot with Ollama")
    st.markdown("Upload PDF documents and chat with them using local Ollama models!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ollama settings
        ollama_host = st.text_input(
            "Ollama Host",
            value="http://localhost:11434",
            help="URL where Ollama is running"
        )

        ollama_model = st.text_input(
            "LLM Model",
            value="gpt-oss:20b",
            help="Ollama model for text generation"
        )

        embedding_model = st.text_input(
            "Embedding Model",
            value="nomic-embed-text",
            help="Ollama model for embeddings"
        )

        working_dir = st.text_input(
            "Working Directory",
            value="./rag_storage",
            help="Directory for RAG storage"
        )

        output_dir = st.text_input(
            "Output Directory",
            value="./output",
            help="Directory for processed documents"
        )

        st.divider()

        # Initialize RAG button
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                try:
                    # Check if Ollama is running
                    import requests
                    response = requests.get(f"{ollama_host}/api/version", timeout=5)
                    st.success(f"✅ Ollama connected: v{response.json()['version']}")

                    # Initialize RAG
                    st.session_state.rag = asyncio.run(
                        initialize_rag(ollama_model, embedding_model, ollama_host, working_dir)
                    )
                    st.success("✅ RAG system initialized!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")

        st.divider()

        # Document upload section
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )

        if uploaded_files and st.session_state.rag is not None:
            if st.button("📤 Process Documents"):
                for uploaded_file in uploaded_files:
                    # Skip if already processed
                    if uploaded_file.name in st.session_state.documents_processed:
                        st.info(f"⏭️ Skipping {uploaded_file.name} (already processed)")
                        continue

                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name

                            # Process document
                            os.makedirs(output_dir, exist_ok=True)
                            asyncio.run(
                                process_document(st.session_state.rag, tmp_path, output_dir)
                            )

                            # Clean up temp file
                            os.unlink(tmp_path)

                            st.session_state.documents_processed.append(uploaded_file.name)
                            st.success(f"✅ Processed: {uploaded_file.name}")

                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

        # Show processed documents
        if st.session_state.documents_processed:
            st.divider()
            st.subheader("📋 Processed Documents")
            for doc in st.session_state.documents_processed:
                st.text(f"✓ {doc}")

            if st.button("🗑️ Clear All Documents"):
                st.session_state.documents_processed = []
                st.session_state.messages = []
                st.rerun()

    # Main chat interface
    if st.session_state.rag is None:
        st.info("👈 Please initialize the RAG system from the sidebar to get started!")
        st.markdown("""
        ### Quick Start Guide:
        1. Make sure Ollama is running: `ollama serve`
        2. Configure your Ollama settings in the sidebar
        3. Click "🚀 Initialize RAG System"
        4. Upload PDF documents
        5. Click "📤 Process Documents"
        6. Start chatting with your documents!
        """)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from RAG
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = asyncio.run(
                            query_rag(st.session_state.rag, prompt, mode="hybrid")
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Show helpful tips if no documents processed
        if not st.session_state.documents_processed:
            st.info("💡 Upload and process some documents to start chatting!")


if __name__ == "__main__":
    main()
