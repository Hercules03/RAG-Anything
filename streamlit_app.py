#!/usr/bin/env python
"""
Streamlit RAG-Anything Chatbot with Ollama
Multi-tab interface for document management, chunk viewing, and chatbot
"""

import streamlit as st
import asyncio
import os
import tempfile
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Import utility modules
from utils import (
    ChunkManager,
    LateChunkingStrategies,
    RegulationExtractor,
    MetadataStore,
    DocumentDatabase
)

# Page configuration
st.set_page_config(
    page_title="RAG-Anything Document Manager",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = None
if "db_manager" not in st.session_state:
    st.session_state.db_manager = None
if "chunk_manager" not in st.session_state:
    st.session_state.chunk_manager = None


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


async def process_single_document(
    rag,
    file_path: str,
    output_dir: str,
    doc_id: str,
    regulation_id: str,
    chunking_strategy: str = "default",
    chunk_size: int = 1200,
    **chunking_kwargs
):
    """Process single document with custom chunking"""

    # Get file metadata
    file_size = os.path.getsize(file_path)

    # Extract regulation ID if not provided
    if not regulation_id:
        reg_extractor = RegulationExtractor()
        regulation_id = reg_extractor.extract_from_pdf(file_path) or Path(file_path).stem

    # Add metadata
    metadata_store = st.session_state.metadata_store
    metadata_store.add_document(
        doc_id=doc_id,
        filename=Path(file_path).name,
        regulation_id=regulation_id,
        file_size=file_size,
        chunking_strategy=chunking_strategy,
        processing_status="processing"
    )

    # Process with default chunking
    if chunking_strategy == "default":
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto"
        )
    else:
        # Process document first to get content
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto"
        )

        # Get full document content
        full_doc = await rag.lightrag.full_docs.get_by_id(doc_id)
        if full_doc:
            # Extract text
            text_content = ""
            content_list = full_doc.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    text_content += item.get("text", "") + "\n"

            # Apply custom chunking
            chunker = LateChunkingStrategies()
            chunks = chunker.apply_strategy(
                text_content,
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                **chunking_kwargs
            )

            # Delete old chunks and insert new ones
            all_chunk_keys = await rag.lightrag.text_chunks.get_all_keys()
            for key in all_chunk_keys:
                chunk_data = await rag.lightrag.text_chunks.get_by_id(key)
                if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                    await rag.lightrag.text_chunks.delete_by_id(key)

            # Insert custom chunks
            await rag.lightrag.ainsert_custom_chunks(
                full_text=text_content,
                text_chunks=chunks,
                doc_id=doc_id
            )

    # Update metadata
    # Get page count from metadata
    reg_extractor = RegulationExtractor()
    meta = reg_extractor.extract_metadata(file_path)

    metadata_store.update_document(
        doc_id,
        {
            "processing_status": "completed",
            "page_count": meta.get("page_count", 0)
        }
    )

    return regulation_id


async def process_batch_documents(
    rag,
    folder_path: str,
    output_dir: str,
    chunking_strategy: str = "default",
    chunk_size: int = 1200,
    **chunking_kwargs
):
    """Process all PDFs in a folder"""
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    for pdf_file in pdf_files:
        doc_id = f"doc-{pdf_file.stem}"

        await process_single_document(
            rag,
            str(pdf_file),
            output_dir,
            doc_id,
            None,  # Auto-extract regulation ID
            chunking_strategy,
            chunk_size,
            **chunking_kwargs
        )


def render_sidebar():
    """Render sidebar configuration"""
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

                    # Initialize utility managers
                    st.session_state.metadata_store = MetadataStore(
                        storage_path=working_dir,
                        lightrag=st.session_state.rag.lightrag
                    )
                    st.session_state.db_manager = DocumentDatabase(
                        st.session_state.rag,
                        st.session_state.metadata_store
                    )
                    st.session_state.chunk_manager = ChunkManager(
                        st.session_state.rag.lightrag
                    )

                    st.success("✅ RAG system initialized!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")

        return working_dir, output_dir


def render_tab1_upload():
    """Tab 1: Document Upload"""
    st.header("📄 Document Upload")

    if st.session_state.rag is None:
        st.warning("Please initialize the RAG system from the sidebar first!")
        return

    # Choose upload mode
    upload_mode = st.radio(
        "Upload Mode",
        ["Single Document", "Batch Processing (Folder)"],
        horizontal=True
    )

    # Chunking strategy selector
    st.subheader("Chunking Strategy")
    col1, col2 = st.columns(2)

    with col1:
        chunking_strategy = st.selectbox(
            "Strategy",
            ["default", "fixed_token", "paragraph", "hybrid"],
            help="Chunking method to use"
        )

    with col2:
        chunk_size = st.number_input(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=4000,
            value=1200,
            step=100,
            help="Target chunk size in tokens"
        )

    # Additional parameters
    if chunking_strategy == "fixed_token":
        overlap = st.slider("Overlap (tokens)", 0, 500, 100)
        chunking_kwargs = {"overlap": overlap}
    elif chunking_strategy == "paragraph":
        min_tokens = st.number_input("Min Tokens", 100, 2000, 500)
        max_tokens = st.number_input("Max Tokens", 500, 3000, 1500)
        chunking_kwargs = {"min_tokens": min_tokens, "max_tokens": max_tokens}
    else:
        chunking_kwargs = {}

    st.divider()

    # Single document upload
    if upload_mode == "Single Document":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a single PDF document"
        )

        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                doc_id = st.text_input("Document ID", value=f"doc-{uploaded_file.name[:-4]}")
            with col2:
                regulation_id = st.text_input(
                    "Regulation ID (optional)",
                    value="",
                    help="Leave blank to auto-extract"
                )

            if st.button("📤 Process Document", type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        # Get output dir from session or use default
                        output_dir = "./output"
                        os.makedirs(output_dir, exist_ok=True)

                        # Process document
                        extracted_reg_id = asyncio.run(
                            process_single_document(
                                st.session_state.rag,
                                tmp_path,
                                output_dir,
                                doc_id,
                                regulation_id,
                                chunking_strategy,
                                chunk_size,
                                **chunking_kwargs
                            )
                        )

                        # Clean up temp file
                        os.unlink(tmp_path)

                        st.success(f"✅ Processed: {uploaded_file.name}")
                        st.info(f"Regulation ID: {extracted_reg_id}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Batch processing
    else:
        folder_path = st.text_input(
            "Folder Path",
            value="./documents",
            help="Path to folder containing PDF files"
        )

        if st.button("📦 Process All PDFs in Folder", type="primary"):
            if not os.path.exists(folder_path):
                st.error(f"Folder not found: {folder_path}")
            else:
                pdf_files = list(Path(folder_path).glob("*.pdf"))
                if not pdf_files:
                    st.warning(f"No PDF files found in {folder_path}")
                else:
                    st.info(f"Found {len(pdf_files)} PDF files")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, pdf_file in enumerate(pdf_files):
                        status_text.text(f"Processing {pdf_file.name}...")

                        try:
                            doc_id = f"doc-{pdf_file.stem}"
                            output_dir = "./output"
                            os.makedirs(output_dir, exist_ok=True)

                            asyncio.run(
                                process_single_document(
                                    st.session_state.rag,
                                    str(pdf_file),
                                    output_dir,
                                    doc_id,
                                    None,  # Auto-extract
                                    chunking_strategy,
                                    chunk_size,
                                    **chunking_kwargs
                                )
                            )

                        except Exception as e:
                            st.error(f"Error processing {pdf_file.name}: {str(e)}")

                        progress_bar.progress((idx + 1) / len(pdf_files))

                    status_text.text("✅ Batch processing complete!")


def render_tab2_database():
    """Tab 2: Database Management"""
    st.header("🗄️ Database Management")

    if st.session_state.db_manager is None:
        st.warning("Please initialize the RAG system from the sidebar first!")
        return

    db_manager = st.session_state.db_manager

    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search documents", "")
    with col2:
        status_filter = st.selectbox("Status", ["All", "completed", "processing", "failed"])

    # Get documents
    try:
        if status_filter == "All":
            documents = asyncio.run(db_manager.list_documents(query=search_query))
        else:
            documents = asyncio.run(db_manager.list_documents(query=search_query, status=status_filter))

        # Display statistics
        stats = asyncio.run(db_manager.get_statistics())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Total Pages", stats.get("total_pages", 0))
        with col3:
            st.metric("Total Chunks", stats.get("total_chunks_in_lightrag", 0))
        with col4:
            size_mb = stats.get("total_size_bytes", 0) / (1024 * 1024)
            st.metric("Total Size", f"{size_mb:.1f} MB")

        st.divider()

        # Documents table
        if documents:
            st.subheader(f"📋 Documents ({len(documents)})")

            for doc in documents:
                with st.expander(f"**{doc.get('regulation_id', 'Unknown')}** - {doc.get('filename', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.text(f"Doc ID: {doc.get('doc_id', 'N/A')}")
                        st.text(f"Status: {doc.get('processing_status', 'Unknown')}")
                        st.text(f"Chunking: {doc.get('chunking_strategy', 'default')}")

                    with col2:
                        st.text(f"Pages: {doc.get('page_count', 0)}")
                        size_kb = doc.get('file_size', 0) / 1024
                        st.text(f"Size: {size_kb:.1f} KB")
                        st.text(f"Uploaded: {doc.get('upload_date', 'N/A')[:10]}")

                    with col3:
                        # Action buttons
                        col_a, col_b = st.columns(2)

                        with col_a:
                            if st.button("🔄 Re-process", key=f"reprocess_{doc['doc_id']}"):
                                with st.spinner("Re-processing..."):
                                    try:
                                        success = asyncio.run(
                                            db_manager.reprocess_document(
                                                doc['doc_id'],
                                                chunking_strategy="default"
                                            )
                                        )
                                        if success:
                                            st.success("✅ Re-processed!")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")

                        with col_b:
                            if st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}"):
                                with st.spinner("Deleting..."):
                                    try:
                                        success = asyncio.run(
                                            db_manager.delete_document(doc['doc_id'])
                                        )
                                        if success:
                                            st.success("✅ Deleted!")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
        else:
            st.info("No documents found. Upload some documents in Tab 1!")

        # Bulk operations
        st.divider()
        st.subheader("🔧 Bulk Operations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Sync with LightRAG"):
                with st.spinner("Syncing..."):
                    try:
                        results = asyncio.run(db_manager.sync_with_lightrag())
                        st.success("✅ Sync complete!")
                        st.json(results)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        with col2:
            if st.button("📊 Validate Database"):
                with st.spinner("Validating..."):
                    try:
                        validation = asyncio.run(db_manager.validate_database())
                        if validation["is_valid"]:
                            st.success("✅ Database is valid!")
                        else:
                            st.warning("⚠️ Issues found:")
                            for issue in validation["issues"]:
                                st.error(issue)

                        if validation["warnings"]:
                            for warning in validation["warnings"]:
                                st.warning(warning)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")


def render_tab3_chunks():
    """Tab 3: Chunk Viewer"""
    st.header("🔍 Chunk Viewer")

    if st.session_state.chunk_manager is None:
        st.warning("Please initialize the RAG system from the sidebar first!")
        return

    chunk_manager = st.session_state.chunk_manager
    metadata_store = st.session_state.metadata_store

    # Select document
    documents = metadata_store.get_all_documents()

    if not documents:
        st.info("No documents found. Upload some documents in Tab 1!")
        return

    doc_options = {f"{doc['regulation_id']} ({doc['filename']})": doc['doc_id'] for doc in documents}
    selected_doc = st.selectbox("Select Document", list(doc_options.keys()))
    doc_id = doc_options[selected_doc]

    st.divider()

    # Get chunks and statistics
    try:
        chunks = asyncio.run(chunk_manager.get_chunks_by_doc_id(doc_id))
        stats = asyncio.run(chunk_manager.get_chunk_statistics(doc_id))
        validation = asyncio.run(chunk_manager.validate_chunks(doc_id))

        # Display statistics
        st.subheader("📊 Chunk Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        with col2:
            st.metric("Avg Tokens", f"{stats.get('avg_tokens', 0):.0f}")
        with col3:
            st.metric("Min Tokens", stats.get("min_tokens", 0))
        with col4:
            st.metric("Max Tokens", stats.get("max_tokens", 0))

        # Validation results
        st.divider()
        st.subheader("✅ Validation")

        if validation["is_valid"]:
            st.success("✅ All chunks are valid!")
        else:
            for issue in validation["issues"]:
                st.error(f"❌ {issue}")

        for warning in validation["warnings"]:
            st.warning(f"⚠️ {warning}")

        for rec in validation["recommendations"]:
            st.info(f"💡 {rec}")

        # Search in chunks
        st.divider()
        st.subheader("🔎 Search in Chunks")

        search_text = st.text_input("Search for text within chunks")
        if search_text:
            matches = asyncio.run(
                chunk_manager.search_in_chunks(doc_id, search_text)
            )
            st.info(f"Found {len(matches)} matching chunks")

            for match in matches:
                st.text(f"Chunk {match['chunk_order_index']} ({match['match_count']} matches)")

        # Display chunks
        st.divider()
        st.subheader(f"📄 Chunks ({len(chunks)})")

        for chunk in chunks:
            with st.expander(f"Chunk {chunk['chunk_order_index']} ({chunk['tokens']} tokens)"):
                st.text_area(
                    "Content",
                    chunk['content'],
                    height=200,
                    key=f"chunk_{chunk['key']}"
                )

        # Export option
        st.divider()
        if st.button("📥 Export Chunks to Text"):
            export_path = f"./chunks_{doc_id}.txt"
            success = asyncio.run(
                chunk_manager.export_chunks_to_text(doc_id, export_path)
            )
            if success:
                st.success(f"✅ Exported to {export_path}")

    except Exception as e:
        st.error(f"Error loading chunks: {str(e)}")


def render_tab4_chat():
    """Tab 4: Enhanced Chat"""
    st.header("💬 Chat with Documents")

    if st.session_state.rag is None:
        st.warning("Please initialize the RAG system from the sidebar first!")
        return

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
                        st.session_state.rag.aquery(prompt, mode="hybrid")
                    )

                    # Display response
                    st.markdown(response)

                    # Try to extract regulation references
                    metadata_store = st.session_state.metadata_store
                    documents = metadata_store.get_all_documents()

                    # Find mentioned regulations
                    mentioned_regs = []
                    for doc in documents:
                        reg_id = doc.get("regulation_id", "")
                        if reg_id and reg_id in response:
                            mentioned_regs.append(doc)

                    # Display referenced documents
                    if mentioned_regs:
                        st.divider()
                        st.markdown("**📚 Referenced Regulations:**")
                        for doc in mentioned_regs:
                            st.markdown(f"- **{doc['regulation_id']}**: {doc['filename']}")

                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def main():
    st.title("📚 RAG-Anything Document Manager")
    st.markdown("Manage building regulations, view chunks, and chat with your documents")

    # Render sidebar
    working_dir, output_dir = render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Upload",
        "🗄️ Database",
        "🔍 Chunks",
        "💬 Chat"
    ])

    with tab1:
        render_tab1_upload()

    with tab2:
        render_tab2_database()

    with tab3:
        render_tab3_chunks()

    with tab4:
        render_tab4_chat()


if __name__ == "__main__":
    main()
