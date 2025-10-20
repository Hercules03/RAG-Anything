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
from dotenv import load_dotenv

# Load .env before anything else (matches LightRAG pattern)
load_dotenv(dotenv_path=".env", override=False)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Import utility modules
from utils import (
    ChunkManager,
    ChunkingStrategies,
    RegulationExtractor,
    MetadataStore,
    DocumentDatabase,
    run_async
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
            from lightrag.utils import compute_mdhash_id
            chunker = ChunkingStrategies()  # Fixed class name
            chunks = chunker.apply_strategy(
                text_content,
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                **chunking_kwargs
            )
            # chunks is now List[Dict[str, Any]] matching TextChunkSchema format

            # Delete old chunks and insert new ones
            all_chunk_keys = await rag.lightrag.text_chunks.get_all_keys()
            for key in all_chunk_keys:
                chunk_data = await rag.lightrag.text_chunks.get_by_id(key)
                if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                    await rag.lightrag.text_chunks.delete_by_id(key)

            # Insert custom chunks with proper format
            for chunk in chunks:
                chunk["full_doc_id"] = doc_id  # Add document ID
                chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
                await rag.lightrag.text_chunks.upsert(chunk_id, chunk)

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
            value="gemma3:27b",
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
                    st.session_state.rag = run_async(
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
            ["default", "fixed_token", "paragraph", "hybrid", "hierarchical"],
            help="Chunking method to use. Hierarchical is best for structured documents with headings."
        )

    with col2:
        # Get default from .env
        default_chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        chunk_size = st.number_input(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=4000,
            value=default_chunk_size,
            step=100,
            help=f"Target chunk size in tokens (default from .env: {default_chunk_size})"
        )

    # Additional parameters based on strategy
    if chunking_strategy == "fixed_token":
        default_overlap = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
        overlap = st.slider(
            "Overlap (tokens)",
            0,
            500,
            default_overlap,
            help=f"Overlapping tokens between chunks (default from .env: {default_overlap})"
        )
        chunking_kwargs = {"overlap": overlap}

    elif chunking_strategy == "paragraph":
        min_tokens = st.number_input("Min Tokens", 100, 2000, default_chunk_size // 2)
        max_tokens = st.number_input("Max Tokens", 500, 3000, int(default_chunk_size * 1.5))
        chunking_kwargs = {"min_tokens": min_tokens, "max_tokens": max_tokens}

    elif chunking_strategy == "hierarchical":
        st.info("📊 Hierarchical chunking creates parent-child relationships (10-15% better for structured docs)")

        col_a, col_b = st.columns(2)
        with col_a:
            default_parent = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
            parent_chunk_size = st.number_input(
                "Parent Chunk Size (tokens)",
                min_value=500,
                max_value=4000,
                value=default_parent,
                step=100,
                help=f"Large contextual units (default from .env: {default_parent})"
            )

        with col_b:
            default_child_overlap = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
            child_overlap = st.number_input(
                "Child Overlap (tokens)",
                min_value=0,
                max_value=200,
                value=default_child_overlap,
                step=10,
                help=f"Overlap for child chunks (default from .env: {default_child_overlap})"
            )

        extract_sections = st.checkbox(
            "Extract Sections",
            value=True,
            help="Detect document structure (headings, numbered sections). Recommended for structured docs."
        )

        use_intermediate = st.checkbox(
            "Use Intermediate Level",
            value=False,
            help="Enable 3-level hierarchy (parent → intermediate → child). More granular but more complex."
        )

        chunking_kwargs = {
            "parent_chunk_size": parent_chunk_size,
            "child_chunk_size": chunk_size,  # Use main chunk_size as child size
            "child_overlap": child_overlap,
            "extract_sections": extract_sections,
            "use_intermediate_level": use_intermediate
        }

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
                        extracted_reg_id = run_async(
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
            value="./pdfs",
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

                            run_async(
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
            documents = run_async(db_manager.list_documents(query=search_query))
        else:
            documents = run_async(db_manager.list_documents(query=search_query, status=status_filter))

        # Display statistics
        stats = run_async(db_manager.get_statistics())

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
                                        success = run_async(
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
                                        success = run_async(
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
                        results = run_async(db_manager.sync_with_lightrag())
                        st.success("✅ Sync complete!")
                        st.json(results)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        with col2:
            if st.button("📊 Validate Database"):
                with st.spinner("Validating..."):
                    try:
                        validation = run_async(db_manager.validate_database())
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
        chunks = run_async(chunk_manager.get_chunks_by_doc_id(doc_id))
        stats = run_async(chunk_manager.get_chunk_statistics(doc_id))
        validation = run_async(chunk_manager.validate_chunks(doc_id))

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
            matches = run_async(
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
            success = run_async(
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
                    response = run_async(
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


def render_tab5_chunking_test():
    """Tab 5: Chunking Method Testing"""
    st.header("🧪 Chunking Method Tester")
    st.markdown("Upload a PDF and test different chunking strategies to see how they split your document.")

    # PDF Upload
    st.subheader("1️⃣ Upload Test Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file to test chunking",
        type=["pdf"],
        help="Upload any PDF to see how different chunking strategies work",
        key="chunking_test_uploader"
    )

    if not uploaded_file:
        st.info("👆 Upload a PDF file to start testing chunking methods")
        return

    # Chunking Strategy Selection
    st.subheader("2️⃣ Select Chunking Strategy")

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Chunking Strategy",
            ["fixed_token", "paragraph", "hybrid", "hierarchical"],
            help="Select the chunking method to test"
        )

    with col2:
        # Get default from .env
        default_chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        chunk_size = st.number_input(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=4000,
            value=default_chunk_size,
            step=100,
            help="Target chunk size in tokens"
        )

    # Strategy-specific parameters
    st.subheader("3️⃣ Configure Parameters")

    chunking_kwargs = {}

    if strategy == "fixed_token":
        col_a, col_b = st.columns(2)
        with col_a:
            overlap = st.slider(
                "Overlap (tokens)",
                0,
                500,
                int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                help="Overlapping tokens between chunks"
            )
            chunking_kwargs["overlap"] = overlap
        with col_b:
            split_sentences = st.checkbox(
                "Split on Sentences",
                value=True,
                help="Try to split on sentence boundaries"
            )
            chunking_kwargs["split_sentences"] = split_sentences

    elif strategy == "paragraph":
        col_a, col_b = st.columns(2)
        with col_a:
            min_tokens = st.number_input(
                "Min Tokens",
                100,
                2000,
                default_chunk_size // 2,
                help="Minimum tokens per chunk"
            )
            chunking_kwargs["min_tokens"] = min_tokens
        with col_b:
            max_tokens = st.number_input(
                "Max Tokens",
                500,
                3000,
                int(default_chunk_size * 1.5),
                help="Maximum tokens per chunk"
            )
            chunking_kwargs["max_tokens"] = max_tokens

    elif strategy == "hybrid":
        hybrid_strategy = st.selectbox(
            "Hybrid Strategy",
            ["balanced", "paragraph-first", "semantic-first"],
            help="Sub-strategy for hybrid approach"
        )
        chunking_kwargs["hybrid_strategy"] = hybrid_strategy

    elif strategy == "hierarchical":
        st.info("📊 Hierarchical chunking creates parent-child relationships")

        col_a, col_b = st.columns(2)
        with col_a:
            parent_chunk_size = st.number_input(
                "Parent Chunk Size",
                500,
                4000,
                int(os.getenv("PARENT_CHUNK_SIZE", "1500")),
                step=100,
                help="Large contextual units"
            )
            chunking_kwargs["parent_chunk_size"] = parent_chunk_size

        with col_b:
            child_overlap = st.number_input(
                "Child Overlap",
                0,
                200,
                int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                step=10,
                help="Overlap for child chunks"
            )
            chunking_kwargs["child_overlap"] = child_overlap

        col_c, col_d = st.columns(2)
        with col_c:
            extract_sections = st.checkbox(
                "Extract Sections",
                value=True,
                help="Detect document structure"
            )
            chunking_kwargs["extract_sections"] = extract_sections

        with col_d:
            use_intermediate = st.checkbox(
                "3-Level Hierarchy",
                value=False,
                help="Enable intermediate level"
            )
            chunking_kwargs["use_intermediate_level"] = use_intermediate

    st.divider()

    # Chunk Button
    if st.button("🔬 Chunk Document", type="primary", use_container_width=True):
        with st.spinner(f"Processing with {strategy} strategy..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Parse PDF
                st.info("📖 Parsing PDF...")
                from raganything import RAGAnything, RAGAnythingConfig

                config = RAGAnythingConfig(
                    working_dir=tempfile.mkdtemp(),
                    parser=os.getenv("PARSER", "mineru"),
                    parse_method=os.getenv("PARSE_METHOD", "auto"),
                )

                temp_rag = RAGAnything(config=config)
                parsed_result = run_async(temp_rag.parse_document(tmp_path))
                text_content = parsed_result["text"]

                # Clean up temp file
                os.unlink(tmp_path)

                if not text_content or not text_content.strip():
                    st.error("❌ No text extracted from PDF")
                    return

                st.success(f"✅ Extracted {len(text_content)} characters")

                # Apply chunking strategy
                st.info(f"✂️ Chunking with {strategy} strategy...")
                chunker = ChunkingStrategies()

                chunks = chunker.apply_strategy(
                    text_content,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    **chunking_kwargs
                )

                st.success(f"✅ Created {len(chunks)} chunks")

                # Store in session state for display
                st.session_state.test_chunks = chunks
                st.session_state.test_strategy = strategy
                st.session_state.test_text_length = len(text_content)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display Results
    if "test_chunks" in st.session_state and st.session_state.test_chunks:
        st.divider()
        st.subheader("📊 Chunking Results")

        chunks = st.session_state.test_chunks
        strategy_used = st.session_state.test_strategy
        text_length = st.session_state.test_text_length

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Chunks", len(chunks))

        with col2:
            avg_tokens = sum(c["tokens"] for c in chunks) / len(chunks)
            st.metric("Avg Tokens/Chunk", f"{avg_tokens:.0f}")

        with col3:
            total_tokens = sum(c["tokens"] for c in chunks)
            st.metric("Total Tokens", total_tokens)

        with col4:
            st.metric("Source Characters", f"{text_length:,}")

        # Hierarchical-specific stats
        if strategy_used == "hierarchical":
            st.markdown("**Hierarchical Structure:**")
            col_a, col_b, col_c = st.columns(3)

            parents = [c for c in chunks if c.get("level") == 0]
            children = [c for c in chunks if c.get("level") in [1, 2]]

            with col_a:
                st.metric("Parent Chunks", len(parents))
            with col_b:
                st.metric("Child Chunks", len(children))
            with col_c:
                avg_children = len(children) / len(parents) if parents else 0
                st.metric("Avg Children/Parent", f"{avg_children:.1f}")

        st.divider()

        # Chunk Display Options
        st.subheader("🔍 Browse Chunks")

        col_view, col_filter = st.columns([2, 1])

        with col_view:
            view_mode = st.radio(
                "View Mode",
                ["Compact", "Detailed", "Metadata Only"],
                horizontal=True
            )

        with col_filter:
            if strategy_used == "hierarchical":
                level_filter = st.selectbox(
                    "Filter by Level",
                    ["All", "Parents Only", "Children Only"]
                )
            else:
                level_filter = "All"

        # Filter chunks
        display_chunks = chunks
        if level_filter == "Parents Only":
            display_chunks = [c for c in chunks if c.get("level") == 0]
        elif level_filter == "Children Only":
            display_chunks = [c for c in chunks if c.get("level") in [1, 2]]

        # Display chunks
        st.markdown(f"**Showing {len(display_chunks)} chunks:**")

        for i, chunk in enumerate(display_chunks):
            with st.expander(
                f"Chunk {chunk['chunk_order_index']} "
                f"({chunk['tokens']} tokens)"
                f"{' - ' + chunk.get('section_title', '') if chunk.get('section_title') else ''}"
            ):
                if view_mode == "Detailed":
                    # Show full content
                    st.markdown("**Content:**")
                    st.text_area(
                        "Chunk Content",
                        chunk["content"],
                        height=200,
                        key=f"chunk_content_{i}",
                        label_visibility="collapsed"
                    )

                    # Show metadata
                    st.markdown("**Metadata:**")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.write(f"🔢 Tokens: {chunk['tokens']}")
                        st.write(f"📍 Index: {chunk['chunk_order_index']}")
                    with col_m2:
                        if strategy_used == "hierarchical":
                            st.write(f"📊 Level: {chunk.get('level', 'N/A')}")
                            st.write(f"🔗 Parent: {chunk.get('parent_id', 'None')}")
                    with col_m3:
                        if strategy_used == "hierarchical":
                            st.write(f"👶 Children: {len(chunk.get('children_ids', []))}")
                            st.write(f"📑 Section: {chunk.get('section_title', 'None')}")

                elif view_mode == "Compact":
                    # Show preview only
                    preview = chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"]
                    st.markdown(f"```\n{preview}\n```")

                    if strategy_used == "hierarchical":
                        st.caption(
                            f"Level: {chunk.get('level')} | "
                            f"Parent: {chunk.get('parent_id', 'None')} | "
                            f"Children: {len(chunk.get('children_ids', []))}"
                        )

                else:  # Metadata Only
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.json({
                            "tokens": chunk["tokens"],
                            "chunk_order_index": chunk["chunk_order_index"],
                            "content_length": len(chunk["content"])
                        })
                    with col_meta2:
                        if strategy_used == "hierarchical":
                            st.json({
                                "level": chunk.get("level"),
                                "parent_id": chunk.get("parent_id"),
                                "children_count": len(chunk.get("children_ids", [])),
                                "section_title": chunk.get("section_title")
                            })


def main():
    st.title("📚 RAG-Anything Document Manager")
    st.markdown("Manage building regulations, view chunks, and chat with your documents")

    # Render sidebar
    working_dir, output_dir = render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Upload",
        "🗄️ Database",
        "🔍 Chunks",
        "💬 Chat",
        "🧪 Chunking Test"
    ])

    with tab1:
        render_tab1_upload()

    with tab2:
        render_tab2_database()

    with tab3:
        render_tab3_chunks()

    with tab4:
        render_tab4_chat()

    with tab5:
        render_tab5_chunking_test()


if __name__ == "__main__":
    main()
