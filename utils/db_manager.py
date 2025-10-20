"""
Document Database Manager - CRUD operations for RAG documents

Provides high-level interface for:
- Viewing all documents with metadata
- Deleting documents (from both LightRAG and metadata)
- Re-processing documents with different chunking strategies
- Searching and filtering documents
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from .metadata_store import MetadataStore
from .regulation_extractor import RegulationExtractor
from .chunking_strategies import ChunkingStrategies


class DocumentDatabase:
    """High-level document database management"""

    def __init__(self, rag_instance, metadata_store: MetadataStore):
        """
        Initialize document database manager

        Args:
            rag_instance: RAGAnything instance
            metadata_store: MetadataStore instance
        """
        self.rag = rag_instance
        self.lightrag = rag_instance.lightrag
        self.metadata = metadata_store
        self.reg_extractor = RegulationExtractor()
        self.chunker = ChunkingStrategies()

    async def list_documents(
        self, query: str = "", status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all documents with metadata

        Args:
            query: Search query (matches filename or regulation_id)
            status: Filter by processing status

        Returns:
            List of document metadata dictionaries
        """
        # Get metadata
        documents = self.metadata.search_documents(query=query, status=status)

        # Enrich with LightRAG status (check if chunks exist)
        for doc in documents:
            doc_id = doc["doc_id"]

            # Check if document has chunks in LightRAG
            try:
                # Get all chunk keys and filter by doc_id
                all_chunk_keys = await self.lightrag.text_chunks.get_all_keys()
                doc_chunks = []
                for key in all_chunk_keys:
                    chunk_data = await self.lightrag.text_chunks.get_by_id(key)
                    if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                        doc_chunks.append(chunk_data)

                if doc_chunks:
                    doc["lightrag_status"] = "indexed"
                    doc["chunk_count"] = len(doc_chunks)
                else:
                    doc["lightrag_status"] = "not_indexed"
                    doc["chunk_count"] = 0
            except Exception:
                doc["lightrag_status"] = "unknown"
                doc["chunk_count"] = 0

        return documents

    async def get_document_details(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a document

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with complete document information:
            {
                "metadata": {...},
                "lightrag_status": {...},
                "chunk_count": int,
                "chunk_stats": {...}
            }
        """
        # Get metadata
        metadata = self.metadata.get_document(doc_id)
        if not metadata:
            return None

        # Get chunk information and LightRAG status
        try:
            from .chunk_manager import ChunkManager

            chunk_mgr = ChunkManager(self.lightrag)
            chunks = await chunk_mgr.get_chunks_by_doc_id(doc_id)
            chunk_stats = await chunk_mgr.get_chunk_statistics(doc_id)

            # Determine LightRAG status based on chunks
            lightrag_status = {
                "indexed": len(chunks) > 0,
                "chunk_count": len(chunks)
            }
        except Exception:
            chunks = []
            chunk_stats = {}
            lightrag_status = {"indexed": False, "chunk_count": 0}

        return {
            "metadata": metadata,
            "lightrag_status": lightrag_status,
            "chunk_count": len(chunks),
            "chunk_stats": chunk_stats,
        }

    async def delete_document(self, doc_id: str, delete_from_lightrag: bool = True) -> bool:
        """
        Delete document from database

        Args:
            doc_id: Document ID
            delete_from_lightrag: Whether to delete from LightRAG storage

        Returns:
            True if successful, False otherwise
        """
        success = True

        # Delete from metadata
        if not self.metadata.delete_document(doc_id):
            print(f"Warning: Failed to delete metadata for {doc_id}")
            success = False

        # Delete from LightRAG storage
        if delete_from_lightrag:
            try:
                # Delete chunks
                all_chunk_keys = await self.lightrag.text_chunks.get_all_keys()
                for key in all_chunk_keys:
                    chunk_data = await self.lightrag.text_chunks.get_by_id(key)
                    if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                        await self.lightrag.text_chunks.delete_by_id(key)

                # Delete document
                await self.lightrag.full_docs.delete_by_id(doc_id)

                # Note: LightRAG doesn't have doc_status, only text_chunks and full_docs

                print(f"Successfully deleted {doc_id} from LightRAG storage")

            except Exception as e:
                print(f"Error deleting from LightRAG: {e}")
                success = False

        return success

    async def delete_multiple_documents(
        self, doc_ids: List[str], delete_from_lightrag: bool = True
    ) -> Dict[str, bool]:
        """
        Delete multiple documents

        Args:
            doc_ids: List of document IDs
            delete_from_lightrag: Whether to delete from LightRAG storage

        Returns:
            Dictionary mapping doc_id to success status
        """
        results = {}

        for doc_id in doc_ids:
            results[doc_id] = await self.delete_document(doc_id, delete_from_lightrag)

        return results

    async def reprocess_document(
        self,
        doc_id: str,
        chunking_strategy: str = "default",
        chunk_size: int = 1200,
        **chunking_kwargs
    ) -> bool:
        """
        Re-process document with different chunking strategy

        Args:
            doc_id: Document ID
            chunking_strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            **chunking_kwargs: Additional chunking parameters

        Returns:
            True if successful, False otherwise
        """
        # Get document metadata
        metadata = self.metadata.get_document(doc_id)
        if not metadata:
            print(f"Document {doc_id} not found in metadata")
            return False

        # Get full document content
        full_doc = await self.lightrag.full_docs.get_by_id(doc_id)
        if not full_doc:
            print(f"Document {doc_id} not found in LightRAG storage")
            return False

        # Extract text content from full_doc
        text_content = ""
        if isinstance(full_doc, dict):
            content_list = full_doc.get("content", [])
            for item in content_list:
                if item.get("type") == "text":
                    text_content += item.get("text", "") + "\n"
        else:
            text_content = str(full_doc)

        if not text_content.strip():
            print(f"No text content found for {doc_id}")
            return False

        try:
            # Delete old chunks
            all_chunk_keys = await self.lightrag.text_chunks.get_all_keys()
            for key in all_chunk_keys:
                chunk_data = await self.lightrag.text_chunks.get_by_id(key)
                if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                    await self.lightrag.text_chunks.delete_by_id(key)

            # Apply new chunking strategy
            if chunking_strategy == "default":
                # Use LightRAG's default chunking - re-insert document
                content_list = full_doc.get("content", [])
                await self.rag.insert_content_list(content_list, file_path=doc_id)
            else:
                # Use custom chunking strategy
                chunks = self.chunker.apply_strategy(
                    text_content,
                    strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    **chunking_kwargs
                )

                # Insert custom chunks
                await self.lightrag.ainsert_custom_chunks(
                    full_text=text_content,
                    text_chunks=chunks,
                    doc_id=doc_id
                )

            # Update metadata
            self.metadata.update_document(
                doc_id,
                {
                    "chunking_strategy": chunking_strategy,
                    "processing_status": "completed",
                }
            )

            print(f"Successfully re-processed {doc_id} with {chunking_strategy} strategy")
            return True

        except Exception as e:
            print(f"Error re-processing document: {e}")
            self.metadata.update_document(
                doc_id,
                {"processing_status": "failed"}
            )
            return False

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        search_in: str = "metadata"
    ) -> List[Dict[str, Any]]:
        """
        Search documents

        Args:
            query: Search query
            limit: Maximum number of results
            search_in: Where to search:
                - "metadata": Search in metadata only
                - "content": Search in document content using RAG

        Returns:
            List of matching documents with metadata
        """
        if search_in == "metadata":
            # Search in metadata
            results = self.metadata.search_documents(query=query)
            return results[:limit]

        elif search_in == "content":
            # Search using RAG query
            try:
                rag_results = await self.rag.query(query, mode="hybrid")

                # Extract doc_ids from results
                doc_ids = set()
                if isinstance(rag_results, dict):
                    # Parse response to find document references
                    response = rag_results.get("response", "")
                    # This is a simple heuristic - you may need to improve this
                    for doc in self.metadata.get_all_documents():
                        if doc.get("regulation_id", "") in response:
                            doc_ids.add(doc["doc_id"])

                # Get metadata for matched documents
                results = [
                    self.metadata.get_document(doc_id)
                    for doc_id in doc_ids
                    if self.metadata.get_document(doc_id)
                ]

                return results[:limit]

            except Exception as e:
                print(f"Error searching with RAG: {e}")
                return []

        else:
            raise ValueError(f"Invalid search_in value: {search_in}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with comprehensive statistics
        """
        # Get metadata statistics
        meta_stats = self.metadata.get_statistics()

        # Get LightRAG storage statistics
        try:
            # Count chunks in LightRAG (no doc_status attribute in LightRAG)
            all_chunks = await self.lightrag.text_chunks.get_all_keys()

            # Count unique documents from chunks
            unique_doc_ids = set()
            for chunk_key in all_chunks:
                chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_key)
                if chunk_data and chunk_data.get("full_doc_id"):
                    unique_doc_ids.add(chunk_data["full_doc_id"])

            lightrag_stats = {
                "total_docs_in_lightrag": len(unique_doc_ids),
                "total_chunks_in_lightrag": len(all_chunks),
            }
        except Exception:
            lightrag_stats = {
                "total_docs_in_lightrag": 0,
                "total_chunks_in_lightrag": 0,
            }

        return {
            **meta_stats,
            **lightrag_stats,
        }

    async def sync_with_lightrag(self) -> Dict[str, Any]:
        """
        Sync metadata with LightRAG storage

        Returns:
            Sync results dictionary
        """
        return await self.metadata.sync_with_lightrag()

    async def export_metadata(self, output_path: str, format: str = "csv") -> bool:
        """
        Export metadata to file

        Args:
            output_path: Path to save file
            format: Export format ("csv" or "json")

        Returns:
            True if successful, False otherwise
        """
        if format == "csv":
            return self.metadata.export_to_csv(output_path)
        elif format == "json":
            import json
            try:
                metadata = self.metadata.get_all_documents()
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"Error exporting to JSON: {e}")
                return False
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def validate_database(self) -> Dict[str, Any]:
        """
        Validate database integrity

        Returns:
            Dictionary with validation results:
            {
                "is_valid": bool,
                "issues": List[str],
                "warnings": List[str],
                "recommendations": List[str]
            }
        """
        issues = []
        warnings = []
        recommendations = []

        # Get all documents from metadata
        metadata_docs = {doc["doc_id"] for doc in self.metadata.get_all_documents()}

        # Get all documents from LightRAG (by scanning chunks)
        try:
            all_chunk_keys = await self.lightrag.text_chunks.get_all_keys()
            lightrag_docs = set()
            for key in all_chunk_keys:
                chunk_data = await self.lightrag.text_chunks.get_by_id(key)
                if chunk_data and chunk_data.get("full_doc_id"):
                    lightrag_docs.add(chunk_data["full_doc_id"])
        except Exception:
            lightrag_docs = set()

        # Check for orphaned metadata
        orphaned_metadata = metadata_docs - lightrag_docs
        if orphaned_metadata:
            issues.append(
                f"Found {len(orphaned_metadata)} documents in metadata but not in LightRAG: {list(orphaned_metadata)[:5]}"
            )
            recommendations.append("Run sync_with_lightrag() to clean up orphaned metadata")

        # Check for missing metadata
        missing_metadata = lightrag_docs - metadata_docs
        if missing_metadata:
            warnings.append(
                f"Found {len(missing_metadata)} documents in LightRAG but not in metadata: {list(missing_metadata)[:5]}"
            )
            recommendations.append("Run sync_with_lightrag() to add missing metadata")

        # Check for documents with no chunks
        from .chunk_manager import ChunkManager
        chunk_mgr = ChunkManager(self.lightrag)

        for doc_id in metadata_docs & lightrag_docs:
            chunks = await chunk_mgr.get_chunks_by_doc_id(doc_id)
            if not chunks:
                warnings.append(f"Document {doc_id} has no chunks")

        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
