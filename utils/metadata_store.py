"""
Metadata Store - Persistent document metadata management

Stores metadata for all documents in the RAG system:
- Document ID, regulation ID, filename
- Upload date, file size, page count
- Processing status and chunking strategy
- Sync with LightRAG's doc_status storage
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class MetadataStore:
    """Manage document metadata with persistence and LightRAG sync"""

    def __init__(self, storage_path: str = "./rag_storage", lightrag=None):
        """
        Initialize metadata store

        Args:
            storage_path: Directory for metadata storage
            lightrag: LightRAG instance for status sync
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        self.lightrag = lightrag

        # Initialize metadata file if not exists
        if not self.metadata_file.exists():
            self._save_metadata({})

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata from file

        Returns:
            Dictionary mapping doc_id to metadata
        """
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> bool:
        """
        Save metadata to file

        Args:
            metadata: Complete metadata dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    def add_document(
        self,
        doc_id: str,
        filename: str,
        regulation_id: Optional[str] = None,
        file_size: int = 0,
        page_count: int = 0,
        chunking_strategy: str = "default",
        **extra_fields
    ) -> bool:
        """
        Add document metadata

        Args:
            doc_id: Document ID (e.g., "doc-abc123")
            filename: Original filename
            regulation_id: Extracted regulation ID (e.g., "APP-1")
            file_size: File size in bytes
            page_count: Number of pages
            chunking_strategy: Chunking strategy used
            **extra_fields: Additional metadata fields

        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()

        metadata[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "regulation_id": regulation_id or filename,
            "file_size": file_size,
            "page_count": page_count,
            "chunking_strategy": chunking_strategy,
            "upload_date": datetime.now().isoformat(),
            "processing_status": "processing",
            **extra_fields
        }

        return self._save_metadata(metadata)

    def update_document(
        self, doc_id: str, updates: Dict[str, Any]
    ) -> bool:
        """
        Update document metadata

        Args:
            doc_id: Document ID
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()

        if doc_id not in metadata:
            print(f"Document {doc_id} not found in metadata")
            return False

        metadata[doc_id].update(updates)
        metadata[doc_id]["last_modified"] = datetime.now().isoformat()

        return self._save_metadata(metadata)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document metadata

        Args:
            doc_id: Document ID

        Returns:
            True if successful, False otherwise
        """
        metadata = self._load_metadata()

        if doc_id in metadata:
            del metadata[doc_id]
            return self._save_metadata(metadata)

        return False

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        metadata = self._load_metadata()
        return metadata.get(doc_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all document metadata

        Returns:
            List of metadata dictionaries
        """
        metadata = self._load_metadata()
        return list(metadata.values())

    def search_documents(
        self,
        query: str = "",
        regulation_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents by criteria

        Args:
            query: Search query (matches filename or regulation_id)
            regulation_id: Filter by regulation ID
            status: Filter by processing status

        Returns:
            List of matching metadata dictionaries
        """
        metadata = self._load_metadata()
        results = []

        for doc_meta in metadata.values():
            # Filter by regulation_id
            if regulation_id and doc_meta.get("regulation_id") != regulation_id:
                continue

            # Filter by status
            if status and doc_meta.get("processing_status") != status:
                continue

            # Filter by query
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in doc_meta.get("filename", "").lower()
                    or query_lower in doc_meta.get("regulation_id", "").lower()
                ):
                    continue

            results.append(doc_meta)

        return results

    def get_by_regulation_id(self, regulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by regulation ID

        Args:
            regulation_id: Regulation ID (e.g., "APP-1")

        Returns:
            Metadata dictionary or None if not found
        """
        metadata = self._load_metadata()

        for doc_meta in metadata.values():
            if doc_meta.get("regulation_id") == regulation_id:
                return doc_meta

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get metadata statistics

        Returns:
            Dictionary with statistics:
            {
                "total_documents": int,
                "total_pages": int,
                "total_size_bytes": int,
                "by_status": Counter,
                "by_regulation": Counter,
                "by_chunking_strategy": Counter
            }
        """
        metadata = self._load_metadata()

        stats = {
            "total_documents": len(metadata),
            "total_pages": 0,
            "total_size_bytes": 0,
            "by_status": {},
            "by_regulation": {},
            "by_chunking_strategy": {},
        }

        for doc_meta in metadata.values():
            # Count pages and size
            stats["total_pages"] += doc_meta.get("page_count", 0)
            stats["total_size_bytes"] += doc_meta.get("file_size", 0)

            # Count by status
            status = doc_meta.get("processing_status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by regulation
            reg_id = doc_meta.get("regulation_id", "unknown")
            stats["by_regulation"][reg_id] = stats["by_regulation"].get(reg_id, 0) + 1

            # Count by chunking strategy
            strategy = doc_meta.get("chunking_strategy", "default")
            stats["by_chunking_strategy"][strategy] = (
                stats["by_chunking_strategy"].get(strategy, 0) + 1
            )

        return stats

    async def sync_with_lightrag(self) -> Dict[str, Any]:
        """
        Sync metadata with LightRAG's doc_status storage

        Returns:
            Dictionary with sync results:
            {
                "synced": int,
                "added_to_metadata": int,
                "updated_status": int,
                "orphaned_in_metadata": int
            }
        """
        if not self.lightrag or not hasattr(self.lightrag, "doc_status"):
            return {
                "synced": 0,
                "added_to_metadata": 0,
                "updated_status": 0,
                "orphaned_in_metadata": 0,
            }

        metadata = self._load_metadata()
        results = {
            "synced": 0,
            "added_to_metadata": 0,
            "updated_status": 0,
            "orphaned_in_metadata": 0,
        }

        # Get all doc_ids from LightRAG
        lightrag_doc_ids = set(await self.lightrag.doc_status.get_all_keys())
        metadata_doc_ids = set(metadata.keys())

        # Add missing documents to metadata
        for doc_id in lightrag_doc_ids - metadata_doc_ids:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if doc_status:
                metadata[doc_id] = {
                    "doc_id": doc_id,
                    "filename": doc_id,
                    "regulation_id": doc_id,
                    "processing_status": doc_status.get("status", "unknown"),
                    "upload_date": datetime.now().isoformat(),
                    "synced_from_lightrag": True,
                }
                results["added_to_metadata"] += 1

        # Update status for existing documents
        for doc_id in lightrag_doc_ids & metadata_doc_ids:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if doc_status:
                old_status = metadata[doc_id].get("processing_status")
                new_status = doc_status.get("status", "unknown")

                if old_status != new_status:
                    metadata[doc_id]["processing_status"] = new_status
                    metadata[doc_id]["last_synced"] = datetime.now().isoformat()
                    results["updated_status"] += 1

        # Count orphaned documents (in metadata but not in LightRAG)
        results["orphaned_in_metadata"] = len(metadata_doc_ids - lightrag_doc_ids)
        results["synced"] = len(lightrag_doc_ids & metadata_doc_ids)

        # Save updated metadata
        self._save_metadata(metadata)

        return results

    def export_to_csv(self, output_path: str) -> bool:
        """
        Export metadata to CSV file

        Args:
            output_path: Path to save CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            import csv

            metadata = self._load_metadata()

            if not metadata:
                return False

            # Get all fields from first document
            fields = list(next(iter(metadata.values())).keys())

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(metadata.values())

            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
