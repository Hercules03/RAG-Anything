"""
Chunk Manager - View, validate, and manage document chunks

Provides tools to:
- Retrieve chunks for a specific document
- Calculate chunk statistics
- Validate chunk quality
- Search within chunks
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics
from pathlib import Path


class ChunkManager:
    """Manage and view document chunks in LightRAG"""

    def __init__(self, lightrag, metadata_store=None):
        """
        Initialize ChunkManager

        Args:
            lightrag: LightRAG instance with initialized storages
            metadata_store: Optional metadata store for ID mapping
        """
        self.lightrag = lightrag
        self.metadata_store = metadata_store

    def _resolve_doc_id(self, doc_id: str) -> str:
        """
        Resolve document ID to the actual LightRAG document ID

        Args:
            doc_id: Metadata document ID (e.g., "doc-APP001")

        Returns:
            Actual LightRAG document ID or original if not found
        """
        if not self.metadata_store:
            return doc_id

        # Try 1: Get from metadata lightrag_doc_id field
        metadata = self.metadata_store.get_document(doc_id)
        if metadata and metadata.get("lightrag_doc_id"):
            return metadata["lightrag_doc_id"]

        # Try 2: Search full_docs by file path or regulation_id
        if metadata and self.lightrag and hasattr(self.lightrag, "full_docs"):
            try:
                # Try synchronous file read as fallback
                full_docs_file = Path("./rag_storage/kv_store_full_docs.json")
                if full_docs_file.exists():
                    with open(full_docs_file, "r", encoding="utf-8") as f:
                        all_docs = json.load(f)

                    filename = metadata.get("filename", "")
                    regulation_id = metadata.get("regulation_id", "")

                    for lightrag_doc_id, doc_data in all_docs.items():
                        if not doc_data:
                            continue

                        # Match by file path or regulation_id
                        doc_file_path = doc_data.get("file_path", "")

                        # Check if filename matches
                        if filename and filename in doc_file_path:
                            print(f"Resolved {doc_id} -> {lightrag_doc_id} (by filename)")
                            # Update metadata with the resolved ID for future use
                            self.metadata_store.update_document(
                                doc_id, {"lightrag_doc_id": lightrag_doc_id}
                            )
                            return lightrag_doc_id

                        # Check if regulation_id matches
                        if regulation_id and regulation_id in doc_file_path:
                            print(f"Resolved {doc_id} -> {lightrag_doc_id} (by regulation_id)")
                            self.metadata_store.update_document(
                                doc_id, {"lightrag_doc_id": lightrag_doc_id}
                            )
                            return lightrag_doc_id

            except Exception as e:
                print(f"Error resolving doc_id from full_docs: {e}")

        return doc_id

    def _get_chunks_from_storage_files(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Fallback method to get chunks directly from storage files

        Args:
            doc_id: Document ID

        Returns:
            List of chunk dictionaries
        """
        try:
            # Resolve the document ID to the actual LightRAG ID
            actual_doc_id = self._resolve_doc_id(doc_id)

            # Try to read chunks from storage file
            chunks_file = Path("./rag_storage/kv_store_text_chunks.json")
            if not chunks_file.exists():
                # Store debug info
                self._last_debug_info = {
                    "queried_doc_id": doc_id,
                    "resolved_doc_id": actual_doc_id,
                    "total_chunks_in_storage": 0,
                    "sample_stored_doc_ids": [],
                    "found_chunks_count": 0,
                    "method": "fallback - storage file not found"
                }
                return []

            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            # Collect debug info
            stored_doc_ids = set()
            for key, chunk_data in chunks_data.items():
                if chunk_data and chunk_data.get("full_doc_id"):
                    stored_doc_ids.add(chunk_data.get("full_doc_id"))
                    if len(stored_doc_ids) >= 10:
                        break

            chunks = []
            for key, chunk_data in chunks_data.items():
                if chunk_data and chunk_data.get("full_doc_id") == actual_doc_id:
                    chunks.append({
                        "key": key,
                        "tokens": chunk_data.get("tokens", 0),
                        "content": chunk_data.get("content", ""),
                        "full_doc_id": chunk_data.get("full_doc_id", ""),
                        "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                    })

            # Sort by chunk order
            chunks.sort(key=lambda x: x["chunk_order_index"])

            # Store debug info
            self._last_debug_info = {
                "queried_doc_id": doc_id,
                "resolved_doc_id": actual_doc_id,
                "total_chunks_in_storage": len(chunks_data),
                "sample_stored_doc_ids": sorted(stored_doc_ids),
                "found_chunks_count": len(chunks),
                "method": "fallback - direct file read"
            }

            return chunks

        except Exception as e:
            print(f"Error reading chunks from storage files: {e}")
            # Store debug info for error case
            self._last_debug_info = {
                "queried_doc_id": doc_id,
                "resolved_doc_id": "error",
                "total_chunks_in_storage": 0,
                "sample_stored_doc_ids": [],
                "found_chunks_count": 0,
                "method": f"fallback - error: {str(e)}"
            }
            return []

    async def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document

        Args:
            doc_id: Document ID (e.g., "doc-APP001" or "doc-abc123")

        Returns:
            List of chunk dictionaries with structure:
            {
                "tokens": int,
                "content": str,
                "full_doc_id": str,
                "chunk_order_index": int
            }
        """
        # Check if LightRAG is properly initialized
        if not self.lightrag:
            print("Warning: LightRAG instance is None, using fallback method")
            return self._get_chunks_from_storage_files(doc_id)
            
        if not hasattr(self.lightrag, "text_chunks"):
            print("Warning: LightRAG instance does not have text_chunks attribute, using fallback method")
            return self._get_chunks_from_storage_files(doc_id)

        # Resolve the document ID to the actual LightRAG ID
        actual_doc_id = self._resolve_doc_id(doc_id)
        
        try:
            # Get all chunks from storage
            # Note: LightRAG uses hash-based keys, so we need to scan all
            all_chunks_dict = await self.lightrag.text_chunks.get_all()
        except Exception as e:
            print(f"Error accessing text_chunks: {e}, using fallback method")
            return self._get_chunks_from_storage_files(doc_id)

        # Collect debug info to return
        debug_info = {
            "queried_doc_id": doc_id,
            "resolved_doc_id": actual_doc_id,
            "total_chunks_in_storage": len(all_chunks_dict),
            "sample_stored_doc_ids": [],
            "method": "async - LightRAG API"
        }

        # Get sample of stored doc_ids (first 10 unique)
        stored_doc_ids = set()
        for key, chunk_data in all_chunks_dict.items():
            if chunk_data and chunk_data.get("full_doc_id"):
                stored_doc_ids.add(chunk_data.get("full_doc_id"))
                if len(stored_doc_ids) >= 10:
                    break
        debug_info["sample_stored_doc_ids"] = sorted(stored_doc_ids)

        chunks = []
        for key, chunk_data in all_chunks_dict.items():
            if chunk_data and chunk_data.get("full_doc_id") == actual_doc_id:
                chunks.append({
                    "key": key,
                    "tokens": chunk_data.get("tokens", 0),
                    "content": chunk_data.get("content", ""),
                    "full_doc_id": chunk_data.get("full_doc_id", ""),
                    "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                })

        # Sort by chunk order
        chunks.sort(key=lambda x: x["chunk_order_index"])

        debug_info["found_chunks_count"] = len(chunks)

        # Store debug info for UI access
        self._last_debug_info = debug_info

        return chunks

    async def get_chunk_statistics(self, doc_id: str) -> Dict[str, Any]:
        """
        Calculate statistics for document chunks

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with statistics:
            {
                "total_chunks": int,
                "total_tokens": int,
                "avg_tokens": float,
                "min_tokens": int,
                "max_tokens": int,
                "median_tokens": float,
                "std_dev_tokens": float,
                "token_distribution": Counter,
            }
        """
        chunks = await self.get_chunks_by_doc_id(doc_id)

        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "median_tokens": 0,
                "std_dev_tokens": 0,
                "token_distribution": Counter(),
            }

        token_counts = [chunk["tokens"] for chunk in chunks]
        total_tokens = sum(token_counts)

        # Calculate token distribution (rounded to nearest 100)
        token_distribution = Counter(
            [(tokens // 100) * 100 for tokens in token_counts]
        )

        stats = {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(chunks) if chunks else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "median_tokens": statistics.median(token_counts) if token_counts else 0,
            "std_dev_tokens": statistics.stdev(token_counts)
            if len(token_counts) > 1
            else 0,
            "token_distribution": token_distribution,
        }

        return stats

    async def validate_chunks(self, doc_id: str) -> Dict[str, Any]:
        """
        Validate chunk quality and identify potential issues

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with validation results:
            {
                "is_valid": bool,
                "issues": List[str],
                "warnings": List[str],
                "recommendations": List[str]
            }
        """
        chunks = await self.get_chunks_by_doc_id(doc_id)
        issues = []
        warnings = []
        recommendations = []

        if not chunks:
            issues.append("No chunks found for this document")
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "recommendations": ["Re-process the document"],
            }

        # Check 1: Chunk order continuity
        expected_order = list(range(len(chunks)))
        actual_order = [chunk["chunk_order_index"] for chunk in chunks]
        if actual_order != expected_order:
            issues.append(
                f"Chunk order is not continuous. Expected {expected_order}, got {actual_order}"
            )

        # Check 2: Chunk size anomalies
        stats = await self.get_chunk_statistics(doc_id)
        if stats["min_tokens"] < 50:
            warnings.append(
                f"Some chunks are very small (min: {stats['min_tokens']} tokens). "
                "Consider increasing chunk size."
            )
        if stats["max_tokens"] > 5000:
            warnings.append(
                f"Some chunks are very large (max: {stats['max_tokens']} tokens). "
                "Consider decreasing chunk size."
            )

        # Check 3: Standard deviation (consistency)
        if stats["std_dev_tokens"] > stats["avg_tokens"] * 0.5:
            warnings.append(
                f"High variation in chunk sizes (std dev: {stats['std_dev_tokens']:.1f}). "
                "Chunks are inconsistently sized."
            )

        # Check 4: Encoding issues
        for i, chunk in enumerate(chunks):
            try:
                # Try to encode/decode to check for issues
                chunk["content"].encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                issues.append(
                    f"Chunk {i} has encoding issues that may affect search/retrieval"
                )

        # Check 5: Empty chunks
        empty_chunks = [
            i for i, chunk in enumerate(chunks) if not chunk["content"].strip()
        ]
        if empty_chunks:
            issues.append(
                f"Found {len(empty_chunks)} empty chunks at indices: {empty_chunks}"
            )

        # Check 6: Parent document exists
        parent_doc = await self.lightrag.full_docs.get_by_id(doc_id)
        if not parent_doc:
            issues.append("Parent document not found in storage (orphaned chunks)")

        # Recommendations
        if stats["avg_tokens"] < 500:
            recommendations.append(
                "Average chunk size is small. Consider increasing to 1000-1500 tokens for better context."
            )
        if stats["avg_tokens"] > 2000:
            recommendations.append(
                "Average chunk size is large. Consider decreasing to 1000-1500 tokens for better retrieval precision."
            )

        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    async def search_in_chunks(
        self, doc_id: str, query: str, case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for text within document's chunks

        Args:
            doc_id: Document ID
            query: Search query string
            case_sensitive: Whether search should be case-sensitive

        Returns:
            List of chunks containing the query with highlighted matches
        """
        chunks = await self.get_chunks_by_doc_id(doc_id)
        matching_chunks = []

        search_query = query if case_sensitive else query.lower()

        for chunk in chunks:
            search_content = (
                chunk["content"] if case_sensitive else chunk["content"].lower()
            )

            if search_query in search_content:
                # Find match positions
                match_count = search_content.count(search_query)

                matching_chunks.append(
                    {
                        **chunk,
                        "match_count": match_count,
                        "query": query,
                    }
                )

        return matching_chunks

    async def get_chunk_by_index(self, doc_id: str, chunk_index: int) -> Optional[Dict]:
        """
        Get a specific chunk by its index

        Args:
            doc_id: Document ID
            chunk_index: Chunk order index

        Returns:
            Chunk dictionary or None if not found
        """
        chunks = await self.get_chunks_by_doc_id(doc_id)

        for chunk in chunks:
            if chunk["chunk_order_index"] == chunk_index:
                return chunk

        return None

    async def compare_chunking_strategies(
        self, doc_id: str, alternative_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Compare current chunking with an alternative strategy

        Args:
            doc_id: Document ID
            alternative_chunks: List of alternative chunk texts

        Returns:
            Comparison statistics
        """
        current_chunks = await self.get_chunks_by_doc_id(doc_id)
        current_stats = await self.get_chunk_statistics(doc_id)

        # Calculate stats for alternative chunks
        alt_token_counts = [len(chunk.split()) * 1.3 for chunk in alternative_chunks]  # Rough token estimate
        alt_stats = {
            "total_chunks": len(alternative_chunks),
            "avg_tokens": sum(alt_token_counts) / len(alt_token_counts)
            if alt_token_counts
            else 0,
            "min_tokens": min(alt_token_counts) if alt_token_counts else 0,
            "max_tokens": max(alt_token_counts) if alt_token_counts else 0,
        }

        return {
            "current": current_stats,
            "alternative": alt_stats,
            "chunk_count_diff": alt_stats["total_chunks"]
            - current_stats["total_chunks"],
            "avg_tokens_diff": alt_stats["avg_tokens"] - current_stats["avg_tokens"],
        }

    async def export_chunks_to_text(self, doc_id: str, output_path: str) -> bool:
        """
        Export all chunks to a text file for inspection

        Args:
            doc_id: Document ID
            output_path: Path to save the text file

        Returns:
            True if successful, False otherwise
        """
        try:
            chunks = await self.get_chunks_by_doc_id(doc_id)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Document ID: {doc_id}\n")
                f.write(f"Total Chunks: {len(chunks)}\n")
                f.write("=" * 80 + "\n\n")

                for chunk in chunks:
                    f.write(f"Chunk {chunk['chunk_order_index']}\n")
                    f.write(f"Tokens: {chunk['tokens']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(chunk["content"])
                    f.write("\n" + "=" * 80 + "\n\n")

            return True
        except Exception as e:
            print(f"Error exporting chunks: {e}")
            return False
