"""
Chunk Manager - View, validate, and manage document chunks

Provides tools to:
- Retrieve chunks for a specific document
- Calculate chunk statistics
- Validate chunk quality
- Search within chunks
"""

import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics


class ChunkManager:
    """Manage and view document chunks in LightRAG"""

    def __init__(self, lightrag):
        """
        Initialize ChunkManager

        Args:
            lightrag: LightRAG instance with initialized storages
        """
        self.lightrag = lightrag

    async def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document

        Args:
            doc_id: Document ID (e.g., "doc-abc123")

        Returns:
            List of chunk dictionaries with structure:
            {
                "tokens": int,
                "content": str,
                "full_doc_id": str,
                "chunk_order_index": int
            }
        """
        if not hasattr(self.lightrag, "text_chunks"):
            return []

        # Get all chunks from storage
        # Note: LightRAG uses hash-based keys, so we need to scan all
        all_chunks_dict = await self.lightrag.text_chunks.get_all()

        # DEBUG: Log what we're searching for
        print(f"\n{'='*70}")
        print(f"DEBUG [get_chunks_by_doc_id]: Searching for doc_id: '{doc_id}'")
        print(f"DEBUG [get_chunks_by_doc_id]: Total chunks in storage: {len(all_chunks_dict)}")

        # DEBUG: Show sample of stored doc_ids (first 10 unique)
        stored_doc_ids = set()
        for key, chunk_data in all_chunks_dict.items():
            if chunk_data and chunk_data.get("full_doc_id"):
                stored_doc_ids.add(chunk_data.get("full_doc_id"))
                if len(stored_doc_ids) >= 10:
                    break
        print(f"DEBUG [get_chunks_by_doc_id]: Sample stored doc_ids (up to 10):")
        for sample_id in sorted(stored_doc_ids):
            print(f"  - '{sample_id}'")

        chunks = []
        for key, chunk_data in all_chunks_dict.items():
            if chunk_data and chunk_data.get("full_doc_id") == doc_id:
                chunks.append({
                    "key": key,
                    "tokens": chunk_data.get("tokens", 0),
                    "content": chunk_data.get("content", ""),
                    "full_doc_id": chunk_data.get("full_doc_id", ""),
                    "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                })

        # Sort by chunk order
        chunks.sort(key=lambda x: x["chunk_order_index"])

        # DEBUG: Log what we found
        print(f"DEBUG [get_chunks_by_doc_id]: Found {len(chunks)} chunks for doc_id: '{doc_id}'")
        print(f"{'='*70}\n")

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
