"""
Chunking Strategies - Pre-embedding text chunking methods for RAG

Provides various chunking strategies applied BEFORE embedding (early/naive chunking):
- Fixed token size chunking with overlap
- Paragraph-based chunking
- Semantic chunking (future)
- Contextual chunking with metadata (future)
- Hybrid strategies

Note: This implements "early chunking" (chunk-then-embed workflow).
For true "late chunking" (embed-then-chunk), see: arXiv:2409.04701
"""

import os
import re
from typing import List, Dict, Any, Tuple
import tiktoken
from dotenv import load_dotenv
from .hierarchical_chunking import HierarchicalChunker

# Load .env file (same pattern as LightRAG)
load_dotenv(dotenv_path=".env", override=False)


class ChunkingStrategies:
    """Custom chunking strategies for pre-embedding text processing"""

    def __init__(
        self,
        encoding_name: str = None,
        chunk_size: int = None,
        overlap: int = None
    ):
        """
        Initialize chunking strategies

        Configuration priority:
        1. Explicit parameters passed to __init__
        2. Environment variables from .env
        3. Default values

        Args:
            encoding_name: Tiktoken encoding name (default: from TIKTOKEN_ENCODING or cl100k_base)
            chunk_size: Default chunk size in tokens (default: from CHUNK_SIZE or 1200)
            overlap: Default overlap size in tokens (default: from CHUNK_OVERLAP_SIZE or 100)
        """
        # Load from .env with fallback to defaults (matching LightRAG pattern)
        self.encoding_name = encoding_name or os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
        self.default_chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1200"))
        self.default_overlap = overlap or int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))

        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            # Fallback to rough word-based estimation
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.3)

    def _format_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Convert plain text chunks to LightRAG-compatible format

        Matches LightRAG's TextChunkSchema format:
        {
            "tokens": int,
            "content": str,
            "chunk_order_index": int
        }

        Note: full_doc_id will be added by LightRAG during insertion

        Args:
            chunks: List of plain text chunks

        Returns:
            List of formatted chunk dictionaries
        """
        formatted = []
        for index, chunk_text in enumerate(chunks):
            formatted.append({
                "tokens": self.count_tokens(chunk_text),
                "content": chunk_text.strip(),
                "chunk_order_index": index,
            })
        return formatted

    def fixed_token_chunking(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
        split_sentences: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by fixed token size with overlap

        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk (default: from .env CHUNK_SIZE or 1200)
            overlap: Number of overlapping tokens between chunks (default: from .env CHUNK_OVERLAP_SIZE or 100)
            split_sentences: Try to split on sentence boundaries

        Returns:
            List[Dict[str, Any]]: Formatted chunks matching LightRAG's TextChunkSchema
        """
        # Use instance defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        chunks = []

        if split_sentences:
            # Split into sentences first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)

                # If adding this sentence exceeds chunk size, save current chunk
                if current_tokens + sentence_tokens > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))

                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > 1:
                        # Keep last few sentences for overlap
                        overlap_text = " ".join(current_chunk[-2:])
                        overlap_tokens = self.count_tokens(overlap_text)

                        if overlap_tokens <= overlap:
                            current_chunk = current_chunk[-2:]
                            current_tokens = overlap_tokens
                        else:
                            current_chunk = []
                            current_tokens = 0
                    else:
                        current_chunk = []
                        current_tokens = 0

                current_chunk.append(sentence)
                current_tokens += sentence_tokens

            # Add remaining chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        else:
            # Simple word-based chunking
            words = text.split()
            current_chunk = []

            for word in words:
                current_chunk.append(word)
                chunk_text = " ".join(current_chunk)

                if self.count_tokens(chunk_text) >= chunk_size:
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    if overlap > 0:
                        overlap_words = int(overlap / 1.3)  # Rough word count
                        current_chunk = current_chunk[-overlap_words:]
                    else:
                        current_chunk = []

            # Add remaining
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        # Format and return
        return self._format_chunks(chunks)

    def paragraph_chunking(self, text: str, min_tokens: int = None, max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Chunk text by paragraphs, combining small paragraphs and splitting large ones

        Args:
            text: Text to chunk
            min_tokens: Minimum tokens per chunk (default: CHUNK_SIZE / 2)
            max_tokens: Maximum tokens per chunk (default: CHUNK_SIZE * 1.5)

        Returns:
            List[Dict[str, Any]]: Formatted chunks matching LightRAG's TextChunkSchema
        """
        # Use instance defaults if not specified
        min_tokens = min_tokens or (self.default_chunk_size // 2)
        max_tokens = max_tokens or int(self.default_chunk_size * 1.5)

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If paragraph is too large, split it
            if para_tokens > max_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph
                sub_chunks = self.fixed_token_chunking(para, chunk_size=max_tokens, overlap=100)
                chunks.extend(sub_chunks)

            # If adding paragraph exceeds max, save current chunk
            elif current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

            # Add paragraph to current chunk
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Format and return
        return self._format_chunks(chunks)

    def semantic_chunking(self, text: str, model: Any = None) -> List[Dict[str, Any]]:
        """
        Chunk by semantic similarity (placeholder for future implementation)

        This would use sentence embeddings to group semantically similar
        sentences together.

        Args:
            text: Text to chunk
            model: Sentence transformer model (future)

        Returns:
            List of text chunks

        Note:
            Currently falls back to paragraph chunking.
            Future: Implement with sentence-transformers library.
        """
        # TODO: Implement semantic chunking with sentence embeddings
        # Would require sentence-transformers library
        # For now, use paragraph chunking as approximation
        return self.paragraph_chunking(text)

    def contextual_chunking(
        self, text: str, context_sentences: int = 2, chunk_size: int = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk with surrounding context (Anthropic-style contextual retrieval)

        Note: Currently simplified implementation. Returns standard format with
        context information as extra field.

        Args:
            text: Text to chunk
            context_sentences: Number of sentences to include as context (reserved for future use)
            chunk_size: Target chunk size (default: from .env CHUNK_SIZE or 1200)

        Returns:
            List[Dict[str, Any]]: Formatted chunks matching LightRAG's TextChunkSchema
        """
        # Use instance defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size

        # Get base chunks (already in correct format)
        base_chunks = self.fixed_token_chunking(text, chunk_size=chunk_size, overlap=0)

        # Note: context_sentences parameter reserved for future enhancement
        # Would add surrounding sentence context here to improve retrieval
        _ = context_sentences  # Mark as intentionally unused for now

        return base_chunks

    def hybrid_chunking(
        self, text: str, chunk_size: int = None, strategy: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Hybrid chunking combining multiple strategies

        Args:
            text: Text to chunk
            chunk_size: Target chunk size (default: from .env CHUNK_SIZE or 1200)
            strategy: One of "balanced", "semantic-first", "paragraph-first"

        Returns:
            List of text chunks
        """
        # Use instance defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size

        if strategy == "paragraph-first":
            # Try paragraph chunking first, fall back to fixed if chunks too large
            chunks = self.paragraph_chunking(text, max_tokens=chunk_size)
        elif strategy == "semantic-first":
            # Would use semantic chunking if implemented
            chunks = self.semantic_chunking(text)
        else:  # balanced
            # Use paragraph chunking with reasonable defaults
            chunks = self.paragraph_chunking(
                text, min_tokens=chunk_size // 2, max_tokens=chunk_size * 1.5
            )

        return chunks

    def hierarchical_chunking(
        self,
        text: str,
        parent_chunk_size: int = None,
        child_chunk_size: int = None,
        extract_sections: bool = True,
        child_overlap: int = None,
        use_intermediate_level: bool = False,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Hierarchical chunking with parent-child relationships

        Creates multi-level document structure for improved retrieval:
        - Parent chunks: Large contextual units (1500-2048 tokens)
        - Child chunks: Precise retrieval units (128-300 tokens)
        - Optional intermediate level (512-1024 tokens)

        Based on research showing 10-15% improvement in retrieval accuracy
        for well-structured documents.

        Args:
            text: Text to chunk
            parent_chunk_size: Parent chunk size in tokens (default: from .env PARENT_CHUNK_SIZE or 1500)
            child_chunk_size: Child chunk size in tokens (default: from .env CHUNK_SIZE or 300)
            extract_sections: Whether to detect document sections (recommended for structured docs)
            child_overlap: Child chunk overlap in tokens (default: from .env CHUNK_OVERLAP_SIZE or 60)
            use_intermediate_level: Enable 3-level hierarchy (default: False for 2-level)
            return_metadata: Return document statistics along with chunks

        Returns:
            List[Dict[str, Any]]: Formatted chunks with hierarchical metadata
            OR
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: Chunks + statistics if return_metadata=True

            Each chunk extends LightRAG's TextChunkSchema with:
            {
                "tokens": int,
                "content": str,
                "chunk_order_index": int,
                "parent_id": str | None,        # ID of parent chunk
                "children_ids": List[str],      # IDs of child chunks (for parents)
                "level": int,                   # Hierarchy level (0=parent, 1=intermediate, 2=child)
                "section_title": str | None     # Extracted section heading if available
            }

        Note:
            Works best with well-structured documents (PDFs with headings, technical manuals,
            legal documents, academic papers). Not recommended for unstructured narrative text.

            For retrieval, typically only child chunks are embedded in vector DB.
            Parent chunks provide context during auto-merging retrieval.
        """
        # Initialize hierarchical chunker with parameters
        chunker = HierarchicalChunker(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
            encoding_name=self.encoding_name,
            use_intermediate_level=use_intermediate_level
        )

        # Chunk document with hierarchical structure
        chunks, stats = chunker.chunk_document(text, extract_sections=extract_sections)

        # Return based on metadata flag
        if return_metadata:
            return chunks, stats
        else:
            return chunks

    def apply_strategy(
        self,
        text: str,
        strategy: str = "fixed_token",
        chunk_size: int = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Apply a chunking strategy to text

        Args:
            text: Text to chunk
            strategy: Strategy name:
                - "fixed_token": Fixed token size with overlap
                - "paragraph": Paragraph-based chunking
                - "semantic": Semantic similarity chunking (future)
                - "contextual": Contextual chunking with surrounding text
                - "hybrid": Hybrid approach
                - "hierarchical": Hierarchical parent-child chunking (10-15% better for structured docs)
            chunk_size: Target chunk size in tokens (default: from .env CHUNK_SIZE or 1200)
            **kwargs: Additional strategy-specific arguments

        Returns:
            List[Dict[str, Any]]: Formatted chunks matching LightRAG's TextChunkSchema
            [
                {
                    "tokens": int,              # Token count
                    "content": str,             # Chunk text (stripped)
                    "chunk_order_index": int,   # Sequential index
                },
                ...
            ]

            For hierarchical strategy, chunks include additional metadata:
            {
                "parent_id": str | None,        # ID of parent chunk
                "children_ids": List[str],      # IDs of child chunks (for parents)
                "level": int,                   # Hierarchy level (0=parent, 1/2=child)
                "section_title": str | None     # Extracted section heading
            }
        """
        # Use instance defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size

        if strategy == "fixed_token":
            return self.fixed_token_chunking(
                text,
                chunk_size=chunk_size,
                overlap=kwargs.get("overlap", 100),
                split_sentences=kwargs.get("split_sentences", True),
            )

        elif strategy == "paragraph":
            return self.paragraph_chunking(
                text,
                min_tokens=kwargs.get("min_tokens", chunk_size // 2),
                max_tokens=kwargs.get("max_tokens", chunk_size * 1.5),
            )

        elif strategy == "semantic":
            return self.semantic_chunking(text, model=kwargs.get("model"))

        elif strategy == "contextual":
            return self.contextual_chunking(
                text,
                context_sentences=kwargs.get("context_sentences", 2),
                chunk_size=chunk_size,
            )

        elif strategy == "hybrid":
            return self.hybrid_chunking(
                text,
                chunk_size=chunk_size,
                strategy=kwargs.get("hybrid_strategy", "balanced"),
            )

        elif strategy == "hierarchical":
            return self.hierarchical_chunking(
                text,
                parent_chunk_size=kwargs.get("parent_chunk_size"),
                child_chunk_size=kwargs.get("child_chunk_size", chunk_size),
                extract_sections=kwargs.get("extract_sections", True),
                child_overlap=kwargs.get("child_overlap"),
                use_intermediate_level=kwargs.get("use_intermediate_level", False),
                return_metadata=kwargs.get("return_metadata", False),
            )

        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available chunking strategies

        Returns:
            Dictionary mapping strategy names to their descriptions and parameters
        """
        return {
            "fixed_token": {
                "name": "Fixed Token Size",
                "description": "Chunk by fixed token count with overlap",
                "status": "available",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1200, "min": 100, "max": 4000},
                    "overlap": {"type": "int", "default": 100, "min": 0, "max": 500},
                    "split_sentences": {"type": "bool", "default": True},
                },
            },
            "paragraph": {
                "name": "Paragraph-Based",
                "description": "Chunk by paragraphs, combining or splitting as needed",
                "status": "available",
                "parameters": {
                    "min_tokens": {"type": "int", "default": 500, "min": 100, "max": 2000},
                    "max_tokens": {"type": "int", "default": 1500, "min": 500, "max": 3000},
                },
            },
            "semantic": {
                "name": "Semantic Similarity",
                "description": "Chunk by semantic similarity (coming soon)",
                "status": "future",
                "parameters": {},
            },
            "contextual": {
                "name": "Contextual Chunking",
                "description": "Add surrounding context to each chunk (experimental)",
                "status": "experimental",
                "parameters": {
                    "context_sentences": {"type": "int", "default": 2, "min": 1, "max": 5},
                    "chunk_size": {"type": "int", "default": 1200, "min": 100, "max": 4000},
                },
            },
            "hybrid": {
                "name": "Hybrid Strategy",
                "description": "Combine multiple chunking approaches",
                "status": "available",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1200, "min": 100, "max": 4000},
                    "hybrid_strategy": {
                        "type": "select",
                        "default": "balanced",
                        "options": ["balanced", "paragraph-first", "semantic-first"],
                    },
                },
            },
            "hierarchical": {
                "name": "Hierarchical Chunking",
                "description": "Multi-level parent-child chunking (10-15% better for structured docs)",
                "status": "available",
                "parameters": {
                    "parent_chunk_size": {"type": "int", "default": 1500, "min": 500, "max": 4000},
                    "child_chunk_size": {"type": "int", "default": 300, "min": 100, "max": 1000},
                    "child_overlap": {"type": "int", "default": 60, "min": 0, "max": 200},
                    "extract_sections": {"type": "bool", "default": True},
                    "use_intermediate_level": {"type": "bool", "default": False},
                },
                "best_for": [
                    "PDFs with headings",
                    "Technical manuals",
                    "Legal documents",
                    "Academic papers",
                    "Building regulations"
                ],
                "not_recommended_for": [
                    "Unstructured narrative text",
                    "Short documents",
                    "Flat content without hierarchy"
                ]
            },
        }
