"""
Late Chunking Strategies - Custom chunking methods for advanced RAG

Provides various chunking strategies that can be applied before inserting into RAG:
- Fixed token size chunking
- Semantic chunking (future)
- Contextual chunking (future)
- Hybrid strategies (future)
"""

import re
from typing import List, Dict, Any, Optional, Callable
import tiktoken


class LateChunkingStrategies:
    """Custom chunking strategies for late chunking workflow"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize late chunking strategies

        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
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

    def fixed_token_chunking(
        self,
        text: str,
        chunk_size: int = 1200,
        overlap: int = 100,
        split_sentences: bool = True,
    ) -> List[str]:
        """
        Chunk text by fixed token size with overlap

        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk
            overlap: Number of overlapping tokens between chunks
            split_sentences: Try to split on sentence boundaries

        Returns:
            List of text chunks
        """
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

        return chunks

    def paragraph_chunking(self, text: str, min_tokens: int = 500, max_tokens: int = 1500) -> List[str]:
        """
        Chunk text by paragraphs, combining small paragraphs and splitting large ones

        Args:
            text: Text to chunk
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
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

        return chunks

    def semantic_chunking(self, text: str, model=None) -> List[str]:
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
        self, text: str, context_sentences: int = 2, chunk_size: int = 1200
    ) -> List[Dict[str, str]]:
        """
        Chunk with surrounding context (Anthropic-style contextual retrieval)

        Each chunk includes:
        - Main content
        - Context from surrounding text
        - Full text suitable for late chunking

        Args:
            text: Text to chunk
            context_sentences: Number of sentences to include as context
            chunk_size: Target chunk size

        Returns:
            List of dictionaries with structure:
            {
                "content": str,  # Main chunk content
                "context": str,  # Surrounding context
                "full_text": str  # Context + content (for insertion)
            }
        """
        # First, get base chunks
        base_chunks = self.fixed_token_chunking(text, chunk_size=chunk_size, overlap=0)

        # Split original text into sentences for context
        sentences = re.split(r'(?<=[.!?])\s+', text)

        contextual_chunks = []

        for i, chunk in enumerate(base_chunks):
            # Find which sentences belong to this chunk
            chunk_start_idx = 0
            chunk_end_idx = 0

            # Simple heuristic: find chunk position in original text
            chunk_pos = text.find(chunk[:50])  # Match first 50 chars

            # Find surrounding sentences
            context_before = []
            context_after = []

            # This is a simplified version - more sophisticated matching would be better
            contextual_chunks.append({
                "content": chunk,
                "context": f"Chunk {i+1} of {len(base_chunks)}",
                "full_text": chunk,  # For now, just use chunk as-is
            })

        return contextual_chunks

    def hybrid_chunking(
        self, text: str, chunk_size: int = 1200, strategy: str = "balanced"
    ) -> List[str]:
        """
        Hybrid chunking combining multiple strategies

        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            strategy: One of "balanced", "semantic-first", "paragraph-first"

        Returns:
            List of text chunks
        """
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

    def apply_strategy(
        self,
        text: str,
        strategy: str = "fixed_token",
        chunk_size: int = 1200,
        **kwargs
    ) -> List[str]:
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
            chunk_size: Target chunk size in tokens
            **kwargs: Additional strategy-specific arguments

        Returns:
            List of text chunks (or list of dicts for contextual)
        """
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
            result = self.contextual_chunking(
                text,
                context_sentences=kwargs.get("context_sentences", 2),
                chunk_size=chunk_size,
            )
            # Return just the full_text for insertion
            return [item["full_text"] for item in result]

        elif strategy == "hybrid":
            return self.hybrid_chunking(
                text,
                chunk_size=chunk_size,
                strategy=kwargs.get("hybrid_strategy", "balanced"),
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
        }
