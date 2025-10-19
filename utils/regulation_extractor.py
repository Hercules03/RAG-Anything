"""
Regulation ID Extractor - Extract regulation identifiers from PDFs

Extracts regulation IDs following patterns like:
- APP-1, APP-2, APP-3 (Appendix)
- BC-001, BC-002 (Building Code)
- FS-2024-01 (Fire Safety)
- Custom patterns

Uses multiple strategies:
1. Text pattern matching in first few pages
2. Filename parsing
3. Metadata extraction
"""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class RegulationExtractor:
    """Extract regulation IDs from PDF documents"""

    # Common regulation ID patterns
    DEFAULT_PATTERNS = [
        r'APP-\d+[A-Z]?',  # APP-1, APP-2A
        r'BC-\d{3,4}[A-Z]?',  # BC-001, BC-1234
        r'FS-\d{4}-\d+',  # FS-2024-01
        r'[A-Z]{2,4}-\d+[A-Z]?',  # Generic: XX-123, XXXX-123A
        r'[A-Z]+ \d+',  # "APPENDIX 1", "ANNEX 2"
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        Initialize regulation extractor

        Args:
            custom_patterns: Additional regex patterns to try
        """
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def extract_from_pdf(
        self,
        pdf_path: str,
        pages_to_check: int = 3,
        fallback_to_filename: bool = True
    ) -> Optional[str]:
        """
        Extract regulation ID from PDF

        Args:
            pdf_path: Path to PDF file
            pages_to_check: Number of pages to scan (default: 3)
            fallback_to_filename: Use filename if no pattern found

        Returns:
            Regulation ID string or None
        """
        regulation_id = None

        # Strategy 1: Extract from PDF text content
        if PdfReader:
            regulation_id = self._extract_from_pdf_content(pdf_path, pages_to_check)

        # Strategy 2: Extract from filename
        if not regulation_id and fallback_to_filename:
            regulation_id = self._extract_from_filename(pdf_path)

        return regulation_id

    def _extract_from_pdf_content(
        self, pdf_path: str, pages_to_check: int
    ) -> Optional[str]:
        """
        Extract regulation ID from PDF text content

        Args:
            pdf_path: Path to PDF file
            pages_to_check: Number of pages to scan

        Returns:
            Regulation ID or None
        """
        if not PdfReader:
            return None

        try:
            reader = PdfReader(pdf_path)
            text = ""

            # Extract text from first N pages
            for i, page in enumerate(reader.pages[:pages_to_check]):
                if i >= pages_to_check:
                    break
                text += page.extract_text() + "\n"

            # Try each pattern
            for pattern in self.patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Return first match, cleaned up
                    regulation_id = matches[0].strip().upper()

                    # Clean up patterns like "APPENDIX 1" to "APP-1"
                    regulation_id = self._normalize_regulation_id(regulation_id)
                    return regulation_id

        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")

        return None

    def _extract_from_filename(self, pdf_path: str) -> Optional[str]:
        """
        Extract regulation ID from filename

        Args:
            pdf_path: Path to PDF file

        Returns:
            Regulation ID or None
        """
        filename = Path(pdf_path).stem  # Get filename without extension

        # Try patterns on filename
        for pattern in self.patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            if matches:
                regulation_id = matches[0].strip().upper()
                return self._normalize_regulation_id(regulation_id)

        # If no pattern matched, use filename as-is (cleaned)
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'(regulation|code|standard|appendix|annex)[-_\s]*', '', filename, flags=re.IGNORECASE)
        cleaned = re.sub(r'[-_\s]+', '-', cleaned).strip('-')

        if cleaned:
            return cleaned.upper()

        return None

    def _normalize_regulation_id(self, regulation_id: str) -> str:
        """
        Normalize regulation ID to standard format

        Args:
            regulation_id: Raw regulation ID

        Returns:
            Normalized regulation ID
        """
        # Convert "APPENDIX 1" to "APP-1"
        regulation_id = re.sub(r'APPENDIX\s+(\d+)', r'APP-\1', regulation_id, flags=re.IGNORECASE)

        # Convert "ANNEX 1" to "ANX-1"
        regulation_id = re.sub(r'ANNEX\s+(\d+)', r'ANX-\1', regulation_id, flags=re.IGNORECASE)

        # Convert spaces to hyphens
        regulation_id = re.sub(r'\s+', '-', regulation_id)

        # Ensure uppercase
        regulation_id = regulation_id.upper()

        return regulation_id

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract full metadata from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata:
            {
                "regulation_id": str,
                "filename": str,
                "page_count": int,
                "file_size": int,
                "title": str or None,
                "author": str or None,
                "creation_date": str or None
            }
        """
        metadata = {
            "regulation_id": None,
            "filename": Path(pdf_path).name,
            "page_count": 0,
            "file_size": 0,
            "title": None,
            "author": None,
            "creation_date": None,
        }

        # Get file size
        try:
            metadata["file_size"] = Path(pdf_path).stat().st_size
        except Exception:
            pass

        # Get PDF metadata
        if PdfReader:
            try:
                reader = PdfReader(pdf_path)
                metadata["page_count"] = len(reader.pages)

                # Get PDF metadata
                pdf_metadata = reader.metadata
                if pdf_metadata:
                    metadata["title"] = pdf_metadata.get("/Title")
                    metadata["author"] = pdf_metadata.get("/Author")
                    metadata["creation_date"] = pdf_metadata.get("/CreationDate")

            except Exception as e:
                print(f"Error reading PDF metadata: {e}")

        # Extract regulation ID
        metadata["regulation_id"] = self.extract_from_pdf(pdf_path)

        return metadata

    def batch_extract(
        self, pdf_paths: List[str], show_progress: bool = False
    ) -> Dict[str, str]:
        """
        Extract regulation IDs from multiple PDFs

        Args:
            pdf_paths: List of PDF file paths
            show_progress: Show progress bar (requires tqdm)

        Returns:
            Dictionary mapping file paths to regulation IDs
        """
        results = {}

        pdf_list = pdf_paths
        if show_progress:
            try:
                from tqdm import tqdm
                pdf_list = tqdm(pdf_paths, desc="Extracting regulation IDs")
            except ImportError:
                pass

        for pdf_path in pdf_list:
            regulation_id = self.extract_from_pdf(pdf_path)
            results[pdf_path] = regulation_id or Path(pdf_path).stem

        return results

    def validate_regulation_id(self, regulation_id: str) -> bool:
        """
        Validate if a string matches regulation ID pattern

        Args:
            regulation_id: String to validate

        Returns:
            True if matches any pattern, False otherwise
        """
        for pattern in self.patterns:
            if re.fullmatch(pattern, regulation_id, re.IGNORECASE):
                return True
        return False

    def suggest_regulation_id(
        self, pdf_path: str, candidates: List[str]
    ) -> Optional[str]:
        """
        Suggest best regulation ID from candidates

        Args:
            pdf_path: Path to PDF file
            candidates: List of candidate regulation IDs

        Returns:
            Best matching regulation ID or None
        """
        # Extract from PDF first
        extracted = self.extract_from_pdf(pdf_path)
        if extracted:
            return extracted

        # If extraction failed, look for best match in candidates
        filename = Path(pdf_path).stem.upper()

        for candidate in candidates:
            if candidate.upper() in filename:
                return candidate

        # Return first candidate if available
        return candidates[0] if candidates else None
