"""
Document Loader Module for Multimodal Chatbot (RAG System)
===========================================================

This module prepares raw files into vector-friendly chunks for Retrieval-Augmented Generation (RAG).

Features:
- Supports PDF, TXT, Markdown, and DOCX
- Extracts and cleans text
- Splits into overlapping chunks
- Attaches metadata (file type, size, page count)
- Handles batch and directory processing
- LangChain 1.0.3 compatible
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod

# ============================================================================
# LANGCHAIN 1.0.3 IMPORTS (COMPATIBLE)
# ============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Optional imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: Dict[str, str] = {
    ".pdf": "PDF Document",
    ".txt": "Text File",
    ".md": "Markdown File",
    ".docx": "Word Document",
}

MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB

# ============================================================================
# BASE FILE LOADER
# ============================================================================

class BaseFileLoader(ABC):
    """Abstract base class for file loaders."""

    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """Check if loader supports this file format."""
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load and extract text from a file."""
        raise NotImplementedError

    @abstractmethod
    def load_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load text and return with metadata."""
        raise NotImplementedError


# ============================================================================
# PDF LOADER
# ============================================================================

class PDFLoader(BaseFileLoader):
    """Extracts text from PDF documents using PyPDF2."""

    def __init__(self):
        if PyPDF2 is None:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")

    def supports_format(self, file_path: str) -> bool:
        """Check if file is PDF."""
        return file_path.lower().endswith(".pdf")

    def load(self, file_path: str) -> str:
        """Extract text from PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        size = os.path.getsize(file_path)
        if size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {size / 1024 / 1024:.2f} MB (max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        logger.info(f"Loading PDF: {os.path.basename(file_path)} ({size / 1024:.2f} KB)")
        text_pages = []

        try:
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                total_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_pages.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i}/{total_pages}: {str(e)}")
                        continue

            if not text_pages:
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""

            combined_text = "\n\n".join(text_pages).strip()
            logger.info(
                f"✅ PDF extracted: {len(combined_text)} chars from {len(pdf.pages)} pages"
            )
            return combined_text

        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {str(e)}")
            raise

    def load_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load PDF with metadata."""
        text = self.load(file_path)
        
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            page_count = len(pdf.pages)

        metadata = {
            "source": os.path.basename(file_path),
            "file_type": "PDF",
            "file_size_bytes": os.path.getsize(file_path),
            "pages": page_count,
            "characters": len(text),
        }
        return text, metadata


# ============================================================================
# TEXT / MARKDOWN LOADER
# ============================================================================

class TextLoader(BaseFileLoader):
    """Loads text (.txt) or markdown (.md) files."""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        self.fallback_encodings = ["latin-1", "iso-8859-1", "cp1252"]

    def supports_format(self, file_path: str) -> bool:
        """Check if file is .txt or .md."""
        ext = file_path.lower()
        return ext.endswith((".txt", ".md"))

    def load(self, file_path: str) -> str:
        """Load text from file with fallback encoding."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        size = os.path.getsize(file_path)
        if size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {size / 1024 / 1024:.2f} MB (max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        file_type = Path(file_path).suffix.upper()
        logger.info(f"Loading {file_type}: {os.path.basename(file_path)} ({size / 1024:.2f} KB)")

        # Try primary encoding
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                text = f.read()
            logger.info(f"✅ Loaded with {self.encoding} encoding")
            return text.strip()

        except UnicodeDecodeError as e:
            logger.warning(
                f"Failed to decode with {self.encoding}, trying fallback encodings"
            )

            # Try fallback encodings
            for encoding in self.fallback_encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"✅ Loaded with {encoding} encoding")
                    return text.strip()
                except UnicodeDecodeError:
                    continue

            # If all fails, raise original error
            raise ValueError(
                f"Could not decode {file_path} with any supported encoding. "
                f"Tried: {[self.encoding] + self.fallback_encodings}"
            )

    def load_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load text file with metadata."""
        text = self.load(file_path)
        
        metadata = {
            "source": os.path.basename(file_path),
            "file_type": "Text" if file_path.endswith(".txt") else "Markdown",
            "file_size_bytes": os.path.getsize(file_path),
            "lines": len(text.splitlines()),
            "characters": len(text),
        }
        return text, metadata


# ============================================================================
# DOCX LOADER
# ============================================================================

class DocxLoader(BaseFileLoader):
    """Extracts text from Word (.docx) documents."""

    def __init__(self):
        if DocxDocument is None:
            raise ImportError(
                "python-docx not installed. Install with: pip install python-docx"
            )

    def supports_format(self, file_path: str) -> bool:
        """Check if file is .docx."""
        return file_path.lower().endswith(".docx")

    def load(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        size = os.path.getsize(file_path)
        if size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {size / 1024 / 1024:.2f} MB (max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        logger.info(f"Loading DOCX: {os.path.basename(file_path)} ({size / 1024:.2f} KB)")

        try:
            doc = DocxDocument(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            if not paragraphs:
                logger.warning(f"No text found in DOCX: {file_path}")
                return ""

            text = "\n".join(paragraphs).strip()
            logger.info(f"✅ DOCX extracted: {len(text)} chars from {len(paragraphs)} paragraphs")
            return text

        except Exception as e:
            logger.error(f"Failed to read DOCX {file_path}: {str(e)}")
            raise

    def load_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load DOCX with metadata."""
        text = self.load(file_path)
        doc = DocxDocument(file_path)

        metadata = {
            "source": os.path.basename(file_path),
            "file_type": "DOCX",
            "file_size_bytes": os.path.getsize(file_path),
            "paragraphs": len(doc.paragraphs),
            "characters": len(text),
        }
        return text, metadata


# ============================================================================
# MAIN DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    """
    Main entry point for loading and chunking multiple file formats.
    
    Supports: PDF, TXT, Markdown (.md), and DOCX
    
    Example:
        loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
        documents = loader.load_file("document.pdf", with_metadata=True)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        encoding: str = "utf-8",
    ):
        """Initialize DocumentLoader with chunking configuration."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = encoding

        # Initialize loaders (with lazy imports)
        self.loaders = {}
        try:
            self.loaders[".pdf"] = PDFLoader()
        except ImportError:
            logger.warning("PyPDF2 not installed, PDF loading disabled")

        self.loaders[".txt"] = TextLoader(encoding)
        self.loaders[".md"] = TextLoader(encoding)

        try:
            self.loaders[".docx"] = DocxLoader()
        except ImportError:
            logger.warning("python-docx not installed, DOCX loading disabled")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"DocumentLoader initialized: chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    def load_file(
        self, 
        file_path: str, 
        with_metadata: bool = False,
    ) -> List[Any]:
        """
        Load and chunk a single file.
        
        Args:
            file_path: Path to file
            with_metadata: Return Document objects with metadata
            
        Returns:
            List of chunks (str) or Document objects
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in self.loaders:
            raise ValueError(
                f"Unsupported format: {file_ext}. Supported: {list(self.loaders.keys())}"
            )

        loader = self.loaders[file_ext]
        text = loader.load(file_path)
        text = self._clean_text(text)

        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return []

        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} chunks")

        if with_metadata:
            _, metadata = loader.load_with_metadata(file_path)
            return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        else:
            return chunks

    def load_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        with_metadata: bool = False,
    ) -> List[Any]:
        """
        Load and chunk all files in a directory.
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            with_metadata: Include metadata
            
        Returns:
            List of chunks or Document objects
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Not a directory: {directory}")

        all_chunks: List[Any] = []
        path_iter = (
            Path(directory).rglob("*") if recursive else Path(directory).glob("*")
        )

        file_count = 0
        for file_path in path_iter:
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                try:
                    chunks = self.load_file(str(file_path), with_metadata=with_metadata)
                    all_chunks.extend(chunks)
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue

        logger.info(
            f"Loaded {file_count} files → {len(all_chunks)} chunks from {directory}"
        )
        return all_chunks

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean unwanted whitespace and special characters."""
        if not text:
            return ""

        # Remove null bytes and replacement characters
        text = text.replace("\x00", "").replace("\ufffd", "")

        # Normalize whitespace
        text = " ".join(text.split())

        return text.strip()

    @staticmethod
    def get_file_stats(file_path: str) -> Dict[str, Any]:
        """Get file statistics."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        size_bytes = os.path.getsize(file_path)
        size_mb = round(size_bytes / (1024 * 1024), 2)

        return {
            "file_name": os.path.basename(file_path),
            "file_type": Path(file_path).suffix.upper(),
            "file_size_bytes": size_bytes,
            "file_size_mb": size_mb,
            "modified_timestamp": os.path.getmtime(file_path),
            "is_supported": Path(file_path).suffix.lower() in SUPPORTED_FORMATS,
        }

    @staticmethod
    def validate_file(file_path: str) -> Tuple[bool, str]:
        """
        Validate if file can be loaded.
        
        Returns:
            (is_valid, message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return False, f"Not a file: {file_path}"

        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            return False, f"File too large (max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"

        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}"

        return True, "✅ File is valid"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported."""
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


def get_supported_formats() -> Dict[str, str]:
    """Get dictionary of supported formats."""
    return SUPPORTED_FORMATS.copy()


def estimate_chunks(
    file_size_mb: float, 
    chunk_size: int = 500,
) -> int:
    """Estimate number of chunks from file size."""
    if file_size_mb <= 0:
        return 0
    return max(1, int((file_size_mb * 1_000_000) / chunk_size))


# ============================================================================
# END OF MODULE
# ============================================================================
