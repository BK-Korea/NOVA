"""Document parser for IR materials (HTML, PPT, PDF)."""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import asyncio
from io import BytesIO

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.documents.elements import Element
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store."""
        return {
            "content": self.content,
            "metadata": self.metadata
        }


class DocumentParser:
    """
    Parse IR materials into searchable chunks.

    Supports:
    - HTML pages (news, IR pages)
    - PDF documents (presentations, reports)
    - PowerPoint presentations (PPT/PPTX)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Parse a file into chunks.

        Args:
            file_path: Path to the file
            material_metadata: Metadata about the source material

        Returns:
            List of DocumentChunk
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        # Determine file type
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".html" or suffix == ".htm":
                return self._parse_html(file_path, material_metadata)
            elif suffix == ".pdf":
                return self._parse_pdf(file_path, material_metadata)
            elif suffix in [".ppt", ".pptx"]:
                return self._parse_pptx(file_path, material_metadata)
            elif suffix in [".doc", ".docx"]:
                return self._parse_docx(file_path, material_metadata)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return []

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _parse_html(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse HTML file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        # Use BeautifulSoup to extract main content
        soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get main content
        main_content = soup.get_text(separator="\n", strip=True)

        # Use unstructured for more detailed parsing
        try:
            elements = partition_html(text=html_content)
            chunks = self._create_chunks_from_elements(
                elements,
                material_metadata,
                file_path
            )
            if chunks:
                return chunks
        except Exception as e:
            logger.debug(f"unstructured HTML parsing failed: {e}, falling back to simple parsing")

        # Fallback: simple chunking
        return self._simple_chunk(
            main_content,
            material_metadata,
            file_path
        )

    def _parse_pdf(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse PDF file."""
        # Try different strategies in order of preference
        strategies = ["fast", "auto", "hi_res"]
        
        for strategy in strategies:
            try:
                elements = partition_pdf(
                    filename=str(file_path),
                    strategy=strategy,
                    extract_images_in_pdf=False,
                )
                if elements:
                    return self._create_chunks_from_elements(
                        elements,
                        material_metadata,
                        file_path
                    )
            except Exception as e:
                logger.debug(f"PDF parsing with strategy '{strategy}' failed: {e}")
                continue
        
        # Fallback: try PyPDF2 if available
        try:
            import PyPDF2
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            if text.strip():
                return self._simple_chunk(text, material_metadata, file_path)
        except ImportError:
            logger.debug("PyPDF2 not available for fallback")
        except Exception as e:
            logger.debug(f"PyPDF2 fallback failed: {e}")
        
        logger.error(f"Failed to parse PDF {file_path}")
        return []

    def _parse_pptx(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse PowerPoint file."""
        try:
            elements = partition_pptx(filename=str(file_path))
            return self._create_chunks_from_elements(
                elements,
                material_metadata,
                file_path
            )
        except Exception as e:
            logger.error(f"Failed to parse PPTX {file_path}: {e}")
            return []

    def _parse_docx(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse Word document (requires unstructured)."""
        try:
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(filename=str(file_path))
            return self._create_chunks_from_elements(
                elements,
                material_metadata,
                file_path
            )
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}")
            return []

    def _create_chunks_from_elements(
        self,
        elements: List[Element],
        material_metadata: Dict[str, Any],
        file_path: Path
    ) -> List[DocumentChunk]:
        """Create chunks from unstructured elements."""
        chunks = []
        current_chunk = ""
        current_metadata = material_metadata.copy()

        for i, element in enumerate(elements):
            text = str(element).strip()
            if not text:
                continue

            # Add element type to metadata
            element_type = element.__class__.__name__
            current_metadata["element_type"] = element_type
            current_metadata["element_id"] = i

            # Get page number if available (handle both dict and object metadata)
            if hasattr(element, "metadata"):
                metadata = element.metadata
                # Try to get page_number - works with both dict and ElementMetadata object
                page_num = None
                if hasattr(metadata, "page_number"):
                    page_num = metadata.page_number
                elif isinstance(metadata, dict):
                    page_num = metadata.get("page_number")
                if page_num:
                    current_metadata["page_number"] = page_num

            # Simple chunking by size
            if len(current_chunk) + len(text) > self.chunk_size:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=current_metadata.copy()
                    ))
                current_chunk = text
            else:
                current_chunk += "\n" + text if current_chunk else text

        # Add remaining chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=current_metadata.copy()
            ))

        logger.debug(f"Created {len(chunks)} chunks from {file_path}")
        return chunks

    def _simple_chunk(
        self,
        text: str,
        material_metadata: Dict[str, Any],
        file_path: Path
    ) -> List[DocumentChunk]:
        """Simple text chunking fallback."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata={
                    **material_metadata,
                    "chunk_index": len(chunks),
                    "file_path": str(file_path)
                }
            ))

            start = end - self.chunk_overlap

        return chunks

    async def parse_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path] = None
    ) -> List[DocumentChunk]:
        """
        Parse content directly from URL.

        Args:
            url: URL to parse
            material_metadata: Metadata about the source
            save_dir: Optional directory to save downloaded content

        Returns:
            List of DocumentChunk
        """
        import httpx

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            async with httpx.AsyncClient(timeout=30, headers=headers) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "").lower()

                if "html" in content_type or "text" in content_type:
                    # Try unstructured first
                    try:
                        elements = partition_html(text=response.text)
                        chunks = self._create_chunks_from_elements(
                            elements,
                            material_metadata,
                            Path(url)
                        )
                        if chunks:
                            return chunks
                    except Exception as e:
                        logger.debug(f"partition_html failed: {e}, using fallback")
                    
                    # Fallback: BeautifulSoup simple parsing
                    soup = BeautifulSoup(response.text, "lxml")
                    
                    # Remove unwanted elements
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    
                    # Get main content
                    text = soup.get_text(separator="\n", strip=True)
                    
                    # Filter out very short or empty content
                    if len(text) > 100:
                        return self._simple_chunk(text, material_metadata, Path(url))
                    else:
                        logger.debug(f"Content too short from {url}")
                        return []
                        
                elif "pdf" in content_type:
                    # Save PDF temporarily
                    if save_dir:
                        import hashlib
                        save_dir.mkdir(parents=True, exist_ok=True)
                        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".pdf"
                        temp_path = save_dir / filename

                        with open(temp_path, "wb") as f:
                            f.write(response.content)

                        chunks = self._parse_pdf(temp_path, material_metadata)
                        return chunks
                    else:
                        logger.warning("save_dir not provided for PDF download")
                        return []
                else:
                    logger.debug(f"Unsupported content type: {content_type} for {url}")
                    return []

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return []
