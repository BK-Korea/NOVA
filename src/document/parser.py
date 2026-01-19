"""Document parser for IR materials (HTML, PPT, PDF, Excel, etc.)."""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import asyncio
from io import BytesIO
from urllib.parse import urlparse

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
            elif suffix in [".xls", ".xlsx"]:
                return self._parse_xlsx(file_path, material_metadata)
            elif suffix == ".txt":
                return self._parse_txt(file_path, material_metadata)
            elif suffix == ".csv":
                return self._parse_csv(file_path, material_metadata)
            elif suffix == ".rtf":
                return self._parse_rtf(file_path, material_metadata)
            elif suffix == ".zip":
                return self._parse_zip(file_path, material_metadata)
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

    def _parse_xlsx(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse Excel file."""
        try:
            # Try unstructured first
            try:
                from unstructured.partition.xlsx import partition_xlsx
                elements = partition_xlsx(filename=str(file_path))
                return self._create_chunks_from_elements(
                    elements,
                    material_metadata,
                    file_path
                )
            except ImportError:
                logger.debug("unstructured xlsx partition not available, trying pandas")
            
            # Fallback: pandas
            try:
                import pandas as pd
                chunks = []
                
                # Read all sheets
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Convert DataFrame to text
                    text_lines = []
                    text_lines.append(f"Sheet: {sheet_name}\n")
                    text_lines.append(df.to_string(index=False))
                    
                    sheet_text = "\n".join(text_lines)
                    sheet_chunks = self._simple_chunk(
                        sheet_text,
                        {**material_metadata, "sheet_name": sheet_name},
                        file_path
                    )
                    chunks.extend(sheet_chunks)
                
                return chunks
            except ImportError:
                logger.warning("pandas not available for Excel parsing")
                return []
        except Exception as e:
            logger.error(f"Failed to parse XLSX {file_path}: {e}")
            return []

    def _parse_txt(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return self._simple_chunk(text, material_metadata, file_path)
        except Exception as e:
            logger.error(f"Failed to parse TXT {file_path}: {e}")
            return []

    def _parse_csv(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
            return self._simple_chunk(text, material_metadata, file_path)
        except ImportError:
            logger.warning("pandas not available for CSV parsing")
            # Fallback: simple text parsing
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return self._simple_chunk(text, material_metadata, file_path)
        except Exception as e:
            logger.error(f"Failed to parse CSV {file_path}: {e}")
            return []

    def _parse_rtf(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse RTF file."""
        try:
            from unstructured.partition.rtf import partition_rtf
            elements = partition_rtf(filename=str(file_path))
            return self._create_chunks_from_elements(
                elements,
                material_metadata,
                file_path
            )
        except ImportError:
            logger.warning("unstructured rtf partition not available")
            # Fallback: simple text extraction
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # Remove RTF control codes (simple)
            import re
            text = re.sub(r'\\[a-z]+\d*\s?', '', text)
            text = re.sub(r'\{[^}]*\}', '', text)
            return self._simple_chunk(text, material_metadata, file_path)
        except Exception as e:
            logger.error(f"Failed to parse RTF {file_path}: {e}")
            return []

    def _parse_zip(
        self,
        file_path: Path,
        material_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Parse ZIP file by extracting and parsing contents."""
        import zipfile
        import tempfile
        from pathlib import Path as PathLib
        
        chunks = []
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract to temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(temp_dir)
                    
                    # Parse each file in the zip
                    for extracted_file in PathLib(temp_dir).rglob('*'):
                        if extracted_file.is_file():
                            # Skip non-document files
                            if extracted_file.suffix.lower() in ['.exe', '.dll', '.so', '.bin']:
                                continue
                            
                            try:
                                file_chunks = self.parse_file(extracted_file, {
                                    **material_metadata,
                                    "zip_file": str(file_path),
                                    "extracted_path": str(extracted_file.relative_to(temp_dir))
                                })
                                chunks.extend(file_chunks)
                            except Exception as e:
                                logger.debug(f"Failed to parse {extracted_file} from ZIP: {e}")
                                continue
            
            return chunks
        except Exception as e:
            logger.error(f"Failed to parse ZIP {file_path}: {e}")
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
                elif "audio" in content_type or url.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
                    # Skip audio files
                    logger.info(f"Audio file detected: {url}. Skipping as requested.")
                    return []
                elif "video" in content_type or url.lower().endswith((".mp4", ".webm", ".mov")):
                    # Skip video files
                    logger.info(f"Video file detected: {url}. Skipping as requested.")
                    return []
                elif "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type or \
                     "application/vnd.ms-excel" in content_type or \
                     url.lower().endswith((".xls", ".xlsx")):
                    # Excel file
                    return await self._parse_excel_url(url, material_metadata, save_dir, response.content)
                elif "text/plain" in content_type or url.lower().endswith(".txt"):
                    # Plain text file
                    text = response.text
                    return self._simple_chunk(text, material_metadata, Path(url))
                elif "text/csv" in content_type or url.lower().endswith(".csv"):
                    # CSV file
                    return await self._parse_csv_url(url, material_metadata, save_dir, response.content)
                elif "application/zip" in content_type or url.lower().endswith(".zip"):
                    # ZIP file - extract and parse contents
                    return await self._parse_zip_url(url, material_metadata, save_dir, response.content)
                elif "application/msword" in content_type or \
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type:
                    # Word file from URL
                    return await self._parse_docx_url(url, material_metadata, save_dir, response.content)
                elif "application/vnd.ms-powerpoint" in content_type or \
                     "application/vnd.openxmlformats-officedocument.presentationml.presentation" in content_type:
                    # PowerPoint file from URL
                    return await self._parse_pptx_url(url, material_metadata, save_dir, response.content)
                else:
                    # Try to parse as generic document
                    logger.debug(f"Unknown content type: {content_type} for {url}. Attempting generic parsing.")
                    # Try saving and parsing based on extension
                    if save_dir:
                        import hashlib
                        save_dir.mkdir(parents=True, exist_ok=True)
                        ext = Path(urlparse(url).path).suffix or ".bin"
                        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ext
                        temp_path = save_dir / filename
                        with open(temp_path, "wb") as f:
                            f.write(response.content)
                        # Try parsing as file
                        return self.parse_file(temp_path, material_metadata)
                    return []

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return []

    async def _parse_excel_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        file_data: bytes
    ) -> List[DocumentChunk]:
        """Parse Excel file from URL."""
        if not save_dir:
            logger.warning("save_dir not provided for Excel download")
            return []
        
        import hashlib
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".xlsx"
        temp_path = save_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        return self._parse_xlsx(temp_path, material_metadata)

    async def _parse_csv_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        file_data: bytes
    ) -> List[DocumentChunk]:
        """Parse CSV file from URL."""
        if not save_dir:
            # Try parsing directly from text
            try:
                text = file_data.decode('utf-8', errors='ignore')
                return self._simple_chunk(text, material_metadata, Path(url))
            except:
                return []
        
        import hashlib
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".csv"
        temp_path = save_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        return self._parse_csv(temp_path, material_metadata)

    async def _parse_zip_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        file_data: bytes
    ) -> List[DocumentChunk]:
        """Parse ZIP file from URL."""
        if not save_dir:
            logger.warning("save_dir not provided for ZIP download")
            return []
        
        import hashlib
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".zip"
        temp_path = save_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        return self._parse_zip(temp_path, material_metadata)

    async def _parse_docx_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        file_data: bytes
    ) -> List[DocumentChunk]:
        """Parse Word document from URL."""
        if not save_dir:
            logger.warning("save_dir not provided for DOCX download")
            return []
        
        import hashlib
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".docx"
        temp_path = save_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        return self._parse_docx(temp_path, material_metadata)

    async def _parse_pptx_url(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        file_data: bytes
    ) -> List[DocumentChunk]:
        """Parse PowerPoint from URL."""
        if not save_dir:
            logger.warning("save_dir not provided for PPTX download")
            return []
        
        import hashlib
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".pptx"
        temp_path = save_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        return self._parse_pptx(temp_path, material_metadata)

    async def _parse_audio(
        self,
        url: str,
        material_metadata: Dict[str, Any],
        save_dir: Optional[Path],
        audio_data: bytes
    ) -> List[DocumentChunk]:
        """
        Parse audio file by transcribing to text using OpenAI Whisper API.
        
        Args:
            url: Audio file URL
            material_metadata: Metadata about the source
            save_dir: Optional directory to save audio file
            audio_data: Audio file content (bytes)
        
        Returns:
            List of DocumentChunk from transcribed text
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Audio transcription skipped.")
            return []
        
        try:
            import openai
            import hashlib
            import tempfile
            
            # Check file size (OpenAI limit: 25MB)
            file_size_mb = len(audio_data) / (1024 * 1024)
            if file_size_mb > 25:
                logger.warning(f"Audio file too large ({file_size_mb:.1f}MB > 25MB). Skipping.")
                return []
            
            # Save audio to temporary file
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".mp3"
                temp_path = save_dir / filename
            else:
                temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            
            with open(temp_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Transcribing audio: {url} ({file_size_mb:.1f}MB)")
            
            # Call Whisper API
            client = openai.OpenAI(api_key=api_key)
            
            with open(temp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",  # Korean (can be "auto" for auto-detection)
                    response_format="text"
                )
            
            logger.info(f"Transcription completed. Length: {len(transcript)} chars")
            
            # Create chunks from transcript
            chunks = self._simple_chunk(
                transcript,
                {
                    **material_metadata,
                    "file_format": "audio",
                    "transcription": True,
                    "audio_duration": None  # Could extract from metadata if available
                },
                temp_path
            )
            
            # Optionally keep audio file for reference
            # temp_path.unlink()  # Uncomment to delete after transcription
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error transcribing audio {url}: {e}")
            return []
