"""Web scraper for corporate IR materials (news, presentations, IR pitches)."""
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class IRMaterial:
    """A single IR material document."""
    url: str
    title: str
    material_type: str  # "news", "presentation", "ir_pitch", "sec_filing", "shareholder_letter", "quarterly_report", "other"
    date: Optional[str] = None
    file_format: Optional[str] = None  # "pdf", "ppt", "html", "doc", etc.
    source_page: Optional[str] = None


class IRScraper:
    """
    Scrapes corporate IR materials from official company websites.

    This scraper handles:
    - News/Press releases
    - Investor presentations
    - IR pitch decks
    - SEC filings (links from official site)
    - Financial reports

    The scraper intelligently navigates different website structures
    by looking for common IR section patterns.
    """

    # Common IR section URL patterns (comprehensive list)
    IR_PATTERNS = [
        # Primary IR paths
        "investors",
        "investor-relations", 
        "investor_relations",
        "ir",
        "stockholders",
        "shareholders",
        # Financial information
        "financial-information",
        "financials",
        "financial-reports",
        "financial-data",
        # News & Press
        "news",
        "press",
        "press-releases",
        "press-room",
        "newsroom",
        "media",
        # Events & Presentations
        "events",
        "events-presentations",
        "events-and-presentations",
        "presentations",
        "webcasts",
        # Reports (excluding SEC filings - handled separately)
        "quarterly-reports",
        "annual-reports",
        "reports",
        "earnings",
        "earnings-releases",
        # Shareholder materials
        "shareholder-letter",
        "shareholder-letters", 
        "letter-to-shareholders",
        "quarterly-letter",
        "ceo-letter",
        "chairman-letter",
        # Corporate governance
        "governance",
        "corporate-governance",
        # Stock information
        "stock",
        "stock-information",
        "share-price",
        # Resources
        "resources",
        "documents",
        "downloads",
    ]
    
    # Common IR subdomains
    IR_SUBDOMAINS = [
        "ir",
        "investors", 
        "investor",
        "investor-relations",
        "shareholdings",
        "finance",
    ]

    # File extension patterns
    FILE_EXTENSIONS = {
        ".pdf": "pdf",
        ".ppt": "ppt",
        ".pptx": "pptx",
        ".doc": "doc",
        ".docx": "docx",
        ".xls": "xls",
        ".xlsx": "xlsx",
        ".html": "html",
        ".htm": "html",
        # Audio files
        ".mp3": "audio",
        ".wav": "audio",
        ".m4a": "audio",
        ".ogg": "audio",
        # Video files (can extract audio)
        ".mp4": "video",
        ".webm": "video",
        ".mov": "video",
    }

    def __init__(self, timeout: int = 30, max_depth: int = 2):
        self.timeout = timeout
        self.max_depth = max_depth  # How deep to crawl from IR pages
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self._visited_urls: Set[str] = set()

    def scrape(
        self,
        company_website: str,
        progress_callback: Optional[callable] = None
    ) -> List[IRMaterial]:
        """
        Scrape IR materials from company website.

        Args:
            company_website: Official company website URL
            progress_callback: Optional progress callback function

        Returns:
            List of IRMaterial found
        """
        materials = []
        self._visited_urls = set()

        # Step 1: Find IR section pages
        if progress_callback:
            progress_callback("Searching for IR sections...")

        ir_pages = self._find_ir_pages(company_website)

        if not ir_pages:
            logger.warning(f"No IR pages found for {company_website}")
            # Try homepage anyway
            ir_pages = [company_website]
        
        if progress_callback:
            progress_callback(f"Found {len(ir_pages)} IR section(s)")

        # Step 2: Scrape materials from each IR page (with depth crawling)
        for page_url in ir_pages:
            if progress_callback:
                progress_callback(f"Scraping: {page_url}")

            page_materials = self._scrape_page_recursive(
                page_url, 
                company_website, 
                depth=0,
                progress_callback=progress_callback
            )
            materials.extend(page_materials)

        # Step 3: Deduplicate by URL
        seen_urls = set()
        unique_materials = []
        for m in materials:
            if m.url not in seen_urls:
                seen_urls.add(m.url)
                unique_materials.append(m)
        
        # Step 4: Filter out non-IR materials
        filtered_materials = self._filter_relevant_materials(unique_materials)

        if progress_callback:
            progress_callback(f"Found {len(filtered_materials)} IR materials")
            
        logger.info(f"Found {len(filtered_materials)} IR materials")
        return filtered_materials
    
    def _filter_relevant_materials(self, materials: List[IRMaterial]) -> List[IRMaterial]:
        """Filter out irrelevant materials (navigation links, SEC filings, etc.)."""
        filtered = []
        skip_patterns = [
            "javascript:", "mailto:", "#", "tel:",
            "facebook.com", "twitter.com", "linkedin.com", "youtube.com",
            "instagram.com", "pinterest.com",
            "/careers", "/jobs", "/contact", "/about-us", "/privacy",
            "/terms", "/cookie", "/login", "/signup", "/register",
        ]
        
        # SEC filing patterns to exclude
        sec_patterns = [
            "/sec-filings",
            "/sec/",
            "sec.gov",
            "10-k", "10-q", "8-k", "form-10", "form-8",
            "xbrl", "xbrl.zip",
            "edgar",
            "/all-sec-filings",
            "/quarterly-filings",
            "/annual-filings",
        ]
        
        for m in materials:
            url_lower = m.url.lower()
            title_lower = m.title.lower()
            
            # Skip SEC filings and related materials
            if m.material_type == "sec_filing":
                logger.debug(f"Excluding SEC filing: {m.url}")
                continue
            
            # Skip URLs containing SEC patterns
            if any(pattern in url_lower for pattern in sec_patterns):
                logger.debug(f"Excluding SEC-related URL: {m.url}")
                continue
            
            # Skip titles mentioning SEC filings
            if any(term in title_lower for term in ["sec filing", "10-k", "10-q", "8-k", "form ", "edgar"]):
                logger.debug(f"Excluding SEC-related title: {m.title}")
                continue
            
            # Skip social media and navigation links
            if any(pattern in url_lower for pattern in skip_patterns):
                continue
            
            # Skip if title is too short (likely a button or icon)
            if len(m.title) < 5 and not m.file_format:
                continue
                
            filtered.append(m)
        
        logger.info(f"Filtered {len(materials)} materials to {len(filtered)} (excluded {len(materials) - len(filtered)} SEC/irrelevant)")
        return filtered

    def _find_ir_pages(self, base_url: str) -> List[str]:
        """Find IR section pages by trying common patterns."""
        ir_pages = []
        parsed = urlparse(base_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        
        # Extract domain parts for subdomain testing
        netloc = parsed.netloc
        domain_parts = netloc.split(".")
        if domain_parts[0] == "www":
            domain_parts = domain_parts[1:]
        root_domain = ".".join(domain_parts)
        
        # If the URL itself looks like an IR subdomain, add it directly
        if any(sub in netloc for sub in self.IR_SUBDOMAINS):
            ir_pages.append(base_url)
            logger.info(f"Input URL is already an IR site: {base_url}")
        
        # Step 1: Try common IR subdomains (ir.company.com, investors.company.com)
        for subdomain in self.IR_SUBDOMAINS:
            subdomain_url = f"{parsed.scheme}://{subdomain}.{root_domain}/"
            if subdomain_url not in ir_pages and self._is_valid_page(subdomain_url):
                logger.info(f"Found IR subdomain: {subdomain_url}")
                ir_pages.append(subdomain_url)

        # Step 2: Try common IR paths on main domain (limit to most common ones first)
        priority_patterns = ["investors", "investor-relations", "ir", "financials", "news"]
        for pattern in priority_patterns:
            test_url = urljoin(base_domain, f"/{pattern}/")
            if test_url not in ir_pages and self._is_valid_page(test_url):
                logger.info(f"Found IR page: {test_url}")
                ir_pages.append(test_url)
        
        # If we found IR pages, also try subpaths within them (excluding SEC filings)
        if ir_pages:
            additional_pages = []
            # Exclude sec-filings from subpaths
            subpaths = ["presentations", "events", "annual-reports", 
                       "quarterly-reports", "press-releases", "news", "earnings"]
            for ir_page in ir_pages[:3]:  # Limit to first 3 IR pages
                for subpath in subpaths:
                    test_url = urljoin(ir_page, f"/{subpath}/")
                    if test_url not in ir_pages and test_url not in additional_pages:
                        if self._is_valid_page(test_url):
                            logger.info(f"Found IR subpage: {test_url}")
                            additional_pages.append(test_url)
            ir_pages.extend(additional_pages)
        
        # Step 3: Crawl main page and IR pages to find more IR links
        ir_pages.extend(self._find_ir_links_from_homepage(base_url))
        
        # Also crawl found IR pages for additional links
        for ir_page in list(ir_pages)[:5]:  # Limit crawling
            found_links = self._find_ir_links_from_homepage(ir_page)
            for link in found_links:
                if link not in ir_pages:
                    ir_pages.append(link)

        return list(set(ir_pages))  # Deduplicate
    
    def _find_ir_links_from_homepage(self, base_url: str) -> List[str]:
        """Crawl homepage to find links to IR sections."""
        ir_links = []
        try:
            response = httpx.get(
                base_url,
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True
            )
            if response.status_code != 200:
                return ir_links
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Look for links containing IR-related keywords (excluding SEC)
            ir_keywords = [
                "investor", "shareholder", "stockholder", "ir", 
                "financial", "annual report", "quarterly"
            ]
            
            # SEC-related keywords to exclude
            sec_keywords = ["sec filing", "sec-filing", "sec/", "edgar", "10-k", "10-q", "8-k"]
            
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                text = link.get_text(strip=True).lower()
                href_lower = href.lower()
                
                # Skip SEC-related links
                if any(sec_kw in text or sec_kw in href_lower for sec_kw in sec_keywords):
                    continue
                
                # Check if link text or href contains IR keywords
                if any(kw in text or kw in href_lower for kw in ir_keywords):
                    full_url = urljoin(base_url, href)
                    if self._is_valid_page(full_url):
                        logger.info(f"Found IR link from homepage: {full_url}")
                        ir_links.append(full_url)
            
        except Exception as e:
            logger.warning(f"Error crawling homepage for IR links: {e}")
        
        return ir_links

    def _is_valid_page(self, url: str) -> bool:
        """Check if URL is a valid accessible page."""
        try:
            response = httpx.head(
                url,
                headers=self.headers,
                timeout=10,  # Shorter timeout for validation
                follow_redirects=True
            )
            if response.status_code == 200:
                return True
            # Some servers don't support HEAD, try GET
            if response.status_code == 405:
                response = httpx.get(
                    url,
                    headers=self.headers,
                    timeout=10,
                    follow_redirects=True
                )
                return response.status_code == 200
            return False
        except Exception:
            return False
    
    def _scrape_page_recursive(
        self, 
        page_url: str, 
        base_url: str, 
        depth: int = 0,
        progress_callback: Optional[callable] = None
    ) -> List[IRMaterial]:
        """Recursively scrape materials from a page and its subpages."""
        if depth > self.max_depth:
            return []
        
        if page_url in self._visited_urls:
            return []
        
        self._visited_urls.add(page_url)
        
        materials = self._scrape_page(page_url, base_url)
        
        # If depth allows, also follow IR-related links on this page
        if depth < self.max_depth:
            try:
                response = httpx.get(
                    page_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    follow_redirects=True
                )
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "lxml")
                    
                    # Find subpage links
                    for link in soup.find_all("a", href=True):
                        href = link.get("href", "")
                        full_url = urljoin(page_url, href)
                        
                        # Only follow IR-related links on same domain
                        if full_url in self._visited_urls:
                            continue
                        
                        parsed_base = urlparse(base_url)
                        parsed_link = urlparse(full_url)
                        
                        # Check if same domain or IR subdomain
                        base_root = ".".join(parsed_base.netloc.split(".")[-2:])
                        link_root = ".".join(parsed_link.netloc.split(".")[-2:])
                        
                        if base_root == link_root and self._is_ir_page(full_url):
                            if progress_callback and depth == 0:
                                progress_callback(f"  Following: {full_url[:60]}...")
                            
                            sub_materials = self._scrape_page_recursive(
                                full_url, base_url, depth + 1, progress_callback
                            )
                            materials.extend(sub_materials)
                            
            except Exception as e:
                logger.debug(f"Error in recursive scrape of {page_url}: {e}")
        
        return materials

    def _scrape_page(self, page_url: str, base_url: str) -> List[IRMaterial]:
        """Scrape materials from a single page."""
        materials = []

        try:
            response = httpx.get(
                page_url,
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Find all links to documents
            links_found = 0
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                if not href:
                    continue

                # Resolve relative URLs
                full_url = urljoin(page_url, href)

                # Check if it's a document we want
                material = self._classify_link(full_url, link, page_url)
                if material:
                    materials.append(material)
                    links_found += 1
            
            # Also check for direct file links in page content (for Korean sites)
            # Look for common file patterns in text
            page_text = soup.get_text().lower()
            if not materials and any(keyword in page_text for keyword in ["pdf", "다운로드", "download", "보고서", "자료"]):
                # Try to find links in script tags or data attributes
                for script in soup.find_all("script"):
                    if script.string:
                        # Look for URLs in JavaScript
                        import re
                        urls = re.findall(r'https?://[^\s"\'<>]+\.(?:pdf|ppt|pptx|doc|docx|xls|xlsx)', script.string)
                        for url in urls:
                            material = self._classify_link(url, None, page_url)
                            if material:
                                materials.append(material)
                                links_found += 1
            
            if links_found > 0:
                logger.debug(f"Found {links_found} materials on {page_url}")
            else:
                logger.debug(f"No materials found on {page_url} (checked {len(list(soup.find_all('a', href=True)))} links)")

        except Exception as e:
            logger.error(f"Error scraping {page_url}: {e}")

        return materials

    def _classify_link(
        self,
        url: str,
        link_element,
        source_page: str
    ) -> Optional[IRMaterial]:
        """Classify a link as an IR material."""
        # Get file extension
        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        # Check for document extensions
        file_format = None
        for ext, fmt in self.FILE_EXTENSIONS.items():
            if path_lower.endswith(ext):
                file_format = fmt
                break

        # If no file extension, might be an HTML page
        # For HTML pages, only include if it's clearly an IR-related content page
        if not file_format:
            # Check if it's a content page (not just navigation)
            if not self._is_ir_page(url):
                return None
            # For HTML pages, check if link text suggests it's actual content
            if link_element:
                link_text = link_element.get_text(strip=True).lower()
                # Korean keywords for documents/reports
                korean_keywords = ["보고서", "자료", "공시", "발표", "실적", "재무", "연례", "분기"]
                # English keywords
                english_keywords = ["report", "presentation", "release", "earnings", "financial", "annual", "quarterly"]
                
                if not any(kw in link_text for kw in korean_keywords + english_keywords):
                    # Might be just a navigation link, skip
                    return None

        # Get title from link text
        if link_element:
            title = link_element.get_text(strip=True)
            # Also check for title attribute or data attributes
            if not title or len(title) < 3:
                title = link_element.get("title", "") or link_element.get("data-title", "")
        else:
            title = ""
        
        if not title or len(title) < 3:
            title = Path(parsed.path).stem or parsed.path.split("/")[-1] or "Document"

        # Classify material type
        material_type = self._classify_material_type(url, title, link_element)

        # Extract date if present
        date = self._extract_date(title, link_element)

        return IRMaterial(
            url=url,
            title=title,
            material_type=material_type,
            date=date,
            file_format=file_format,
            source_page=source_page
        )

    def _is_ir_page(self, url: str) -> bool:
        """Check if URL might be an IR-related page."""
        path_lower = urlparse(url).path.lower()
        ir_keywords = [
            "news", "press", "release", "presentation", "ir",
            "investor", "financial", "earnings", "quarterly",
            "annual", "report", "sec", "filing",
            # Shareholder letter keywords
            "letter", "shareholder", "stockholder", "ceo",
            "update", "resources", "documents",
            # Korean keywords
            "보고서", "자료", "공시", "발표", "실적", "재무", "연례", "분기",
            "투자", "주주", "공시자료", "재무제표"
        ]
        return any(keyword in path_lower for keyword in ir_keywords)

    def _classify_material_type(self, url: str, title: str, link_element) -> str:
        """Classify the type of IR material."""
        url_lower = url.lower()
        title_lower = title.lower()

        # Check for SEC filings
        if "sec" in url_lower or any(f in url_lower for f in ["10-k", "10-q", "8-k", "form", "filing"]):
            return "sec_filing"

        # Check for shareholder letters (check before presentations)
        if any(kw in url_lower or kw in title_lower for kw in
               ["shareholder-letter", "letter-to-shareholder", "stockholder-letter",
                "ceo-letter", "quarterly-letter", "shareholder letter",
                "letter to shareholder", "stockholder letter"]):
            return "shareholder_letter"

        # Check for quarterly/annual reports
        if any(kw in url_lower or kw in title_lower for kw in
               ["quarterly-report", "annual-report", "quarterly report", "annual report",
                "q1", "q2", "q3", "q4", "earnings"]):
            return "quarterly_report"

        # Check for presentations
        if any(kw in url_lower or kw in title_lower for kw in
               ["presentation", "deck", "slide", "investor", "day", "conference"]):
            return "presentation"

        # Check for news/press
        if any(kw in url_lower or kw in title_lower for kw in
               ["news", "press", "release", "announcement"]):
            return "news"

        # Check for IR pitch
        if any(kw in url_lower or kw in title_lower for kw in
               ["pitch", "overview", "intro", "company", "overview"]):
            return "ir_pitch"

        return "other"

    def _extract_date(self, title: str, link_element) -> Optional[str]:
        """Extract date from title or link context."""
        # Try to find date patterns in title
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # 2024-01-15
            r"\d{4}/\d{2}/\d{2}",  # 2024/01/15
            r"\d{4}\.\d{2}\.\d{2}",  # 2024.01.15
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, title)
            if match:
                return match.group(0)

        # Try to get from nearby text (e.g., date column in table)
        parent = link_element.parent
        if parent:
            # Look for date in siblings
            for sibling in parent.find_all(["td", "div", "span"]):
                text = sibling.get_text(strip=True)
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(0)

        return None

    async def download_material(
        self,
        material: IRMaterial,
        save_dir: Path
    ) -> Optional[Path]:
        """
        Download an IR material to local storage.

        Args:
            material: IRMaterial to download
            save_dir: Directory to save to

        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Generate filename
            filename = self._generate_filename(material)
            filepath = save_dir / filename

            # Skip if already exists
            if filepath.exists():
                logger.debug(f"File already exists: {filepath}")
                return filepath

            # Download
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    material.url,
                    headers=self.headers,
                    follow_redirects=True
                )
                response.raise_for_status()

                # Save
                with open(filepath, "wb") as f:
                    f.write(response.content)

                logger.info(f"Downloaded: {filepath}")
                return filepath

        except Exception as e:
            logger.error(f"Failed to download {material.url}: {e}")
            return None

    def _generate_filename(self, material: IRMaterial) -> str:
        """Generate safe filename from material."""
        # Sanitize title
        safe_title = re.sub(r'[^\w\s-]', '', material.title)
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        safe_title = safe_title.strip('-')[:50]  # Limit length

        # Add date if available
        date_prefix = ""
        if material.date:
            date_prefix = material.date.replace("-", "").replace("/", "") + "_"

        # Add extension
        if material.file_format:
            ext = material.file_format
        else:
            ext = "html"

        return f"{date_prefix}{safe_title}.{ext}"
