"""URL validation and verification utility."""
import logging
from typing import List, Dict, Optional
import httpx
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class URLValidator:
    """Validate and verify URLs for citation purposes."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._cache: Dict[str, bool] = {}  # Cache validation results
    
    async def validate_url(self, url: str) -> Dict[str, any]:
        """
        Validate if URL is accessible and returns metadata.
        
        Returns:
            Dict with:
            - valid: bool
            - status_code: int
            - accessible: bool
            - error: str (if any)
        """
        if url in self._cache:
            return {"valid": self._cache[url], "cached": True}
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                result = {"valid": False, "error": "Invalid URL format", "accessible": False}
                self._cache[url] = False
                return result
            
            # Try HEAD request first (faster)
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                try:
                    response = await client.head(url)
                    status_code = response.status_code
                    accessible = 200 <= status_code < 400
                except:
                    # Some servers don't support HEAD, try GET
                    try:
                        response = await client.get(url, timeout=self.timeout)
                        status_code = response.status_code
                        accessible = 200 <= status_code < 400
                    except httpx.TimeoutException:
                        result = {
                            "valid": True,  # URL format is valid
                            "accessible": False,
                            "error": "Timeout",
                            "status_code": None
                        }
                        self._cache[url] = False
                        return result
                    except Exception as e:
                        result = {
                            "valid": True,
                            "accessible": False,
                            "error": str(e)[:100],
                            "status_code": None
                        }
                        self._cache[url] = False
                        return result
            
            result = {
                "valid": True,
                "accessible": accessible,
                "status_code": status_code,
                "error": None if accessible else f"HTTP {status_code}"
            }
            self._cache[url] = accessible
            return result
            
        except Exception as e:
            result = {
                "valid": False,
                "accessible": False,
                "error": str(e)[:100],
                "status_code": None
            }
            self._cache[url] = False
            return result
    
    async def validate_urls_batch(self, urls: List[str]) -> Dict[str, Dict]:
        """Validate multiple URLs in batch."""
        results = {}
        for url in urls:
            results[url] = await self.validate_url(url)
        return results
    
    def get_url_status_icon(self, validation_result: Dict) -> str:
        """Get status icon for URL validation."""
        if not validation_result.get("valid"):
            return "❌"
        if validation_result.get("accessible"):
            return "✅"
        return "⚠️"


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text."""
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates
