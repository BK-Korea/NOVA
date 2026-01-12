"""NOVA Agent - Main agent graph orchestrating the research workflow."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from pydantic import SecretStr

from .state import AgentState, create_initial_state
from .company_matcher import CompanyMatcher, CompanyInfo
from .quality_evaluator import IterativeAnswerer
from ..scrapers.ir_scraper import IRScraper, IRMaterial
from ..document.parser import DocumentParser, DocumentChunk
from ..vectorstore.chroma_store import ChromaStore, RetrievedChunk
from ..llm.glm_client import GLMChat, GLMEmbeddings, OpenAIEmbeddings

logger = logging.getLogger(__name__)


class NovaAgent:
    """
    Main NOVA agent for IR material research and analysis.

    Workflow:
    1. Resolve company query → official company info
    2. Scrape IR materials from official website
    3. Parse and index materials in vector store
    4. Answer questions using RAG + iterative quality refinement
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        data_dir: Path = None,
        progress_callback: Optional[Callable] = None,
        openai_api_key: str = "",
        embedding_provider: str = "openai",
        quality_threshold: int = 8
    ):
        # Initialize LLM
        self.llm = GLMChat(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model="glm-4.7",
            temperature=0.7
        )

        # Initialize embeddings
        if embedding_provider == "openai" and openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                api_key=SecretStr(openai_api_key),
                model="text-embedding-3-small"
            )
        else:
            self.embeddings = GLMEmbeddings(
                api_key=SecretStr(api_key),
                base_url=base_url
            )

        # Set data directory
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = data_dir

        # Initialize components
        self.company_matcher = CompanyMatcher(self.llm)
        self.scraper = IRScraper()
        self.parser = DocumentParser()
        self.vector_store = ChromaStore(
            persist_dir=self.data_dir / "vectordb",
            embeddings=self.embeddings
        )
        self.answerer = IterativeAnswerer(
            llm=self.llm,
            target_score=quality_threshold,
            max_iterations=2
        )

        self.progress_callback = progress_callback
        self.quality_threshold = quality_threshold

        # Current state
        self.state = create_initial_state()

    def resolve_company(self, query: str) -> Dict[str, Any]:
        """
        Resolve company query to official company information.

        Returns:
            Dict with 'company_info' (CompanyInfo) or 'error' (str)
        """
        try:
            if self.progress_callback:
                self.progress_callback(f"Resolving company: {query}")

            result = self.company_matcher.match(query)

            if "error" in result:
                return result

            company_info: CompanyInfo = result["company_info"]

            # Update state - prefer IR website if available
            self.state["company_query"] = query
            self.state["company_name"] = company_info.official_name
            self.state["company_website"] = company_info.ir_website or company_info.website
            self.state["main_website"] = company_info.website

            return result

        except Exception as e:
            logger.error(f"Error resolving company: {e}")
            return {"error": str(e)}

    def fetch_materials(self, company_name: str) -> Dict[str, Any]:
        """
        Fetch IR materials from company website.

        Returns:
            Dict with 'materials' (list) and 'chunks_indexed' (int)
        """
        try:
            website = self.state.get("company_website")
            main_website = self.state.get("main_website")
            
            if not website:
                return {"error": "Company website not set. Call resolve_company first."}

            if self.progress_callback:
                self.progress_callback(f"Scraping IR materials from {website}")

            # Scrape materials from IR website (or main website)
            materials = self.scraper.scrape(
                company_website=website,
                progress_callback=self.progress_callback
            )
            
            # Also scrape main website if different from IR website
            if main_website and main_website != website:
                if self.progress_callback:
                    self.progress_callback(f"Also checking main website: {main_website}")
                main_materials = self.scraper.scrape(
                    company_website=main_website,
                    progress_callback=self.progress_callback
                )
                # Merge and deduplicate
                seen_urls = {m.url for m in materials}
                for m in main_materials:
                    if m.url not in seen_urls:
                        materials.append(m)
                        seen_urls.add(m.url)

            self.state["materials"] = materials

            if not materials:
                return {"error": "No IR materials found on company website."}

            if self.progress_callback:
                self.progress_callback(f"Found {len(materials)} materials")

            # Parse and index
            chunks = self._process_materials(materials, company_name)

            self.state["materials_processed"] = True
            self.state["chunks_indexed"] = chunks

            return {
                "materials": materials,
                "chunks_indexed": chunks
            }

        except Exception as e:
            logger.error(f"Error fetching materials: {e}")
            return {"error": str(e)}

    def _process_materials(
        self,
        materials: list,
        company_name: str
    ) -> int:
        """Process materials: parse and index in vector store."""
        all_chunks = []
        save_dir = self.data_dir / "processed"

        for i, material in enumerate(materials):
            if self.progress_callback:
                self.progress_callback(f"Processing {i+1}/{len(materials)}: {material.title[:50]}")

            # Check if already indexed
            if self.vector_store.check_existing(material.url):
                logger.debug(f"Already indexed: {material.url}")
                continue

            # Prepare metadata
            material_metadata = {
                "url": material.url,
                "title": material.title,
                "material_type": material.material_type,
                "date": material.date or "",
                "file_format": material.file_format or "html",
                "company_name": company_name
            }

            # Try to download and parse
            try:
                # For HTML files, we might parse directly from URL
                # For now, let's try parsing from URL if possible
                import asyncio
                import nest_asyncio

                # Apply nest_asyncio to allow nested event loops
                nest_asyncio.apply()

                async def parse_and_save():
                    chunks = await self.parser.parse_url(
                        url=material.url,
                        material_metadata=material_metadata,
                        save_dir=save_dir
                    )
                    return chunks

                # Run async function
                chunks = asyncio.run(parse_and_save())

                if chunks:
                    all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"Failed to process {material.url}: {e}")
                continue

        # Add to vector store
        if all_chunks:
            if self.progress_callback:
                self.progress_callback(f"Indexing {len(all_chunks)} chunks...")

            indexed = self.vector_store.add_chunks(
                chunks=all_chunks,
                progress_callback=self.progress_callback
            )

            return indexed

        return 0

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about indexed materials.

        Returns:
            Dict with 'current_answer', 'answer_score', and optional 'error'
        """
        try:
            if not self.state.get("materials_processed"):
                stats = self.vector_store.get_collection_stats()
                if stats["total_chunks"] == 0:
                    return {"error": "No materials indexed. Fetch materials first."}

            if self.progress_callback:
                self.progress_callback("Searching relevant materials...")

            # Retrieve relevant chunks
            company_name = self.state.get("company_name", "")
            retrieved = self.vector_store.search(
                query=question,
                top_k=10,
                filter_company=company_name if company_name else None
            )

            if not retrieved:
                return {"error": "No relevant information found in indexed materials."}

            # Build context
            context = "\n\n---\n\n".join([c.to_context_string() for c in retrieved])

            if self.progress_callback:
                self.progress_callback("Generating answer...")

            # Generate answer with iterative refinement
            answer, score, iterations = self.answerer.generate(
                question=question,
                context=context,
                progress_callback=self.progress_callback
            )

            # Update state
            self.state["current_query"] = question
            self.state["current_answer"] = answer
            self.state["answer_score"] = score
            self.state["meets_quality_threshold"] = score >= self.quality_threshold

            return {
                "current_answer": answer,
                "answer_score": score,
                "meets_quality_threshold": score >= self.quality_threshold,
                "iterations_used": iterations
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.vector_store.get_collection_stats()
        stats.update({
            "company_name": self.state.get("company_name", ""),
            "company_website": self.state.get("company_website", ""),
            "materials_count": len(self.state.get("materials", [])),
            "materials_processed": self.state.get("materials_processed", False)
        })
        return stats

    def clear_all(self) -> None:
        """Clear all indexed data."""
        self.vector_store.clear_collection()
        self.state = create_initial_state()
