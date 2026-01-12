"""Agent state definitions for NOVA."""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..scrapers.ir_scraper import IRMaterial


@dataclass
class AgentState:
    """
    State for the NOVA research agent.

    Tracks the entire research session including:
    - Company identification
    - Scraped IR materials
    - Current query and retrieved context
    - Generated answers with quality scores
    """
    # Company information
    company_query: str = ""
    company_name: str = ""
    company_website: str = ""
    company_confirmed: bool = False

    # Materials status
    materials: List[IRMaterial] = field(default_factory=list)
    materials_processed: bool = False
    chunks_indexed: int = 0

    # Query handling
    current_query: str = ""
    retrieved_chunks: List[Dict] = field(default_factory=list)

    # Answer generation
    current_answer: str = ""
    answer_score: int = 0
    meets_quality_threshold: bool = False

    # Error handling
    error: Optional[str] = None

    # Session metadata
    session_active: bool = True


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dictionary."""
    return {
        "company_query": "",
        "company_name": "",
        "company_website": "",
        "company_confirmed": False,
        "materials": [],
        "materials_processed": False,
        "chunks_indexed": 0,
        "current_query": "",
        "retrieved_chunks": [],
        "current_answer": "",
        "answer_score": 0,
        "meets_quality_threshold": False,
        "error": None,
        "session_active": True,
    }
