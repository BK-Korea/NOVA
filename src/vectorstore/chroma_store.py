"""Chroma vector store for RAG with IR materials."""
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from ..document.parser import DocumentChunk


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score."""
    content: str
    metadata: Dict
    score: float

    @property
    def citation(self) -> str:
        """Generate citation string."""
        material_type = self.metadata.get('material_type', 'Unknown')
        title = self.metadata.get('title', 'Unknown')
        date = self.metadata.get('date', '')
        url = self.metadata.get('url', '')

        if date:
            return f"[{material_type} | {title} | {date}]"
        return f"[{material_type} | {title}]"

    def to_context_string(self) -> str:
        """Format chunk for LLM context with URL."""
        url = self.metadata.get('url', '')
        if url:
            return f"{self.citation}\nURL: {url}\n{self.content}"
        return f"{self.citation}\n{self.content}"


class ChromaStore:
    """
    Chroma vector store wrapper for IR materials.

    Features:
    - Persistent storage
    - Metadata filtering (by material type, date, source)
    - Retrieval with citations
    """

    def __init__(
        self,
        persist_dir: Path,
        embeddings: Embeddings,
        collection_name: str = "ir_materials"
    ):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.collection_name = collection_name

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk to add
            batch_size: Batch size for embedding
            progress_callback: Optional progress callback

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data
            texts = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]

            # Generate unique IDs
            ids = [
                f"{c.metadata.get('url', 'unknown')}_{c.metadata.get('chunk_index', i+j)}"
                for j, c in enumerate(batch)
            ]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            total_added += len(batch)

            if progress_callback:
                progress_callback(f"Indexed {total_added}/{len(chunks)} chunks")

        return total_added

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_material_types: Optional[List[str]] = None,
        filter_company: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_material_types: Optional list of material types to filter
            filter_company: Optional company name to filter
            min_score: Minimum similarity score

        Returns:
            List of RetrievedChunk with scores
        """
        # Build where clause for filtering
        where = None
        conditions = []

        if filter_material_types:
            conditions.append({"material_type": {"$in": filter_material_types}})

        if filter_company:
            conditions.append({"company_name": filter_company})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievedChunk
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # Convert distance to similarity score (cosine)
                score = 1 - distance

                if score >= min_score:
                    retrieved.append(RetrievedChunk(
                        content=doc,
                        metadata=meta,
                        score=score
                    ))

        return retrieved

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir)
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_by_company(self, company_name: str) -> int:
        """Delete all chunks for a specific company."""
        # Get IDs to delete
        results = self.collection.get(
            where={"company_name": company_name},
            include=[]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def check_existing(self, url: str) -> bool:
        """Check if a URL has already been indexed."""
        results = self.collection.get(
            where={"url": url},
            limit=1,
            include=[]
        )
        return len(results.get("ids", [])) > 0
