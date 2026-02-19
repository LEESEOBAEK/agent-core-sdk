from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Universal search result schema."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """
    Abstract Base Class for Vector Stores.

    This interface is designed to be:
    1. Backend-Agnostic (Chroma, Vespa, Pinecone, etc.)
    2. Framework-Agnostic (Can be used in any Python project)
    3. Tracing-Ready (Includes standard hooks for observability)
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        """Add documents to the store."""
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """
        Search for similar documents.
        Returns a list of standardized SearchResult objects.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the backend (e.g. 'chroma', 'vespa') for tracing."""
        pass
