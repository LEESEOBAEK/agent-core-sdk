import logging
from typing import List, Dict, Any

try:
    import chromadb
except ImportError:
    chromadb = None

from agent_core.rag.infrastructure.base import VectorStore, SearchResult
from agent_core.rag.embedding import collection_name_for, get_embedding_function

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        collection_name: str = "phoenix-rag-docs",
        embed_provider: str = "chroma",
        embed_model: str = None,
    ):
        if chromadb is None:
            raise ImportError("pip install chromadb  (or: pip install agent-core-sdk[rag])")

        self.embed_provider = embed_provider
        self.embed_model = embed_model

        actual_name = collection_name_for(collection_name, embed_provider)
        embedding_fn = get_embedding_function(embed_provider, embed_model)

        self.client = chromadb.PersistentClient(path=persist_dir)

        kwargs = {"name": actual_name}
        if embedding_fn:
            kwargs["embedding_function"] = embedding_fn
        self.collection = self.client.get_or_create_collection(**kwargs)

        logger.info(f"[ChromaStore] initialized: {actual_name} (docs={self.collection.count()})")

    @property
    def backend_name(self) -> str:
        return "chroma"

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            return True
        except Exception as e:
            logger.error(f"[ChromaStore] add failed: {e}")
            return False

    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        standardized: List[SearchResult] = []

        ids = results["ids"][0]
        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results.get("metadatas", [[]])[0] or []

        for id_, doc, dist, meta in zip(ids, docs, dists, metas):
            score = round(1 - dist, 4)
            standardized.append(SearchResult(
                id=id_,
                content=doc,
                score=score,
                metadata=meta or {},
            ))

        return standardized

    def count(self) -> int:
        return self.collection.count()
