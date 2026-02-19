"""
agent_core/rag/infrastructure/vespa_store.py
=============================================
Vespa implementation of the VectorStore interface.

Supports:
  - Semantic search  — HNSW approximate nearest-neighbour (default)
  - Keyword search   — BM25 full-text search
  - Hybrid search    — weighted combination of both

Quick start
-----------
1. Start Vespa:
     cd docker/vespa && docker compose up -d

2. Deploy the schema (one-time, or use deploy=True):
     VespaStore(deploy=True)

3. Use as a drop-in replacement for ChromaStore:
     store = VespaStore()
     store.add_documents(documents, metadatas, ids)
     results = store.query("your question")

Environment variables
---------------------
  VESPA_URL       — Vespa HTTP endpoint (default: http://localhost:8080)
  OLLAMA_BASE_URL — Ollama server base URL for embeddings
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from agent_core.rag.infrastructure.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

try:
    from vespa.application import Vespa as _Vespa
    _PYVESPA_AVAILABLE = True
except ImportError:
    _PYVESPA_AVAILABLE = False


class VespaStore(VectorStore):
    """
    Vespa backend for the VectorStore interface.

    Parameters
    ----------
    url : str
        Vespa HTTP API endpoint.
        Defaults to VESPA_URL env var, then ``http://localhost:8080``.
    application : str
        Vespa application/tenant name (used in document IDs and deployment).
    schema : str
        Vespa schema (document-type) name — must match the .sd file.
    embed_fn : callable, optional
        Custom embedding function: ``(List[str]) -> List[List[float]]``.
        When omitted, falls back to Ollama via ``embed_model``.
    embed_model : str
        Ollama embedding model name (used only when ``embed_fn`` is None).
    embed_base_url : str, optional
        Ollama base URL.  Falls back to ``OLLAMA_BASE_URL`` env var.
    embedding_dim : int
        Output dimension of the embedding model (must match the schema).
    default_search_mode : str
        One of ``"semantic"``, ``"keyword"``, ``"hybrid"``.
    deploy : bool
        If ``True``, deploy the Vespa application schema on init
        (requires Docker — uses pyvespa's VespaDocker helper).
    container_name : str
        Docker container name to deploy to when ``deploy=True``.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        application: str = "rag",
        schema: str = "rag_document",
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        embed_model: str = "nomic-embed-text",
        embed_base_url: Optional[str] = None,
        embedding_dim: int = 768,
        default_search_mode: str = "semantic",
        deploy: bool = False,
        container_name: str = "vespa-local",
    ):
        if not _PYVESPA_AVAILABLE:
            raise ImportError(
                "pyvespa is not installed. "
                "Run: pip install pyvespa  "
                "(or: pip install agent-core-sdk[vespa])"
            )

        self._url = url or os.getenv("VESPA_URL", "http://localhost:8080")
        self._application = application
        self._schema = schema
        self._embedding_dim = embedding_dim
        self._default_search_mode = default_search_mode

        # Resolve embedding function
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from agent_core.rag.embedding import OllamaEmbeddingFunction
            self._embed_fn = OllamaEmbeddingFunction(
                model=embed_model,
                base_url=embed_base_url,
            )

        if deploy:
            self._deploy(container_name)

        self._app = _Vespa(url=self._url)

        logger.info(
            "[VespaStore] initialized "
            f"(url={self._url}, schema={schema}, "
            f"dim={embedding_dim}, mode={default_search_mode})"
        )

    # ------------------------------------------------------------------ #
    # VectorStore interface                                                #
    # ------------------------------------------------------------------ #

    @property
    def backend_name(self) -> str:
        return "vespa"

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> bool:
        """
        Embed and feed documents into Vespa.

        Returns True only when every document is accepted (HTTP 200/201).
        """
        if not documents:
            return True

        try:
            embeddings = self._embed_fn(documents)
        except Exception as exc:
            logger.error(f"[VespaStore] Embedding failed: {exc}")
            return False

        failed = 0
        for doc_id, content, meta, emb in zip(ids, documents, metadatas, embeddings):
            fields = {
                "id": doc_id,
                "content": content,
                "metadata": json.dumps(meta, ensure_ascii=False),
                "embedding": {"values": emb},
            }
            try:
                resp = self._app.feed_data_point(
                    schema=self._schema,
                    data_id=doc_id,
                    fields=fields,
                )
                if resp.status_code not in (200, 201):
                    logger.warning(
                        f"[VespaStore] Feed '{doc_id}' → "
                        f"HTTP {resp.status_code}: {resp.json}"
                    )
                    failed += 1
            except Exception as exc:
                logger.error(f"[VespaStore] Feed '{doc_id}' failed: {exc}")
                failed += 1

        if failed:
            logger.warning(f"[VespaStore] {failed}/{len(ids)} documents failed")
        return failed == 0

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Retrieve documents similar to *query_text*.

        Keyword argument ``search_mode`` overrides ``default_search_mode``
        for this call: ``"semantic"`` | ``"keyword"`` | ``"hybrid"``.
        """
        mode = kwargs.get("search_mode", self._default_search_mode)

        try:
            if mode == "keyword":
                return self._keyword_query(query_text, top_k)

            # semantic and hybrid both require an embedding
            query_emb = self._embed_fn([query_text])[0]
            if mode == "hybrid":
                return self._hybrid_query(query_text, query_emb, top_k)
            return self._semantic_query(query_emb, top_k)

        except Exception as exc:
            logger.error(f"[VespaStore] Query failed: {exc}")
            return []

    def count(self) -> int:
        """Return the total number of documents stored in this schema."""
        try:
            resp = self._app.query(
                body={
                    "yql": f"select * from {self._schema} where true",
                    "hits": 0,
                }
            )
            return (
                resp.json
                .get("root", {})
                .get("fields", {})
                .get("totalCount", 0)
            )
        except Exception as exc:
            logger.error(f"[VespaStore] count() failed: {exc}")
            return 0

    # ------------------------------------------------------------------ #
    # Private query helpers                                                #
    # ------------------------------------------------------------------ #

    def _semantic_query(
        self, query_emb: List[float], top_k: int
    ) -> List[SearchResult]:
        yql = (
            f"select id, content, metadata "
            f"from {self._schema} "
            f"where ({{targetHits:{top_k}}}nearestNeighbor(embedding, embedding))"
        )
        body = {
            "yql": yql,
            "input.query(embedding)": {"values": query_emb},
            "ranking": "semantic",
            "hits": top_k,
        }
        return self._parse_response(self._app.query(body=body))

    def _keyword_query(self, query_text: str, top_k: int) -> List[SearchResult]:
        # Use @q parameter binding to safely inject the query string
        yql = (
            f"select id, content, metadata "
            f"from {self._schema} "
            f"where content contains @q"
        )
        body = {
            "yql": yql,
            "q": query_text,
            "ranking": "keyword",
            "hits": top_k,
        }
        return self._parse_response(self._app.query(body=body))

    def _hybrid_query(
        self, query_text: str, query_emb: List[float], top_k: int
    ) -> List[SearchResult]:
        # Retrieve more ANN candidates than needed, then re-rank
        ann_hits = top_k * 2
        yql = (
            f"select id, content, metadata "
            f"from {self._schema} "
            f"where ({{targetHits:{ann_hits}}}nearestNeighbor(embedding, embedding)) "
            f"or (content contains @q)"
        )
        body = {
            "yql": yql,
            "q": query_text,
            "input.query(embedding)": {"values": query_emb},
            "ranking": "hybrid",
            "hits": top_k,
        }
        return self._parse_response(self._app.query(body=body))

    # ------------------------------------------------------------------ #
    # Private utilities                                                    #
    # ------------------------------------------------------------------ #

    def _parse_response(self, response) -> List[SearchResult]:
        results: List[SearchResult] = []
        for hit in response.hits or []:
            fields = hit.get("fields", {})
            results.append(
                SearchResult(
                    id=fields.get("id", ""),
                    content=fields.get("content", ""),
                    score=round(float(hit.get("relevance", 0.0)), 4),
                    metadata=self._deserialize_metadata(fields.get("metadata")),
                )
            )
        return results

    @staticmethod
    def _deserialize_metadata(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _deploy(self, container_name: str) -> None:
        """
        Build and deploy the Vespa application package to a running container.

        Uses pyvespa's VespaDocker helper, which talks to the config server
        on port 19071.  The Docker container must already be running
        (``docker compose up -d``).
        """
        try:
            from vespa.deployment import VespaDocker
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                FieldSet,
                HNSW,
                RankProfile,
                Schema,
            )
        except ImportError as exc:
            raise ImportError(
                "pyvespa[full] is required for auto-deploy. "
                "Run: pip install 'pyvespa>=0.45'"
            ) from exc

        dim = self._embedding_dim
        tensor_type = f"tensor<float>(x[{dim}])"

        schema = Schema(
            name=self._schema,
            document=Document(
                fields=[
                    Field(
                        name="id",
                        type="string",
                        indexing=["summary", "attribute"],
                    ),
                    Field(
                        name="content",
                        type="string",
                        indexing=["summary", "index"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="metadata",
                        type="string",
                        indexing=["summary"],
                    ),
                    Field(
                        name="embedding",
                        type=tensor_type,
                        indexing=["summary", "attribute", "index"],
                        attribute=["distance-metric: angular"],
                        ann=HNSW(
                            distance_metric="angular",
                            max_links_per_node=16,
                            neighbors_to_explore_at_insert=200,
                        ),
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["content"])],
            rank_profiles=[
                RankProfile(
                    name="keyword",
                    first_phase="bm25(content)",
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(embedding)", tensor_type)],
                    first_phase="closeness(field, embedding)",
                ),
                RankProfile(
                    name="hybrid",
                    inherits="semantic",
                    first_phase=(
                        "0.7 * closeness(field, embedding) + 0.3 * bm25(content)"
                    ),
                ),
            ],
        )

        app_package = ApplicationPackage(
            name=self._application,
            schema=[schema],
        )

        logger.info(
            f"[VespaStore] Deploying schema '{self._schema}' "
            f"to container '{container_name}' …"
        )
        try:
            vespa_docker = VespaDocker.from_container_name_or_id(container_name)
            self._app = vespa_docker.deploy(
                application_package=app_package,
                max_wait=120,
            )
            logger.info("[VespaStore] Schema deployed successfully.")
        except Exception as exc:
            logger.error(f"[VespaStore] Schema deployment failed: {exc}")
            raise
