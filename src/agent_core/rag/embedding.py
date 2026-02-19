"""
agent_core/rag/embedding.py  —  ChromaDB embedding function wrappers
=====================================================================
Implements the ChromaDB embedding_function interface.
Supported providers:
  - chroma  : ChromaDB built-in embedding (no extra setup)
  - ollama  : Ollama local embedding model (e.g. nomic-embed-text)
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """
    Wraps an Ollama local embedding model for use as a ChromaDB embedding_function.
    Uses the Ollama OpenAI-compatible /v1/embeddings endpoint.

    Parameters
    ----------
    model : str
        Ollama embedding model name (default: "nomic-embed-text")
    base_url : str | None
        Ollama server base URL. Falls back to OLLAMA_BASE_URL env var,
        then "http://localhost:11434/v1".
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = None,
    ):
        self.model = model
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if not raw_url.rstrip("/").endswith("/v1"):
            raw_url = raw_url.rstrip("/") + "/v1"
        self._base_url = raw_url
        logger.info(f"[OllamaEmbedding] initialized (model={model}, url={self._base_url})")

    def name(self) -> str:
        """ChromaDB EmbeddingFunction interface method."""
        return f"ollama/{self.model}"

    def embed_query(self, input) -> List[List[float]]:
        """Query text embedding (ChromaDB _embed(is_query=True) interface)."""
        texts = list(input) if not isinstance(input, list) else input
        texts = [str(t) for t in texts]
        logger.debug(f"[OllamaEmbedding] embed_query: {len(texts)} texts")
        return self(texts)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Convert a list of texts to embedding vectors.
        Implements the ChromaDB EmbeddingFunction interface.

        Parameters
        ----------
        input : List[str]
            Texts to embed

        Returns
        -------
        List[List[float]]
            Embedding vectors
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(base_url=self._base_url, api_key="ollama")
        response = client.embeddings.create(model=self.model, input=input)
        embeddings = [item.embedding for item in response.data]

        dim = len(embeddings[0]) if embeddings else 0
        logger.debug(f"[OllamaEmbedding] {len(embeddings)} texts embedded (dim={dim})")
        return embeddings


def get_embedding_function(provider: str, model: str = None):
    """
    Return an appropriate embedding_function for the given provider.

    Parameters
    ----------
    provider : str
        "chroma" or "ollama"
    model : str | None
        Embedding model name. None selects the provider default.

    Returns
    -------
    callable | None
        embedding_function for ChromaDB get_or_create_collection().
        Returns None for "chroma" to use ChromaDB's built-in embedding.
    """
    if provider == "chroma":
        logger.info("[Embedding] Using ChromaDB built-in embedding")
        return None

    if provider == "ollama":
        embed_model = model or "nomic-embed-text"
        return OllamaEmbeddingFunction(model=embed_model)

    raise ValueError(
        f"Unsupported embed_provider: '{provider}'. Supported: chroma, ollama"
    )


def collection_name_for(base_name: str, provider: str) -> str:
    """
    Return a collection name scoped to the embedding provider.
    Different embedding spaces cannot share a collection, so a suffix
    is appended when the provider is not the default ("chroma").

    Examples
    --------
    collection_name_for("phoenix-rag-docs", "chroma")  → "phoenix-rag-docs"
    collection_name_for("phoenix-rag-docs", "ollama")  → "phoenix-rag-docs-ollama"
    """
    if provider == "chroma":
        return base_name
    return f"{base_name}-{provider}"
