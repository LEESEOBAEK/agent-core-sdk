# agent_core.rag.infrastructure package
import os

from .base import VectorStore, SearchResult

__all__ = ["VectorStore", "SearchResult", "create_store"]


def create_store(backend: str = None, **kwargs) -> VectorStore:
    """
    Factory that instantiates the appropriate VectorStore backend.

    The active backend is resolved in this order:
    1. ``backend`` argument (explicit)
    2. ``RAG_BACKEND`` environment variable
    3. Default: ``"chroma"``

    Parameters
    ----------
    backend : str, optional
        ``"chroma"`` or ``"vespa"``.
    **kwargs
        Forwarded verbatim to the store constructor.

    Returns
    -------
    VectorStore
        A fully initialised store instance.

    Examples
    --------
    # Use ChromaDB (default)
    store = create_store()

    # Use Vespa via env var
    # RAG_BACKEND=vespa python my_app.py

    # Use Vespa explicitly
    store = create_store("vespa", embedding_dim=768)
    """
    resolved = backend or os.getenv("RAG_BACKEND", "chroma")

    if resolved == "chroma":
        from .chroma_store import ChromaStore
        return ChromaStore(**kwargs)

    if resolved == "vespa":
        from .vespa_store import VespaStore
        return VespaStore(**kwargs)

    raise ValueError(
        f"Unsupported RAG_BACKEND: '{resolved}'. "
        f"Supported values: 'chroma', 'vespa'."
    )
