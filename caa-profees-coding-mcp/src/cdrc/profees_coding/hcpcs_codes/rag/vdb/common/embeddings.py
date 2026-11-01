"""
Shared embedding helpers for HCPCS VDB tooling.

We keep imports lightweight at module import time so that downstream scripts can
be analysed without requiring LangChain / FAISS to be installed locally. Heavy
dependencies are imported lazily when the resolver is called.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def resolve_embedding_model(model_name: str, *, proxy: Optional[str] = None) -> Any:
    """
    Resolve an embedding model by name.

    Resolution order:
    1. Codex embedding registry (see `cdrc.common.embeddings_utils`).
    2. HuggingFace sentence transformers via LangChain.

    Parameters
    ----------
    model_name:
        The name of the embedding deployment or HuggingFace model.
    proxy:
        Optional HTTP proxy forwarded to the registry.

    Returns
    -------
    langchain.embeddings.base.Embeddings
        A LangChain-compatible embedding object.

    Raises
    ------
    RuntimeError
        If no embedding backend is available locally.
    """

    try:
        from cdrc.common.embeddings_utils import get_embedding_models  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency missing locally
        embedding_registry: Optional[Callable[..., dict[str, Callable[..., Any]]]] = None
        embedding_registry_error = exc
    else:
        embedding_registry = get_embedding_models  # type: ignore[assignment]
        embedding_registry_error = None

    if embedding_registry is not None:
        catalogue = embedding_registry(proxy=proxy)
        if model_name in catalogue:
            return catalogue[model_name]()

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        if embedding_registry is None:
            raise RuntimeError(
                f"Unable to resolve embedding model '{model_name}'. Neither the Codex "
                "embedding registry nor HuggingFace embeddings are available. "
                "Install project dependencies (requests, langchain-community, etc.) "
                "or run the build inside the SDS environment."
            ) from (embedding_registry_error or exc)
        raise RuntimeError(f"Embedding model '{model_name}' is not registered and HuggingFace embeddings " "are unavailable locally.") from exc

    return HuggingFaceEmbeddings(model_name=model_name)
