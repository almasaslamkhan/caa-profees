"""
Small FAISS backend for HCPCS retrieval artifacts.

The runtime keeps FAISS and embedding resolution isolated here so the retrieval
service stays focused on ranking and payload assembly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


def resolve_embedding_model(model_name: str, *, proxy: Optional[str] = None) -> Any:
    try:
        from cdrc.common.embeddings_utils import get_embedding_models  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
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
                f"Unable to resolve embedding model '{model_name}'. Neither the Codex " "embedding registry nor HuggingFace embeddings are available."
            ) from (embedding_registry_error or exc)
        raise RuntimeError(f"Embedding model '{model_name}' is not registered and HuggingFace embeddings are unavailable locally.") from exc

    return HuggingFaceEmbeddings(model_name=model_name)


def _require_numpy():
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for HCPCS retrieval.") from exc
    return np


def _require_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required for HCPCS retrieval.") from exc
    return faiss


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    metadata: dict[str, Any]
    rank: int


@dataclass
class LoadedVectorStore:
    path: Path
    manifest: dict[str, Any]
    index: Any
    metadata: list[dict[str, Any]]


def load_vector_store(index_dir: str | Path) -> LoadedVectorStore:
    faiss = _require_faiss()
    path = Path(index_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found under {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = path / str(manifest.get("metadata_file") or "metadata.jsonl")
    index_path = path / str(manifest.get("faiss_index") or "index.faiss")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    metadata: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                metadata.append(json.loads(line))

    return LoadedVectorStore(
        path=path,
        manifest=manifest,
        index=faiss.read_index(str(index_path)),
        metadata=metadata,
    )


def embed_text(text: str, *, model_name: str, proxy: str | None = None) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Text is empty; cannot build a retrieval query.")

    embedder = resolve_embedding_model(model_name, proxy=proxy)
    if hasattr(embedder, "embed_query"):
        return list(embedder.embed_query(text))
    if hasattr(embedder, "embed_documents"):
        return list(embedder.embed_documents([text])[0])
    raise AttributeError(f"Embedding model '{model_name}' does not expose embed_query or embed_documents.")


def search_vector_store(
    store: LoadedVectorStore,
    query_vector: Sequence[float],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    np = _require_numpy()
    faiss = _require_faiss()

    if not query_vector:
        raise ValueError("Query vector is empty.")

    vec = np.array([query_vector], dtype="float32")
    if bool(store.manifest.get("normalize_embeddings", True)):
        faiss.normalize_L2(vec)

    scores, indices = store.index.search(vec, top_k)
    hits: list[RetrievalHit] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        metadata = store.metadata[idx] if idx < len(store.metadata) else {"vector_index": idx}
        hits.append(RetrievalHit(score=float(score), metadata=dict(metadata), rank=rank))
    return hits


__all__ = [
    "LoadedVectorStore",
    "RetrievalHit",
    "embed_text",
    "load_vector_store",
    "resolve_embedding_model",
    "search_vector_store",
]
