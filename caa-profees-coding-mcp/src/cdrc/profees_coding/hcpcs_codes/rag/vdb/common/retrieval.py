"""
Utilities for loading VDB manifests and performing FAISS retrieval.

The module intentionally delays importing heavy dependencies (numpy, faiss,
langchain) until runtime so that static analysis tooling can operate without
requiring the full environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .embeddings import resolve_embedding_model
from .schema import GCodeVDBRecord


def _require_numpy():
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "numpy is required for retrieval. Install project dependencies or execute " "inside the SDS runtime before calling retrieval helpers."
        ) from exc
    return np


def _require_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required for retrieval. Install `faiss-cpu` (or the SDS package) " "before calling retrieval helpers.") from exc
    return faiss


@dataclass
class VDBManifest:
    embedding_model: str
    vector_count: int
    normalize_embeddings: bool
    metadata_file: str
    faiss_index: str
    full_sidecar: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VDBManifest":
        return cls(
            embedding_model=payload["embedding_model"],
            vector_count=int(payload["vector_count"]),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            metadata_file=payload["metadata_file"],
            faiss_index=payload["faiss_index"],
            full_sidecar=payload.get("full_sidecar"),
        )


@dataclass
class VDBIndex:
    path: Path
    manifest: VDBManifest
    index: Any
    metadata: List[dict[str, Any]]


@dataclass
class RetrievalHit:
    score: float
    metadata: dict[str, Any]
    rank: int


def load_vdb(vdb_dir: str | Path) -> VDBIndex:
    """
    Load a persisted FAISS VDB directory.

    Parameters
    ----------
    vdb_dir:
        Directory containing `manifest.json`, `index.faiss`, `metadata.jsonl`.
    """

    faiss = _require_faiss()

    vdb_path = Path(vdb_dir)
    manifest_path = vdb_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found under {vdb_path}")

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = VDBManifest.from_json(manifest_payload)

    index_path = vdb_path / manifest.faiss_index
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    metadata_path = vdb_path / manifest.metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    index = faiss.read_index(str(index_path))

    metadata: List[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metadata.append(json.loads(line))

    if metadata and len(metadata) != manifest.vector_count:
        # We do not fail hard but warn via ValueError to highlight mismatch.
        raise ValueError(
            f"Metadata length ({len(metadata)}) does not match manifest.vector_count "
            f"({manifest.vector_count}). Ensure VDB was built with matching artifacts."
        )

    return VDBIndex(path=vdb_path, manifest=manifest, index=index, metadata=metadata)


def embed_summary(summary: str, *, model_name: Optional[str] = None, proxy: Optional[str] = None) -> List[float]:
    """
    Embed a single summary string using the requested embedding model.
    """

    if not summary or not summary.strip():
        raise ValueError("Summary text is empty – cannot embed.")

    if not model_name:
        raise ValueError("An embedding model name must be provided to embed the summary.")

    embedder = resolve_embedding_model(model_name, proxy=proxy)

    if hasattr(embedder, "embed_query"):
        vector = embedder.embed_query(summary)  # type: ignore[no-any-return]
    elif hasattr(embedder, "embed_documents"):
        vector = embedder.embed_documents([summary])[0]  # type: ignore[index,assignment]
    else:  # pragma: no cover
        raise AttributeError(f"Embedding model '{model_name}' does not expose embed_query or embed_documents.")

    return vector


def search_vdb(
    vdb: VDBIndex,
    query_vector: Sequence[float],
    *,
    top_k: int = 6,
) -> List[RetrievalHit]:
    """
    Perform a cosine-similarity search against the FAISS index.

    Returns a list of RetrievalHit sorted by score (descending).
    """

    np = _require_numpy()
    faiss = _require_faiss()

    if not query_vector:
        raise ValueError("Query vector is empty.")

    vec = np.array([query_vector], dtype="float32")
    if vdb.manifest.normalize_embeddings:
        faiss.normalize_L2(vec)

    scores, indices = vdb.index.search(vec, top_k)
    hits: List[RetrievalHit] = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        metadata = vdb.metadata[idx] if idx < len(vdb.metadata) else {"vector_index": idx}
        hits.append(RetrievalHit(score=float(score), metadata=metadata, rank=rank))

    return hits


def load_sidecar(vdb: VDBIndex) -> Iterable[dict[str, Any]]:
    """
    Stream the full-summary sidecar, if present.
    """

    if not vdb.manifest.full_sidecar:
        return []

    sidecar_path = vdb.path / vdb.manifest.full_sidecar
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar file declared but missing: {sidecar_path}")

    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hits_to_gcode_records(
    hits: Sequence[RetrievalHit],
    *,
    g_code: str,
) -> List[GCodeVDBRecord]:
    """
    Convert retrieval hits into `GCodeVDBRecord` instances.

    This assumes the metadata JSONL contains the justification text and labels,
    which may not be true for all VDBs yet. Missing fields default to None.
    """

    records: List[GCodeVDBRecord] = []
    for hit in hits:
        md = hit.metadata or {}
        justification = md.get("justification_text") or md.get("summary_compact") or ""
        metadata_items = {
            k: v
            for k, v in md.items()
            if k
            not in {
                "record_id",
                "example_id",
                "summary_compact",
                "justification_text",
                "g_code",
                "g_code_label",
                "gold_g2211",
            }
        }
        metadata_items["retrieval_score"] = hit.score
        metadata_items["rank"] = hit.rank
        metadata = {str(k): "" if v is None else str(v) for k, v in metadata_items.items()}

        records.append(
            GCodeVDBRecord(
                record_id=md.get("record_id") or md.get("encounter_id"),
                example_id=md.get("example_id") or md.get("encounter_id"),
                justification_text=justification,
                g_code=g_code,
                g_code_label=md.get("g_code_label") or ("YES" if md.get("gold_g2211") else "NO"),
                counterfactual=md.get("counterfactual"),
                metadata=metadata,
            )
        )
    return records


def retrieve_similar_encounters(
    summary_text: str,
    *,
    vdb_dir: str | Path,
    top_k: int = 6,
    embedding_model_override: Optional[str] = None,
    embedding_proxy: Optional[str] = None,
) -> List[RetrievalHit]:
    """
    Convenience wrapper: load VDB, embed the summary, and return top-k hits.
    """

    vdb = load_vdb(vdb_dir)
    model_name = embedding_model_override or vdb.manifest.embedding_model

    query_vector = embed_summary(summary_text, model_name=model_name, proxy=embedding_proxy)
    return search_vdb(vdb, query_vector, top_k=top_k)
