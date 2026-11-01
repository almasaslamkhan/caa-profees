from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactManifest:
    target_code: str
    build_id: str
    created_at: str
    summary_model_name: str | None
    model_kwargs: dict[str, Any]
    summary_prompt_path: str | None
    summary_prompt_hash: str | None
    dataset_paths: list[str]
    dataset_fingerprint: str | None
    embedding_model: str
    vector_count: int
    normalize_embeddings: bool
    metadata_file: str
    faiss_index: str
    summary_cache_file: str | None = None
    full_sidecar: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        target_code: str,
        build_id: str,
    ) -> "ArtifactManifest":
        return cls(
            target_code=str(payload.get("target_code") or payload.get("feature_name") or target_code),
            build_id=str(payload.get("build_id") or build_id),
            created_at=str(payload.get("created_at") or ""),
            summary_model_name=payload.get("summary_model_name"),
            model_kwargs=dict(payload.get("model_kwargs") or {}),
            summary_prompt_path=payload.get("summary_prompt_path"),
            summary_prompt_hash=payload.get("summary_prompt_hash"),
            dataset_paths=[str(path) for path in payload.get("dataset_paths") or []],
            dataset_fingerprint=payload.get("dataset_fingerprint"),
            embedding_model=str(payload["embedding_model"]),
            vector_count=int(payload.get("vector_count", 0)),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            metadata_file=str(payload.get("metadata_file") or "metadata.jsonl"),
            faiss_index=str(payload.get("faiss_index") or "index.faiss"),
            summary_cache_file=payload.get("summary_cache_file"),
            full_sidecar=payload.get("full_sidecar"),
        )


@dataclass(frozen=True)
class ArtifactSelection:
    target_code: str
    build_id: str
    source: str
    artifact_root: Path
    build_dir: Path
    index_dir: Path
    manifest_path: Path
    manifest: ArtifactManifest
    summary_cache_path: Path | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact_manifest(
    manifest_path: str | Path,
    *,
    target_code: str,
    build_id: str | None = None,
) -> ArtifactManifest:
    path = Path(manifest_path)
    payload = _read_json(path)
    return ArtifactManifest.from_payload(
        payload,
        target_code=target_code,
        build_id=build_id or path.parent.name,
    )


def _build_selection(
    *,
    manifest_path: Path,
    target_code: str,
    build_id: str,
    source: str,
    artifact_root: Path,
    build_dir: Path,
    summary_cache_path: Path | None,
) -> ArtifactSelection:
    manifest = load_artifact_manifest(manifest_path, target_code=target_code, build_id=build_id)
    return ArtifactSelection(
        target_code=manifest.target_code,
        build_id=manifest.build_id,
        source=source,
        artifact_root=artifact_root,
        build_dir=build_dir,
        index_dir=build_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        summary_cache_path=summary_cache_path,
    )


def _registry_path(artifact_root: Path) -> Path:
    return artifact_root / "registry.json"


def _resolve_from_registry(
    *,
    artifact_root: Path,
    target_code: str,
    build_id: str | None = None,
) -> ArtifactSelection | None:
    registry_path = _registry_path(artifact_root)
    if not registry_path.exists():
        return None

    registry_payload = _read_json(registry_path)
    builds = list(registry_payload.get("builds") or [])
    matching = [
        build for build in builds if str(build.get("target_code") or "").upper() == target_code.upper() and str(build.get("status") or "") == "ready"
    ]
    if build_id:
        matching = [build for build in matching if str(build.get("build_id") or "") == build_id]
    matching.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    if not matching:
        return None

    build = matching[0]
    build_dir = artifact_root / str(build["artifact_dir"])
    manifest_path = build_dir / "manifest.json"
    summary_cache_path = build_dir / str(build.get("summary_cache_file") or "summaries.jsonl")
    if not summary_cache_path.exists():
        summary_cache_path = None
    return _build_selection(
        manifest_path=manifest_path,
        target_code=target_code,
        build_id=str(build["build_id"]),
        source="latest_registry" if build_id is None else "registry_pin",
        artifact_root=artifact_root,
        build_dir=build_dir,
        summary_cache_path=summary_cache_path,
    )


def _resolve_legacy_selection(
    *,
    vdb_dir: str | Path,
    target_code: str,
    summary_cache_path: str | Path | None = None,
) -> ArtifactSelection:
    build_dir = Path(vdb_dir)
    manifest_path = build_dir / "manifest.json"
    cache_path = Path(summary_cache_path) if summary_cache_path else None
    return _build_selection(
        manifest_path=manifest_path,
        target_code=target_code,
        build_id=f"legacy-{build_dir.name}",
        source="legacy_pin",
        artifact_root=build_dir.parent,
        build_dir=build_dir,
        summary_cache_path=cache_path if cache_path and cache_path.exists() else None,
    )


def resolve_artifact(config: dict[str, Any], *, target_code: str) -> ArtifactSelection | None:
    artifact_root_raw = str(config.get("artifact_root") or "").strip()
    artifact_root = Path(artifact_root_raw) if artifact_root_raw else None
    artifact_build_id = str(config.get("artifact_build_id") or "").strip() or None
    artifact_selection = str(config.get("artifact_selection") or "latest_registry").strip().lower()
    vdb_dir = str(config.get("vdb_dir") or "").strip() or None
    summary_cache_path = str(config.get("summary_cache_path") or "").strip() or None

    if artifact_root and artifact_build_id:
        selection = _resolve_from_registry(
            artifact_root=artifact_root,
            target_code=target_code,
            build_id=artifact_build_id,
        )
        if selection is not None:
            return selection

    if artifact_root and artifact_selection == "latest_registry":
        selection = _resolve_from_registry(artifact_root=artifact_root, target_code=target_code)
        if selection is not None:
            return selection

    if vdb_dir:
        return _resolve_legacy_selection(
            vdb_dir=vdb_dir,
            target_code=target_code,
            summary_cache_path=summary_cache_path,
        )

    if artifact_root:
        return _resolve_from_registry(artifact_root=artifact_root, target_code=target_code)

    return None


def register_ready_artifact(
    *,
    artifact_root: str | Path,
    build_dir: str | Path,
    manifest: ArtifactManifest,
) -> Path:
    artifact_root_path = Path(artifact_root)
    artifact_root_path.mkdir(parents=True, exist_ok=True)
    registry_path = _registry_path(artifact_root_path)

    if registry_path.exists():
        registry_payload = _read_json(registry_path)
    else:
        registry_payload = {"schema_version": REGISTRY_SCHEMA_VERSION, "builds": []}

    builds = [build for build in registry_payload.get("builds") or [] if str(build.get("build_id")) != manifest.build_id]
    relative_build_dir = Path(build_dir).relative_to(artifact_root_path)
    builds.append(
        {
            "build_id": manifest.build_id,
            "target_code": manifest.target_code,
            "artifact_dir": str(relative_build_dir),
            "created_at": manifest.created_at,
            "summary_model_name": manifest.summary_model_name,
            "embedding_model": manifest.embedding_model,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "vector_count": manifest.vector_count,
            "summary_cache_file": manifest.summary_cache_file,
            "status": "ready",
        }
    )
    builds.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    registry_payload["schema_version"] = REGISTRY_SCHEMA_VERSION
    registry_payload["builds"] = builds
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")
    return registry_path


__all__ = [
    "ArtifactManifest",
    "ArtifactSelection",
    "REGISTRY_SCHEMA_VERSION",
    "load_artifact_manifest",
    "register_ready_artifact",
    "resolve_artifact",
]
