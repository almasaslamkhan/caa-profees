from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from cdrc.common.utils import load_from_yaml, merge_config
from cdrc.profees_coding.hcpcs_codes.rag.artifacts import ArtifactManifest, register_ready_artifact
from cdrc.profees_coding.hcpcs_codes.rag.builder import g2211
from cdrc.profees_coding.hcpcs_codes.rag.profiles import resolve_retrieval_profile
from cdrc.profees_coding.hcpcs_codes.rag.vector_store import resolve_embedding_model

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "config" / "g2211.yaml"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Builder config YAML.")
    parser.add_argument("--feature_name", type=str, default=argparse.SUPPRESS, help="Target retrieval code.")
    parser.add_argument("--data_paths", nargs="+", default=argparse.SUPPRESS, help="Dataset paths.")
    parser.add_argument("--artifact_root", type=str, default=argparse.SUPPRESS, help="Artifact root.")
    parser.add_argument("--build_label", type=str, default=argparse.SUPPRESS, help="Optional build label.")
    parser.add_argument("--model_name", type=str, default=argparse.SUPPRESS, help="Offline summary model.")
    parser.add_argument("--model_kwargs", type=str, default=argparse.SUPPRESS, help="JSON string model kwargs.")
    parser.add_argument("--prompt_path", type=str, default=argparse.SUPPRESS, help="Summary prompt YAML.")
    parser.add_argument("--embedding_model_name", type=str, default=argparse.SUPPRESS, help="Embedding model.")
    parser.add_argument("--embedding_proxy", type=str, default=argparse.SUPPRESS, help="Embedding proxy.")
    parser.add_argument("--logging_level", type=str, default=argparse.SUPPRESS, help="Logging level.")
    return parser


def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "feature_name": "G2211",
        "artifact_root": "./cdrc/profees_coding/hcpcs_codes/artifacts/rag/g2211",
        "build_label": "",
        "logging_level": "INFO",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value).strip("-") or "default"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_fingerprint(data_paths: list[str]) -> str:
    payload_parts: list[str] = []
    for path in data_paths:
        file_path = Path(path)
        stat = file_path.stat()
        payload_parts.append(f"{file_path}:{stat.st_size}:{int(stat.st_mtime)}")
    return _sha256_text("|".join(payload_parts))


def _prompt_hash(prompt_path: str) -> str:
    return _sha256_text(Path(prompt_path).read_text(encoding="utf-8"))


def _build_id(*, model_name: str, embedding_model_name: str, build_label: str) -> tuple[str, str]:
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    pieces = [
        created_at.replace(":", "").replace("+00:00", "Z"),
        _slug(model_name),
        _slug(embedding_model_name),
    ]
    if build_label:
        pieces.append(_slug(build_label))
    return "-".join(pieces), created_at


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_faiss_artifact(
    *,
    rows: list[dict[str, Any]],
    build_dir: Path,
    embedding_model_name: str,
    embedding_proxy: str | None,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    embedder = resolve_embedding_model(embedding_model_name, proxy=embedding_proxy)

    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy and faiss are required to build HCPCS retrieval artifacts.") from exc

    texts = [row["encounter_summary"] for row in rows]
    vectors = embedder.embed_documents(texts)
    vecs = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(build_dir / "index.faiss"))


def build_artifact(config: dict[str, Any]) -> Path:
    feature_name = str(config["feature_name"])
    profile = resolve_retrieval_profile(feature_name=feature_name)
    if profile.target_code != "G2211":
        raise NotImplementedError("This first production implementation supports G2211 artifact builds only.")

    data_paths = [str(path) for path in config.get("data_paths") or []]
    if not data_paths:
        raise ValueError("At least one data path is required to build an artifact.")

    model_name = str(config["model_name"])
    model_kwargs = dict(config.get("model_kwargs") or {})
    prompt_path = str(config["prompt_path"])
    artifact_root = Path(str(config["artifact_root"]))
    embedding_model_name = str(config["embedding_model_name"])
    embedding_proxy = config.get("embedding_proxy")
    build_label = str(config.get("build_label") or "")

    build_id, created_at = _build_id(
        model_name=model_name,
        embedding_model_name=embedding_model_name,
        build_label=build_label,
    )
    build_dir = artifact_root / "builds" / build_id
    build_dir.mkdir(parents=True, exist_ok=True)

    encounters = g2211.load_encounters_from_json(data_paths, target_code=profile.target_code)
    summary_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    full_sidecar_rows: list[dict[str, Any]] = []

    for encounter in encounters:
        summary_obj = g2211.generate_summary(
            encounter=encounter,
            model_name=model_name,
            model_kwargs=model_kwargs,
            prompt_path=prompt_path,
            proxy=embedding_proxy,
        )
        summary_compact = g2211.flatten_summary_compact(summary_obj)
        summary_full = g2211.flatten_summary_full(summary_obj)
        summary_rows.append(
            {
                "encounter_id": encounter["encounter_id"],
                "encounter_source": encounter.get("encounter_source"),
                "target_code": profile.target_code,
                "label": encounter["label"],
                "summary_compact": summary_compact,
                "summary_full": summary_full,
            }
        )
        metadata_rows.append(
            g2211.build_metadata_record(
                encounter=encounter,
                encounter_summary=summary_compact,
                target_code=profile.target_code,
            )
        )
        full_sidecar_rows.append(
            {
                "encounter_id": encounter["encounter_id"],
                "summary_full": summary_full,
            }
        )

    _write_jsonl(build_dir / "summaries.jsonl", summary_rows)
    _write_jsonl(build_dir / "metadata.jsonl", metadata_rows)
    _write_jsonl(build_dir / "full_summary_sidecar.jsonl", full_sidecar_rows)
    _build_faiss_artifact(
        rows=metadata_rows,
        build_dir=build_dir,
        embedding_model_name=embedding_model_name,
        embedding_proxy=embedding_proxy,
    )

    manifest = ArtifactManifest(
        target_code=profile.target_code,
        build_id=build_id,
        created_at=created_at,
        summary_model_name=model_name,
        model_kwargs=model_kwargs,
        summary_prompt_path=prompt_path,
        summary_prompt_hash=_prompt_hash(prompt_path),
        dataset_paths=data_paths,
        dataset_fingerprint=_dataset_fingerprint(data_paths),
        embedding_model=embedding_model_name,
        vector_count=len(metadata_rows),
        normalize_embeddings=True,
        metadata_file="metadata.jsonl",
        faiss_index="index.faiss",
        summary_cache_file="summaries.jsonl",
        full_sidecar="full_summary_sidecar.jsonl",
    )
    (build_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    register_ready_artifact(artifact_root=artifact_root, build_dir=build_dir, manifest=manifest)
    return build_dir


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    config = load_from_yaml(args.config_path)
    args = merge_config(args, unknown_args, config)
    args = _apply_defaults(args)

    logger.remove()
    logger.add(lambda message: print(message, end=""), level=str(args.logging_level).upper())

    build_dir = build_artifact(vars(args))
    logger.info("Built HCPCS retrieval artifact at {}", build_dir)
    return str(build_dir)


if __name__ == "__main__":
    main()
