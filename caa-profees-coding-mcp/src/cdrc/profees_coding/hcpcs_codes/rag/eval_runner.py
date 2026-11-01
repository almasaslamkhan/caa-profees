from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.utils import load_from_yaml, merge_config
from cdrc.profees_coding.hcpcs_codes.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.hcpcs_codes.data.schema import InputSchema
from cdrc.profees_coding.hcpcs_codes.rag.schema import RagEncounterPayloadModel, RetrievalRequestModel
from cdrc.profees_coding.hcpcs_codes.rag.service import RetrievalService


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to RAG runner config YAML.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument(
        "--filter_name",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(AVAILABLE_FILTERS.keys()),
        help=f"Name of filter to apply. Available filters: {list(AVAILABLE_FILTERS.keys())}",
    )
    parser.add_argument("--filter_kwargs", type=str, default=argparse.SUPPRESS, help="JSON string of filter args.")
    parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Debug mode.")
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default=argparse.SUPPRESS,
        help="Set logging verbosity.",
    )
    parser.add_argument("--feature_name", type=str, default=argparse.SUPPRESS, help="Retrieval feature profile to use.")
    parser.add_argument("--candidate_codes", type=str, default=argparse.SUPPRESS, help="Backward-compatible alias for feature selection.")
    parser.add_argument("--summary_cache_path", type=str, default=argparse.SUPPRESS, help="Legacy summary cache path.")
    parser.add_argument("--vdb_dir", type=str, default=argparse.SUPPRESS, help="Legacy FAISS VDB directory.")
    parser.add_argument("--artifact_selection", type=str, default=argparse.SUPPRESS, help="Artifact selection mode.")
    parser.add_argument("--artifact_root", type=str, default=argparse.SUPPRESS, help="Artifact root with registry.json.")
    parser.add_argument("--artifact_build_id", type=str, default=argparse.SUPPRESS, help="Optional pinned artifact build id.")
    parser.add_argument("--embedding_model_override", type=str, default=argparse.SUPPRESS, help="Override embedding model.")
    parser.add_argument("--embedding_proxy", type=str, default=argparse.SUPPRESS, help="Optional embedding proxy.")
    parser.add_argument("--retrieval_top_k", type=int, default=argparse.SUPPRESS, help="Final number of exemplars.")
    parser.add_argument(
        "--retrieval_candidate_pool_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Dense retrieval pool size before reranking.",
    )
    parser.add_argument("--per_label_limit", type=int, default=argparse.SUPPRESS, help="Legacy alias for symmetric positive/negative limits.")
    parser.add_argument("--positive_example_limit", type=int, default=argparse.SUPPRESS, help="Max positive examples.")
    parser.add_argument("--negative_example_limit", type=int, default=argparse.SUPPRESS, help="Max negative examples.")
    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS, help="Run name.")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Output directory.")
    parser.add_argument("--contributor", type=str, default=argparse.SUPPRESS, help="Contributor name.")
    return parser


def _apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    legacy_limit = getattr(args, "per_label_limit", None)
    if legacy_limit is not None:
        if not hasattr(args, "positive_example_limit"):
            setattr(args, "positive_example_limit", legacy_limit)
        if not hasattr(args, "negative_example_limit"):
            setattr(args, "negative_example_limit", legacy_limit)

    defaults = {
        "filter_name": None,
        "filter_kwargs": {},
        "debug": False,
        "logging_level": "INFO",
        "feature_name": "G2211",
        "artifact_selection": "latest_registry",
        "artifact_root": "./cdrc/profees_coding/hcpcs_codes/artifacts/rag/g2211",
        "retrieval_top_k": 4,
        "retrieval_candidate_pool_size": 24,
        "positive_example_limit": 2,
        "negative_example_limit": 2,
        "run_name": "hcpcs-rag",
        "output_dir": "./results_hcpcs/rag",
        "contributor": getpass.getuser(),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def build_retrieval_service(args: argparse.Namespace) -> RetrievalService:
    return RetrievalService(vars(args))


def _load_raw_dataset(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as handle:
        if data_path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        if data_path.endswith(".json"):
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected list payload in {data_path}")
            return payload
    raise ValueError("Unsupported file format.")


def _merge_input_fields(input_item: dict[str, Any], gold_item: dict[str, Any]) -> dict[str, Any]:
    def _merge_field(value, gold_value):
        if isinstance(value, list) or isinstance(gold_value, list):
            base_list = list(value or [])
            gold_list = list(gold_value or [])
            return base_list + gold_list if gold_list else base_list
        return gold_value if value is None else value

    return {field: _merge_field(input_item.get(field), gold_item.get(field)) for field in InputSchema.model_fields.keys()}


def _gold_feature_label(gold_item: dict[str, Any], feature_code: str) -> str | None:
    procedures = gold_item.get("procedure") or []
    if not isinstance(procedures, list):
        return None
    for procedure in procedures:
        if str((procedure or {}).get("code") or "").upper() == feature_code.upper():
            return "YES"
    return "NO" if procedures else None


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_example_id",
        "target_code",
        "query_gold_label",
        "retrieval_status",
        "retrieval_status_reason",
        "query_summary_source",
        "query_signal_count",
        "artifact_build_id",
        "artifact_source",
        "retrieval_rank",
        "retrieved_example_id",
        "retrieved_encounter_id",
        "retrieved_label",
        "retrieved_source",
        "similarity_score",
        "matched_signals",
        "teaching_justification",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _flatten_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = record["retrieval"]
    diagnostics = retrieval.get("diagnostics") or {}
    base = {
        "query_example_id": record.get("query_example_id"),
        "target_code": retrieval.get("target_code") or retrieval.get("feature_code"),
        "query_gold_label": record.get("query_gold_label"),
        "retrieval_status": retrieval.get("retrieval_status"),
        "retrieval_status_reason": retrieval.get("retrieval_status_reason") or "",
        "query_summary_source": (retrieval.get("query_summary") or {}).get("source") or "",
        "query_signal_count": len(retrieval.get("query_signals") or []),
        "artifact_build_id": diagnostics.get("artifact_build_id") or "",
        "artifact_source": diagnostics.get("artifact_source") or "",
    }
    encounters = retrieval.get("retrieved_encounters") or []
    if not encounters:
        return [{**base}]

    rows: list[dict[str, Any]] = []
    for encounter in encounters:
        rows.append(
            {
                **base,
                "retrieval_rank": encounter.get("rank"),
                "retrieved_example_id": encounter.get("example_id") or "",
                "retrieved_encounter_id": encounter.get("encounter_id") or "",
                "retrieved_label": encounter.get("label") or "",
                "retrieved_source": encounter.get("encounter_source") or "",
                "similarity_score": encounter.get("similarity_score"),
                "matched_signals": "|".join(encounter.get("matched_signals") or []),
                "teaching_justification": encounter.get("teaching_justification") or "",
            }
        )
    return rows


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(record["retrieval"].get("retrieval_status") or "unknown" for record in records)
    retrieved_label_counts = Counter()
    selected_counts: list[int] = []
    for record in records:
        retrieval = record["retrieval"]
        retrieved_label_counts.update(retrieval.get("label_distribution") or {})
        selected_counts.append(len(retrieval.get("retrieved_encounters") or []))
    return {
        "num_queries": len(records),
        "status_counts": dict(status_counts),
        "retrieved_label_distribution": dict(retrieved_label_counts),
        "avg_selected_count": round(sum(selected_counts) / len(selected_counts), 4) if selected_counts else 0.0,
    }


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    config = load_from_yaml(args.config_path)
    config.setdefault("filter_name", None)
    config.setdefault("filter_kwargs", {})
    args = merge_config(args, unknown_args, config)
    args = _apply_runtime_defaults(args)

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_debug(args.debug)
    logger.remove()
    level = args.logging_level.upper()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    logger.info("Logger initialized with level: {}", level)

    data = _load_raw_dataset(args.data_path)
    filter_function = AVAILABLE_FILTERS.get(args.filter_name)
    if filter_function is not None:
        original_count = len(data)
        data = filter_function(data, **args.filter_kwargs)
        logger.info("Filtered RAG dataset from {} to {} records using filter '{}'.", original_count, len(data), args.filter_name)

    service = build_retrieval_service(args)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item in data:
        input_item = item.get("input", item)
        gold_item = item.get("gold", {}) or {}
        input_model = InputSchema(**_merge_input_fields(input_item, gold_item))
        request = RetrievalRequestModel(
            feature_name=str(args.feature_name),
            encounter=RagEncounterPayloadModel.model_validate(input_model.model_dump()),
            overrides=vars(args),
        )
        result = service.retrieve(request).model_dump()
        feature_code = str(result.get("target_code") or result.get("feature_code") or args.feature_name)
        record = {
            "query_example_id": str(input_model.example_id) if input_model.example_id is not None else None,
            "query_gold_label": _gold_feature_label(gold_item, feature_code),
            "retrieval": result,
        }
        records.append(record)
        rows.extend(_flatten_rows(record))

    results_path = run_dir / "retrieval_results.jsonl"
    csv_path = run_dir / "retrieval_report.csv"
    summary_path = run_dir / "retrieval_summary.json"
    _write_jsonl(results_path, records)
    _write_csv(csv_path, rows)

    summary_payload = {
        **_build_summary(records),
        "run_name": args.run_name,
        "data_path": args.data_path,
        "feature_name": args.feature_name,
        "contributor": args.contributor,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.info("Wrote RAG outputs to {}", run_dir)
    return summary_payload


if __name__ == "__main__":
    main()
