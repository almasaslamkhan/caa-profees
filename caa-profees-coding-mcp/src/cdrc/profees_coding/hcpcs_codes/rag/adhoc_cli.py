from __future__ import annotations

import argparse
import sys
from itertools import count
from pathlib import Path
from typing import Callable, TextIO
from uuid import uuid4

from loguru import logger

from cdrc.common.utils import load_from_yaml, merge_config
from cdrc.profees_coding.hcpcs_codes.rag.eval_runner import _apply_runtime_defaults
from cdrc.profees_coding.hcpcs_codes.rag.schema import RagEncounterPayloadModel, RetrievalRequestModel, RetrievalResponseModel
from cdrc.profees_coding.hcpcs_codes.rag.service import RetrievalService

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "runner.yaml"
QUIT_COMMAND = ":quit"
END_TOKEN = "END"
TARGET_CODE = "G2211"
AUTO_END_BLANK_LINE_THRESHOLD = 3


class QuitRequested(Exception):
    pass


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive ad hoc HCPCS retrieval CLI.")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to retrieval config YAML.")
    parser.add_argument("--artifact_root", type=str, default=argparse.SUPPRESS, help="Artifact root with registry.json.")
    parser.add_argument("--artifact_build_id", type=str, default=argparse.SUPPRESS, help="Optional pinned artifact build id.")
    parser.add_argument("--retrieval_top_k", type=int, default=argparse.SUPPRESS, help="Final number of retrieved examples.")
    parser.add_argument("--positive_example_limit", type=int, default=argparse.SUPPRESS, help="Maximum positive examples.")
    parser.add_argument("--negative_example_limit", type=int, default=argparse.SUPPRESS, help="Maximum negative examples.")
    parser.add_argument("--embedding_model_override", type=str, default=argparse.SUPPRESS, help="Override embedding model.")
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default=argparse.SUPPRESS,
        help="Set logging verbosity.",
    )
    return parser


def _configure_logging(level: str | None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=str(level or "INFO").upper())


def load_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = get_parser()
    args, unknown_args = parser.parse_known_args(argv)
    config = load_from_yaml(args.config_path)
    config.setdefault("filter_name", None)
    config.setdefault("filter_kwargs", {})
    merged = merge_config(args, unknown_args, config)
    merged = _apply_runtime_defaults(merged)
    merged.feature_name = TARGET_CODE
    return merged


def build_cli_service(args: argparse.Namespace) -> RetrievalService:
    return RetrievalService(vars(args))


def _write(output: TextIO, text: str = "") -> None:
    print(text, file=output)
    output.flush()


def _read_line(prompt: str, input_func: Callable[[str], str]) -> str:
    value = input_func(prompt)
    if value.strip() == QUIT_COMMAND:
        raise QuitRequested
    return value


def _prompt_note(*, input_func: Callable[[str], str], output: TextIO) -> str:
    while True:
        _write(
            output,
            "Paste SOAP note text. Enter END on its own line to finish. "
            f"{AUTO_END_BLANK_LINE_THRESHOLD} blank lines will also submit. Type :quit to exit.",
        )
        lines: list[str] = []
        blank_line_streak = 0
        while True:
            prompt = "soap> " if not lines else ""
            line = _read_line(prompt, input_func)
            if line.strip() == END_TOKEN:
                break
            if not line.strip():
                blank_line_streak += 1
                lines.append("")
                if lines and blank_line_streak >= AUTO_END_BLANK_LINE_THRESHOLD:
                    while lines and not lines[-1].strip():
                        lines.pop()
                    break
                continue
            blank_line_streak = 0
            lines.append(line)
        note = "\n".join(lines).strip()
        if note:
            return note
        _write(output, "SOAP note cannot be empty. Please try again.")


def _prompt_client_name(*, input_func: Callable[[str], str], output: TextIO) -> str:
    _write(output, "Client name is optional. Leave blank to skip.")
    return _read_line("client> ", input_func).strip()


def _make_example_id(query_index: int) -> str:
    return f"adhoc-{uuid4().hex[:8]}-{query_index}"


def build_request(*, note: str, client_name: str, query_index: int) -> RetrievalRequestModel:
    additional_input = {"customer_name": client_name} if client_name else None
    return RetrievalRequestModel(
        feature_name=TARGET_CODE,
        encounter=RagEncounterPayloadModel(
            example_id=_make_example_id(query_index),
            text=note,
            encounter_metadata={"type": "ambulatory"},
            additional_input=additional_input,
        ),
    )


def _indent_block(text: str, prefix: str = "    ") -> str:
    if not text:
        return ""
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def format_response(response: RetrievalResponseModel) -> str:
    diagnostics = response.diagnostics
    lines = [
        "",
        "=== Retrieval Result ===",
        f"Target code: {response.target_code or response.feature_code}",
        f"Status: {response.retrieval_status}",
    ]
    if response.retrieval_status_reason:
        lines.append(f"Reason: {response.retrieval_status_reason}")
    if diagnostics.artifact_build_id:
        lines.append(f"Artifact build id: {diagnostics.artifact_build_id}")
    lines.append(f"Query summary source: {response.query_summary.source}")
    lines.append("Query signals: " + (", ".join(response.query_signals) if response.query_signals else "(none)"))

    if response.retrieval_status in {"missing_artifact", "error", "not_applicable"}:
        return "\n".join(lines)

    if not response.retrieved_encounters:
        lines.append("No similar encounters found.")
        return "\n".join(lines)

    for encounter in response.retrieved_encounters:
        lines.extend(
            [
                "",
                f"{encounter.rank}. Similarity score: {encounter.similarity_score:.4f}",
                f"   Label: {encounter.label or 'UNKNOWN'}",
                f"   Encounter id: {encounter.encounter_id or '-'}",
                f"   Example id: {encounter.example_id or '-'}",
                f"   Encounter source: {encounter.encounter_source or '-'}",
                "   Matched signals: " + (", ".join(encounter.matched_signals) if encounter.matched_signals else "(none)"),
            ]
        )
        if encounter.encounter_summary:
            lines.append("   Encounter summary:")
            lines.append(_indent_block(encounter.encounter_summary))
        if encounter.teaching_justification:
            lines.append("   Teaching justification:")
            lines.append(_indent_block(encounter.teaching_justification))
    return "\n".join(lines)


def _print_banner(*, args: argparse.Namespace, output: TextIO) -> None:
    _write(output, "HCPCS Interactive Retrieval CLI")
    _write(output, f"Target code: {TARGET_CODE}")
    _write(output, f"Artifact selection: {args.artifact_selection}")
    _write(output, f"Artifact root: {args.artifact_root}")
    _write(output, f"Retrieval top-k: {args.retrieval_top_k}")
    _write(output, "All ad hoc queries are treated as ambulatory encounters.")
    _write(output, "")


def run_repl(
    *,
    service: RetrievalService,
    args: argparse.Namespace,
    input_func: Callable[[str], str],
    output: TextIO,
) -> int:
    _print_banner(args=args, output=output)
    for query_index in count(start=1):
        try:
            note = _prompt_note(input_func=input_func, output=output)
            client_name = _prompt_client_name(input_func=input_func, output=output)
        except QuitRequested:
            _write(output, "Exiting HCPCS retrieval CLI.")
            return 0

        request = build_request(note=note, client_name=client_name, query_index=query_index)
        response = service.retrieve(request)
        _write(output, format_response(response))
        _write(output, "")


def main(
    argv: list[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO | None = None,
) -> int:
    output_stream = output or sys.stdout
    try:
        args = load_cli_args(argv)
        _configure_logging(getattr(args, "logging_level", None))
        service = build_cli_service(args)
        return run_repl(service=service, args=args, input_func=input_func, output=output_stream)
    except KeyboardInterrupt:
        _write(output_stream, "")
        _write(output_stream, "Exiting HCPCS retrieval CLI.")
        return 0


__all__ = [
    "AUTO_END_BLANK_LINE_THRESHOLD",
    "TARGET_CODE",
    "build_cli_service",
    "build_request",
    "format_response",
    "get_parser",
    "load_cli_args",
    "main",
    "run_repl",
]
