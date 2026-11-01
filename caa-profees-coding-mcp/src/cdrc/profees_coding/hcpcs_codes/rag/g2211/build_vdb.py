"""Compatibility wrapper for the G2211 retrieval artifact builder."""

from cdrc.profees_coding.hcpcs_codes.rag.builder.g2211 import (
    build_metadata_record,
    build_teaching_justification,
    flatten_summary_compact,
    flatten_summary_full,
    generate_summary,
)
from cdrc.profees_coding.hcpcs_codes.rag.builder.g2211 import load_encounters_from_json as load_encounters_from_JSON
from cdrc.profees_coding.hcpcs_codes.rag.builder.service import DEFAULT_CONFIG_PATH, build_artifact, get_parser, main

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "build_artifact",
    "build_metadata_record",
    "build_teaching_justification",
    "flatten_summary_compact",
    "flatten_summary_full",
    "generate_summary",
    "get_parser",
    "load_encounters_from_JSON",
    "main",
]


if __name__ == "__main__":
    main()
