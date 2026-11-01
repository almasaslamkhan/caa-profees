# cdrc/profees_coding/hcpcs_codes/VDB/common/build.py
from __future__ import annotations

from typing import Literal

from .schema import GCodeLabel, GCodeVDBRecord


def extract_gcode_label(gold: dict, g_code: str) -> GCodeLabel:
    """
    Reads gold.procedure and returns YES if the selected G code exists (HCPCS), else NO.
    """
    has_code = any(p.get("system") == "HCPCS" and p.get("code") == g_code for p in gold.get("procedure", []))
    return "YES" if has_code else "NO"


def build_gcode_vdb_record(
    *,
    example_id: str,
    g_code: str,
    justification_text: str,
    g_code_label: GCodeLabel,
    counterfactual: str | None,
    prompt_version: str,
    model_version: str,
    source: str = "",
) -> GCodeVDBRecord:
    """
    Creates a schema-validated VDB record. The embedding will be computed over `justification_text`.
    """
    return GCodeVDBRecord(
        record_id=f"{example_id}_{g_code}_{g_code_label}",
        example_id=example_id,
        justification_text=justification_text,
        g_code=g_code,
        g_code_label=g_code_label,
        counterfactual=counterfactual,
        metadata={
            "prompt_version": prompt_version,
            "model_version": model_version,
            "source": source,
        },
    )
