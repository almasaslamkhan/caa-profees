from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

SECTION_ALIASES = {
    "g2211 evidence / longitudinal care": "feature_evidence",
    "notes relevant to g2211": "feature_evidence",
    "continuity indicators": "continuity",
    "key chronic conditions": "chronic_conditions",
    "chronic conditions": "chronic_conditions",
    "management actions today": "management_actions",
    "key acute/symptoms": "acute_symptoms",
    "acute/symptom conditions": "acute_symptoms",
}

EMPTY_SECTIONS = {
    "visit_type": [],
    "feature_evidence": [],
    "continuity": [],
    "chronic_conditions": [],
    "management_actions": [],
    "acute_symptoms": [],
    "raw_context": [],
}


@lru_cache(maxsize=4)
def load_summary_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary cache not found at {cache_path}")

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            key = str(payload.get("encounter_id") or payload.get("example_id") or "")
            if key:
                records[key] = payload
    return records


def lookup_summary_record(example_id: Any, cache_path: str | None) -> dict[str, Any]:
    if example_id is None or not cache_path:
        return {}
    try:
        return load_summary_cache(cache_path).get(str(example_id), {})
    except FileNotFoundError:
        return {}


def parse_summary_sections(summary_text: str) -> dict[str, list[str]]:
    sections = {key: list(value) for key, value in EMPTY_SECTIONS.items()}
    current_section: str | None = None

    for raw_line in (summary_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("visit type:"):
            value = line.split(":", 1)[1].strip()
            if value:
                sections["visit_type"].append(value)
            current_section = None
            continue

        if line.endswith(":"):
            canonical = SECTION_ALIASES.get(line[:-1].strip().lower())
            current_section = canonical
            continue

        if line.startswith("- "):
            item = line[2:].strip()
            if current_section:
                sections[current_section].append(item)
            else:
                sections["raw_context"].append(item)
            continue

        if current_section:
            sections[current_section].append(line)
        else:
            sections["raw_context"].append(line)

    if not any(sections[key] for key in ("feature_evidence", "continuity", "chronic_conditions", "management_actions")) and summary_text:
        sections["raw_context"] = [summary_text.strip()]
    return sections


def _summary_from_additional_input(additional_input: Any) -> dict[str, str]:
    if not isinstance(additional_input, Mapping):
        return {}

    compact = str(
        additional_input.get("retrieval_summary_compact")
        or additional_input.get("summary_compact")
        or additional_input.get("encounter_summary")
        or ""
    ).strip()
    full = str(additional_input.get("retrieval_summary_full") or additional_input.get("summary_full") or compact or "").strip()
    if not compact and not full:
        return {}
    return {
        "summary_compact": compact or full,
        "summary_full": full or compact,
        "source": "additional_input",
    }


def build_summary_bundle(
    *,
    example_id: Any,
    text: str,
    cache_path: str | None,
    additional_input: Any = None,
) -> dict[str, Any]:
    record = {}
    if cache_path and example_id is not None:
        record = lookup_summary_record(example_id, cache_path)
    if not record:
        record = _summary_from_additional_input(additional_input)

    if record:
        summary_compact = str(record.get("summary_compact") or "").strip()
        summary_full = str(record.get("summary_full") or summary_compact).strip()
        source = str(record.get("source") or record.get("encounter_source") or (Path(cache_path).name if cache_path else "cache"))
    else:
        raw_text = (text or "").strip()
        summary_compact = raw_text[:2000]
        summary_full = raw_text
        source = "raw_note_fallback"

    return {
        "summary_compact": summary_compact,
        "summary_full": summary_full,
        "source": source,
        "sections": parse_summary_sections(summary_compact or summary_full),
    }
