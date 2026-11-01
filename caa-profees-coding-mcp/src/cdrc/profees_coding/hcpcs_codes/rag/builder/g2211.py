from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from cdrc.common.llm_utils import get_llm
from cdrc.profees_coding.hcpcs_codes.rag.ranking import build_signals_from_sections
from cdrc.profees_coding.hcpcs_codes.rag.summaries import parse_summary_sections


def load_encounters_from_json(file_paths: list[str], *, target_code: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in file_paths:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{file_path} must contain a list of encounters")

        for encounter in payload:
            input_item = encounter.get("input", {}) or {}
            gold_item = encounter.get("gold", {}) or {}
            procedures = gold_item.get("procedure", []) or []
            label = "YES" if any(str((procedure or {}).get("code") or "").upper() == target_code.upper() for procedure in procedures) else "NO"
            records.append(
                {
                    "encounter_id": str(input_item.get("example_id") or ""),
                    "soap_note": str(input_item.get("text") or ""),
                    "orders_display": input_item.get("orders_display"),
                    "patient_type": input_item.get("patient_type"),
                    "patient_age": input_item.get("patient_age"),
                    "patient_gender": input_item.get("patient_sex"),
                    "encounter_source": input_item.get("source"),
                    "encounter_metadata": input_item.get("encounter_metadata") or {},
                    "label": label,
                }
            )
    return records


def load_prompt_config(prompt_path: str) -> dict[str, Any]:
    import yaml

    with open(prompt_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_summary(
    *,
    encounter: dict[str, Any],
    model_name: str,
    model_kwargs: dict[str, Any],
    prompt_path: str,
    proxy: str | None = None,
) -> dict[str, Any]:
    prompts = load_prompt_config(prompt_path)
    prompt_block = prompts["soap_summary"]
    system_prompt = prompt_block["system"]
    human_template = prompt_block["user"]

    llm = get_llm(proxy=proxy)[model_name](**model_kwargs)
    human_prompt = human_template.format(
        encounter_id=encounter["encounter_id"],
        patient_type=encounter.get("patient_type"),
        patient_age=encounter.get("patient_age"),
        patient_gender=encounter.get("patient_gender"),
        encounter_source=encounter.get("encounter_source"),
        gold_g2211=encounter["label"] == "YES",
        soap_note=encounter["soap_note"],
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=human_prompt.strip()),
        ]
    )
    raw_text = response.content if hasattr(response, "content") else str(response)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON output for encounter {encounter['encounter_id']}") from exc


def flatten_summary_compact(
    summary: dict[str, Any],
    max_conditions: int = 10,
    max_actions: int = 10,
    max_symptoms: int = 8,
    max_g2211_notes: int = 6,
    max_continuity: int = 6,
) -> str:
    def take_list(value: Any, limit: int) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value[:limit]]
        return [str(value)]

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    visit_type = summary.get("visit_type", "")
    continuity = take_list(summary.get("continuity_indicators"), max_continuity)
    g2211_notes = take_list(summary.get("notes_relevant_to_G2211"), max_g2211_notes)
    chronic = take_list(summary.get("chronic_conditions"), max_conditions)
    actions = take_list(summary.get("management_actions_today"), max_actions)
    symptoms = take_list(summary.get("acute_or_symptom_conditions"), max_symptoms)

    parts: list[str] = []
    if visit_type:
        parts.append(f"Visit type: {visit_type}")
    if g2211_notes:
        parts.append("G2211 evidence / longitudinal care:")
        parts.append(bullets(g2211_notes))
    if continuity:
        parts.append("Continuity indicators:")
        parts.append(bullets(continuity))
    if chronic:
        parts.append("Key chronic conditions:")
        parts.append(bullets(chronic))
    if actions:
        parts.append("Management actions today:")
        parts.append(bullets(actions))
    if symptoms:
        parts.append("Key acute/symptoms:")
        parts.append(bullets(symptoms))
    return "\n".join(parts).strip()


def flatten_summary_full(summary: dict[str, Any]) -> str:
    def format_list(title: str, items: Any) -> str:
        if not items:
            return ""
        if isinstance(items, list):
            return f"{title}:\n" + "\n".join(f"- {item}" for item in items)
        return f"{title}: {items}"

    parts = []
    if summary.get("visit_type"):
        parts.append(f"Visit type: {summary['visit_type']}")
    parts.append(format_list("Continuity indicators", summary.get("continuity_indicators")))
    parts.append(format_list("Chronic conditions", summary.get("chronic_conditions")))
    parts.append(format_list("Acute/symptom conditions", summary.get("acute_or_symptom_conditions")))
    parts.append(format_list("Management actions today", summary.get("management_actions_today")))
    parts.append(format_list("Notes relevant to G2211", summary.get("notes_relevant_to_G2211")))
    return "\n".join(part for part in parts if part).strip()


def build_teaching_justification(*, encounter_summary: str, label: str) -> str:
    sections = parse_summary_sections(encounter_summary)
    continuity = "; ".join(sections.get("continuity") or [])[:160]
    feature_evidence = "; ".join(sections.get("feature_evidence") or [])[:220]
    if label == "YES":
        if feature_evidence:
            return f"G2211=YES because the encounter shows longitudinal-care evidence: {feature_evidence}"
        if continuity:
            return f"G2211=YES because ongoing responsibility and follow-up are documented: {continuity}"
        return "G2211=YES because the encounter documents ongoing longitudinal chronic-care management."
    if feature_evidence:
        return f"G2211=NO boundary case: {feature_evidence}"
    if continuity:
        return f"G2211=NO boundary case despite follow-up cues: {continuity}"
    return "G2211=NO boundary case because the encounter does not clearly document qualifying longitudinal care."


def build_metadata_record(*, encounter: dict[str, Any], encounter_summary: str, target_code: str) -> dict[str, Any]:
    sections = parse_summary_sections(encounter_summary)
    from cdrc.profees_coding.hcpcs_codes.rag.profiles import G2211_PROFILE

    signals = build_signals_from_sections(
        profile=G2211_PROFILE,
        sections=sections,
        orders_display=str(encounter.get("orders_display") or ""),
        encounter_route="ambulatory",
    )
    label = str(encounter["label"])
    record_id = f"{encounter['encounter_id']}_{target_code}_{label}"
    return {
        "record_id": record_id,
        "example_id": encounter["encounter_id"],
        "encounter_id": encounter["encounter_id"],
        "encounter_source": encounter.get("encounter_source"),
        "target_code": target_code,
        "label": label,
        "g_code": target_code,
        "g_code_label": label,
        "encounter_summary": encounter_summary,
        "summary_compact": encounter_summary,
        "teaching_justification": build_teaching_justification(
            encounter_summary=encounter_summary,
            label=label,
        ),
        "signals": signals,
    }


__all__ = [
    "build_metadata_record",
    "build_teaching_justification",
    "flatten_summary_compact",
    "flatten_summary_full",
    "generate_summary",
    "load_encounters_from_json",
]
