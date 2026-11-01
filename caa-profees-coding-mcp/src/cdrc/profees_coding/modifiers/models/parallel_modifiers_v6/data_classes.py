import json
from typing import Any, Iterable

from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSIONS = "prompt_versions"
    PROMPT_VERSION = "prompt_version"
    PROMPT_VERSION_GY = "prompt_version_gy"
    PROMPT_VERSION_95 = "prompt_version_95"
    PROMPT_VERSION_P1_P6 = "prompt_version_p1_p6"
    PROMPT_VERSION_50_LT_RT = "prompt_version_50_lt_rt"
    BRANCH_SELECTION_RETRY_ENABLED = "branch_selection_retry_enabled"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    NUM_REPEATS = "num_repeats"
    EHR_DATA_IN_SCOPE = "ehr_data_in_scope"
    MODIFIER_RELEVANT_EHR_SCOPE = "modifier_relevant_ehr_scope"


class PROMPTS(StringEnum):
    PARALLEL_MODIFIERS = "parallel_modifiers"


DEFAULT_EHR_DATA_IN_SCOPE = [
    "patient_sex",
    "patient_age",
    "payer_name",
    "encounter_metadata",
    "health_plan",
    "facility_info",
    "provider_info",
    "lab_info",
    "anaesthesia_staff_info",
    "anaesthesia_concurrent_staff_info",
]

ALLOWED_EHR_DATA_IN_SCOPE = set(DEFAULT_EHR_DATA_IN_SCOPE)


MODIFIER_RELEVANT_EHR_DATA_IN_SCOPE = {
    # Facility/off-campus modifiers: facility_info was the strongest unique
    # signal in the EHR ablations and is required for PO/PN.
    "PO": ["facility_info"],
    "PN": ["facility_info"],
    # Teaching/provider-role modifiers.
    "GC": ["provider_info"],
    # Lab routing and waived-test modifiers.
    "90": ["lab_info"],
    "QW": ["payer_name", "facility_info", "lab_info"],
    # Anesthesia staffing modifiers.
    "QK": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "QX": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    # Anesthesia physical status modifiers mostly use note/problem evidence, but
    # anesthesia staff context is the most relevant structured EHR support.
    "P1": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "P2": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "P3": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "P4": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "P5": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    "P6": ["provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info"],
    # Telehealth modifier: usually note-driven, with payer/plan/encounter
    # metadata as the targeted structured support.
    "95": ["payer_name", "encounter_metadata", "health_plan"],
    # E/M distinct-service modifier is mostly note/order driven; keep only the
    # targeted same-day service context seen in ablations.
    "25": ["encounter_metadata", "lab_info"],
    # Payer/coverage modifiers.
    "GY": ["payer_name", "health_plan"],
    "UD": ["payer_name", "health_plan"],
    # Therapy discipline and assistant modifiers.
    "GP": ["encounter_metadata", "provider_info"],
    "GO": ["encounter_metadata", "provider_info"],
    "GN": ["encounter_metadata", "provider_info"],
    "CQ": ["encounter_metadata", "provider_info"],
    "CO": ["encounter_metadata", "provider_info"],
    # Laterality/bilateral modifiers are note/procedure driven in the ablation
    # pass, so no structured EHR fields are included by default.
    "50": [],
    "LT": [],
    "RT": [],
}


def _coerce_ehr_data_items(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_EHR_DATA_IN_SCOPE)

    if isinstance(raw, bool):
        return list(DEFAULT_EHR_DATA_IN_SCOPE) if raw else []

    if isinstance(raw, str):
        stripped = raw.strip()
        lowered = stripped.lower()
        if not stripped or lowered in {"default", "all", "true", "yes", "on"}:
            return list(DEFAULT_EHR_DATA_IN_SCOPE)
        if lowered in {"none", "false", "no", "off"}:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in stripped.strip("[]").split(",")]
            return _coerce_ehr_data_items(parsed)
        return [item.strip() for item in stripped.split(",")]

    if isinstance(raw, Iterable):
        return [str(item).strip() for item in raw]

    return [str(raw).strip()]


def parse_ehr_data_in_scope(raw: Any) -> list[str]:
    selected: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()

    for item in _coerce_ehr_data_items(raw):
        normalized = str(item or "").strip()
        if not normalized:
            continue
        if normalized not in ALLOWED_EHR_DATA_IN_SCOPE:
            unknown.append(normalized)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)

    if unknown:
        allowed = ", ".join(DEFAULT_EHR_DATA_IN_SCOPE)
        raise ValueError(f"Unsupported ehr_data_in_scope value(s): {', '.join(unknown)}. " f"Allowed values: {allowed}, all, default, none")

    return selected


def get_ehr_data_in_scope(config: dict) -> list[str]:
    configurable = (config.get("configurable", {}) if isinstance(config, dict) else {}) or {}
    raw = configurable.get(CONFIG.EHR_DATA_IN_SCOPE)
    return parse_ehr_data_in_scope(raw)


def modifier_relevant_ehr_scope_enabled(config: dict) -> bool:
    configurable = (config.get("configurable", {}) if isinstance(config, dict) else {}) or {}
    raw = configurable.get(CONFIG.MODIFIER_RELEVANT_EHR_SCOPE, True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _normalize_modifier(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _ordered_intersection(candidates: Iterable[str], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    selected: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in allowed_set or item in seen:
            continue
        seen.add(item)
        selected.append(item)
    return selected


def get_relevant_ehr_data_in_scope(config: dict, modifier_names: Iterable[str]) -> list[str]:
    globally_allowed = get_ehr_data_in_scope(config)
    if not modifier_relevant_ehr_scope_enabled(config):
        return globally_allowed

    relevant: list[str] = []
    seen: set[str] = set()
    unknown_modifier_seen = False

    for modifier_name in modifier_names or []:
        modifier = _normalize_modifier(modifier_name)
        if modifier not in MODIFIER_RELEVANT_EHR_DATA_IN_SCOPE:
            unknown_modifier_seen = True
            continue
        for field in MODIFIER_RELEVANT_EHR_DATA_IN_SCOPE[modifier]:
            if field not in seen:
                seen.add(field)
                relevant.append(field)

    if unknown_modifier_seen and not relevant:
        relevant = list(DEFAULT_EHR_DATA_IN_SCOPE)

    return _ordered_intersection(DEFAULT_EHR_DATA_IN_SCOPE, _ordered_intersection(relevant, globally_allowed))


def get_relevant_ehr_data_in_scope_for_modifier_groups(config: dict, modifier_groups: Iterable[Iterable[str]]) -> list[str]:
    globally_allowed = get_ehr_data_in_scope(config)
    if not modifier_relevant_ehr_scope_enabled(config):
        return globally_allowed

    relevant: list[str] = []
    seen: set[str] = set()

    for modifier_group in modifier_groups or []:
        for field in get_relevant_ehr_data_in_scope(config, modifier_group):
            if field not in seen:
                seen.add(field)
                relevant.append(field)

    return _ordered_intersection(DEFAULT_EHR_DATA_IN_SCOPE, _ordered_intersection(relevant, globally_allowed))
