from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, Field

from cdrc.common.data.schema import AWVHistory, EncounterMetaData
from cdrc.profees_coding.hcpcs_codes.data.schema import HealthPlan, Problem, Procedure

AMBULATORY_ROUTE = "ambulatory"
FACILITY_ROUTE = "facility"
AWV_CODES = {"G0402", "G0438", "G0439"}
HCPCS_IN_SCOPE_CONFIG_KEY = "hcpcs_in_scope"
INPUT_ABLATION_MODE_CONFIG_KEY = "input_ablation_mode"
INPUT_ABLATION_NONE = "none"
INPUT_ABLATION_SOAP_NOTE_ONLY = "soap_note_only"
HCPCS_FAMILIES = {"g2211", "awv", "other_hcpcs"}
HCPCS_SCOPE_ALIASES = {
    "all": HCPCS_FAMILIES,
    "g2211": {"g2211"},
    "g2211_only": {"g2211"},
    "awv": {"awv"},
    "awv_only": {"awv"},
    "awvs": {"awv"},
    "other": {"other_hcpcs"},
    "others": {"other_hcpcs"},
    "other_hcpcs": {"other_hcpcs"},
    "facility": {"other_hcpcs"},
}


class HCPCSInputGraphState(TypedDict, total=False):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders_display: Optional[str]
    health_plan: Optional[List[HealthPlan]]
    awv_history: Optional[List[AWVHistory]]
    encounter_date: Optional[str]
    encounter_metadata: Optional[EncounterMetaData]
    awv_eligibility: Optional[Mapping[str, Any]]
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")
    additional_input: Optional[Mapping[str, Any]] = Field(None, description="Additional input fields")


class HCPCSOutputGraphState(TypedDict, total=False):
    procedure: List[Procedure] = Field(default_factory=list, description="List of HCPCS procedure codes")
    additional_output: Mapping[str, Any]


class HCPCSGraphState(HCPCSInputGraphState, HCPCSOutputGraphState, total=False):
    encounter_route: Literal["ambulatory", "facility"]
    awv_precheck_passed: bool
    g2211_result: Mapping[str, Any]
    awv_result: Mapping[str, Any]
    other_hcpcs_result: Mapping[str, Any]


def _split_scope_items(raw_scope: Any) -> Iterable[str]:
    if raw_scope is None:
        return []
    if isinstance(raw_scope, str):
        return [item.strip() for item in raw_scope.replace(";", ",").split(",")]
    if isinstance(raw_scope, Iterable):
        return [str(item).strip() for item in raw_scope]
    return [str(raw_scope).strip()]


def normalize_hcpcs_in_scope(raw_scope: Any) -> set[str]:
    items = [item.lower() for item in _split_scope_items(raw_scope) if item]
    if not items:
        return set(HCPCS_FAMILIES)

    normalized: set[str] = set()
    for item in items:
        aliases = HCPCS_SCOPE_ALIASES.get(item)
        if aliases is None:
            raise ValueError(f"Unsupported HCPCS scope family '{item}'.")
        normalized.update(aliases)

    return normalized or set(HCPCS_FAMILIES)


def get_configurable(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, Mapping) else config


def is_hcpcs_family_enabled(raw_scope: Any, family: str) -> bool:
    return family in normalize_hcpcs_in_scope(raw_scope)


def is_hcpcs_family_enabled_for_config(config: Mapping[str, Any] | None, family: str) -> bool:
    configurable = get_configurable(config)
    return is_hcpcs_family_enabled(configurable.get(HCPCS_IN_SCOPE_CONFIG_KEY), family)


def normalize_input_ablation_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or INPUT_ABLATION_NONE).strip().lower()
    if mode not in {INPUT_ABLATION_NONE, INPUT_ABLATION_SOAP_NOTE_ONLY}:
        raise ValueError(f"Unsupported input ablation mode '{mode}'.")
    return mode


def _customer_name_from_additional_input(additional_input: Any) -> Optional[str]:
    if isinstance(additional_input, Mapping):
        value = additional_input.get("customer_name")
    elif isinstance(additional_input, BaseModel):
        value = additional_input.model_dump().get("customer_name")
    elif additional_input is not None:
        value = getattr(additional_input, "customer_name", None)
    else:
        value = None

    text = str(value).strip() if value is not None else ""
    return text or None


def apply_input_ablation_mode(state: Mapping[str, Any], config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    configurable = get_configurable(config)
    mode = normalize_input_ablation_mode(configurable.get(INPUT_ABLATION_MODE_CONFIG_KEY))
    if mode == INPUT_ABLATION_NONE:
        return state

    customer_name = _customer_name_from_additional_input(state.get("additional_input"))
    ablated_state: dict[str, Any] = {
        "example_id": state.get("example_id"),
        "text": state.get("text", ""),
        "patient_age": state.get("patient_age"),
        "patient_sex": state.get("patient_sex"),
    }
    if "encounter_route" in state:
        ablated_state["encounter_route"] = state.get("encounter_route")
    if customer_name:
        ablated_state["additional_input"] = {"customer_name": customer_name}
    return ablated_state


def encounter_type_from_state(state: Mapping[str, Any]) -> str:
    metadata = state.get("encounter_metadata")
    if isinstance(metadata, dict):
        encounter_type = metadata.get("type")
    elif isinstance(metadata, BaseModel):
        encounter_type = metadata.model_dump().get("type")
    elif metadata is not None:
        encounter_type = getattr(metadata, "type", None)
    else:
        encounter_type = None
    return str(encounter_type or "").strip().lower()


def route_encounter(state: Mapping[str, Any]) -> dict[str, str]:
    encounter_type = encounter_type_from_state(state)
    if encounter_type in {"inpatient", "hospital", "ed", "emergency", "emergency department"}:
        return {"encounter_route": FACILITY_ROUTE}
    return {"encounter_route": AMBULATORY_ROUTE}


def sort_procedures(items: List[dict]) -> List[dict]:
    def sort_key(item: dict) -> tuple[int, Any]:
        code = str(item.get("code") or "").upper()
        if code == "G2211":
            return (0, 0)
        if code in AWV_CODES:
            return (1, {"G0402": 0, "G0438": 1, "G0439": 2}[code])
        return (2, code)

    return sorted(items, key=sort_key)


def merge_results(state: Mapping[str, Any]) -> dict[str, Any]:
    branch_results = {
        "g2211": state.get("g2211_result"),
        "awv": state.get("awv_result"),
        "other_hcpcs": state.get("other_hcpcs_result"),
    }

    deduped: dict[tuple[str, str], dict] = {}
    additional_output: dict[str, Any] = {}
    customer_name: Optional[str] = None

    for branch_name, result in branch_results.items():
        if not isinstance(result, Mapping):
            continue

        for procedure in result.get("procedure") or []:
            procedure_dict = procedure.model_dump() if hasattr(procedure, "model_dump") else dict(procedure)
            system = str(procedure_dict.get("system") or "").upper()
            code = str(procedure_dict.get("code") or "").upper()
            if not system or not code:
                continue
            deduped.setdefault((system, code), procedure_dict)

        branch_additional = result.get("additional_output")
        if branch_additional:
            additional_output[branch_name] = branch_additional

        if not customer_name:
            customer_name = result.get("customer_name")

    procedures = sort_procedures(list(deduped.values()))
    if customer_name:
        additional_output["customer_name"] = customer_name

    output = {"procedure": procedures}
    if additional_output:
        output["additional_output"] = additional_output
    return output
