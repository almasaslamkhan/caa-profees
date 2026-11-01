import json
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.modifiers.data.schema import MEDICARE_PAYER_NAMES
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.data_classes import CONFIG, get_relevant_ehr_data_in_scope_for_modifier_groups
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.model_selection import get_model_and_correction_model
from cdrc.profees_coding.modifiers.models.utils import dedupe_codes, dump, get_cpt_codes, normalize_code, print_runnable


class ModifierBranchPlanItem(BaseModel):
    branch_name: str = Field(..., description="Exact branch name chosen from the available branch list.")
    eligible_cpt_codes: List[str] = Field(
        default_factory=list,
        description="Only CPT/HCPCS codes from the provided candidate list that should be evaluated by this branch.",
    )
    evidence: str = Field(
        default="",
        description="Short justification for why this branch should be evaluated.",
    )


class ModifierBranchPlanOutput(BaseModel):
    branches: List[ModifierBranchPlanItem] = Field(
        default_factory=list,
        description="Modifier branches that should run for this encounter and the CPT/HCPCS codes each branch should evaluate.",
    )
    # unselected_branches: List[ModifierBranchPlanItem] = Field(
    #     default_factory=list,
    #     description="Modifier branches that should not be run for this encounter and the CPT/HCPCS codes each branch should evaluate.",
    # )


class ModifierBranchPlanState(TypedDict, total=False):
    text: str
    patient_sex: Optional[str]
    patient_age: Optional[int]
    orders_display: Optional[str]
    orders: Optional[List[Any]]
    procedure: List[Any]
    em: List[Any]
    encounter_metadata: Any
    health_plan: Any
    facility_info: Any
    provider_info: Any
    lab_info: Any
    anaesthesia_staff_info: Any
    anaesthesia_concurrent_staff_info: Any
    modifier_branch_plan: Dict[str, List[str]]


def _normalize_branch_name(raw: Any) -> str:
    parts = [normalize_code(part) for part in str(raw or "").split(",")]
    return ",".join([part for part in parts if part])


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


def _branch_selection_retry_enabled(config: RunnableConfig) -> bool:
    raw = (config.get("configurable", {}) or {}).get(CONFIG.BRANCH_SELECTION_RETRY_ENABLED, False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _describe_branch(modifier_names: Sequence[str]) -> str:
    normalized = {normalize_code(name) for name in (modifier_names or [])}

    if normalized == {"50", "LT", "RT"}:
        return (
            "Laterality and bilateral procedure modifiers. "
            "Consider left/right anatomy, bilateral procedures, paired organs, extremities, eyes, ears, breasts, limbs."
        )

    if normalized == {"PO", "PN"}:
        return (
            "Off-campus outpatient hospital versus non-excepted off-campus outpatient modifiers. "
            "Consider place of service, outpatient hospital department, provider-based clinic, off-campus facility context."
        )

    if normalized == {"QK", "QX"}:
        return (
            "Anesthesia medical direction and CRNA service modifiers. "
            "Consider anesthesia staffing, CRNA involvement, medical direction, anesthesiologist supervision."
        )

    if normalized == {"QW", "90"}:
        return (
            "CLIA-waived lab and reference lab modifiers. "
            "Consider lab testing, waived testing, send-out/reference lab, outside laboratory performance."
        )

    if normalized == {"GY", "UD"}:
        return (
            "Statutory exclusion and Medicaid or payer-specific drug/utilization modifiers. "
            "Consider coverage exclusions, payer restrictions, Medicaid-specific drug or utilization context."
        )

    # if normalized == {"GY", "UD"}:
    #     return (
    #         "Modifiers for non-covered services (GY) versus Medicaid or payer-specific drug/program identification (UD). "
    #         "Do NOT use UD for standard professional services (e.g., office visits, psychiatric evaluations like CPT 90791). "
    #         "Use GY when the service is statutorily excluded from Medicare. "
    #         "Use UD ONLY when documentation indicates Medicaid-specific billing, state drug programs, or payer-defined categories "
    #         "(typically drugs, injectables, or supplies). "
    #         "Look for keywords such as Medicaid, drug, injection, supply, NDC, or state program. "
    #         "If none of these are present, do not select UD."
    #     )

    if normalized == {"GP", "GO", "GN", "CQ", "CO"}:
        return "Therapy discipline and assistant modifiers. " "Consider PT, OT, ST, therapy plans of care, therapy assistants, rehab documentation."

    if "95" in normalized:
        return "Telehealth modifier. " "Consider video visits, telemedicine, virtual visits, remote synchronous audiovisual care."

    if normalized.intersection({"P1", "P2", "P3", "P4", "P5", "P6"}):
        return (
            "Anesthesia physical status modifiers. " "Consider anesthesia cases with ASA physical status or perioperative anesthesia documentation."
        )

    return f"Modifier branch for {', '.join(modifier_names)}."


def _serialize_orders(state: ModifierBranchPlanState) -> str:
    orders_display = state.get("orders_display")
    if orders_display:
        return str(orders_display)

    orders = state.get("orders")
    if not orders:
        return "None"

    lines: List[str] = []
    for order in orders:
        payload = dump(order)
        try:
            lines.append(json.dumps(payload, indent=2, default=str))
        except TypeError:
            lines.append(str(payload))
    return "\n".join(lines) or "None"


def _serialize_ehr_context(state: ModifierBranchPlanState, ehr_data_in_scope: Sequence[str]) -> str:
    lines: List[str] = []

    for key in ehr_data_in_scope:
        if key == "patient_sex":
            patient_sex = state.get("patient_sex")
            if patient_sex:
                lines.append(f"patient_sex: {patient_sex}")
            continue

        if key == "patient_age":
            patient_age = state.get("patient_age")
            if patient_age is not None:
                lines.append(f"patient_age: {patient_age}")
            continue

        if key == "payer_name":
            payer_names = state.get("payer_name")
            if payer_names:
                lines.append("payer_name:")
                if isinstance(payer_names, list):
                    for payer in payer_names:
                        if isinstance(payer, dict):
                            payer_name = payer.get("payer_name", "")
                            is_medicare = payer.get("is_medicare")
                            lines.append(f"- payer_name: {payer_name}, is_medicare: {is_medicare}")
                        else:
                            payer_name = str(payer)
                            lines.append(f"- payer_name: {payer_name}, is_medicare: {payer_name in MEDICARE_PAYER_NAMES}")
                else:
                    payer_name = str(payer_names)
                    lines.append(f"- payer_name: {payer_name}, is_medicare: {payer_name in MEDICARE_PAYER_NAMES}")
            continue

        value = state.get(key)
        if not value:
            continue

        lines.append(f"{key}:")
        if isinstance(value, list):
            for item in value:
                payload = dump(item)
                try:
                    lines.append(json.dumps(payload, indent=2, default=str))
                except TypeError:
                    lines.append(str(payload))
        else:
            payload = dump(value)
            try:
                lines.append(json.dumps(payload, indent=2, default=str))
            except TypeError:
                lines.append(str(payload))

    return "\n".join(lines) or "None"


def _build_fallback_plan(branch_modifier_groups: Sequence[Sequence[str]], cpt_codes: Sequence[str]) -> Dict[str, List[str]]:
    normalized_codes = dedupe_codes(cpt_codes)
    if not normalized_codes:
        return {}
    return {",".join(group): list(normalized_codes) for group in branch_modifier_groups if normalized_codes}


def _should_retry_for_sparse_output(
    cleaned_plan: Dict[str, List[str]],
    branch_modifier_groups: Sequence[Sequence[str]],
    all_cpt_codes: Sequence[str],
    state: ModifierBranchPlanState,
    ehr_data_in_scope: Sequence[str],
) -> bool:
    # Simple semantic retry trigger.
    # Tune these thresholds however you want.
    has_meaningful_context = bool(
        (state.get("text") or "").strip() or state.get("orders") or state.get("orders_display") or any(state.get(key) for key in ehr_data_in_scope)
    )
    if not has_meaningful_context:
        return False

    if not all_cpt_codes:
        return False

    if len(branch_modifier_groups) <= 1:
        return False

    return len(cleaned_plan) == 0


def _clean_response_plan(
    response: ModifierBranchPlanOutput,
    available_lookup: Dict[str, str],
    all_cpt_codes: List[str],
) -> Dict[str, List[str]]:
    allowed_codes = set(all_cpt_codes)
    cleaned_plan: Dict[str, List[str]] = {}

    for branch in response.branches or []:
        normalized_branch = _normalize_branch_name(branch.branch_name)
        branch_name = available_lookup.get(normalized_branch)
        if not branch_name:
            continue

        eligible_codes = [code for code in dedupe_codes(branch.eligible_cpt_codes) if code in allowed_codes]

        # Key recall change:
        # If the model selected the branch but failed to attach codes,
        # do not throw the branch away. Keep it alive with all candidate codes.
        if not eligible_codes:
            eligible_codes = list(all_cpt_codes)

        cleaned_plan.setdefault(branch_name, [])
        merged_codes = cleaned_plan[branch_name] + eligible_codes
        cleaned_plan[branch_name] = dedupe_codes(merged_codes)

    return cleaned_plan


def _build_prompt(
    config: RunnableConfig,
    template_variant: str = "primary",
) -> str:
    if template_variant == "retry":
        prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["branch_selection_retry"]
        prompt_template = config["configurable"][CONFIG.PROMPTS]["branch_selection_retry"][prompt_version]

        return prompt_template

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["branch_selection"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["branch_selection"][prompt_version]

    return prompt_template


@track_tokens("model_usage")
def modifier_branch_extraction_node(
    state: ModifierBranchPlanState,
    config: RunnableConfig,
    branch_modifier_groups: Sequence[Sequence[str]],
) -> Dict[str, Dict[str, List[str]]]:
    all_cpt_codes = dedupe_codes(get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure")))
    if not all_cpt_codes:
        return {"modifier_branch_plan": {}}

    available_branches: List[str] = []
    available_lookup: Dict[str, str] = {}
    ehr_data_in_scope = get_relevant_ehr_data_in_scope_for_modifier_groups(config, branch_modifier_groups)

    for group in branch_modifier_groups:
        branch_name = ",".join(group)
        normalized_branch = _normalize_branch_name(branch_name)
        available_lookup[normalized_branch] = branch_name
        available_branches.append(f"- {branch_name}: {_describe_branch(group)}")

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model, correction_model = get_model_and_correction_model(config["configurable"][CONFIG.MODEL_NAME], model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierBranchPlanOutput,
        correction_model=correction_model,
        num_retries=3,
    )

    def invoke_once(template_variant: str) -> ModifierBranchPlanOutput:
        prompt = ChatPromptTemplate.from_template(
            _build_prompt(config, template_variant),
            template_format="jinja2",
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | model | parser
        return chain.invoke(
            {
                "available_branches": "\n".join(available_branches),
                "cpt_codes": ", ".join(all_cpt_codes),
                "soap_note": state.get("text") or "",
                "orders": _serialize_orders(state),
                "ehr_context": _serialize_ehr_context(state, ehr_data_in_scope),
            }
        )

    try:
        first_response = invoke_once("primary")
        cleaned_plan = _clean_response_plan(
            response=first_response,
            available_lookup=available_lookup,
            all_cpt_codes=all_cpt_codes,
        )

        if _branch_selection_retry_enabled(config) and _should_retry_for_sparse_output(
            cleaned_plan=cleaned_plan,
            branch_modifier_groups=branch_modifier_groups,
            all_cpt_codes=all_cpt_codes,
            state=state,
            ehr_data_in_scope=ehr_data_in_scope,
        ):
            retry_response = invoke_once("retry")
            retry_plan = _clean_response_plan(
                response=retry_response,
                available_lookup=available_lookup,
                all_cpt_codes=all_cpt_codes,
            )

            # Union the plans to bias toward recall
            for branch_name, codes in retry_plan.items():
                cleaned_plan.setdefault(branch_name, [])
                cleaned_plan[branch_name] = dedupe_codes(cleaned_plan[branch_name] + codes)

        return {"modifier_branch_plan": cleaned_plan}

    except Exception:
        return {
            "modifier_branch_plan": _build_fallback_plan(
                branch_modifier_groups,
                all_cpt_codes,
            )
        }
