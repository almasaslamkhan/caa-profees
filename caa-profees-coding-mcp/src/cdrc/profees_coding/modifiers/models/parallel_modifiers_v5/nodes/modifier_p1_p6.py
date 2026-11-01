import json
from argparse import Namespace
from collections import Counter, defaultdict
from math import ceil
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v5.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v5.nodes.modifiers import MEDICARE_PAYER_NAMES
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes, is_anesthesia_cpt

P_STATUS_MODIFIERS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
P_STATUS_ORDER = {code: idx for idx, code in enumerate(P_STATUS_MODIFIERS, start=1)}


class EligibleCPTOutput(BaseModel):
    cpt_codes: List[str] = Field(default_factory=list, description="Only anesthesia CPT codes eligible to be evaluated for P1-P6 modifiers.")


class AssignedModifier(BaseModel):
    code: str = Field(..., description="Assigned modifier code (P1-P6).")
    justification: Optional[str] = Field(None, description="Brief reasoning for applying the modifier.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT code.")
    modifiers: List[AssignedModifier] = Field(default_factory=list, description="Assigned modifiers for the CPT code.")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Assigned modifiers by CPT code.")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    payer_name: Optional[List[str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_p1_p6: Optional[ModifierOutput]
    eligible_p1_p6_cpt_codes: Optional[List[str]]


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _upsert_p_modifier(mods: List[Any], mj: AssignedModifier) -> List[Any]:
    chosen_code = _normalize_code(getattr(mj, "code", None) or _dump(mj).get("code"))
    if chosen_code not in P_STATUS_ORDER:
        return list(mods or [])

    chosen_justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    output_mods: List[Any] = []
    seen_chosen = False

    for mod in list(mods or []):
        mod_data = _dump(mod)
        mod_code = _normalize_code(mod_data.get("code") or getattr(mod, "code", None))

        if mod_code in P_STATUS_ORDER and mod_code != chosen_code:
            continue

        if mod_code == chosen_code:
            seen_chosen = True
            existing_justification = (mod_data.get("justification") or "").strip()
            mod_data["code"] = chosen_code
            mod_data["justification"] = existing_justification or chosen_justification
            output_mods.append(Modifier(**mod_data))
            continue

        output_mods.append(mod)

    if not seen_chosen:
        output_mods.append(Modifier(code=chosen_code, justification=chosen_justification))

    return output_mods


def _select_best_modifier(modifiers: List[AssignedModifier], allowed_modifiers: Set[str]) -> Optional[AssignedModifier]:
    valid = [m for m in (modifiers or []) if _normalize_code(m.code) in allowed_modifiers]
    if not valid:
        return None
    return max(valid, key=lambda m: P_STATUS_ORDER.get(_normalize_code(m.code), -1))


def _extract_payer_names(state: ModifierGraphState, config: RunnableConfig) -> List[str]:
    state_value = state.get("payer_name")
    if isinstance(state_value, list):
        names = [str(x).strip() for x in state_value if str(x).strip()]
        if names:
            return names
    if isinstance(state_value, str):
        raw = state_value.strip()
        if raw:
            return [raw]

    cfg_value = (config.get("configurable", {}) or {}).get("payer")
    if isinstance(cfg_value, list):
        return [str(x).strip() for x in cfg_value if str(x).strip()]
    if isinstance(cfg_value, str) and cfg_value.strip():
        return [cfg_value.strip()]
    return []


@track_tokens("model_usage")
def modifier_p1_p6_decision_node(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> Dict[str, ModifierOutput]:
    eligible_cpt_codes = [_normalize_code(code) for code in (state.get("eligible_p1_p6_cpt_codes") or []) if _normalize_code(code)]
    if eligible_cpt_codes:
        eligible_cpt_codes = list(dict.fromkeys(eligible_cpt_codes))
    else:
        cpt_codes = [_normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code]
        eligible_cpt_codes = []
        seen: Set[str] = set()
        for code in cpt_codes:
            if not is_anesthesia_cpt(code) or code in seen:
                continue
            seen.add(code)
            eligible_cpt_codes.append(code)

    if not eligible_cpt_codes:
        # print(f"No eligible codes for modifier P1-P6.")
        return {"output_p1_p6": ModifierOutput()}

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["P1-P6"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_p1_p6"][prompt_version]

    additional_info = []
    payer_names = state.get("payer_name")
    if payer_names:
        _payer_names = []
        for payer in _extract_payer_names(state, config):
            _payer_names.append(payer + ", Medicare: " + str(payer in MEDICARE_PAYER_NAMES))
        additional_info.append("- Payer Name:")
        additional_info.append(",".join(_payer_names))

    encounter_metadata = state.get("encounter_metadata")
    if encounter_metadata:
        additional_info.append("- Encounter Metadata:")
        additional_info.append(encounter_metadata.model_dump_json(indent=2))

    health_plan = state.get("health_plan")
    if health_plan:
        additional_info.append("- Health Plan:")
        for fac in health_plan:
            additional_info.append(fac.model_dump_json(indent=2))

    facility_info = state.get("facility_info")
    if facility_info:
        additional_info.append("- Facility Info:")
        for fac in facility_info:
            additional_info.append(fac.model_dump_json(indent=2))

    provider_info = state.get("provider_info")
    if provider_info:
        additional_info.append("- Provider Info:")
        for fac in provider_info:
            additional_info.append(fac.model_dump_json(indent=2))

    lab_info = state.get("lab_info")
    if lab_info:
        additional_info.append("- Lab Info:")
        for fac in lab_info:
            additional_info.append(fac.model_dump_json(indent=2))

    anaesthesia_staff_info = state.get("anaesthesia_staff_info")
    if anaesthesia_staff_info:
        additional_info.append("- Anaesthesia Staff Info:")
        for fac in anaesthesia_staff_info:
            additional_info.append(fac.model_dump_json(indent=2))

    anaesthesia_record_info = state.get("anaesthesia_record_info")
    if anaesthesia_record_info:
        additional_info.append("- Anaesthesia Record Info:")
        for fac in anaesthesia_record_info:
            additional_info.append(fac.model_dump_json(indent=2))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(), additional_info="\n".join(additional_info)
    )
    chain = prompt | model | parser

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            "cpt_code": ",".join(eligible_cpt_codes),
        }
    )
    return {"output_p1_p6": response}


def validate_and_apply_p1_p6(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> ModifierGraphOutput:
    del config

    allowed = set(P_STATUS_MODIFIERS)
    modifier_raw = state.get("output_p1_p6")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decision_by_cpt: Dict[str, AssignedModifier] = {}

    for cam in modifier_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        if not is_anesthesia_cpt(cpt):
            continue

        cam_modifiers = [mj for mj in (cam.modifiers or []) if _normalize_code(mj.code) in allowed]
        best_modifier = _select_best_modifier(cam_modifiers, allowed)
        if best_modifier:
            decision_by_cpt[cpt] = best_modifier

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj:
            mods = _upsert_p_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj:
            mods = _upsert_p_modifier(mods, mj)
        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
    }


def _get_proc_key(p) -> str:
    return (
        getattr(p, "code", None)
        or getattr(p, "cpt_code", None)
        or (p.get("code") if isinstance(p, dict) else None)
        or (p.get("cpt_code") if isinstance(p, dict) else None)
        or str(p)
    )


def _get_em_key(e) -> str:
    return (
        getattr(e, "code", None)
        or getattr(e, "cpt_code", None)
        or (e.get("code") if isinstance(e, dict) else None)
        or (e.get("cpt_code") if isinstance(e, dict) else None)
        or str(e)
    )


def _get_mod(e) -> str:
    return getattr(e, "code", None) or (e.get("code") if isinstance(e, dict) else None) or str(e)


def aggregate_parallel_results(parallel_results: List[ModifierGraphOutput]) -> ModifierGraphOutput:
    if not parallel_results:
        raise ValueError("No results to aggregate")

    n = len(parallel_results)
    threshold = ceil(n / 2)

    em_lookups = []
    proc_lookups = []
    for res in parallel_results:
        em_map = {_get_em_key(em): em for em in (res.get("em") or [])}
        proc_map = {_get_proc_key(p): p for p in (res.get("procedure") or [])}
        em_lookups.append(em_map)
        proc_lookups.append(proc_map)

    base_em = parallel_results[0].get("em") or []
    aggregated_em: List[EMItem] = []
    for base in base_em:
        key = _get_em_key(base)
        mod_counts: Counter[str] = Counter()
        mod_examples: Dict[str, List[Any]] = defaultdict(list)

        for run_idx in range(n):
            em_in_run = em_lookups[run_idx].get(key)
            if not em_in_run:
                continue
            seen_this_run = set()
            for m in getattr(em_in_run, "modifiers", []) or []:
                mod_code = _get_mod(m)
                if not mod_code or mod_code in seen_this_run:
                    continue
                seen_this_run.add(mod_code)
                mod_counts[mod_code] += 1
                mod_examples[mod_code].append(m)

        majority_mods: List[Any] = []
        for mod_code, cnt in mod_counts.items():
            if cnt >= threshold and mod_examples[mod_code]:
                candidates = mod_examples[mod_code]
                rep = next((c for c in candidates if getattr(c, "justification", None)), candidates[0])
                majority_mods.append(rep)

        d = (_dump(base) or {}).copy()
        d["modifiers"] = majority_mods
        aggregated_em.append(EMItem(**d))

    base_proc = parallel_results[0].get("procedure") or []
    aggregated_proc: List[Procedure] = []
    for base in base_proc:
        key = _get_proc_key(base)
        mod_counts: Counter[str] = Counter()
        mod_examples: Dict[str, List[Any]] = defaultdict(list)

        for run_idx in range(n):
            p_in_run = proc_lookups[run_idx].get(key)
            if not p_in_run:
                continue
            seen_this_run = set()
            for m in getattr(p_in_run, "modifiers", []) or []:
                mod_code = _get_mod(m)
                if not mod_code or mod_code in seen_this_run:
                    continue
                seen_this_run.add(mod_code)
                mod_counts[mod_code] += 1
                mod_examples[mod_code].append(m)

        majority_mods: List[Any] = []
        for mod_code, cnt in mod_counts.items():
            if cnt >= threshold and mod_examples[mod_code]:
                candidates = mod_examples[mod_code]
                rep = next((c for c in candidates if getattr(c, "justification", None)), candidates[0])
                majority_mods.append(rep)

        d = (_dump(base) or {}).copy()
        d["modifiers"] = majority_mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        aggregated_proc.append(Procedure(**d))

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
    }


def build_modifier_p1_p6_subgraph(args: Namespace, modifier_names: List[str]):
    requested = {_normalize_code(name) for name in (modifier_names or [])}
    allowed_modifiers = requested.intersection(P_STATUS_MODIFIERS)
    if not allowed_modifiers:
        raise ValueError(f"modifier_p1_p6 subgraph only supports P1-P6, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_p1_p6_decision", modifier_p1_p6_decision_node)
    workflow.add_node("modifier_p1_p6_validate", validate_and_apply_p1_p6)

    workflow.add_edge(START, "modifier_p1_p6_decision")
    workflow.add_edge("modifier_p1_p6_decision", "modifier_p1_p6_validate")
    workflow.add_edge("modifier_p1_p6_validate", END)

    try:
        num_repeats = int(getattr(args, "num_repeats", 1))
    except Exception:
        num_repeats = 1

    if num_repeats <= 1:
        return workflow.compile()

    def batch_and_aggregate(state: ModifierGraphState, config: RunnableConfig):
        single_flow = workflow.compile()
        invocations = [state for _ in range(num_repeats)]
        cfg = {
            **(config or {}),
            "configurable": {**((config or {}).get("configurable", {}))},
            "max_concurrency": 999,
        }
        parallel_res = single_flow.batch(invocations, cfg, return_exceptions=True)
        if all(isinstance(res, Exception) for res in parallel_res):
            raise ValueError(f"All modifier P1-P6 parallel executions failed: {parallel_res}")
        successful_results = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(successful_results)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
