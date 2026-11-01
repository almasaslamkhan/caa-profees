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
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v3.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v3.nodes.modifiers import MEDICARE_PAYER_NAMES
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes, is_procedure_code

LATERALITY_MODIFIERS: tuple[str, ...] = ("50", "LT", "RT")
LATERALITY_PRIORITY = {"50": 3, "LT": 2, "RT": 1}


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code (50, LT, RT).")
    evidence: Optional[str] = Field(None, description="Snippet from the encounter that supports the decision.")
    justification: Optional[str] = Field(None, description="Brief reasoning for applying or rejecting the modifier.")
    assigned: bool = Field(..., description="True if the modifier should be applied to this CPT code.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT code.")
    modifiers: List[ModifierJustification] = Field(default_factory=list, description="Modifiers assigned to the CPT code.")


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
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_50_lt_rt: Optional[ModifierOutput]


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


def _to_modifier(modifier_like: Any) -> Modifier:
    return Modifier(**_dump(modifier_like))


def _select_best_assigned_modifier(modifiers: List[ModifierJustification]) -> Optional[ModifierJustification]:
    assigned = [m for m in (modifiers or []) if m.assigned]
    if not assigned:
        return None
    return max(assigned, key=lambda m: LATERALITY_PRIORITY.get(_normalize_code(m.code), -1))


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


def _prefix_justification_with_cpt(modifier: ModifierJustification, cpt_code: str) -> ModifierJustification:
    normalized_cpt = _normalize_code(cpt_code)
    data = _dump(modifier).copy()
    justification = data.get("justification") or ""
    prefix = f"{normalized_cpt}: "
    if normalized_cpt and not justification.startswith(prefix):
        data["justification"] = f"{prefix}{justification}" if justification else prefix.rstrip()
    return ModifierJustification(**data)


def _apply_modifiers(items: List[Any], modifiers_by_code: Dict[str, List[Any]], item_type: Any, clear_linked: bool = False):
    output = []
    for item in items or []:
        item_data = _dump(item).copy()
        code = item_data.get("code") or item_data.get("cpt_code") or getattr(item, "code", None) or getattr(item, "cpt_code", None)
        normalized_code = _normalize_code(code)
        item_data["modifiers"] = modifiers_by_code.get(normalized_code, [])
        if clear_linked and "linked_diagnosis" in item_data:
            item_data["linked_diagnosis"] = []
        output.append(item_type(**item_data))
    return output


def _merge_50_lt_rt_response(response: ModifierOutput) -> tuple[ModifierOutput, List[ModifierJustification]]:
    aggregated_cam: List[CodeAndModifier] = []
    unassigned: List[ModifierJustification] = []
    seen_unassigned: set[tuple[str, str]] = set()

    for cam in response.code_and_modifiers or []:
        cpt_code = _normalize_code(cam.cpt_code)
        filtered_modifiers = [modifier for modifier in (cam.modifiers or []) if _normalize_code(modifier.code) in LATERALITY_PRIORITY]
        selected_modifier = _select_best_assigned_modifier(filtered_modifiers)

        if selected_modifier and is_procedure_code(cpt_code):
            aggregated_cam.append(CodeAndModifier(cpt_code=cpt_code, modifiers=[selected_modifier]))

        for modifier in filtered_modifiers:
            mod_code = _normalize_code(_dump(modifier).get("code") or getattr(modifier, "code", None))
            assigned_flag = getattr(modifier, "assigned", False) or _dump(modifier).get("assigned", False)
            key = (cpt_code, mod_code)
            if not mod_code or assigned_flag or key in seen_unassigned:
                continue
            seen_unassigned.add(key)
            unassigned.append(_prefix_justification_with_cpt(modifier, cpt_code))

    return ModifierOutput(code_and_modifiers=aggregated_cam), _merge_unassigned_by_code(unassigned)


@track_tokens("model_usage")
def modifier_50_lt_rt_decision_node(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> Dict[str, ModifierOutput]:
    cpt_codes = [_normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code]
    eligible_cpt_codes: List[str] = []
    seen: Set[str] = set()
    for code in cpt_codes:
        if not is_procedure_code(code) or code in seen:
            continue
        seen.add(code)
        eligible_cpt_codes.append(code)

    if not eligible_cpt_codes:
        # print("No eligible codes for modifier 50, LT, RT")
        return {"output_50_lt_rt": ModifierOutput()}

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["50_LT_RT"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_50_lt_rt"][prompt_version]

    additional_info = []
    payer_names = state.get("payer_name")
    if payer_names:
        _payer_names = []
        for payer in _extract_payer_names(state, config):
            _payer_names.append(payer + ", Medicare: " + str(payer in MEDICARE_PAYER_NAMES))
        additional_info.append("- Payer Name:")
        additional_info.append(",".join(_payer_names))

    for ehr_data in [
        "encounter_metadata",
        "health_plan",
        "facility_info",
        "provider_info",
        "lab_info",
        "anaesthesia_staff_info",
        "anaesthesia_concurrent_staff_info",
        "anaesthesia_record_info",
    ]:
        _metadata = state.get(ehr_data)
        if _metadata:
            ehr_data_name = ehr_data.replace("_", " ").capitalize()
            additional_info.append(f"- {ehr_data_name}:")
            if isinstance(_metadata, list):
                for _m in _metadata:
                    additional_info.append(_m.model_dump_json(indent=2))
            else:
                additional_info.append(_metadata.model_dump_json(indent=2))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(),
        additional_info="\n".join(additional_info),
    )
    chain = prompt | model | parser

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            "cpt_code": ",".join(eligible_cpt_codes),
            "modifier_name": ", ".join(sorted(set(LATERALITY_MODIFIERS), key=lambda m: LATERALITY_PRIORITY.get(m, 0), reverse=True)),
        }
    )
    return {"output_50_lt_rt": response}


def validate_and_apply_50_lt_rt(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> ModifierGraphOutput:
    modifier_raw = state.get("output_50_lt_rt")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))
    merged, unassigned = _merge_50_lt_rt_response(modifier_output)

    proc_modifiers: Dict[str, List[Modifier]] = {}
    for cam in merged.code_and_modifiers:
        mods = [_to_modifier(m) for m in cam.modifiers if getattr(m, "assigned", False) or _dump(m).get("assigned", False)]
        code = _normalize_code(cam.cpt_code)
        if is_procedure_code(code):
            proc_modifiers[code] = mods

    return {
        "em": _apply_modifiers(state.get("em"), {}, EMItem),
        "procedure": _apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": unassigned,
    }


def _merge_unassigned_by_code(unassigned: List[ModifierJustification]) -> List[ModifierJustification]:
    if not unassigned:
        return []
    grouped: Dict[str, List[ModifierJustification]] = defaultdict(list)
    for item in unassigned:
        mod_code = _normalize_code(_dump(item).get("code") or getattr(item, "code", None))
        if mod_code:
            grouped[mod_code].append(item)

    merged: List[ModifierJustification] = []
    for mod_code, items in grouped.items():
        rep = items[0]
        seen_evidence: set[str] = set()
        seen_justification: set[str] = set()
        evidence_parts: List[str] = []
        justification_parts: List[str] = []
        for item in items:
            evidence = _dump(item).get("evidence") or getattr(item, "evidence", None) or ""
            justification = _dump(item).get("justification") or getattr(item, "justification", None) or ""
            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                evidence_parts.append(evidence)
            if justification and justification not in seen_justification:
                seen_justification.add(justification)
                justification_parts.append(justification)

        rep_data = _dump(rep).copy()
        rep_data["code"] = mod_code
        rep_data["assigned"] = False
        if evidence_parts:
            rep_data["evidence"] = "\n".join(evidence_parts)
        if justification_parts:
            rep_data["justification"] = "\n".join(justification_parts)
        merged.append(ModifierJustification(**rep_data))

    return merged


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

    unassigned_by_code: Dict[str, List[Any]] = defaultdict(list)
    for res in parallel_results:
        for it in res.get("unassigned_modifiers") or []:
            code = _get_mod(it)
            if code:
                unassigned_by_code[code].append(it)

    modifier_output: List[Any] = []
    for _, items in unassigned_by_code.items():
        modifier_output.extend(items)

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
        "unassigned_modifiers": _merge_unassigned_by_code(modifier_output),
    }


def build_modifier_50_lt_rt_subgraph(args: Namespace, modifier_names: List[str]):
    requested = {_normalize_code(name) for name in (modifier_names or [])}
    allowed_modifiers = requested.intersection(LATERALITY_MODIFIERS)
    if not allowed_modifiers:
        raise ValueError(f"modifier_50_lt_rt subgraph only supports 50/LT/RT, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_50_lt_rt_decision", modifier_50_lt_rt_decision_node)
    workflow.add_node("modifier_50_lt_rt_validate", validate_and_apply_50_lt_rt)

    workflow.add_edge(START, "modifier_50_lt_rt_decision")
    workflow.add_edge("modifier_50_lt_rt_decision", "modifier_50_lt_rt_validate")
    workflow.add_edge("modifier_50_lt_rt_validate", END)

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
            raise ValueError(f"All modifier 50/LT/RT parallel executions failed: {parallel_res}")
        successful_results = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(successful_results)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
