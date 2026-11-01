import json
import random
from argparse import Namespace
from collections import Counter, defaultdict
from functools import partial
from math import ceil
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AnaesthesiaRecordInfo, AnaesthesiaStaffInfo, EncounterMetaData, FacilityInfo, HealthPlan, LabInfo, ProviderInfo
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v3.data_classes import CONFIG, PROMPTS
from cdrc.profees_coding.modifiers.models.utils import (
    MODIFIERS_IN_SCOPE,
    get_cpt_codes,
    get_cpt_codes_description,
    get_guidelines_strings,
    get_modifiers_in_scope,
    is_em_code,
)

MEDICARE_PAYER_NAMES = {
    "MEDICARE",
    "MEDICARE FFS",
    "CMS",
    "MEDICARE PART B",
    "MEDICARE PART A",
    # companies that offers medicare plans
    "BCBS",
    "UHC",
    "UMR",
    "HEALTHPARTNERS",
    "WPS",
    "MVA",
    "AETNA",
}


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code, e.g. '25', 'LT', etc.")
    evidence: str = Field(..., description="Snippet from the encounter that supports the decision")
    justification: str = Field(..., description="Brief reasoning for applying or rejecting the modifier")
    assigned: bool = Field(..., description="True if the modifier should be applied to this CPT code")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT code")
    modifiers: List[ModifierJustification] = Field(default_factory=list, description="Modifiers assigned to the CPT code")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Assigned modifiers by CPT code")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]

    payer_name: List[str]
    encounter_metadata: Optional[EncounterMetaData]
    health_plan: Optional[List[HealthPlan]]
    facility_info: Optional[List[FacilityInfo]]
    provider_info: Optional[List[ProviderInfo]]
    lab_info: Optional[List[LabInfo]]
    anaesthesia_staff_info: Optional[List[AnaesthesiaStaffInfo]]
    anaesthesia_record_info: Optional[List[AnaesthesiaRecordInfo]]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    pass


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    # num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    # if num_repeats > 1:
    #     model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def _to_modifier(modifier_like: Any) -> Modifier:
    return Modifier(**_dump(modifier_like))


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


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


def _merge_responses(responses: List[ModifierOutput]) -> tuple[ModifierOutput, List[ModifierJustification]]:
    if not responses:
        raise ValueError("No modifier responses provided")
    if len(responses) == 1:
        single = responses[0]
        unassigned_single: List[ModifierJustification] = []
        seen: set[tuple[str, str]] = set()
        for cam in single.code_and_modifiers:
            cpt_code = _normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = _normalize_code(_dump(modifier).get("code") or getattr(modifier, "code", None))
                assigned_flag = getattr(modifier, "assigned", False) or _dump(modifier).get("assigned", False)
                key = (cpt_code, mod_code)
                if not mod_code or assigned_flag or key in seen:
                    continue
                seen.add(key)
                unassigned_single.append(_prefix_justification_with_cpt(modifier, cpt_code))
        return single, _merge_unassigned_by_code(unassigned_single)

    threshold = ceil(len(responses) / 2)
    modifiers_by_cpt: Dict[str, List[ModifierJustification]] = defaultdict(list)
    counts_by_cpt: Dict[str, Counter[str]] = defaultdict(Counter)

    for response in responses:
        for cam in response.code_and_modifiers:
            code = _normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = _normalize_code(getattr(modifier, "code", None) or _dump(modifier).get("code"))
                assigned = getattr(modifier, "assigned", False) or _dump(modifier).get("assigned", False)
                if not mod_code or not assigned:
                    continue
                counts_by_cpt[code][mod_code] += 1
                modifiers_by_cpt[code].append(modifier)

    aggregated_cam: List[CodeAndModifier] = []
    for code, counter in counts_by_cpt.items():
        selected_modifiers: List[ModifierJustification] = []
        for mod_code, count in counter.items():
            if count < threshold:
                continue
            example = next(
                (m for m in modifiers_by_cpt[code] if _normalize_code(_dump(m).get("code") or getattr(m, "code", None)) == mod_code),
                None,
            )
            if example:
                selected_modifiers.append(example)
        aggregated_cam.append(CodeAndModifier(cpt_code=code, modifiers=selected_modifiers))

    assigned_codes = {
        (_normalize_code(cam.cpt_code), _normalize_code(_dump(m).get("code") or getattr(m, "code", None)))
        for cam in aggregated_cam
        for m in cam.modifiers
    }

    unassigned: List[ModifierJustification] = []
    seen_unassigned: set[tuple[str, str]] = set()
    for response in responses:
        for cam in response.code_and_modifiers:
            cpt_code = _normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = _normalize_code(_dump(modifier).get("code") or getattr(modifier, "code", None))
                assigned_flag = getattr(modifier, "assigned", False) or _dump(modifier).get("assigned", False)
                key = (cpt_code, mod_code)
                if not mod_code or assigned_flag or key in assigned_codes or key in seen_unassigned:
                    continue
                seen_unassigned.add(key)
                unassigned.append(_prefix_justification_with_cpt(modifier, cpt_code))

    # Attach derived unassigned list to output via a lightweight wrapper
    result = ModifierOutput(code_and_modifiers=aggregated_cam)
    return result, _merge_unassigned_by_code(unassigned)


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
    for code, items in unassigned_by_code.items():
        modifier_output.extend(items)

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
        "unassigned_modifiers": _merge_unassigned_by_code(modifier_output),
    }


@track_tokens("model_usage")
def modifiers_classifier(state: ModifierGraphState, config: RunnableConfig, modifier_names: List[str]) -> ModifierGraphOutput:
    prompt_versions = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])
    mod_str = "_".join(modifier_names)
    if mod_str in prompt_versions:
        prompt_version = prompt_versions[mod_str]
    else:
        prompt_version = prompt_versions["ALL"]

    prompt_template = config["configurable"][CONFIG.PROMPTS][f"modifier_{mod_str}".lower()][prompt_version]

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

    parser = PydanticOutputParserWithRetries(pydantic_object=ModifierOutput, correction_model=model, num_retries=3)

    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(),
        additional_info="\n".join(additional_info),
    )
    from cdrc.profees_coding.modifiers.models.utils import print_runnable

    chain = prompt | model | parser

    cpt_codes = get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "cpt_code": ",".join(cpt_codes),
            "orders": state.get("orders_display"),
        }
    )

    merged, unassigned = _merge_responses([response])

    em_modifiers: Dict[str, List[Modifier]] = {}
    proc_modifiers: Dict[str, List[Modifier]] = {}
    for cam in merged.code_and_modifiers:
        mods = [_to_modifier(m) for m in cam.modifiers if getattr(m, "assigned", False) or _dump(m).get("assigned", False)]
        code = _normalize_code(cam.cpt_code)
        if is_em_code(code):
            em_modifiers[code] = mods
        else:
            proc_modifiers[code] = mods

    return {
        "em": _apply_modifiers(state.get("em"), em_modifiers, EMItem),
        "procedure": _apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": unassigned,
    }


def build_modifiers_subgraph(args: Namespace, modifier_names: List[str]):
    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)

    classifier = partial(modifiers_classifier, modifier_names=modifier_names)
    workflow.add_node("classifier", classifier)
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)

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
            **config,
            "configurable": {**config.get("configurable", {})},
            "max_concurrency": 999,
        }
        parallel_res = single_flow.batch(invocations, cfg, return_exceptions=True)
        if all(isinstance(res, Exception) for res in parallel_res):
            raise ValueError(f"All parallel executions failed, no valid results to aggregate, {parallel_res}")
        parallel_res = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(parallel_res)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
