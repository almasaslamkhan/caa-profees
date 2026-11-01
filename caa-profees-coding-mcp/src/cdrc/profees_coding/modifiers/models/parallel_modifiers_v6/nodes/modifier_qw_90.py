import json
from argparse import Namespace
from collections import Counter, defaultdict
from math import ceil
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AnaesthesiaStaffInfo, EncounterMetaData, FacilityInfo, HealthPlan, LabInfo, ProviderInfo
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.modifiers.data.schema import MEDICARE_PAYER_NAMES, EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.data_classes import CONFIG, get_relevant_ehr_data_in_scope
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.model_selection import get_model_and_correction_model
from cdrc.profees_coding.modifiers.models.utils import build_ehr_data, get_cpt_codes

ALLOWED_MODIFIERS = ("QW", "90")


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code, either 'QW' or '90'.")
    evidence: str = Field(..., description="Snippet from the encounter that supports the decision")
    justification: str = Field(
        ...,
        description="Concise decision reason only: less than 20 words, focused on the decisive rule/fact.",
    )
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


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    pass


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


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
                if mod_code not in ALLOWED_MODIFIERS or assigned_flag or key in seen:
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
                if mod_code not in ALLOWED_MODIFIERS or not assigned:
                    continue
                counts_by_cpt[code][mod_code] += 1
                modifiers_by_cpt[code].append(modifier)

    aggregated_cam: List[CodeAndModifier] = []
    for code, counter in counts_by_cpt.items():
        selected_modifiers: List[ModifierJustification] = []
        for mod_code in ALLOWED_MODIFIERS:
            count = counter.get(mod_code, 0)
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
                if mod_code not in ALLOWED_MODIFIERS or assigned_flag or key in assigned_codes or key in seen_unassigned:
                    continue
                seen_unassigned.add(key)
                unassigned.append(_prefix_justification_with_cpt(modifier, cpt_code))

    result = ModifierOutput(code_and_modifiers=aggregated_cam)
    return result, _merge_unassigned_by_code(unassigned)


def _resolve_qw_90_conflict(modifiers: List[Any], cpt_code: str) -> tuple[List[Modifier], List[ModifierJustification]]:
    assigned_by_code: Dict[str, Any] = {}

    for modifier in modifiers or []:
        data = _dump(modifier)
        mod_code = _normalize_code(data.get("code") or getattr(modifier, "code", None))
        if mod_code not in ALLOWED_MODIFIERS:
            continue

        assigned_flag = data.get("assigned")
        if assigned_flag is None and not hasattr(modifier, "assigned"):
            assigned_flag = True
        if not assigned_flag:
            continue

        assigned_by_code.setdefault(mod_code, modifier)

    if "QW" in assigned_by_code and "90" in assigned_by_code:
        suppressed_data = _dump(assigned_by_code["QW"]).copy()
        existing_justification = (suppressed_data.get("justification") or "").strip()
        conflict_note = "Suppressed because modifier 90 was also selected for the same CPT; QW and 90 are mutually exclusive."
        suppressed_data["code"] = "QW"
        suppressed_data["assigned"] = False
        suppressed_data["justification"] = f"{existing_justification} {conflict_note}".strip()
        suppressed = _prefix_justification_with_cpt(ModifierJustification(**suppressed_data), cpt_code)
        return [_to_modifier(assigned_by_code["90"])], [suppressed]

    selected: List[Modifier] = []
    for mod_code in ALLOWED_MODIFIERS:
        modifier = assigned_by_code.get(mod_code)
        if modifier:
            selected.append(_to_modifier(modifier))
    return selected, []


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
                mod_code = _normalize_code(_get_mod(m))
                if mod_code not in ALLOWED_MODIFIERS or mod_code in seen_this_run:
                    continue
                seen_this_run.add(mod_code)
                mod_counts[mod_code] += 1
                mod_examples[mod_code].append(m)

        selected_codes = [mod_code for mod_code in ALLOWED_MODIFIERS if mod_counts.get(mod_code, 0) >= threshold]
        if "QW" in selected_codes and "90" in selected_codes:
            selected_codes = ["90"] if mod_counts["90"] >= mod_counts["QW"] else ["QW"]

        majority_mods: List[Any] = []
        for mod_code in selected_codes:
            if mod_examples[mod_code]:
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
                mod_code = _normalize_code(_get_mod(m))
                if mod_code not in ALLOWED_MODIFIERS or mod_code in seen_this_run:
                    continue
                seen_this_run.add(mod_code)
                mod_counts[mod_code] += 1
                mod_examples[mod_code].append(m)

        selected_codes = [mod_code for mod_code in ALLOWED_MODIFIERS if mod_counts.get(mod_code, 0) >= threshold]
        if "QW" in selected_codes and "90" in selected_codes:
            selected_codes = ["90"] if mod_counts["90"] >= mod_counts["QW"] else ["QW"]

        majority_mods: List[Any] = []
        for mod_code in selected_codes:
            if mod_examples[mod_code]:
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
            code = _normalize_code(_get_mod(it))
            if code in ALLOWED_MODIFIERS:
                unassigned_by_code[code].append(it)

    modifier_output: List[Any] = []
    for _, items in unassigned_by_code.items():
        modifier_output.extend(items)

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
        "unassigned_modifiers": _merge_unassigned_by_code(modifier_output),
    }


@track_tokens("model_usage")
def modifier_90_qw_classifier(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    prompt_versions = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])
    prompt_version = prompt_versions["QW_90"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_qw_90"][prompt_version]

    ehr_info = build_ehr_data(state, config, ehr_list=get_relevant_ehr_data_in_scope(config, ALLOWED_MODIFIERS))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model, correction_model = get_model_and_correction_model(config["configurable"][CONFIG.MODEL_NAME], model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=correction_model,
        num_retries=3,
    )

    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(),
        additional_info=ehr_info,
    )
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

    proc_modifiers: Dict[str, List[Modifier]] = {}
    conflict_unassigned: List[ModifierJustification] = []
    for cam in merged.code_and_modifiers:
        code = _normalize_code(cam.cpt_code)
        selected_modifiers, suppressed = _resolve_qw_90_conflict(cam.modifiers, code)
        if selected_modifiers:
            proc_modifiers[code] = selected_modifiers
        conflict_unassigned.extend(suppressed)

    return {
        "em": _apply_modifiers(state.get("em"), {}, EMItem),
        "procedure": _apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": _merge_unassigned_by_code(unassigned + conflict_unassigned),
    }


def build_modifier_qw_90_subgraph(args: Namespace, modifier_names: List[str]):
    normalized = {_normalize_code(name) for name in (modifier_names or [])}
    if not normalized or not normalized.issubset(set(ALLOWED_MODIFIERS)):
        raise ValueError(f"modifier_90_qw subgraph only supports QW/90, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("classifier", modifier_90_qw_classifier)
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
            raise ValueError(f"All QW/90 parallel executions failed: {parallel_res}")
        parallel_res = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(parallel_res)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
