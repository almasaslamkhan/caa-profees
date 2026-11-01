import json
from argparse import Namespace
from collections import Counter, defaultdict
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
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.nodes.modifier_qw import MEDICARE_PAYER_NAMES
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes

ALLOWED_MODIFIERS = ("QW", "90")


class ModifierDecision(BaseModel):
    code: str = Field(..., description="Selected modifier code, either 'QW' or '90'.")
    justification: str = Field(..., description="Brief reasoning for applying the modifier")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT code")
    modifiers: List[ModifierDecision] = Field(default_factory=list, description="Modifiers selected for the CPT code")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Selected modifiers by CPT code")


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


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_qw_90: Optional[ModifierOutput]


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


def _merge_responses(responses: List[ModifierOutput]) -> ModifierOutput:
    if not responses:
        raise ValueError("No modifier responses provided")

    if len(responses) == 1:
        single = responses[0]
        filtered = [
            CodeAndModifier(
                cpt_code=_normalize_code(cam.cpt_code),
                modifiers=[modifier for modifier in (cam.modifiers or []) if _normalize_code(modifier.code) in ALLOWED_MODIFIERS],
            )
            for cam in single.code_and_modifiers
            if any(_normalize_code(mod.code) in ALLOWED_MODIFIERS for mod in (cam.modifiers or []))
        ]
        return ModifierOutput(code_and_modifiers=filtered)

    threshold = ceil(len(responses) / 2)
    modifiers_by_cpt: Dict[str, List[ModifierDecision]] = defaultdict(list)
    counts_by_cpt: Dict[str, Counter[str]] = defaultdict(Counter)

    for response in responses:
        for cam in response.code_and_modifiers:
            code = _normalize_code(cam.cpt_code)
            seen_this_code = set()
            for modifier in cam.modifiers:
                mod_code = _normalize_code(modifier.code)
                if mod_code not in ALLOWED_MODIFIERS or mod_code in seen_this_code:
                    continue
                seen_this_code.add(mod_code)
                counts_by_cpt[code][mod_code] += 1
                modifiers_by_cpt[code].append(modifier)

    aggregated_cam: List[CodeAndModifier] = []
    for code, counter in counts_by_cpt.items():
        selected_modifiers: List[ModifierDecision] = []
        for mod_code in ALLOWED_MODIFIERS:
            if counter.get(mod_code, 0) < threshold:
                continue
            example = next(
                (m for m in modifiers_by_cpt[code] if _normalize_code(m.code) == mod_code),
                None,
            )
            if example:
                selected_modifiers.append(example)

        if selected_modifiers:
            aggregated_cam.append(CodeAndModifier(cpt_code=code, modifiers=selected_modifiers))

    return ModifierOutput(code_and_modifiers=aggregated_cam)


def _resolve_qw_90_conflict(modifiers: List[Any]) -> List[Modifier]:
    selected_by_code: Dict[str, Any] = {}

    for modifier in modifiers or []:
        data = _dump(modifier)
        mod_code = _normalize_code(data.get("code") or getattr(modifier, "code", None))
        if mod_code not in ALLOWED_MODIFIERS:
            continue
        selected_by_code.setdefault(mod_code, modifier)

    if "QW" in selected_by_code and "90" in selected_by_code:
        return [_to_modifier(selected_by_code["90"])]

    selected: List[Modifier] = []
    for mod_code in ALLOWED_MODIFIERS:
        modifier = selected_by_code.get(mod_code)
        if modifier:
            selected.append(_to_modifier(modifier))
    return selected


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

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
    }


def _build_additional_info(state: ModifierGraphState, config: RunnableConfig) -> str:
    additional_info = []
    payer_names = state.get("payer_name")
    if payer_names:
        normalized_payers = []
        for payer in _extract_payer_names(state, config):
            normalized_payers.append(payer + ", Medicare: " + str(_normalize_code(payer) in MEDICARE_PAYER_NAMES))
        additional_info.append("- Payer Name:")
        additional_info.append(",".join(normalized_payers))

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
        metadata = state.get(ehr_data)
        if metadata:
            ehr_data_name = ehr_data.replace("_", " ").capitalize()
            additional_info.append(f"- {ehr_data_name}:")
            if isinstance(metadata, list):
                for item in metadata:
                    additional_info.append(item.model_dump_json(indent=2))
            else:
                additional_info.append(metadata.model_dump_json(indent=2))
    return "\n".join(additional_info)


@track_tokens("model_usage")
def modifier_90_qw_classifier(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    prompt_versions = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])
    prompt_version = prompt_versions["QW_90"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_qw_90"][prompt_version]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=3,
    )

    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(),
        additional_info=_build_additional_info(state, config),
    )
    chain = prompt | model | parser

    cpt_codes = get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))
    cpt_codes = [_normalize_code(code) for code in cpt_codes if _normalize_code(code)]
    if not cpt_codes:
        return {
            "em": _apply_modifiers(state.get("em"), {}, EMItem),
            "procedure": _apply_modifiers(state.get("procedure"), {}, Procedure, clear_linked=True),
        }

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "cpt_code": ",".join(cpt_codes),
            "orders": state.get("orders_display"),
        }
    )
    merged = _merge_responses([response])

    proc_modifiers: Dict[str, List[Modifier]] = {}
    for cam in merged.code_and_modifiers:
        code = _normalize_code(cam.cpt_code)
        selected_modifiers = _resolve_qw_90_conflict(cam.modifiers)
        if selected_modifiers:
            proc_modifiers[code] = selected_modifiers

    return {
        "em": _apply_modifiers(state.get("em"), {}, EMItem),
        "procedure": _apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
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
