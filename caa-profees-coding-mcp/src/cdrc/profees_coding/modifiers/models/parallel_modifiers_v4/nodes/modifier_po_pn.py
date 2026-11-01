import json
from argparse import Namespace
from collections import Counter, defaultdict
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Set, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.data.schema import FacilityInfo
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes

PO_PN_MODIFIERS: tuple[str, ...] = ("PO", "PN")


class ModifierDecision(BaseModel):
    code: str = Field(..., description="Selected modifier code (PO or PN).")
    justification: Optional[str] = Field(None, description="Brief reasoning explaining why the modifier is appropriate.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT or HCPCS code evaluated for modifiers PO/PN.")
    modifiers: List[ModifierDecision] = Field(default_factory=list, description="List of modifier decisions for this CPT code.")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Selected modifiers by CPT/HCPCS code.")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    payer_name: Optional[List[str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    facility_info: Optional[List[FacilityInfo]]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_po_pn: Optional[ModifierOutput]


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


def _upsert_po_pn_modifier(mods: List[Any], mj: ModifierDecision, allowed_modifiers: Iterable[str] = PO_PN_MODIFIERS) -> List[Any]:
    chosen_code = _normalize_code(getattr(mj, "code", None) or _dump(mj).get("code"))
    allowed = {_normalize_code(c) for c in (allowed_modifiers or [])}
    if chosen_code not in allowed:
        return list(mods or [])

    justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    updated = list(mods or [])

    for idx, mod in enumerate(updated):
        mod_data = _dump(mod)
        mod_code = _normalize_code(mod_data.get("code") or getattr(mod, "code", None))
        if mod_code != chosen_code:
            continue
        existing_justification = (mod_data.get("justification") or "").strip()
        mod_data["code"] = chosen_code
        mod_data["justification"] = existing_justification or justification
        updated[idx] = Modifier(**mod_data)
        return updated

    updated.append(Modifier(code=chosen_code, justification=justification))
    return updated


def _candidate_cpt_codes(state: ModifierGraphState) -> List[str]:
    cpt_codes = [_normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if _normalize_code(code)]

    deduped_codes: List[str] = []
    seen: Set[str] = set()
    for code in cpt_codes:
        if code in seen:
            continue
        seen.add(code)
        deduped_codes.append(code)
    return deduped_codes


@track_tokens("model_usage")
def modifier_po_pn_decision_node(state: ModifierGraphState, config: RunnableConfig) -> Dict[str, ModifierOutput]:
    candidate_cpt_codes = _candidate_cpt_codes(state)

    if not candidate_cpt_codes:
        return {"output_po_pn": ModifierOutput()}

    # prompt_version = config["configurable"].get("prompt_version_po_pn", "v1")
    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["PO-PN"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_po_pn"][prompt_version]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    chain_inputs = {
        "soap_note": state.get("text"),
        "orders": state.get("orders_display"),
        "facility_info": state.get("facility_info"),
        "cpt_code": ",".join(candidate_cpt_codes),
    }

    # # logging model inputs
    # try:
    #     messages = prompt.format_messages(**chain_inputs)
    #     rendered = [{"role": getattr(m, "type", m.__class__.__name__), "content": getattr(m, "content", None)} for m in messages]
    #     # logger.info(f"LLM_PROMPT modifier_qk_qx prompt_version={prompt_version}\n messages={rendered}")
    #     logger.info(chain_inputs)
    # except Exception as e:
    #     logger.warning(f"Failed to render/log LLM prompt for modifier_qk_qx: {e}")

    response = chain.invoke(chain_inputs)

    return {"output_po_pn": response}


def validate_and_apply_po_pn(
    state: ModifierGraphState, config: RunnableConfig, allowed_modifiers: Optional[Iterable[str]] = None
) -> ModifierGraphOutput:
    del config
    allowed = {_normalize_code(c) for c in (allowed_modifiers or PO_PN_MODIFIERS)}
    allowed_codes = set(_candidate_cpt_codes(state))

    modifier_raw = state.get("output_po_pn")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decision_by_cpt: Dict[str, List[ModifierDecision]] = defaultdict(list)

    for cam in modifier_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        if cpt not in allowed_codes:
            continue

        for mj in cam.modifiers or []:
            mod_code = _normalize_code(mj.code)
            if mod_code in allowed:
                decision_by_cpt[cpt].append(mj)

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        for mj in decision_by_cpt.get(cpt, []):
            mods = _upsert_po_pn_modifier(mods, mj, allowed_modifiers=allowed)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        for mj in decision_by_cpt.get(cpt, []):
            mods = _upsert_po_pn_modifier(mods, mj, allowed_modifiers=allowed)
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


def build_modifier_po_pn_subgraph(args: Namespace, modifier_names: List[str]):
    requested = {_normalize_code(name) for name in (modifier_names or [])}
    allowed_modifiers = requested.intersection(PO_PN_MODIFIERS)
    if not allowed_modifiers:
        raise ValueError(f"modifier_po_pn subgraph only supports PO/PN, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_po_pn_decision", modifier_po_pn_decision_node)
    workflow.add_node(
        "modifier_po_pn_validate",
        lambda state, config: validate_and_apply_po_pn(state, config, allowed_modifiers=allowed_modifiers),
    )

    workflow.add_edge(START, "modifier_po_pn_decision")
    workflow.add_edge("modifier_po_pn_decision", "modifier_po_pn_validate")
    workflow.add_edge("modifier_po_pn_validate", END)

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
            raise ValueError(f"All modifier PO/PN parallel executions failed: {parallel_res}")
        successful_results = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(successful_results)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
