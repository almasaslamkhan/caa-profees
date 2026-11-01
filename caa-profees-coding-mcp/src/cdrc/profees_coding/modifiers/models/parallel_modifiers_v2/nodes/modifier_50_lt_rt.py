from argparse import Namespace
from collections import Counter, defaultdict
from functools import partial
from math import ceil
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers.data_classes import CONFIG
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


def _upsert_laterality_modifier(mods: List[Any], mj: ModifierJustification) -> List[Any]:
    chosen_code = _normalize_code(getattr(mj, "code", None) or _dump(mj).get("code"))
    if chosen_code not in LATERALITY_PRIORITY:
        return list(mods or [])

    chosen_justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    output_mods: List[Any] = []
    seen_chosen = False

    for mod in list(mods or []):
        mod_data = _dump(mod)
        mod_code = _normalize_code(mod_data.get("code") or getattr(mod, "code", None))

        if mod_code in LATERALITY_PRIORITY and mod_code != chosen_code:
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


def _select_best_assigned_modifier(modifiers: List[ModifierJustification]) -> Optional[ModifierJustification]:
    assigned = [m for m in (modifiers or []) if m.assigned]
    if not assigned:
        return None
    return max(assigned, key=lambda m: LATERALITY_PRIORITY.get(_normalize_code(m.code), -1))


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

    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION_50_LT_RT]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_50_lt_rt"][prompt_version]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(format_instructions=parser.get_format_instructions())
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
    allowed = set(LATERALITY_MODIFIERS)
    modifier_raw = state.get("output_50_lt_rt")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decision_by_cpt: Dict[str, ModifierJustification] = {}
    unassigned: List[ModifierJustification] = []

    for cam in modifier_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        cam_modifiers = [mj for mj in (cam.modifiers or []) if _normalize_code(mj.code) in allowed]

        for mj in cam_modifiers:
            if mj.assigned:
                continue
            mj_data = _dump(mj).copy()
            justification = (mj_data.get("justification") or "").strip()
            mj_data["justification"] = f"{cpt}: {justification}" if justification else f"{cpt}:"
            unassigned.append(ModifierJustification(**mj_data))

        best_assigned = _select_best_assigned_modifier(cam_modifiers)
        if best_assigned and is_procedure_code(cpt):
            decision_by_cpt[cpt] = best_assigned

    all_cpt_codes: Set[str] = set()
    for item in (state.get("em") or []) + (state.get("procedure") or []):
        d = _dump(item) or {}
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        if cpt:
            all_cpt_codes.add(cpt)

    for cpt in sorted(all_cpt_codes):
        if is_procedure_code(cpt):
            continue
        for code in sorted(allowed):
            unassigned.append(
                ModifierJustification(
                    code=code,
                    evidence=None,
                    justification=f"{cpt}: Not evaluated for {code} because this is not a procedure code.",
                    assigned=False,
                )
            )

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj and mj.assigned:
            mods = _upsert_laterality_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj and mj.assigned:
            mods = _upsert_laterality_modifier(mods, mj)
        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
        "unassigned_modifiers": _merge_unassigned_by_code((state.get("unassigned_modifiers") or []) + unassigned),
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
