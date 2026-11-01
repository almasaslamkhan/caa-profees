from argparse import Namespace
from collections import Counter, defaultdict
from math import ceil
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers.data_classes import CONFIG, PROMPTS
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v2.nodes.modifiers import MEDICARE_PAYER_NAMES
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes


class ModifierJustification(BaseModel):
    code: str = Field("GY", description="Modifier code. Always 'GY' for this decision task.")
    evidence: Optional[str] = Field(None, description="Direct quote (≤25 words) from the documentation supporting the decision, if applicable.")
    detailed_justification: Optional[str] = Field(
        None, description="Detailed step-by-step reasoning explaining why GY is or is not appropriate, based on Medicare statutory coverage rules."
    )
    justification: Optional[str] = Field(
        None, description="Brief reasoning explaining why GY is or is not appropriate, based on Medicare statutory coverage rules."
    )
    assigned: bool = Field(..., description="True if Modifier GY should be applied; False otherwise.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT or HCPCS code evaluated for Modifier GY.")
    modifiers: List[ModifierJustification] = Field(
        default_factory=list, description="List of modifier decisions for this CPT code (GY only in this task)."
    )


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Modifier GY decisions grouped by CPT code.")


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
    output_gy: Optional[ModifierOutput]


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


def _is_medicare_payer(state: ModifierGraphState, config: RunnableConfig) -> bool:
    return any(_normalize_code(payer) in MEDICARE_PAYER_NAMES for payer in _extract_payer_names(state, config))


def _upsert_gy_modifier(mods: List[Any], mj: ModifierJustification) -> List[Any]:
    justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    updated = list(mods or [])

    for idx, mod in enumerate(updated):
        mod_data = _dump(mod)
        if _normalize_code(mod_data.get("code") or getattr(mod, "code", None)) != "GY":
            continue
        # Preserve existing text if present; otherwise backfill from the decision node.
        existing_justification = (mod_data.get("justification") or "").strip()
        if existing_justification:
            return updated
        mod_data["code"] = "GY"
        mod_data["justification"] = justification
        updated[idx] = Modifier(**mod_data)
        return updated

    updated.append(Modifier(code="GY", justification=justification))
    return updated


@track_tokens("model_usage")
def gy_decision_node(state: ModifierGraphState, config: RunnableConfig) -> Dict[str, ModifierOutput]:
    # GY is Medicare-only. Skip LLM call entirely for non-Medicare payers.
    if not _is_medicare_payer(state, config):
        # print("Payer not eligible for modifier GY")
        return {"gy_output": ModifierOutput()}

    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION_GY]
    prompt_template = config["configurable"][CONFIG.PROMPTS][PROMPTS.MODIFIER_GY][prompt_version]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )

    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    cpt_codes = get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            "cpt_code": ",".join(cpt_codes),
        }
    )
    return {"output_gy": response}


def validate_and_apply_gy(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    gy_raw = state.get("output_gy")
    gy_output: ModifierOutput = gy_raw if isinstance(gy_raw, ModifierOutput) else ModifierOutput(**(gy_raw or {}))

    if not _is_medicare_payer(state, config):
        payer_names = _extract_payer_names(state, config)
        payer_text = ", ".join(payer_names) if payer_names else "Payer not specified"
        gy_unassigned: List[ModifierJustification] = []

        for item in (state.get("em") or []) + (state.get("procedure") or []):
            d = _dump(item) or {}
            cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
            if not cpt:
                continue
            gy_unassigned.append(
                ModifierJustification(
                    code="GY",
                    evidence=f"Payer context: {payer_text}.",
                    justification=f"{cpt}: GY is a Medicare-only modifier and should not be assigned for non-Medicare payer context.",
                    assigned=False,
                )
            )

        return {
            "em": state.get("em") or [],
            "procedure": state.get("procedure") or [],
            "unassigned_modifiers": _merge_unassigned_by_code((state.get("unassigned_modifiers") or []) + gy_unassigned),
        }

    exclusions = set(_normalize_code(c) for c in config["configurable"].get("medicare_statutory_exclusions", []))

    # Build decision map CPT -> ModifierJustification
    decision_by_cpt: Dict[str, ModifierJustification] = {}
    gy_unassigned: List[ModifierJustification] = []
    for cam in gy_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        chosen = None
        for mj in cam.modifiers or []:
            if _normalize_code(mj.code) != "GY":
                continue
            if mj.assigned:
                chosen = mj
                break
            if chosen is None:
                chosen = mj
        if chosen is not None:
            decision_by_cpt[cpt] = chosen
            if not chosen.assigned:
                chosen_data = _dump(chosen).copy()
                justification = (chosen_data.get("justification") or "").strip()
                detailed_justification = (chosen_data.get("detailed_justification") or "").strip()
                chosen_data["justification"] = f"{cpt}: {justification}" if justification else f"{cpt}:"
                chosen_data["detailed_justification"] = f"{cpt}: {detailed_justification}" if detailed_justification else f"{cpt}:"
                gy_unassigned.append(ModifierJustification(**chosen_data))

    # Apply to EM + Procedure while preserving everything
    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)

        if mj and mj.assigned and (not exclusions or cpt in exclusions):
            mods = _upsert_gy_modifier(mods, mj)

        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)

        if mj and mj.assigned and (not exclusions or cpt in exclusions):
            mods = _upsert_gy_modifier(mods, mj)

        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
        "unassigned_modifiers": _merge_unassigned_by_code((state.get("unassigned_modifiers") or []) + gy_unassigned),
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
        seen_detailed_justification: set[str] = set()
        evidence_parts: List[str] = []
        justification_parts: List[str] = []
        detailed_justification_parts: List[str] = []
        for item in items:
            evidence = _dump(item).get("evidence") or getattr(item, "evidence", None) or ""
            justification = _dump(item).get("justification") or getattr(item, "justification", None) or ""
            detailed_justification = _dump(item).get("detailed_justification") or getattr(item, "detailed_justification", None) or ""
            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                evidence_parts.append(evidence)
            if justification and justification not in seen_justification:
                seen_justification.add(justification)
                justification_parts.append(justification)
            if detailed_justification and detailed_justification not in seen_detailed_justification:
                seen_justification.add(detailed_justification)
                detailed_justification_parts.append(detailed_justification)

        rep_data = _dump(rep).copy()
        rep_data["code"] = mod_code
        rep_data["assigned"] = False
        if evidence_parts:
            rep_data["evidence"] = "\n".join(evidence_parts)
        if justification_parts:
            rep_data["justification"] = "\n".join(justification_parts)
        if detailed_justification_parts:
            rep_data["detailed_justification"] = "\n".join(detailed_justification_parts)
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
    for code, items in unassigned_by_code.items():
        modifier_output.extend(items)

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
        "unassigned_modifiers": _merge_unassigned_by_code(modifier_output),
    }


def build_modifier_gy_subgraph(args: Namespace, modifier_names: List[str]):
    normalized = {_normalize_code(name) for name in (modifier_names or [])}
    if "GY" not in normalized:
        raise ValueError(f"modifier_gy subgraph only supports GY, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("gy_decision", gy_decision_node)
    workflow.add_node("gy_validate", validate_and_apply_gy)

    workflow.add_edge(START, "gy_decision")
    workflow.add_edge("gy_decision", "gy_validate")
    workflow.add_edge("gy_validate", END)

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
            raise ValueError(f"All GY parallel executions failed: {parallel_res}")
        successful_results = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(successful_results)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
