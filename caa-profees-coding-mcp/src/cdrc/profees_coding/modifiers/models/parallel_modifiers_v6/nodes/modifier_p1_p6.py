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
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.data_classes import CONFIG, get_relevant_ehr_data_in_scope
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.model_selection import get_model_and_correction_model
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.modifiers import aggregate_parallel_results
from cdrc.profees_coding.modifiers.models.utils import build_ehr_data, get_cpt_codes, is_anesthesia_cpt

P_STATUS_MODIFIERS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
P_STATUS_ORDER = {code: idx for idx, code in enumerate(P_STATUS_MODIFIERS, start=1)}


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code (P1-P6).")
    evidence: Optional[str] = Field(None, description="Snippet from the encounter that supports the decision.")
    justification: Optional[str] = Field(
        None,
        description="Concise decision reason only: less than 20 words, focused on the decisive rule/fact.",
    )
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

    eligible_cpt_codes: Optional[List[str]]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_p1_p6: Optional[ModifierOutput]


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


def _upsert_p_modifier(mods: List[Any], mj: ModifierJustification) -> List[Any]:
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


def _select_best_assigned_modifier(modifiers: List[ModifierJustification], allowed_modifiers: Set[str]) -> Optional[ModifierJustification]:
    assigned = [m for m in (modifiers or []) if _normalize_code(m.code) in allowed_modifiers and m.assigned]
    if not assigned:
        return None
    return max(assigned, key=lambda m: P_STATUS_ORDER.get(_normalize_code(m.code), -1))


@track_tokens("model_usage")
def modifier_p1_p6_decision_node(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> Dict[str, ModifierOutput]:
    cpt_codes = [_normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code]
    # eligible_cpt_codes: List[str] = []
    # seen: Set[str] = set()
    # for code in cpt_codes:
    #     if not is_anesthesia_cpt(code) or code in seen:
    #         continue
    #     seen.add(code)
    #     eligible_cpt_codes.append(code)

    eligible_cpt_codes = [_normalize_code(code) for code in (state.get("eligible_cpt_codes") or []) if _normalize_code(code)]
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
        # print("No eligible codes for modifier P1-P6")
        return {"output_p1_p6": ModifierOutput()}

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["P1_P6"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_p1_p6"][prompt_version]

    additional_info = build_ehr_data(state, config, ehr_list=get_relevant_ehr_data_in_scope(config, P_STATUS_MODIFIERS))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model, correction_model = get_model_and_correction_model(config["configurable"][CONFIG.MODEL_NAME], model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=correction_model,
        num_retries=3,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(), additional_info=additional_info
    )
    chain = prompt | model | parser

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            # "cpt_code": ",".join(eligible_cpt_codes),
            "cpt_code": ",".join(cpt_codes),
        }
    )
    return {"output_p1_p6": response}


def validate_and_apply_p1_p6(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> ModifierGraphOutput:
    del config  # kept for signature parity across nodes

    allowed = set(P_STATUS_MODIFIERS)
    modifier_raw = state.get("output_p1_p6")
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

        best_assigned = _select_best_assigned_modifier(cam_modifiers, allowed)
        if best_assigned and is_anesthesia_cpt(cpt):
            decision_by_cpt[cpt] = best_assigned

    all_cpt_codes: Set[str] = set()
    for item in (state.get("em") or []) + (state.get("procedure") or []):
        d = _dump(item) or {}
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        if cpt:
            all_cpt_codes.add(cpt)

    for cpt in sorted(all_cpt_codes):
        if is_anesthesia_cpt(cpt):
            continue
        for code in sorted(allowed, key=lambda c: P_STATUS_ORDER.get(c, 0)):
            unassigned.append(
                ModifierJustification(
                    code=code,
                    evidence=None,
                    justification=f"{cpt}: Not evaluated for {code} because this CPT is not an anesthesia base code.",
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
            mods = _upsert_p_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj and mj.assigned:
            mods = _upsert_p_modifier(mods, mj)
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
