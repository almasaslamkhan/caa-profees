import json
from argparse import Namespace
from typing import Any, Dict, List, Optional, TypedDict, Union

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
from cdrc.profees_coding.modifiers.models.utils import build_ehr_data, dedupe_codes, dump, get_cpt_codes, merge_unassigned_by_code, normalize_code


class ModifierJustification(BaseModel):
    code: str = Field("95", description="Modifier code. Always '95' for this decision task.")
    evidence: Optional[str] = Field(None, description="Direct quote (<=25 words) from the documentation supporting the decision, if applicable.")
    justification: Optional[str] = Field(
        None,
        description="Concise decision reason only: less than 20 words, focused on the decisive rule/fact.",
    )
    assigned: bool = Field(..., description="True if Modifier 95 should be applied; False otherwise.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT or HCPCS code evaluated for Modifier 95.")
    modifiers: List[ModifierJustification] = Field(
        default_factory=list, description="List of modifier decisions for this CPT code (95 only in this task)."
    )


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Modifier 95 decisions grouped by CPT code.")


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
    output_95: Optional[ModifierOutput]


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    return model_kwargs


def _upsert_95_modifier(mods: List[Any], mj: ModifierJustification) -> List[Any]:
    justification = (getattr(mj, "justification", None) or dump(mj).get("justification") or "").strip() or None
    updated = list(mods or [])

    for idx, mod in enumerate(updated):
        mod_data = dump(mod)
        if normalize_code(mod_data.get("code") or getattr(mod, "code", None)) != "95":
            continue
        existing_justification = (mod_data.get("justification") or "").strip()
        if existing_justification:
            return updated
        mod_data["code"] = "95"
        mod_data["justification"] = justification
        updated[idx] = Modifier(**mod_data)
        return updated

    updated.append(Modifier(code="95", justification=justification))
    return updated


@track_tokens("model_usage")
def modifier_95_decision_node(state: ModifierGraphState, config: RunnableConfig) -> Dict[str, ModifierOutput]:
    cpt_codes = dedupe_codes([normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code])

    eligible_cpt_codes = dedupe_codes([normalize_code(code) for code in (state.get("eligible_cpt_codes") or []) if normalize_code(code)])
    if not eligible_cpt_codes:
        eligible_cpt_codes = cpt_codes

    # Skip LLM call when encounter has no CPT codes to evaluate.
    if not eligible_cpt_codes:
        return {"output_95": ModifierOutput()}

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["95"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_95"][prompt_version]

    ehr_info = build_ehr_data(state, config, ehr_list=get_relevant_ehr_data_in_scope(config, ("95",)))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model, correction_model = get_model_and_correction_model(config["configurable"][CONFIG.MODEL_NAME], model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=correction_model,
        num_retries=3,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(), additional_info=ehr_info
    )
    chain = prompt | model | parser

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            "cpt_code": ",".join(eligible_cpt_codes),
        }
    )
    return {"output_95": response}


def validate_and_apply_95(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    modifier_raw = state.get("output_95")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decision_by_cpt: Dict[str, ModifierJustification] = {}
    unassigned_95: List[ModifierJustification] = []
    for cam in modifier_output.code_and_modifiers or []:
        cpt = normalize_code(cam.cpt_code)
        chosen = None
        for mj in cam.modifiers or []:
            if normalize_code(mj.code) != "95":
                continue
            if mj.assigned:
                chosen = mj
                break
            if chosen is None:
                chosen = mj
        if chosen is not None:
            decision_by_cpt[cpt] = chosen
            if not chosen.assigned:
                chosen_data = dump(chosen).copy()
                justification = (chosen_data.get("justification") or "").strip()
                chosen_data["justification"] = f"{cpt}: {justification}" if justification else f"{cpt}:"
                unassigned_95.append(ModifierJustification(**chosen_data))

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (dump(em) or {}).copy()
        cpt = normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj and mj.assigned:
            mods = _upsert_95_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (dump(p) or {}).copy()
        cpt = normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj and mj.assigned:
            mods = _upsert_95_modifier(mods, mj)
        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
        "unassigned_modifiers": merge_unassigned_by_code((state.get("unassigned_modifiers") or []) + unassigned_95, ModifierJustification),
    }


def build_modifier_95_subgraph(args: Namespace, modifier_names: List[str]):
    normalized = {normalize_code(name) for name in (modifier_names or [])}
    if "95" not in normalized:
        raise ValueError(f"modifier_95 subgraph only supports 95, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_95_decision", modifier_95_decision_node)
    workflow.add_node("modifier_95_validate", validate_and_apply_95)

    workflow.add_edge(START, "modifier_95_decision")
    workflow.add_edge("modifier_95_decision", "modifier_95_validate")
    workflow.add_edge("modifier_95_validate", END)

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
            raise ValueError(f"All modifier 95 parallel executions failed: {parallel_res}")
        successful_results = [res for res in parallel_res if not isinstance(res, Exception)]
        return aggregate_parallel_results(successful_results)

    parallel_workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    parallel_workflow.add_node("batch_and_aggregate", batch_and_aggregate)
    parallel_workflow.add_edge(START, "batch_and_aggregate")
    parallel_workflow.add_edge("batch_and_aggregate", END)
    return parallel_workflow.compile()
