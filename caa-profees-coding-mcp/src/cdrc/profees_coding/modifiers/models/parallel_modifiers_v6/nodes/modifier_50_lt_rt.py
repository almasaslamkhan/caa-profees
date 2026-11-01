import json
from argparse import Namespace
from typing import Dict, List, Optional, TypedDict, Union

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
from cdrc.profees_coding.modifiers.models.utils import (
    apply_modifiers,
    build_ehr_data,
    dedupe_codes,
    dump,
    get_cpt_codes,
    is_procedure_code,
    merge_unassigned_by_code,
    normalize_code,
    to_modifier,
)

LATERALITY_MODIFIERS: tuple[str, ...] = ("50", "LT", "RT")
LATERALITY_PRIORITY = {"50": 3, "LT": 2, "RT": 1}


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code (50, LT, RT).")
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
    output_50_lt_rt: Optional[ModifierOutput]


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


def _select_best_assigned_modifier(modifiers: List[ModifierJustification]) -> Optional[ModifierJustification]:
    assigned = [m for m in (modifiers or []) if m.assigned]
    if not assigned:
        return None
    return max(assigned, key=lambda m: LATERALITY_PRIORITY.get(normalize_code(m.code), -1))


def _prefix_justification_with_cpt(modifier: ModifierJustification, cpt_code: str) -> ModifierJustification:
    normalized_cpt = normalize_code(cpt_code)
    data = dump(modifier).copy()
    justification = data.get("justification") or ""
    prefix = f"{normalized_cpt}: "
    if normalized_cpt and not justification.startswith(prefix):
        data["justification"] = f"{prefix}{justification}" if justification else prefix.rstrip()
    return ModifierJustification(**data)


def _merge_50_lt_rt_response(response: ModifierOutput) -> tuple[ModifierOutput, List[ModifierJustification]]:
    aggregated_cam: List[CodeAndModifier] = []
    unassigned: List[ModifierJustification] = []
    seen_unassigned: set[tuple[str, str]] = set()

    for cam in response.code_and_modifiers or []:
        cpt_code = normalize_code(cam.cpt_code)
        filtered_modifiers = [modifier for modifier in (cam.modifiers or []) if normalize_code(modifier.code) in LATERALITY_PRIORITY]
        selected_modifier = _select_best_assigned_modifier(filtered_modifiers)

        if selected_modifier and is_procedure_code(cpt_code):
            aggregated_cam.append(CodeAndModifier(cpt_code=cpt_code, modifiers=[selected_modifier]))

        for modifier in filtered_modifiers:
            mod_code = normalize_code(dump(modifier).get("code") or getattr(modifier, "code", None))
            assigned_flag = getattr(modifier, "assigned", False) or dump(modifier).get("assigned", False)
            key = (cpt_code, mod_code)
            if not mod_code or assigned_flag or key in seen_unassigned:
                continue
            seen_unassigned.add(key)
            unassigned.append(_prefix_justification_with_cpt(modifier, cpt_code))

    return ModifierOutput(code_and_modifiers=aggregated_cam), merge_unassigned_by_code(unassigned, ModifierJustification)


@track_tokens("model_usage")
def modifier_50_lt_rt_decision_node(
    state: ModifierGraphState,
    config: RunnableConfig,
) -> Dict[str, ModifierOutput]:
    cpt_codes = dedupe_codes([normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code])
    eligible_cpt_codes = [normalize_code(code) for code in (state.get("eligible_cpt_codes") or []) if normalize_code(code)]

    if not eligible_cpt_codes:
        # print("No eligible codes for modifier 50, LT, RT")
        return {"output_50_lt_rt": ModifierOutput()}

    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["50_LT_RT"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_50_lt_rt"][prompt_version]

    ehr_info = build_ehr_data(state, config, ehr_list=get_relevant_ehr_data_in_scope(config, LATERALITY_MODIFIERS))

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

    response = chain.invoke(
        {
            "soap_note": state.get("text"),
            "orders": state.get("orders_display"),
            "cpt_code": ",".join(cpt_codes),
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
        mods = [to_modifier(m) for m in cam.modifiers if getattr(m, "assigned", False) or dump(m).get("assigned", False)]
        code = normalize_code(cam.cpt_code)
        if is_procedure_code(code):
            proc_modifiers[code] = mods

    return {
        "em": apply_modifiers(state.get("em"), {}, EMItem),
        "procedure": apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": unassigned,
    }


def build_modifier_50_lt_rt_subgraph(args: Namespace, modifier_names: List[str]):
    requested = {normalize_code(name) for name in (modifier_names or [])}
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
