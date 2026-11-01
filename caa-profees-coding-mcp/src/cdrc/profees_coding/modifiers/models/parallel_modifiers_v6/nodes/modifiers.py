import json
from argparse import Namespace
from collections import Counter, defaultdict
from functools import partial
from math import ceil
from typing import Any, Dict, List, Optional, Type, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AnaesthesiaStaffInfo, EncounterMetaData, FacilityInfo, HealthPlan, LabInfo, ProviderInfo
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.data_classes import CONFIG, get_relevant_ehr_data_in_scope
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.nodes.model_selection import get_model_and_correction_model
from cdrc.profees_coding.modifiers.models.utils import (
    apply_modifiers,
    build_ehr_data,
    dedupe_codes,
    dump,
    get_cpt_codes,
    get_gold_key,
    get_mod,
    is_em_code,
    merge_unassigned_by_code,
    normalize_code,
    print_runnable,
    to_modifier,
)


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    return model_kwargs


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code, e.g. '25', 'LT', etc.")
    evidence: str = Field(
        ...,
        description="One plain-text string containing all snippet from the encounter that supports the decision. Do not use lists, semicolons, or multiple quoted snippets.",
    )  # Snippet from the encounter that supports the decision
    justification: str = Field(
        ...,
        description="Concise decision reason only: less than 20 words, focused on the decisive rule/fact.",
    )
    assigned: bool = Field(..., description="True if the modifier should be applied to this CPT code")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="A single CPT code")  # CPT code
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

    eligible_cpt_codes: Optional[List[str]]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    pass


def _prefix_justification_with_cpt(modifier: ModifierJustification, cpt_code: str) -> ModifierJustification:
    normalized_cpt = normalize_code(cpt_code)
    data = dump(modifier).copy()
    justification = data.get("justification") or ""
    prefix = f"{normalized_cpt}: "
    if normalized_cpt and not justification.startswith(prefix):
        data["justification"] = f"{prefix}{justification}" if justification else prefix.rstrip()
    return ModifierJustification(**data)


def _merge_responses(responses: List[ModifierOutput]) -> tuple[ModifierOutput, List[ModifierJustification]]:
    if not responses:
        raise ValueError("No modifier responses provided")

    if len(responses) == 1:
        single = responses[0]
        unassigned_single: List[ModifierJustification] = []
        seen: set[tuple[str, str]] = set()
        for cam in single.code_and_modifiers:
            cpt_code = normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = normalize_code(dump(modifier).get("code") or getattr(modifier, "code", None))
                assigned_flag = getattr(modifier, "assigned", False) or dump(modifier).get("assigned", False)
                key = (cpt_code, mod_code)
                if not mod_code or assigned_flag or key in seen:
                    continue
                seen.add(key)
                unassigned_single.append(_prefix_justification_with_cpt(modifier, cpt_code))
        return single, merge_unassigned_by_code(unassigned_single, ModifierJustification)

    threshold = ceil(len(responses) / 2)
    modifiers_by_cpt: Dict[str, List[ModifierJustification]] = defaultdict(list)
    counts_by_cpt: Dict[str, Counter[str]] = defaultdict(Counter)

    for response in responses:
        for cam in response.code_and_modifiers:
            code = normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = normalize_code(getattr(modifier, "code", None) or dump(modifier).get("code"))
                assigned = getattr(modifier, "assigned", False) or dump(modifier).get("assigned", False)
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
                (m for m in modifiers_by_cpt[code] if normalize_code(dump(m).get("code") or getattr(m, "code", None)) == mod_code),
                None,
            )
            if example:
                selected_modifiers.append(example)
        aggregated_cam.append(CodeAndModifier(cpt_code=code, modifiers=selected_modifiers))

    assigned_codes = {
        (normalize_code(cam.cpt_code), normalize_code(dump(m).get("code") or getattr(m, "code", None)))
        for cam in aggregated_cam
        for m in cam.modifiers
    }

    unassigned: List[ModifierJustification] = []
    seen_unassigned: set[tuple[str, str]] = set()
    for response in responses:
        for cam in response.code_and_modifiers:
            cpt_code = normalize_code(cam.cpt_code)
            for modifier in cam.modifiers:
                mod_code = normalize_code(dump(modifier).get("code") or getattr(modifier, "code", None))
                assigned_flag = getattr(modifier, "assigned", False) or dump(modifier).get("assigned", False)
                key = (cpt_code, mod_code)
                if not mod_code or assigned_flag or key in assigned_codes or key in seen_unassigned:
                    continue
                seen_unassigned.add(key)
                unassigned.append(_prefix_justification_with_cpt(modifier, cpt_code))

    # Attach derived unassigned list to output via a lightweight wrapper
    result = ModifierOutput(code_and_modifiers=aggregated_cam)
    return result, merge_unassigned_by_code(unassigned, ModifierJustification)


def aggregate_parallel_results(parallel_results: List[ModifierGraphOutput]) -> ModifierGraphOutput:
    def _match_lookups(parallel_results: List[ModifierGraphOutput], key):
        lookups = []
        for res in parallel_results:
            lookup = {get_gold_key(item): item for item in (res.get(key) or [])}
            lookups.append(lookup)
        return lookups

    def _get_aggregated(result_key: str, item_class: Type[Any], parallel_results: List[ModifierGraphOutput], lookups) -> List[Any]:
        n = len(parallel_results)
        threshold = ceil(n / 2)

        base_items = parallel_results[0].get(result_key) or []
        aggregated_list: List[Any] = []

        for base_item in base_items:
            gold_key = get_gold_key(base_item)
            mod_counts: Counter[str] = Counter()
            mod_examples: Dict[str, List[Any]] = defaultdict(list)

            for run_idx in range(n):
                item_in_run = lookups[run_idx].get(gold_key)
                if not item_in_run:
                    continue

                seen_this_run = set()
                for modifier in getattr(item_in_run, "modifiers", []) or []:
                    mod_code = get_mod(modifier)
                    if not mod_code or mod_code in seen_this_run:
                        continue

                    seen_this_run.add(mod_code)
                    mod_counts[mod_code] += 1
                    mod_examples[mod_code].append(modifier)

            majority_mods: List[Any] = []
            for mod_code, count in mod_counts.items():
                if count >= threshold and mod_examples[mod_code]:
                    candidates = mod_examples[mod_code]
                    representative = next(
                        (c for c in candidates if getattr(c, "justification", None)),
                        candidates[0],
                    )
                    majority_mods.append(representative)

            item_data = (dump(base_item) or {}).copy()
            item_data["modifiers"] = majority_mods
            aggregated_list.append(item_class(**item_data))

        return aggregated_list

    if not parallel_results:
        raise ValueError("No results to aggregate")

    em_lookups = _match_lookups(parallel_results, "em")
    proc_lookups = _match_lookups(parallel_results, "procedure")

    aggregated_em: List[EMItem] = _get_aggregated(
        result_key="em",
        item_class=EMItem,
        parallel_results=parallel_results,
        lookups=em_lookups,
    )

    aggregated_proc: List[Procedure] = _get_aggregated(
        result_key="procedure",
        item_class=Procedure,
        parallel_results=parallel_results,
        lookups=proc_lookups,
    )

    unassigned_by_code: Dict[str, List[Any]] = defaultdict(list)
    for res in parallel_results:
        for it in res.get("unassigned_modifiers") or []:
            code = get_mod(it)
            if code:
                unassigned_by_code[code].append(it)

    modifier_output: List[Any] = []
    for code, items in unassigned_by_code.items():
        modifier_output.extend(items)
    unassigned_modifiers = merge_unassigned_by_code(modifier_output, ModifierJustification)

    return {
        "em": aggregated_em,
        "procedure": aggregated_proc,
        "unassigned_modifiers": unassigned_modifiers,
    }


@track_tokens("model_usage")
def modifiers_classifier(state: ModifierGraphState, config: RunnableConfig, modifier_names: List[str]) -> ModifierGraphOutput:
    prompt_versions = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])
    mod_str = "_".join(modifier_names)
    prompt_version = prompt_versions[mod_str] if mod_str in prompt_versions else prompt_versions["ALL"]
    prompt_template = config["configurable"][CONFIG.PROMPTS][f"modifier_{mod_str}".lower()][prompt_version]

    ehr_info = build_ehr_data(state, config, ehr_list=get_relevant_ehr_data_in_scope(config, modifier_names))

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model, correction_model = get_model_and_correction_model(config["configurable"][CONFIG.MODEL_NAME], model_kwargs)

    parser = PydanticOutputParserWithRetries(pydantic_object=ModifierOutput, correction_model=correction_model, num_retries=3)
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
        format_instructions=parser.get_format_instructions(),
        additional_info=ehr_info,
    )
    chain = prompt | model | parser

    cpt_codes = dedupe_codes([normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code])

    eligible_cpt_codes = state.get("eligible_cpt_codes") or cpt_codes
    eligible_cpt_codes = [normalize_code(code) for code in eligible_cpt_codes if normalize_code(code)]

    if eligible_cpt_codes:
        response = chain.invoke(
            {
                "soap_note": state.get("text"),
                "cpt_code": ",".join(cpt_codes),
                "orders": state.get("orders_display"),
            }
        )
    else:
        response = ModifierOutput(code_and_modifiers=[])

    merged, unassigned = _merge_responses([response])

    em_modifiers: Dict[str, List[Modifier]] = {}
    proc_modifiers: Dict[str, List[Modifier]] = {}
    for cam in merged.code_and_modifiers:
        mods = [to_modifier(m) for m in cam.modifiers if getattr(m, "assigned", False) or dump(m).get("assigned", False)]
        code = normalize_code(cam.cpt_code)
        if is_em_code(code):
            em_modifiers[code] = mods
        else:
            proc_modifiers[code] = mods

    return {
        "em": apply_modifiers(state.get("em"), em_modifiers, EMItem),
        "procedure": apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": unassigned,
    }


def build_modifiers_subgraph(args: Namespace, modifier_names: List[str]):
    classifier = partial(modifiers_classifier, modifier_names=modifier_names)

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
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
