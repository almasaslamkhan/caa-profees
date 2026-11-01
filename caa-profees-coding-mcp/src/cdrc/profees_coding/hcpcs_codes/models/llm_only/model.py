from argparse import Namespace
from typing import Any, List, Literal, Mapping, Optional, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AWVHistory, EncounterMetaData
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.hcpcs_codes.data.schema import HealthPlan, Problem, Procedure
from cdrc.profees_coding.hcpcs_codes.models.helix.model import build_helix_graph, build_llm_only_graph
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG

from .constants import HCPCS_AWV_PROMPT_VERSION, HCPCS_BRANCH_NAME, HCPCS_G2211_PROMPT_VERSION, HCPCS_OTHER_PROMPT_VERSION
from .nodes.awv_eligibility import awv_eligibility_precheck
from .nodes.hcpcs_code import hcpcs_code

AMBULATORY_ROUTE = "ambulatory"
FACILITY_ROUTE = "facility"
AWV_CODES = {"G0402", "G0438", "G0439"}


class HCPCSInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders_display: Optional[str]
    health_plan: Optional[List[HealthPlan]]
    awv_history: Optional[List[AWVHistory]]
    encounter_date: Optional[str]
    encounter_metadata: Optional[EncounterMetaData]
    awv_eligibility: Optional[Mapping[str, Any]]
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")
    additional_input: Optional[Mapping[str, Any]] = Field(None, description="Additional input fields")


class HCPCSOutputGraphState(TypedDict, total=False):
    procedure: List[Procedure] = Field(default_factory=list, description="List of HCPCS procedure codes")
    additional_output: Mapping[str, Any]


class HCPCSGraphState(HCPCSInputGraphState, HCPCSOutputGraphState, total=False):
    encounter_route: Literal["ambulatory", "facility"]
    g2211_result: Mapping[str, Any]
    awv_result: Mapping[str, Any]
    other_hcpcs_result: Mapping[str, Any]


def _encounter_type(state: Mapping[str, Any]) -> str:
    metadata = state.get("encounter_metadata")
    if isinstance(metadata, dict):
        encounter_type = metadata.get("type")
    elif isinstance(metadata, BaseModel):
        encounter_type = metadata.model_dump().get("type")
    elif metadata is not None:
        encounter_type = getattr(metadata, "type", None)
    else:
        encounter_type = None
    return str(encounter_type or "").strip().lower()


def route_encounter(state: Mapping[str, Any]) -> dict:
    encounter_type = _encounter_type(state)
    if encounter_type in {"inpatient", "hospital", "ed", "emergency", "emergency department"}:
        return {"encounter_route": FACILITY_ROUTE}
    return {"encounter_route": AMBULATORY_ROUTE}


def _branch_config(
    config: RunnableConfig,
    *,
    branch_name: str,
    prompt_config_key: str,
    default_prompt_version: Any,
) -> RunnableConfig:
    branch_config = dict(config or {})
    configurable = dict(branch_config.get("configurable") or {})
    configurable[CONFIG.PROMPT_VERSION] = configurable.get(prompt_config_key, default_prompt_version)
    configurable[HCPCS_BRANCH_NAME] = branch_name
    branch_config["configurable"] = configurable
    return branch_config


def _run_branch(
    state: Mapping[str, Any],
    config: RunnableConfig,
    *,
    result_key: str,
    branch_name: str,
    prompt_config_key: str,
    default_prompt_version: Any,
    allowed_route: str,
) -> dict:
    if state.get("encounter_route") != allowed_route:
        return {}

    result = hcpcs_code(
        state,
        _branch_config(
            config,
            branch_name=branch_name,
            prompt_config_key=prompt_config_key,
            default_prompt_version=default_prompt_version,
        ),
    )
    return {result_key: result}


def run_g2211_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    return _run_branch(
        state,
        config,
        result_key="g2211_result",
        branch_name="g2211",
        prompt_config_key=HCPCS_G2211_PROMPT_VERSION,
        default_prompt_version="v4.4",
        allowed_route=AMBULATORY_ROUTE,
    )


def run_awv_eligibility_precheck(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    if state.get("encounter_route") != AMBULATORY_ROUTE:
        return {}
    return awv_eligibility_precheck(state, config)


def run_awv_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    return _run_branch(
        state,
        config,
        result_key="awv_result",
        branch_name="awv",
        prompt_config_key=HCPCS_AWV_PROMPT_VERSION,
        default_prompt_version={"version": "v5.1", "codes_supported": ["G0402", "G0438", "G0439"]},
        allowed_route=AMBULATORY_ROUTE,
    )


def run_other_hcpcs_classifier(state: Mapping[str, Any], config: RunnableConfig) -> dict:
    return _run_branch(
        state,
        config,
        result_key="other_hcpcs_result",
        branch_name="other_hcpcs",
        prompt_config_key=HCPCS_OTHER_PROMPT_VERSION,
        default_prompt_version={"version": "v6.1", "codes_supported": ["Other HCPCS"]},
        allowed_route=FACILITY_ROUTE,
    )


def _sort_procedures(items: List[dict]) -> List[dict]:
    def sort_key(item: dict) -> tuple[int, Any]:
        code = str(item.get("code") or "").upper()
        if code == "G2211":
            return (0, 0)
        if code in AWV_CODES:
            return (1, {"G0402": 0, "G0438": 1, "G0439": 2}[code])
        return (2, code)

    return sorted(items, key=sort_key)


def merge_results(state: Mapping[str, Any]) -> dict:
    branch_results = {
        "g2211": state.get("g2211_result"),
        "awv": state.get("awv_result"),
        "other_hcpcs": state.get("other_hcpcs_result"),
    }

    deduped: dict[tuple[str, str], dict] = {}
    additional_output: dict[str, Any] = {}
    customer_name: Optional[str] = None

    for branch_name, result in branch_results.items():
        if not isinstance(result, Mapping):
            continue

        for procedure in result.get("procedure") or []:
            procedure_dict = procedure.model_dump() if hasattr(procedure, "model_dump") else dict(procedure)
            system = str(procedure_dict.get("system") or "").upper()
            code = str(procedure_dict.get("code") or "").upper()
            if not system or not code:
                continue
            deduped.setdefault((system, code), procedure_dict)

        branch_additional = result.get("additional_output")
        if branch_additional:
            additional_output[branch_name] = branch_additional

        if not customer_name:
            customer_name = result.get("customer_name")

    procedures = _sort_procedures(list(deduped.values()))
    if customer_name:
        additional_output["customer_name"] = customer_name

    output = {"procedure": procedures}
    if additional_output:
        output["additional_output"] = additional_output
    return output


def build_llm_only_graph_legacy(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(HCPCSGraphState, input=HCPCSInputGraphState, output=HCPCSOutputGraphState)

    workflow.add_node("route_encounter", route_encounter)
    workflow.add_node("g2211_classifier", run_g2211_classifier)
    workflow.add_node("awv_eligibility_precheck", run_awv_eligibility_precheck)
    workflow.add_node("awv_classifier", run_awv_classifier)
    workflow.add_node("other_hcpcs_classifier", run_other_hcpcs_classifier)
    workflow.add_node("merge_results", merge_results)

    workflow.add_edge(START, "route_encounter")
    workflow.add_edge("route_encounter", "g2211_classifier")
    workflow.add_edge("route_encounter", "awv_eligibility_precheck")
    workflow.add_edge("route_encounter", "other_hcpcs_classifier")
    workflow.add_edge("awv_eligibility_precheck", "awv_classifier")
    workflow.add_edge("g2211_classifier", "merge_results")
    workflow.add_edge("awv_classifier", "merge_results")
    workflow.add_edge("other_hcpcs_classifier", "merge_results")
    workflow.add_edge("merge_results", END)

    return workflow.compile(), model_config
