from typing import Dict, List, Optional, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.icd10_cm_tools.agent_tools import get_full_data as tool_full
from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.agent_tools import is_valid_item as tool_is_valid
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.icd10cm.models.llm_multistage.model import (
    CodePredictionResults,
    _format_messages,
    _get_llm_and_parser,
    _maybe_collapse_system_for_o1,
)
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG, PROMPTS


class InputState(dict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    patient_age: Optional[int]
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")


class IntermediateState(dict):
    z_code_enhancer_predictions: List[ICD10CM]
    code_predictions: List[ICD10CM]


class OutputState(dict):
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class ICD10CMCodeGraphStates(InputState, IntermediateState, OutputState):
    pass


@track_tokens("model_usage")
def node_coding(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Combined extraction of contexts + per-problem search queries."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.ICD_CODING]
    prompt_version = config["configurable"][CONFIG.ICD_10_CM_PROMPT_VERSION]
    model, parser = _get_llm_and_parser(config, CodePredictionResults)
    chain = model | parser

    system_tmpl = prompts[prompt_version]["system"]
    user_tmpl = prompts[prompt_version]["user"]
    logger.debug(f"[coding] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
        em=state.get("em", []),
        procedure=state.get("procedure", []),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: CodePredictionResults = chain.invoke(messages)

    # Persist contexts
    coded = [c.model_dump() for c in out.results]
    logger.debug(f"Coding predictions NO. of problems: {len(coded)}")

    return {
        "code_predictions": coded,
    }


@track_tokens("model_usage")
def node_z_code_enhancer(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Enhance Z codes based on existing codes in the state.
    This is a placeholder for future implementation.
    """
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.Z_CODE_ENHANCER]
    prompt_version = config["configurable"][CONFIG.ICD_10_CM_PROMPT_VERSION]
    model, parser = _get_llm_and_parser(config, CodePredictionResults)
    chain = model | parser

    system_tmpl = prompts[prompt_version]["system"]
    user_tmpl = prompts[prompt_version]["user"]
    logger.debug(f"[z_code_enhancer] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
        em_codes=state.get("em", []),
        procedure_codes=state.get("procedure", []),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out = chain.invoke(messages)

    # Persist contexts
    coded = [c.model_dump() for c in out.results]
    logger.debug(f"Z Code Enhancer predictions NO. of problems: {len(coded)}")

    return {
        "z_code_enhancer_predictions": coded,
    }


def node_validate_and_filter(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Validate existence and billability, preserving typed CandidateForChoice,
    and record filtered candidates per problem/query.
    """

    def _is_code_kept(raw_code: str) -> bool:
        code = (raw_code or "").strip()
        if not code:
            return False

        is_valid = tool_is_valid(code=code)
        # is_valid_skg = _tool_valid(code=code)
        is_valid_skg = True
        is_billable = tool_is_billable(code=code)

        keep = bool(is_valid and is_billable and is_valid_skg)
        if not keep:
            logger.debug(f"Code {code} valid status: {is_valid}, " f"billable status: {is_billable}, " f"skg status: {is_valid_skg}")
        return keep

    # ---- flat filtered list (used downstream for reranker/choose) ----
    filtered: Dict[str, ICD10CM] = {}
    main_results = state.get("code_predictions", [])
    enhancer_results = state.get("z_code_enhancer_predictions", [])

    merged_results = main_results + enhancer_results

    for pred in merged_results:
        raw_code = pred.get("code", "").strip()
        if _is_code_kept(raw_code):
            # fetch full code data
            code_descriptions = tool_full(code=raw_code)
            icd_code = ICD10CM(
                code=raw_code,
                is_primary=True,
                description=code_descriptions,
                system="ICD-10-CM",
                justification=pred.get("justification", ""),
            )
            filtered[icd_code.code] = icd_code

    return {
        "diagnosis": list(filtered.values()),
    }


def build_llm_singlestage_enhanced_graph(args):
    g = StateGraph(
        ICD10CMCodeGraphStates,
        input=InputState,
        output=OutputState,
    )

    g.add_node("code_prediction", node_coding)
    # g.add_node("z_code_enhancer", node_z_code_enhancer)
    g.add_node("code_validation", node_validate_and_filter)

    g.add_edge(START, "code_prediction")
    # g.add_edge(START, "z_code_enhancer")
    # g.add_edge("z_code_enhancer", "code_validation")
    g.add_edge("code_prediction", "code_validation")
    g.add_edge("code_validation", END)

    return g.compile()
