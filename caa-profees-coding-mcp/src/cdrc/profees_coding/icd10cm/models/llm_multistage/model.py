# cdrc/profees_coding/icd10cm/models/workflow/workflow_icd10cm.py
from __future__ import annotations

import json
import random
from argparse import Namespace
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# Tool shims
from cdrc.common.icd10_cm_tools.agent_tools import get_full_data as tool_full
from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.agent_tools import is_valid_item as tool_is_valid
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.icd10cm.data.schema import ICD10CM, Order, Problem
from cdrc.profees_coding.icd10cm.models.llm_multistage.data_classes import CONFIG, PROMPTS

# ------------------------ Pydantic IO models ------------------------


class ExtractedProblems(BaseModel):
    problem: str
    evidence: str


class ExtractedICDCodes(BaseModel):
    justification: str
    code: str
    description: str


class ExtractedProblemList(BaseModel):
    results: List[ExtractedProblems]


class CodePredictionResults(BaseModel):
    results: List[ExtractedICDCodes] = Field(..., description="list of predicted ICD codes.")


# ------------------------ State ------------------------


class InputState(TypedDict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    patient_age: Optional[int]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]


class OutputState(TypedDict):
    diagnosis: List[ICD10CM]
    extracted_problems: ExtractedProblemList
    z_code_enhancer_predictions: List[ICD10CM]
    code_predictions: List[ICD10CM]
    # (optional but helpful) keep any future extras instead of dropping silently
    model_config = ConfigDict(extra="allow")


class ICD10CMCodeGraphStates(InputState, OutputState):
    pass


# ------------------------ LLM utils ------------------------


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


def _get_llm_and_parser(config: dict, pydantic_model: Any):
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=pydantic_model, correction_model=model, num_retries=3)
    return model, parser


def _format_messages(system_tmpl: str, user_tmpl: str, parser, **vars_):
    system_prompt = SystemMessagePromptTemplate.from_template(system_tmpl, template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(user_tmpl, template_format="jinja2")
    return [system_prompt.format(), user_prompt.format(format_instructions=parser.get_format_instructions(), **vars_)]


def _maybe_collapse_system_for_o1(messages: List, config: dict):
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)
    return messages


@track_tokens("model_usage")
def node_problem_extraction(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Combined extraction of contexts + per-problem search queries."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_MULTI_STAGE]
    model, parser = _get_llm_and_parser(config, ExtractedProblemList)
    chain = model | parser

    system_tmpl = prompts["problem_extraction"][config["configurable"][CONFIG.PROBLEM_EXTRACTION_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["problem_extraction"][config["configurable"][CONFIG.PROBLEM_EXTRACTION_PROMPT_VERSION]]["user"]
    logger.debug(f"[node_problem_extraction] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: ExtractedProblemList = chain.invoke(messages)

    # Persist contexts
    extracted = [c.model_dump() for c in out.results]
    logger.debug(f"Extracted problems NO. of problems: {len(extracted)}")

    return {
        "extracted_problems": extracted,
    }


@track_tokens("model_usage")
def node_coding(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Combined extraction of contexts + per-problem search queries."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_MULTI_STAGE]
    model, parser = _get_llm_and_parser(config, CodePredictionResults)
    chain = model | parser

    system_tmpl = prompts["coding"][config["configurable"][CONFIG.CODING_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["coding"][config["configurable"][CONFIG.CODING_PROMPT_VERSION]]["user"]
    problem_list = [p["problem"] for p in state.get("extracted_problems", [])] if state.get("extracted_problems") else []
    logger.debug(f"[coding] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        problem_list=problem_list,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: CodePredictionResults = chain.invoke(messages)

    # Persist contexts
    coded = [c.model_dump() for c in out.results]
    logger.debug(f"Coding predictions NO. of problems: {len(coded)}")

    return {
        "code_predictions": coded,
    }


def node_z_code_enhancer(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Enhance Z codes based on existing codes in the state.
    This is a placeholder for future implementation.
    """
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_MULTI_STAGE]
    model, parser = _get_llm_and_parser(config, CodePredictionResults)
    chain = model | parser

    system_tmpl = prompts["z_code_enhancer"][config["configurable"][CONFIG.ENHANCER_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["z_code_enhancer"][config["configurable"][CONFIG.ENHANCER_PROMPT_VERSION]]["user"]
    problem_list = [p["problem"] for p in state.get("extracted_problems", [])] if state.get("extracted_problems") else []
    logger.debug(f"[z_code_enhancer] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        problem_list=problem_list,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
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


# ------------------------ Graph builder ------------------------


def build_llm_multistage_graph(args: Namespace):
    """
    Returns (CompiledStateGraph, model_config).
    Expects your prompts dict to include the three blocks exactly as provided:
      WORKFLOW.extract_and_queries.v1, WORKFLOW.direct.v1, WORKFLOW.choose.v1
    """
    prompts = getattr(args, "prompts", None) or load_from_yaml(args.prompt_path)
    config = vars(args)

    # load combo code rules
    combo_rules = None
    combo_rules_path = getattr(args, "combo_rules_path", None)
    if combo_rules_path:
        combo_rules = load_from_yaml(combo_rules_path)
        logger.info(
            f"Loaded combo code rules from: {combo_rules_path} " f"({len(combo_rules) if isinstance(combo_rules, dict) else 'unknown'} rules)"
        )

    model_config = {"configurable": {"prompts": prompts, **config}}
    if combo_rules is not None:
        # store rules dict into config so nodes can use it
        model_config["configurable"]["combo_rules"] = combo_rules

    g = StateGraph(
        ICD10CMCodeGraphStates,
        input=InputState,
        output=OutputState,
    )

    g.add_node("problem_extraction", node_problem_extraction)
    g.add_node("code_prediction", node_coding)
    g.add_node("code_validation", node_validate_and_filter)

    g.add_edge(START, "problem_extraction")
    g.add_edge("problem_extraction", "code_prediction")
    g.add_edge("code_prediction", "code_validation")
    g.add_edge("code_validation", END)

    return g.compile(), model_config


def build_llm_singlestage_graph(args: Namespace):
    """
    Returns (CompiledStateGraph, model_config).
    Expects your prompts dict to include the three blocks exactly as provided:
      WORKFLOW.extract_and_queries.v1, WORKFLOW.direct.v1, WORKFLOW.choose.v1
    """
    prompts = getattr(args, "prompts", None) or load_from_yaml(args.prompt_path)
    config = vars(args)

    # load combo code rules
    combo_rules = None
    combo_rules_path = getattr(args, "combo_rules_path", None)
    if combo_rules_path:
        combo_rules = load_from_yaml(combo_rules_path)
        logger.info(
            f"Loaded combo code rules from: {combo_rules_path} " f"({len(combo_rules) if isinstance(combo_rules, dict) else 'unknown'} rules)"
        )

    model_config = {"configurable": {"prompts": prompts, **config}}
    if combo_rules is not None:
        # store rules dict into config so nodes can use it
        model_config["configurable"]["combo_rules"] = combo_rules

    g = StateGraph(
        ICD10CMCodeGraphStates,
        input=InputState,
        output=OutputState,
    )

    g.add_node("code_prediction", node_coding)
    g.add_node("code_validation", node_validate_and_filter)

    g.add_edge(START, "code_prediction")
    g.add_edge("code_prediction", "code_validation")
    g.add_edge("code_validation", END)

    return g.compile(), model_config


def build_llm_singlestage_enhanced_graph(args: Namespace):
    """
    Returns (CompiledStateGraph, model_config).
    Expects your prompts dict to include the three blocks exactly as provided:
      WORKFLOW.extract_and_queries.v1, WORKFLOW.direct.v1, WORKFLOW.choose.v1
    """
    prompts = getattr(args, "prompts", None) or load_from_yaml(args.prompt_path)
    config = vars(args)

    # load combo code rules
    combo_rules = None
    combo_rules_path = getattr(args, "combo_rules_path", None)
    if combo_rules_path:
        combo_rules = load_from_yaml(combo_rules_path)
        logger.info(
            f"Loaded combo code rules from: {combo_rules_path} " f"({len(combo_rules) if isinstance(combo_rules, dict) else 'unknown'} rules)"
        )

    model_config = {"configurable": {"prompts": prompts, **config}}
    if combo_rules is not None:
        # store rules dict into config so nodes can use it
        model_config["configurable"]["combo_rules"] = combo_rules

    g = StateGraph(
        ICD10CMCodeGraphStates,
        input=InputState,
        output=OutputState,
    )

    g.add_node("code_prediction", node_coding)
    g.add_node("z_code_enhancer", node_z_code_enhancer)
    g.add_node("code_validation", node_validate_and_filter)

    g.add_edge(START, "code_prediction")
    g.add_edge(START, "z_code_enhancer")
    g.add_edge("z_code_enhancer", "code_validation")
    g.add_edge("code_prediction", "code_validation")
    g.add_edge("code_validation", END)

    return g.compile(), model_config
