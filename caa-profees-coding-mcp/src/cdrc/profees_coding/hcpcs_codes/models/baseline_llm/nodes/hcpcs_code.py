import json
import random
from argparse import Namespace
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import AWVHistory
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.hcpcs_codes.data.schema import HealthPlan, Problem
from cdrc.profees_coding.hcpcs_codes.models.baseline_llm.constants import HCPCS_BRANCH_NAME
from cdrc.profees_coding.hcpcs_codes.models.utils import CONFIG, PROMPTS


class ProcedurePayload(BaseModel):
    description: str = Field(..., description="HCPCS procedure code display")
    code: str = Field(..., description="HCPCS procedure code (e.g., G2211)")
    system: Literal["CPT", "HCPCS"]
    justification: Optional[str] = Field(
        None,
        description=(
            "Brief explanation (3-5 sentences) of why this procedure code ,quote from the note for evidence supported, and is appropriate for the visit, grounded in the note."
            "One sentence (no longer than 15 words) explanation of why this procedure code is appropriate for the visit summarising key aspects from the note."
        ),
    )


class HCPCSDebug(BaseModel):
    assigned: bool = Field(..., description="Whether any HCPCS code should be assigned")
    justification: str = Field(..., description="5-7 sentences grounded in the note explaining why assigned or not")
    reason_codes: List[str] = Field(default_factory=list, description="Short stable codes for analytics")
    evidence: List[str] = Field(default_factory=list, description="Short quotes or section refs")


class OutputModel(BaseModel):
    procedure: List[ProcedurePayload] = Field(default_factory=list, description="List of HCPCS procedure codes")
    hcpcs_debug: Optional[HCPCSDebug] = None


class InputState(dict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    health_plan: Optional[List[HealthPlan]]
    awv_history: Optional[List[AWVHistory]]
    encounter_date: Optional[str]
    awv_eligibility: Optional[Mapping[str, Any]]
    all_patient_problems: Optional[List[Problem]]
    additional_input: Optional[Mapping[str, Any]]


class OutputState(dict):
    procedure: List[ProcedurePayload] = Field(default_factory=list, description="List of HCPCS procedure codes")
    customer_name: Optional[str]
    additional_output: Optional[Mapping[str, Any]] = None


class HCPCSCodeGraphStates(InputState, OutputState):
    pass


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


def _get_customer_name_from_state(state: HCPCSCodeGraphStates) -> Optional[str]:
    additional_input = state.get("additional_input", {})
    if isinstance(additional_input, dict):
        customer_name = additional_input.get("customer_name", None)
    elif isinstance(additional_input, BaseModel):
        customer_name = additional_input.customer_name
    else:
        customer_name = None

    return customer_name


def _coerce_awv_history(item: Any) -> Optional[AWVHistory]:
    """Convert various payload types into AWVHistory, requiring both hcpcs_code and claim_date."""
    if item is None:
        return None

    try:
        if isinstance(item, AWVHistory):
            history = item
        elif isinstance(item, BaseModel):
            history = AWVHistory(**item.model_dump())
        elif isinstance(item, dict):
            history = AWVHistory(**item)
        else:
            return None
    except (TypeError, ValueError, ValidationError):
        return None

    if not history.hcpcs_code or not history.claim_date:
        return None

    return history


def _get_awv_history_from_state(state: HCPCSCodeGraphStates) -> Optional[List[AWVHistory]]:
    """Return AWV history regardless of whether it's stored in standard input or additional_input."""
    raw_history: Optional[Any] = state.get("awv_history")

    if not raw_history:
        additional_input = state.get("additional_input")
        if isinstance(additional_input, dict):
            raw_history = additional_input.get("awv_history") or additional_input.get("hcpcs_history")
        elif isinstance(additional_input, BaseModel):
            raw_history = getattr(additional_input, "awv_history", None) or getattr(additional_input, "hcpcs_history", None)

    if raw_history is None:
        return None

    if isinstance(raw_history, BaseModel):
        raw_history = [raw_history]

    if isinstance(raw_history, (set, tuple)):
        raw_history = [item for item in raw_history if item is not None]
    elif not isinstance(raw_history, list):
        raw_history = [raw_history]

    normalized: List[AWVHistory] = []
    for item in raw_history:
        history = _coerce_awv_history(item)
        if history:
            normalized.append(history)

    return normalized or None


def _normalize_prompt_version(prompt_version_config: Union[str, Mapping[str, Any]]) -> Tuple[str, str, Optional[Tuple[str, ...]]]:
    """Normalize prompt config into template key, label, and supported codes."""
    if isinstance(prompt_version_config, str):
        return prompt_version_config, prompt_version_config, None

    if not isinstance(prompt_version_config, Mapping):
        raise TypeError("prompt_version must be a string or mapping with 'version' key")

    template_key = prompt_version_config.get("version")
    if not template_key:
        raise ValueError("prompt_version mapping must include a 'version' field")

    raw_codes: Union[None, str, Sequence[str]] = prompt_version_config.get("codes_supported")
    if raw_codes is None:
        raw_codes = prompt_version_config.get("codes")

    codes_tuple: Optional[Tuple[str, ...]]
    if raw_codes is None:
        codes_tuple = None
    elif isinstance(raw_codes, str):
        codes_tuple = tuple(code.strip() for code in raw_codes.split(",") if code.strip()) or None
    else:
        codes_tuple = tuple(str(code) for code in raw_codes)

    label = prompt_version_config.get("label")
    if not label:
        if codes_tuple:
            label = f"{template_key} | {', '.join(codes_tuple)}"
        else:
            label = template_key

    return template_key, label, codes_tuple


def _stable_prompt_value(value: Any) -> str:
    """Render prompt inputs deterministically to improve cache hit stability."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


@track_tokens("model_usage")
def hcpcs_code(state: HCPCSCodeGraphStates, config: RunnableConfig) -> dict:
    prompt_version_raw = config["configurable"][CONFIG.PROMPT_VERSION]
    prompt_version_key, prompt_version_label, supported_codes = _normalize_prompt_version(prompt_version_raw)
    branch_name = config["configurable"].get(HCPCS_BRANCH_NAME)
    include_justification = _as_bool(config["configurable"].get("include_justification", False), default=False)
    enable_prompt_cache_layout = _as_bool(config["configurable"].get("enable_prompt_cache_layout", True), default=True)

    awv_history = _get_awv_history_from_state(state)
    awv_history_for_prompt = awv_history if awv_history else ""
    awv_eligibility_for_prompt = state.get("awv_eligibility") or ""

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=OutputModel, correction_model=model, num_retries=3)
    chain = model | parser
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version_key]["system"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version_key]["user"], template_format="jinja2"
    )

    format_instructions = parser.get_format_instructions()
    if include_justification:
        format_instructions += (
            "\n\nDEBUG MODE INSTRUCTION:\n"
            "IMPORTANT: Always include `hcpcs_debug` even if procedure list is empty.\n"
            "- For every procedure object in the 'procedure' list, include a "
            "non-empty 'justification' field.\n"
            "- The justification must be 1-3 sentences.\n"
            "- The justification must be clear, concise and no leading justification.\n"
        )

    if enable_prompt_cache_layout:
        cacheable_output_schema_message = HumanMessage(
            content=(
                "## Output Format\n"
                "Return your answer as a JSON object matching this schema:\n"
                f"{format_instructions}\n"
                "DO NOT include any other text in your response. Only return the JSON object."
            )
        )
        messages = [
            system_prompt.format(),
            cacheable_output_schema_message,
            user_prompt.format(
                note=_stable_prompt_value(state.get("text", "")),
                patient_age=_stable_prompt_value(state.get("patient_age", "")),
                patient_sex=_stable_prompt_value(state.get("patient_sex", "")),
                orders_display=_stable_prompt_value(state.get("orders_display", "")),
                health_plan=_stable_prompt_value(state.get("health_plan", "")),
                awv_history=_stable_prompt_value(awv_history_for_prompt),
                encounter_date=_stable_prompt_value(state.get("encounter_date", "")),
                awv_eligibility=_stable_prompt_value(awv_eligibility_for_prompt),
                additional_input=_stable_prompt_value(state.get("additional_input", "")),
                format_instructions="(Output schema is provided in the previous message.)",
            ),
        ]
    else:
        messages = [
            system_prompt.format(),
            user_prompt.format(
                note=state.get("text", ""),
                patient_age=state.get("patient_age", ""),
                patient_sex=state.get("patient_sex", ""),
                orders_display=state.get("orders_display", ""),
                health_plan=state.get("health_plan", ""),
                awv_history=awv_history_for_prompt,
                encounter_date=state.get("encounter_date", ""),
                awv_eligibility=awv_eligibility_for_prompt,
                additional_input=state.get("additional_input", ""),
                format_instructions=format_instructions,
            ),
        ]

    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        merged_parts = [messages[0].content]
        if enable_prompt_cache_layout and len(messages) > 2:
            merged_parts.append(messages[1].content)
            user_message_index = 2
        else:
            user_message_index = 1
        merged_parts.append(messages[user_message_index].content)
        merged_content = "\n\n".join(part for part in merged_parts if part)
        user_message = messages[user_message_index]
        user_message.content = merged_content
        messages = [user_message]

    response = chain.invoke(messages)

    hcpcs_assigned = bool(response.procedure)

    additional_output = None
    if include_justification:
        additional_output = {}

        debug_payload = (
            response.hcpcs_debug.model_dump()
            if response.hcpcs_debug is not None
            else {
                "assigned": hcpcs_assigned,
                "justification": "DEBUG: Model did not provide hcpcs_debug despite instruction.",
                "reason_codes": ["MISSING_DEBUG_PAYLOAD"],
                "evidence": [],
            }
        )

        debug_payload["assigned"] = hcpcs_assigned

        additional_output["hcpcs_debug"] = debug_payload
        if awv_eligibility_for_prompt:
            additional_output["awv_eligibility"] = awv_eligibility_for_prompt
        additional_output["debug_meta"] = {
            "branch_name": branch_name,
            "prompt_version": prompt_version_label,
            "prompt_version_key": prompt_version_key,
            "codes_supported": list(supported_codes) if supported_codes else None,
            "model_name": config["configurable"][CONFIG.MODEL_NAME],
            "seed": model_kwargs.get("seed"),
        }

    out = {
        "procedure": [p.model_dump() for p in response.procedure],
        "customer_name": _get_customer_name_from_state(state),
    }
    if additional_output is not None:
        out["additional_output"] = additional_output

    logger.debug("hcpcs_code output for example_id=%s -> %s", state.get("example_id"), out)

    return out


def build_hcpcs_code_subgraph(args: Namespace):
    workflow = StateGraph(HCPCSCodeGraphStates)
    workflow.add_node("classifier", hcpcs_code)
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)
    return workflow.compile()
