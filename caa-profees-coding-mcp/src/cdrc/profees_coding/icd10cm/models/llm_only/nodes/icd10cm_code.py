import random
from argparse import Namespace
from typing import Any, List, Optional, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.em.models.utils import get_chain  # reuses your chain factory
from cdrc.profees_coding.icd10cm.data.schema import ICD10CM, Order, Problem
from cdrc.profees_coding.icd10cm.models.llm_only.data_classes import CONFIG, PROMPTS

# -------- Pydantic IO models for the LLM chain (clean & minimal) --------


class InputModel(BaseModel):
    example_id: Optional[Union[int, str]] = None
    text: str
    patient_sex: Optional[str] = None
    orders: Optional[List[Order]] = None
    orders_display: Optional[str] = None
    all_patient_problems: Optional[List[Problem]] = None


class OutputModel(BaseModel):
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of ICD-10-CM codes")


# -------------- LangGraph state (TypedDict) for this subgraph --------------


class InputState(dict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]


class OutputState(dict):
    diagnosis: List[ICD10CM]


class ICD10CMCodeGraphStates(InputState, OutputState):
    pass


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


# ------------------------ Node ---------------------------


@track_tokens("model_usage")
def icd10cm_code(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=OutputModel, correction_model=model, num_retries=3)
    chain = model | parser
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["system"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.LLM_ONLY][prompt_version]["user"], template_format="jinja2"
    )
    messages = [
        system_prompt.format(),
        user_prompt.format(
            note=state["text"],
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        # o1-preview and mini do not support system messages
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    response = chain.invoke(messages)

    return {"diagnosis": response.diagnosis}


# ------------------------ Subgraph builder ---------------------------


def build_icd10cm_code_subgraph(args: Namespace):
    workflow = StateGraph(ICD10CMCodeGraphStates)
    workflow.add_node("classifier", icd10cm_code)
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)
    return workflow.compile()
