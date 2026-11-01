import random
from argparse import Namespace
from typing import Dict, List, TypedDict, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.em.data.schema import EMTimeSpent, EncounterTypeDetails, EncounterTypesDetails, PatientType
from cdrc.profees_coding.em.models.utils import get_chain

from ..data_classes import CONFIG, PROMPTS


def _get_model_kwargs_and_possibly_change_seed(config: Dict) -> Dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


# ------------------------ defining the graph state ---------------------------


class TimeStatementInputGraphState(TypedDict):
    text: str


class TimeStatementOutputGraphState(TypedDict):
    time_spent: EMTimeSpent


class TimeStatementGraphState(TimeStatementInputGraphState, TimeStatementOutputGraphState):
    """Full graph state for time statement subgraph."""

    pass


# ------------------------ defining the graph node ---------------------------


@track_tokens("model_usage")
def time_statement_node(state: TimeStatementInputGraphState, config: RunnableConfig) -> TimeStatementOutputGraphState:
    prompt_version = config["configurable"][CONFIG.TIME_STATEMENT_PROMPT_VERSION]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    # Parser now targets EMTimeSpent
    parser = PydanticOutputParserWithRetries(
        pydantic_object=EMTimeSpent,
        correction_model=model,
        num_retries=3,
    )
    chain = model | parser

    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.TIME_STATEMENT][prompt_version]["system"],
        template_format="jinja2",
    )

    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.TIME_STATEMENT][prompt_version]["user"],
        template_format="jinja2",
    )

    messages = [
        system_prompt.format(),
        user_prompt.format(
            soap_note=state["text"],
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    # o1 models cannot handle system messages
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    response: EMTimeSpent = chain.invoke(messages)

    return {
        "time_spent": response,
    }


# ------------------------ defining the graph builder ---------------------------


def build_time_statement_subgraph(args: Namespace):
    workflow = StateGraph(
        TimeStatementGraphState,
        input=TimeStatementInputGraphState,
        output=TimeStatementOutputGraphState,
    )

    workflow.add_node("time_statement", time_statement_node)

    workflow.add_edge(START, "time_statement")
    workflow.add_edge("time_statement", END)

    return workflow.compile()
