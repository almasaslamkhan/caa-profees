from argparse import Namespace
from typing import Annotated, Any, List, Literal, Optional, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.icd10cm.data.schema import ICD10CM, Order, Problem

from .data_classes import CONFIG
from .nodes import build_icd10cm_code_subgraph


# ------------------------ defining the graph state ---------------------------
class ICD10CMInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(description="The list of orders for the encounter displayed as a single string for convenience.")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")


class ICD10CMOutputGraphState(TypedDict):
    diagnosis: List[ICD10CM] = Field(None, description="List of ICD-10-CM codes")


class ICD10CMGraphState(ICD10CMInputGraphState, ICD10CMOutputGraphState):
    pass


def get_icd10cm_code_subgraph(args: Namespace):
    graph = build_icd10cm_code_subgraph(args)
    return graph


# ------------------------ defining the graph builder ---------------------------
def build_llm_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(ICD10CMGraphState, input=ICD10CMInputGraphState, output=ICD10CMOutputGraphState)

    workflow.add_node("icd10cm_code_classifier", get_icd10cm_code_subgraph(args))

    workflow.add_edge(START, "icd10cm_code_classifier")

    workflow.add_edge("icd10cm_code_classifier", END)

    return workflow.compile(), model_config
