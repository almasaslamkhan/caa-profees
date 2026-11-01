from argparse import Namespace
from typing import Annotated, Any, List, Literal, Optional, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.cpt_procedure.data.schema import Order, Problem, Procedure

from .data_classes import CONFIG
from .nodes import build_cpt_procedure_code_subgraph


# ------------------------ defining the graph state ---------------------------
class CPTProcedureInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(description="The list of orders for the encounter displayed as a single string for convenience.")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")


class CPTProcedureOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT Procedure codes")


class CPTProcedureGraphState(CPTProcedureInputGraphState, CPTProcedureOutputGraphState):
    pass


def get_cpt_procedure_code_subgraph(args: Namespace):
    graph = build_cpt_procedure_code_subgraph(args)
    return graph


# ------------------------ defining the graph builder ---------------------------
def build_llm_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(CPTProcedureGraphState, input=CPTProcedureInputGraphState, output=CPTProcedureOutputGraphState)

    workflow.add_node("cpt_procedure_code_classifier", get_cpt_procedure_code_subgraph(args))

    workflow.add_edge(START, "cpt_procedure_code_classifier")

    workflow.add_edge("cpt_procedure_code_classifier", END)

    return workflow.compile(), model_config
