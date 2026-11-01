from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import ICD10CM, EMItem, Order, Problem, Procedure

from .nodes import build_modifiers_subgraph
from .nodes.modifiers import ModifierJustification


# ------------------------ defining the graph state ---------------------------
class ModifiersInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(description="The list of orders for the encounter displayed as a single string for convenience.")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class ModifiersOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    unassigned_modifiers: List[ModifierJustification]


class ModifiersGraphState(ModifiersInputGraphState, ModifiersOutputGraphState):
    pass


def get_modifiers_subgraph(args: Namespace):
    graph = build_modifiers_subgraph(args)
    return graph


# ------------------------ defining the graph builder ---------------------------
def build_llm_only_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args)
    model_config = {"configurable": {"prompts": prompts, **config}}

    workflow = StateGraph(ModifiersGraphState, input=ModifiersInputGraphState, output=ModifiersOutputGraphState)

    workflow.add_node("modifiers_classifier", get_modifiers_subgraph(args))
    workflow.add_edge(START, "modifiers_classifier")
    workflow.add_edge("modifiers_classifier", END)

    return workflow.compile(), model_config
