from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Order, Problem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.llm_only_v1.nodes.association_coding import build_associate_coding_subgraph


# Represents the input state for the ICD-10-CM association graph.
class AssociationCodingInputState(TypedDict):

    # The SOAP note and EHR data from an encounter.
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(description="The list of orders for the encounter displayed as a single string for convenience.")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")

    # The output from the procedure, CPT-E/M, diagnosis model.
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


# Represents the output state from the ICD-10-CM association graph.
class AssociationCodingOutputState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


# Merged state combining input and output, used by the StateGraph as the canonical data object.
class AssociationCodingGraphState(AssociationCodingInputState, AssociationCodingOutputState):
    pass


# Defining the Langgraph graph.
def build_llm_only_v1_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:

    # Load prompt templates from a YAML configuration file specified in args.
    prompts = load_from_yaml(args.prompt_path)

    # Convert Namespace (usually argparse output) to a dictionary for configuration.
    config = vars(args)

    # Combine prompts and other configs into a LLM/configurable structure for LangGraph.
    model_config = {"configurable": {"prompts": prompts, **config}}

    # Defines expected input and output schemas for the graph.
    workflow = StateGraph(AssociationCodingGraphState, input=AssociationCodingInputState, output=AssociationCodingOutputState)

    # Add the main node: an association coding classifier built dynamically according to args.
    workflow.add_node("association_coding_classifier", build_associate_coding_subgraph(args))

    # Connect the starting edge to the classifier node.
    workflow.add_edge(START, "association_coding_classifier")

    # Connect the classifier's output to the END state.
    workflow.add_edge("association_coding_classifier", END)

    # Compile and return the graph with config for downstream execution.
    return workflow.compile(), model_config
