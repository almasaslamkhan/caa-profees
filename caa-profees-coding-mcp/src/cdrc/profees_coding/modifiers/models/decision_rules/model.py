from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.data.schema import (
    AnaesthesiaConcurrentStaffInfo,
    AnaesthesiaRecordInfo,
    AnaesthesiaStaffInfo,
    EncounterMetaData,
    FacilityInfo,
    HealthPlan,
    LabInfo,
    ProviderInfo,
)
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import ICD10CM, EMItem, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.decision_rules.data_classes import CONFIG

from .nodes.modifiers import ModifierJustification, build_decision_rules_subgraph


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
    payer_name: Optional[List[str]]
    encounter_metadata: Optional[EncounterMetaData]
    health_plan: Optional[List[HealthPlan]]
    facility_info: Optional[List[FacilityInfo]]
    provider_info: Optional[List[ProviderInfo]]
    lab_info: Optional[List[LabInfo]]
    anaesthesia_staff_info: Optional[List[AnaesthesiaStaffInfo]]
    anaesthesia_concurrent_staff_info: Optional[List[AnaesthesiaConcurrentStaffInfo]]
    anaesthesia_record_info: Optional[List[AnaesthesiaRecordInfo]]


class ModifiersOutputGraphState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    unassigned_modifiers: List[ModifierJustification]


class ModifiersGraphState(ModifiersInputGraphState, ModifiersOutputGraphState):
    pass


def get_decision_rules_subgraph(args: Namespace):
    del args
    graph = build_decision_rules_subgraph()
    return graph if hasattr(graph, "invoke") else graph.compile()


def build_decision_rules_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:
    prompts = load_from_yaml(args.prompt_path)
    config = vars(args).copy()
    model_config = {"configurable": {CONFIG.PROMPTS: prompts, **config}}

    workflow = StateGraph(ModifiersGraphState, input=ModifiersInputGraphState, output=ModifiersOutputGraphState)
    workflow.add_node("decision_rules", get_decision_rules_subgraph(args))
    workflow.add_edge(START, "decision_rules")
    workflow.add_edge("decision_rules", END)

    return workflow.compile(), model_config
