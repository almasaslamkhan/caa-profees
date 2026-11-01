import json
from argparse import Namespace
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Order, Problem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.code_validation import validate_code_node
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.cpt_ncd_compliance import cpt_procedure_ncd_filtering_node
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.diagnosis_regeneration import diagnosis_regenerator
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.hcpcs_ncd_compliance import hcpcs_procedure_ncd_filtering_node
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.icd_association_coding import build_merged_association_coding_subgraph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.nodes.payload_rebuilding import payload_contruction_node


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

    # The output from the CPT-E/M, and procedure models.
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")


# Represents the output state from the ICD-10-CM association graph.
class AssociationCodingOutputState(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")
    intermediate_raw_associations: Optional[dict] = Field(
        default=None,
        description="Snapshot of raw LLM association output before post-processing.",
    )
    intermediate_validated_associations: Optional[dict] = Field(
        default=None,
        description="Snapshot of associations after ICD-10 validity/billability filtering.",
    )
    intermediate_cpt_ncd_associations: Optional[dict] = Field(
        default=None,
        description="Snapshot of associations after CPT NCD filtering.",
    )
    intermediate_hcpcs_ncd_associations: Optional[dict] = Field(
        default=None,
        description="Snapshot of associations after HCPCS NCD filtering.",
    )
    intermediate_regenerated_associations: Optional[dict] = Field(
        default=None,
        description="Snapshot of associations after diagnosis regeneration.",
    )
    intermediate_regeneration_recommendations: Optional[dict] = Field(
        default=None,
        description="Rule-based recommendation candidates used by diagnosis regeneration.",
    )


class AssociationCodingIntermediateState(TypedDict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")


# Merged state combining input and output, used by the StateGraph as the canonical data object.
class AssociationCodingGraphState(AssociationCodingInputState, AssociationCodingIntermediateState, AssociationCodingOutputState):
    pass


def build_llm_only_merged_v4_graph(args: Namespace) -> tuple[CompiledStateGraph, dict]:

    # Load prompt templates from a YAML configuration file specified in args.
    prompts = load_from_yaml(args.prompt_path)

    # Loading the association coding resources.
    resources = load_association_coding_resources(args)

    # Convert Namespace (usually argparse output) to a dictionary for configuration.
    config = vars(args)

    # Combine prompts and other configs into a LLM/configurable structure for LangGraph.
    model_config = {"configurable": {"prompts": prompts, **config, "resources": resources}}

    # Defining the association coding state graph.
    workflow = StateGraph(AssociationCodingGraphState, input=AssociationCodingInputState, output=AssociationCodingOutputState)

    # Defining the nodes of the graph.
    workflow.add_node("icd_association_coding_classifier", build_merged_association_coding_subgraph(args))
    workflow.add_node("code_validation_stage", validate_code_node)
    workflow.add_node("cpt_ncd_compliance_stage", cpt_procedure_ncd_filtering_node)
    workflow.add_node("hcpcs_ncd_compliance_stage", hcpcs_procedure_ncd_filtering_node)
    workflow.add_node("diagnosis_regeneration_stage", diagnosis_regenerator)
    workflow.add_node("payload_rebuilding_stage", payload_contruction_node)

    # Defining the edges of the graph.
    workflow.add_edge(START, "icd_association_coding_classifier")
    workflow.add_edge("icd_association_coding_classifier", "code_validation_stage")
    workflow.add_edge("code_validation_stage", "cpt_ncd_compliance_stage")
    workflow.add_edge("cpt_ncd_compliance_stage", "hcpcs_ncd_compliance_stage")
    workflow.add_edge("hcpcs_ncd_compliance_stage", "diagnosis_regeneration_stage")
    workflow.add_edge("diagnosis_regeneration_stage", "payload_rebuilding_stage")
    workflow.add_edge("payload_rebuilding_stage", END)

    # Compile and return the graph with config for downstream execution.
    return workflow.compile(), model_config


# --------------------------------------------------------------------------------
# Resource Loading Utility Functions
# --------------------------------------------------------------------------------


def load_association_coding_resources(args):

    # Defining the NCD eligability mapping file paths.
    mapping_paths = {"cpt_icd_mapping": args.cpt_icd_mapping_path, "hcpcs_icd_mapping": args.hcpcs_icd_mapping_path}

    # Defining code description file paths.
    description_paths = {
        "cpt_descriptions": args.cpt_description_path,
        "hcpcs_descriptions": args.hcpcs_description_path,
        "icd10_descriptions": args.icd10_descriptions_path,
    }

    # Merge all loaded data into one resource dictionary.
    return {**load_code_mappings(mapping_paths), **load_code_descriptions(description_paths)}


def load_code_mappings(mapping_paths):
    mappings = {}

    # Iterating over each file and loading it.
    for key, path in mapping_paths.items():

        # Loading the file into memory and converting inner list to a set for fast lookup.
        try:
            with open(path, "r") as f:
                data = json.load(f)
                mappings[key] = {proc: set(icds) for proc, icds in data.items()}

        # If loading fails return an empty dictionary/mapping.
        except Exception:
            mappings[key] = {}

    # Returning all the mappings.
    return mappings


def load_code_descriptions(description_paths):
    descriptions = {}

    # Iterating over each file and loading it.
    for key, path in description_paths.items():

        # Loading the code to description mapping into memory.
        try:
            with open(path, "r") as f:
                descriptions[key] = json.load(f)

        # If loading fails return an empty dictionary/mapping.
        except Exception:
            descriptions[key] = {}

    return descriptions
