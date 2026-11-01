from typing import List, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure

# --------------------------------------------------------------------------------
# Defining the LangGraph subgraph.
# --------------------------------------------------------------------------------


class PostProcessingStateGraph(TypedDict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


def build_post_processing_subgraph(args):

    # Creating the association coding LangGraph.
    workflow = StateGraph(PostProcessingStateGraph)

    # Creating the nodes for the graph.
    workflow.add_node("cpt_procedure_filtering", cpt_procedure_filtering_node)
    workflow.add_node("hcpcs_procedure_filtering", hcpcs_procedure_filtering_node)
    workflow.add_node("diagnosis_filtering", diagnosis_filtering_node)

    # Defining the graph flow from the nodes.
    workflow.add_edge(START, "cpt_procedure_filtering")
    workflow.add_edge("cpt_procedure_filtering", "hcpcs_procedure_filtering")
    workflow.add_edge("hcpcs_procedure_filtering", "diagnosis_filtering")
    workflow.add_edge("diagnosis_filtering", END)

    return workflow.compile()


def cpt_procedure_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if filtering is not enabled for CPT procedures.
    if not config["configurable"].get("cpt_procedure_filtering", False):
        return state

    # Accessing the resources value.
    resources = config["configurable"].get("resources", {})

    # Accessing the CPT->ICD national coverage mapping.
    cpt_icd_mapping = resources.get("cpt_icd_mapping", {})

    # The updated procedure list, with compliant CPT procedures.
    updated_procedures = []

    # Iterating over each of the procedures in the list.
    for procedure in state.get("procedure") or []:

        # Dumping the procedure object into a dictionary.
        procedure_dictionary = _dump_to_dictionary(procedure)

        # Checking to see if the procedure is a CPT procedure.
        if procedure_dictionary.get("system", "") == "CPT":

            # Extracting the procedure code, and linked diagnosis.
            code = procedure_dictionary.get("code", "")
            linked_diagnoses = procedure_dictionary.get("linked_diagnosis", [])

            # If the code is not in the map or the code has an empty list then skip.
            if code in cpt_icd_mapping and len(cpt_icd_mapping[code]) > 0:

                # The updated list of linked diagnoses.
                compliant_linked_diagnoses = []

                # Iterating over each of the linked diagnoses
                for linked_dx in linked_diagnoses:

                    # Checking to see the dx code is NCD compliant for this procedure.
                    if linked_dx.get("code", "") in cpt_icd_mapping[code]:
                        compliant_linked_diagnoses.append(linked_dx)

                # Replace the old linked_diagnosis list with the compliant linked dx list.
                procedure_dictionary["linked_diagnosis"] = compliant_linked_diagnoses

        # Adding the updated procedure to the list.
        updated_procedures.append(procedure_dictionary)

    # Updating the state with the new procedures.
    state["procedure"] = updated_procedures

    return state


def hcpcs_procedure_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if filtering is not enabled for HCPCS procedures.
    if not config["configurable"].get("hcpcs_procedure_filtering", False):
        return state

    # Safely accessing the resources value.
    resources = config["configurable"].get("resources", {})

    # Safely accessing the HCPCS-ICD national coverage mapping.
    hcpcs_icd_mapping = resources.get("hcpcs_icd_mapping", {})

    # The updated procedure list, with compliant HCPCS procedures.
    updated_procedures = []

    # Iterating over each of the procedures in the list.
    for procedure in state.get("procedure") or []:

        # Dumping the procedure object into a dictionary.
        procedure_dictionary = _dump_to_dictionary(procedure)

        # Checking to see if the procedure is a HCPCS procedure.
        if procedure_dictionary.get("system", "") == "HCPCS":

            # Extracting the procedure code, and linked diagnosis.
            code = procedure_dictionary.get("code", "")
            linked_diagnoses = procedure_dictionary.get("linked_diagnosis", [])

            # If the code is not in the map or the code has an empty list then skip.
            if code in hcpcs_icd_mapping and len(hcpcs_icd_mapping[code]) > 0:

                # The updated list of linked diagnoses.
                compliant_linked_diagnoses = []

                # Iterating over each of the linked diagnoses
                for linked_dx in linked_diagnoses:

                    # Checking to see the dx code is NCD compliant for this procedure.
                    if linked_dx.get("code", "") in hcpcs_icd_mapping[code]:
                        compliant_linked_diagnoses.append(linked_dx)

                # Replace the old linked_diagnosis list with the compliant linked dx list.
                procedure_dictionary["linked_diagnosis"] = compliant_linked_diagnoses

        # Adding the updated procedure to the list.
        updated_procedures.append(procedure_dictionary)

    # Updating the state with the new procedures.
    state["procedure"] = updated_procedures

    return state


def diagnosis_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if diagnosis filtering is not enabled
    if not config["configurable"].get("remove_unassociated_diagnoses", False):
        return state

    # Extracting out the current diagnosis list.
    diagnoses = state.get("diagnosis", [])

    # Extracting out the charges (em, and procedure codes).
    charges = state.get("em", []) + state.get("procedure", [])

    # Set which is storing all the unique associated diagnoses.
    associated_diagnoses = set()

    # Iterating over each of the charges.
    for charge in charges:

        # Dumping the charge object into a dictionary.
        charge_dictionary = _dump_to_dictionary(charge)

        # Iterating over each of the linked diagnoses for that charge.
        for diagnosis in charge_dictionary.get("linked_diagnosis", []):

            # Adding the ICD-10 code.
            code = diagnosis.get("code", "")
            associated_diagnoses.add(code)

    # The filterd list of diagnoses (i.e., no unassociated diagnoses).
    filtered_diagnoses = []

    # Dumping the charge object into a dictionary.
    diagnoses_dictionary = _dump_to_dictionary(diagnoses)

    # Iterating over all the original diagnoses.
    for diagnosis in diagnoses_dictionary:

        # If the diagnosis is in the associated set then add to updated list.
        if diagnosis.code in associated_diagnoses:
            filtered_diagnoses.append(diagnosis)

    return state


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def _dump_to_dictionary(x):
    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)
