import re
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.agent_tools import is_valid_item as tool_is_valid
from cdrc.common.llm_utils import track_tokens
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure

# --------------------------------------------------------------------------------
# Defining the LangGraph subgraph.
# --------------------------------------------------------------------------------


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class PostProcessingStateGraph(IntermediateState, OutputState):
    pass


def build_post_processing_subgraph(args):

    # Creating the association coding LangGraph.
    workflow = StateGraph(PostProcessingStateGraph)

    # Creating the nodes for the graph.
    workflow.add_node("validate_code", validate_code_node)
    workflow.add_node("cpt_procedure_filtering", cpt_procedure_ncd_filtering_node)
    workflow.add_node("hcpcs_procedure_filtering", hcpcs_procedure_ncd_filtering_node)
    workflow.add_node("payload_contruction", payload_contruction_node)

    # Defining the graph flow from the nodes.
    workflow.add_edge(START, "validate_code")
    workflow.add_edge("validate_code", "cpt_procedure_filtering")
    workflow.add_edge("cpt_procedure_filtering", "hcpcs_procedure_filtering")
    workflow.add_edge("hcpcs_procedure_filtering", "payload_contruction")
    workflow.add_edge("payload_contruction", END)

    return workflow.compile()


# --------------------------------------------------------------------------------
# Code validation nodes
# --------------------------------------------------------------------------------


def validate_code_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if diagnosis code validation is not enabled.
    if not config["configurable"].get("validate_codes", True):
        return state

    # Cache normalized code -> keep status to avoid repeated validator lookups.
    code_validation_cache = {}

    # Determine whether the diagnosis code exists and is billable.
    def _is_valid_and_billable(raw_code):

        # Normalizing the incoming diagnosis code.
        diagnosis_code = str(raw_code or "").strip().upper()

        # Empty diagnosis codes are always invalid.
        if not diagnosis_code:
            return False

        # Reuse previous validation result when the diagnosis code was already seen.
        if diagnosis_code not in code_validation_cache:
            code_validation_cache[diagnosis_code] = bool(tool_is_valid(code=diagnosis_code) and tool_is_billable(code=diagnosis_code))

        # Return the cached validity/billable status.
        return code_validation_cache[diagnosis_code]

    # Validate linked diagnoses for both E/M and procedure association lists.
    for association_field in ["em_associations", "procedure_associations"]:

        # Updated list of associations after diagnosis-level validation.
        updated_associations = []

        # Iterate over each association item and retain only valid billable diagnoses.
        for association in state.get(association_field) or []:

            # Converting each association payload into dictionary format.
            association_dictionary = _dump_to_dictionary(association)

            # Extracting the linked diagnosis list from the association output.
            linked_diagnoses = association_dictionary.get("linked_diagnoses", [])

            # Collecting only valid and billable diagnoses for this association.
            validated_linked_diagnoses = []

            # Iterating over each linked diagnosis predicted for the current association.
            for linked_dx in linked_diagnoses:

                # Converting linked diagnosis payload into dictionary format.
                linked_dx_dictionary = _dump_to_dictionary(linked_dx)

                # Extracting and normalizing the diagnosis code for validation.
                diagnosis_code = str(linked_dx_dictionary.get("code", "")).strip().upper()

                # Retain only linked diagnoses that are both valid and billable.
                if _is_valid_and_billable(diagnosis_code):
                    linked_dx_dictionary["code"] = diagnosis_code
                    validated_linked_diagnoses.append(linked_dx_dictionary)

            # Replace original linked diagnoses with the validated subset.
            association_dictionary["linked_diagnoses"] = validated_linked_diagnoses

            # Appending the updated association to the tracking list.
            updated_associations.append(association_dictionary)

        # Updating the state with validated associations.
        state[association_field] = updated_associations

    return state


def cpt_procedure_ncd_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if filtering is not enabled for CPT procedures.
    if not config["configurable"].get("cpt_procedure_filtering", False):
        return state

    # Accessing the resources value.
    resources = config["configurable"].get("resources", {})

    # Accessing the CPT->ICD national coverage mapping.
    cpt_icd_mapping = resources.get("cpt_icd_mapping", {})

    # Build code->system lookup from the procedure payload.
    procedure_system_lookup = _build_procedure_system_lookup(state.get("procedure", []))

    # The updated procedure association list, with CPT-compliant linked diagnoses.
    updated_procedure_associations = []

    # Iterate over each procedure association item predicted by the model.
    for procedure_association in state.get("procedure_associations") or []:
        association_dictionary = _dump_to_dictionary(procedure_association)
        code = str(association_dictionary.get("code", "")).strip().upper()
        linked_diagnoses = association_dictionary.get("linked_diagnoses", [])

        # If this association belongs to a CPT procedure, apply NCD filtering.
        if procedure_system_lookup.get(code, "") == "CPT":

            # If the code is not in the map or has an empty mapping list, leave as-is.
            if code in cpt_icd_mapping and len(cpt_icd_mapping[code]) > 0:
                compliant_linked_diagnoses = []
                for linked_dx in linked_diagnoses:
                    linked_dx_dictionary = _dump_to_dictionary(linked_dx)
                    dx_code = str(linked_dx_dictionary.get("code", "")).strip().upper()
                    if dx_code in cpt_icd_mapping[code]:
                        compliant_linked_diagnoses.append(linked_dx_dictionary)

                # Replace model-linked diagnoses with NCD-compliant subset.
                association_dictionary["linked_diagnoses"] = compliant_linked_diagnoses

        updated_procedure_associations.append(association_dictionary)

    # Updating the state with the new procedure associations.
    state["procedure_associations"] = updated_procedure_associations

    return state


def hcpcs_procedure_ncd_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Early exit if filtering is not enabled for HCPCS procedures.
    if not config["configurable"].get("hcpcs_procedure_filtering", False):
        return state

    # Safely accessing the resources value.
    resources = config["configurable"].get("resources", {})

    # Safely accessing the HCPCS-ICD national coverage mapping.
    hcpcs_icd_mapping = resources.get("hcpcs_icd_mapping", {})

    # Build code->system lookup from the procedure payload.
    procedure_system_lookup = _build_procedure_system_lookup(state.get("procedure", []))

    # The updated procedure association list, with HCPCS-compliant linked diagnoses.
    updated_procedure_associations = []

    # Iterate over each procedure association item predicted by the model.
    for procedure_association in state.get("procedure_associations") or []:
        association_dictionary = _dump_to_dictionary(procedure_association)
        code = str(association_dictionary.get("code", "")).strip().upper()
        linked_diagnoses = association_dictionary.get("linked_diagnoses", [])

        # If this association belongs to an HCPCS procedure, apply NCD filtering.
        if procedure_system_lookup.get(code, "") == "HCPCS":

            # If the code is not in the map or has an empty mapping list, leave as-is.
            if code in hcpcs_icd_mapping and len(hcpcs_icd_mapping[code]) > 0:
                compliant_linked_diagnoses = []
                for linked_dx in linked_diagnoses:
                    linked_dx_dictionary = _dump_to_dictionary(linked_dx)
                    dx_code = str(linked_dx_dictionary.get("code", "")).strip().upper()
                    if dx_code in hcpcs_icd_mapping[code]:
                        compliant_linked_diagnoses.append(linked_dx_dictionary)

                # Replace model-linked diagnoses with NCD-compliant subset.
                association_dictionary["linked_diagnoses"] = compliant_linked_diagnoses

        updated_procedure_associations.append(association_dictionary)

    # Updating the state with the new procedure associations.
    state["procedure_associations"] = updated_procedure_associations

    return state


# --------------------------------------------------------------------------------
# Payload Construction Node
# --------------------------------------------------------------------------------


@track_tokens("model_usage")
def payload_contruction_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Accessing the ICD-10 description mapping resource.
    resources = config["configurable"].get("resources", {})
    icd10_descriptions = resources.get("icd10_descriptions", {})

    # Rebuilding full diagnosis objects from linked diagnoses in the association output.
    diagnosis = _rebuild_diagnosis_list(state.get("em_associations", []), state.get("procedure_associations", []), icd10_descriptions)

    # Build a lookup dictionary of all the ICD-10 codes and their content.
    diagnoses = _build_diagnosis_mapping(diagnosis)

    # Build a lookup dictionary of all CPT-E/M and procedure charges.
    em_associations = _build_association_mapping(state.get("em_associations", []), diagnoses)
    procedure_associations = _build_association_mapping(state.get("procedure_associations", []), diagnoses)

    # Building the final em list with the new linked diagnoses field.
    em = []
    for em_charge in state.get("em") or []:
        charge = _dump_to_dictionary(em_charge) or {}
        charge_code = str(charge.get("code", "")).strip().upper()
        charge["linked_diagnosis"] = em_associations.get(charge_code, [])
        em.append(EMItem(**charge))

    # Building the final procedure list with the new linked diagnoses field.
    procedure = []
    for procedure_charge in state.get("procedure") or []:
        charge = _dump_to_dictionary(procedure_charge) or {}
        charge_code = str(charge.get("code", "")).strip().upper()
        charge["linked_diagnosis"] = procedure_associations.get(charge_code, [])
        procedure.append(Procedure(**charge))

    # Updating the procedure and em state.
    state["em"] = em
    state["procedure"] = procedure
    state["diagnosis"] = diagnosis

    # Trimming intermediate association state from the final payload.
    state.pop("em_associations", None)
    state.pop("procedure_associations", None)

    return state


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def _dump_to_dictionary(x):
    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_procedure_system_lookup(procedure_list):

    # Build code -> system lookup from the incoming procedure list.
    lookup = {}
    for procedure in procedure_list or []:
        procedure_dictionary = _dump_to_dictionary(procedure)
        code = str(procedure_dictionary.get("code", "")).strip().upper()
        system = str(procedure_dictionary.get("system", "")).strip().upper()
        if code and system:
            lookup[code] = system
    return lookup


def _build_diagnosis_mapping(diagnosis_list):

    # Build a lookup dictionary of all the ICD-10 codes and their content.
    diagnoses = {}

    # Iterating over the diagnosis list.
    for dx in diagnosis_list:

        # Converting the dx object into a dictionary.
        dx_dictionary = _dump_to_dictionary(dx)

        # Extracting the ICD-10 code for the diagnosis dictionary.
        diagnosis_code = str(dx_dictionary.get("code", "")).strip().upper()

        # Mapping from ICD-10 code to content (is_primary, system, description, justification).
        diagnoses[diagnosis_code] = dx_dictionary

    return diagnoses


def _rebuild_diagnosis_list(em_associations, procedure_associations, icd10_descriptions):

    # The deduped diagnosis list keyed by normalized ICD-10 code.
    seen_codes = set()
    diagnoses = []

    # Iterating over linked diagnoses in E/M associations first, followed by procedure associations.
    for charge_list in [em_associations or [], procedure_associations or []]:
        for charge in charge_list:
            charge_dictionary = _dump_to_dictionary(charge)
            for linked_dx in charge_dictionary.get("linked_diagnoses") or []:
                linked_diagnosis = _dump_to_dictionary(linked_dx)
                diagnosis_code = str(linked_diagnosis.get("code", "")).strip().upper()

                # Skip empty, malformed, and duplicate diagnosis codes.
                if not diagnosis_code or not re.match(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$", diagnosis_code):
                    continue
                if diagnosis_code in seen_codes:
                    continue

                # Tracking the diagnosis code so only the first instance is retained.
                seen_codes.add(diagnosis_code)

                # Building the full diagnosis schema from compact association output.
                diagnoses.append(
                    {
                        "code": diagnosis_code,
                        "is_primary": False,
                        "system": "ICD-10-CM",
                        "description": icd10_descriptions.get(diagnosis_code, ""),
                        "justification": linked_diagnosis.get("justification", ""),
                    }
                )

    return [ICD10CM(**diagnosis) for diagnosis in diagnoses]


def _build_association_mapping(charge_list, diagnoses):

    # Dictionary storing the mapping between E/M or procedure code to their linked diagnoses.
    associations = {}

    # Iterating over each CPT-E/M code and processing its associations.
    for charge in charge_list:

        # Converting to dictionary and normalizing codes.
        charge_dictionary = _dump_to_dictionary(charge)

        # Extracting the code from the charge item.
        charge_code = str(charge_dictionary.get("code", "")).strip().upper()

        # Build lookup table which maps each code to their linked diagnoses.
        if charge_code:

            # Storing the updated linked diagnoses.
            linked_diagnoses = []

            # Iterating over each linked diagnosis for the current code.
            for linked_dx in charge_dictionary.get("linked_diagnoses") or []:

                # Dumping each linked dx to a dictionary.
                linked_diagnosis = _dump_to_dictionary(linked_dx)

                # Extracting the linked diagnose code.
                dx_code = str(linked_diagnosis.get("code", "")).strip().upper()

                # Only include the association if the dx code is in the diagnoses list.
                if dx_code in diagnoses:

                    # Mark primary based on the first retained linked diagnosis after filtering.
                    is_primary = len(linked_diagnoses) == 0

                    # Extracting the code system field.
                    system = diagnoses.get(dx_code, {}).get("system", "")

                    # Extracting the diagnosis description field.
                    description = diagnoses.get(dx_code, {}).get("description", "")

                    # Extracting the association justification.
                    justification = linked_diagnosis.get("justification", "")

                    # Appending to the tracking list.
                    linked_diagnoses.append(
                        {"code": dx_code, "is_primary": is_primary, "system": system, "description": description, "justification": justification}
                    )

            # Adding new code -> linked diagnosis mapping.
            associations[charge_code] = linked_diagnoses

    return associations
