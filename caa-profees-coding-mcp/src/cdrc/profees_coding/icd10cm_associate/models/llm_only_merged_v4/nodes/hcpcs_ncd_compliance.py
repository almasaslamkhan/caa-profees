import copy
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")
    not_hcpcs_ncd_compliant_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to HCPCS NCD non-compliance."
    )


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class PostProcessingStateGraph(IntermediateState, OutputState):
    pass


# --------------------------------------------------------------------------------
# HCPCS NCD Filtering Node
# --------------------------------------------------------------------------------


def hcpcs_procedure_ncd_filtering_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Tracking HCPCS NCD-noncompliant diagnoses from this filtering stage.
    not_hcpcs_ncd_compliant_associations = list(state.get("not_hcpcs_ncd_compliant_associations") or [])

    # Early exit if filtering is not enabled for HCPCS procedures.
    if not config["configurable"].get("hcpcs_procedure_filtering", False):
        state["not_hcpcs_ncd_compliant_associations"] = not_hcpcs_ncd_compliant_associations
        if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
            state["intermediate_hcpcs_ncd_associations"] = _build_association_snapshot(
                state.get("em_associations", []),
                state.get("procedure_associations", []),
            )
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

                    # Normalizing each linked diagnosis payload before compliance checks.
                    linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}
                    dx_code = str(linked_dx_dictionary.get("code", "")).strip().upper()
                    linked_dx_dictionary["code"] = dx_code

                    # Keeping only diagnosis codes that satisfy HCPCS NCD eligibility rules.
                    if dx_code in hcpcs_icd_mapping[code]:
                        compliant_linked_diagnoses.append(linked_dx_dictionary)
                    else:
                        # Logging diagnosis dropped because it failed HCPCS NCD compliance.
                        if code and dx_code:
                            not_hcpcs_ncd_compliant_associations.append((code, dx_code))

                # Replace model-linked diagnoses with NCD-compliant subset.
                association_dictionary["linked_diagnoses"] = compliant_linked_diagnoses

        updated_procedure_associations.append(association_dictionary)

    # Updating the state with the new procedure associations.
    state["procedure_associations"] = updated_procedure_associations
    state["not_hcpcs_ncd_compliant_associations"] = not_hcpcs_ncd_compliant_associations

    # Persist a deep-copied snapshot for report export when debugging is enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_hcpcs_ncd_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    return state


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def _dump_to_dictionary(x):
    # Helper: Obtain dict representation from pydantic or regular objects.
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_association_snapshot(em_associations, procedure_associations):

    # Snapshot as plain deep-copied dictionaries so later node mutations don't back-propagate.
    return {
        "em_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (em_associations or [])],
        "procedure_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (procedure_associations or [])],
    }


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
