import re
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")
    not_valid_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to invalid ICD-10-CM codes."
    )
    not_billable_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to non-billable ICD-10-CM codes."
    )
    not_cpt_ncd_compliant_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to CPT NCD non-compliance."
    )
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
# Payload Construction Node
# --------------------------------------------------------------------------------


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
                        {
                            "code": dx_code,
                            "is_primary": is_primary,
                            "system": system,
                            "description": description,
                            "justification": justification,
                        }
                    )

            # Adding new code -> linked diagnosis mapping.
            associations[charge_code] = linked_diagnoses

    return associations
