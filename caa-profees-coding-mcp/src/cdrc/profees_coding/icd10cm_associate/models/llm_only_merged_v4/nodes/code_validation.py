import copy
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.agent_tools import is_valid_item as tool_is_valid
from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")
    not_valid_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to invalid ICD-10-CM codes."
    )
    not_billable_associations: Optional[List] = Field(
        default_factory=list, description="List of dropped (charge_code, diagnosis_code) pairs due to non-billable ICD-10-CM codes."
    )


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class PostProcessingStateGraph(IntermediateState, OutputState):
    pass


# --------------------------------------------------------------------------------
# Validate Code Node
# --------------------------------------------------------------------------------


def validate_code_node(state: PostProcessingStateGraph, config: RunnableConfig):

    # Preserve the incoming associations as the raw stage snapshot if not already set upstream.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False) and state.get("intermediate_raw_associations") is None:
        state["intermediate_raw_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    # Tracking dropped diagnoses from this validation stage.
    not_valid_associations = list(state.get("not_valid_associations") or [])
    not_billable_associations = list(state.get("not_billable_associations") or [])

    # Early exit if diagnosis code validation is not enabled.
    if not config["configurable"].get("validate_codes", True):
        state["not_valid_associations"] = not_valid_associations
        state["not_billable_associations"] = not_billable_associations
        if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
            state["intermediate_validated_associations"] = _build_association_snapshot(
                state.get("em_associations", []),
                state.get("procedure_associations", []),
            )
        return state

    # Cache normalized code -> keep status to avoid repeated validator lookups.
    code_validation_cache = {}

    # Determine whether the diagnosis code exists and is billable.
    def _get_validation_status(raw_code):

        # Normalizing the incoming diagnosis code.
        diagnosis_code = str(raw_code or "").strip().upper()

        # Empty diagnosis codes are always invalid.
        if not diagnosis_code:
            return diagnosis_code, False, None

        # Reuse previous validation result when the diagnosis code was already seen.
        if diagnosis_code not in code_validation_cache:
            is_valid = bool(tool_is_valid(code=diagnosis_code))
            is_billable = bool(tool_is_billable(code=diagnosis_code)) if is_valid else None
            code_validation_cache[diagnosis_code] = (is_valid, is_billable)

        # Return normalized code and cached validity/billable status.
        is_valid, is_billable = code_validation_cache[diagnosis_code]
        return diagnosis_code, is_valid, is_billable

    # Validate linked diagnoses for both E/M and procedure association lists.
    for association_field in ["em_associations", "procedure_associations"]:

        # Updated list of associations after diagnosis-level validation.
        updated_associations = []

        # Iterate over each association item and retain only valid billable diagnoses.
        for association in state.get(association_field) or []:

            # Converting each association payload into dictionary format.
            association_dictionary = _dump_to_dictionary(association)
            charge_code = str(association_dictionary.get("code", "")).strip().upper()

            # Extracting the linked diagnosis list from the association output.
            linked_diagnoses = association_dictionary.get("linked_diagnoses", [])

            # Tracking diagnoses that survive validation for the final association payload.
            validated_linked_diagnoses = []

            # Iterating over each linked diagnosis predicted for the current association.
            for linked_dx in linked_diagnoses:

                # Converting linked diagnosis payload into dictionary format.
                linked_dx_dictionary = _dump_to_dictionary(linked_dx) or {}

                # Extracting and normalizing the diagnosis code for validation.
                diagnosis_code = str(linked_dx_dictionary.get("code", "")).strip().upper()
                normalized_code, is_valid, is_billable = _get_validation_status(diagnosis_code)

                # Persisting the normalized code representation before branching.
                linked_dx_dictionary["code"] = normalized_code

                # Retain only linked diagnoses that are both valid and billable.
                if is_valid and is_billable:
                    validated_linked_diagnoses.append(linked_dx_dictionary)

                elif not is_valid:
                    # Logging diagnosis dropped because it is not a valid ICD-10-CM code.
                    if charge_code and normalized_code:
                        not_valid_associations.append((charge_code, normalized_code))
                else:
                    # Logging diagnosis dropped because it is valid but not billable.
                    if charge_code and normalized_code:
                        not_billable_associations.append((charge_code, normalized_code))

            # Replace original linked diagnoses with the validated subset.
            association_dictionary["linked_diagnoses"] = validated_linked_diagnoses

            # Appending the updated association to the tracking list.
            updated_associations.append(association_dictionary)

        # Updating the state with validated associations.
        state[association_field] = updated_associations

    # Persisting dropped-code tracking payloads to state for downstream nodes.
    state["not_valid_associations"] = not_valid_associations
    state["not_billable_associations"] = not_billable_associations

    # Persist a deep-copied snapshot for report export when debugging is enabled.
    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_validated_associations"] = _build_association_snapshot(
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
