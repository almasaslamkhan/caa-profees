import copy
from typing import List, Optional

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from cdrc.profees_coding.icd10cm_associate.data.schema import ICD10CM, EMItem, Procedure
from cdrc.profees_coding.icd10cm_associate.models.combo_code_rule import icd10_code_mapping_many
from cdrc.profees_coding.icd10cm_associate.models.utils import CONFIG


class IntermediateState(dict):
    procedure_associations: Optional[List] = Field(default_factory=list, description="List of CPT/HCPCS procedure associations.")
    em_associations: Optional[List] = Field(default_factory=list, description="List of CPT-E/M code associations.")


class OutputState(dict):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class PostProcessingStateGraph(IntermediateState, OutputState):
    pass


# --------------------------------------------------------------------------------
# Combo Code Rule Node
# --------------------------------------------------------------------------------


def combo_code_rule_node(state: PostProcessingStateGraph, config: RunnableConfig):
    for association_field in ["em_associations", "procedure_associations"]:
        state[association_field] = [_apply_combo_rules_to_association(association) for association in state.get(association_field) or []]

    if config["configurable"].get(CONFIG.EXPORT_INTERMEDIATE_PREDICTIONS, False):
        state["intermediate_combo_rule_associations"] = _build_association_snapshot(
            state.get("em_associations", []),
            state.get("procedure_associations", []),
        )

    return state


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def _apply_combo_rules_to_association(association):
    association_dictionary = _dump_to_dictionary(association) or {}
    linked_diagnoses = [_dump_to_dictionary(linked_dx) or {} for linked_dx in association_dictionary.get("linked_diagnoses") or []]
    input_codes = [str(linked_dx.get("code", "")).strip().upper() for linked_dx in linked_diagnoses if str(linked_dx.get("code", "")).strip()]

    if not input_codes:
        association_dictionary["linked_diagnoses"] = []
        return association_dictionary

    evidence_codes_by_output = icd10_code_mapping_many(input_codes)
    original_diagnosis_by_code = _build_original_diagnosis_lookup(linked_diagnoses)

    association_dictionary["linked_diagnoses"] = _build_linked_diagnoses(
        input_codes,
        evidence_codes_by_output,
        original_diagnosis_by_code,
    )
    return association_dictionary


def _build_original_diagnosis_lookup(linked_diagnoses):
    diagnosis_by_code = {}
    for linked_dx in linked_diagnoses:
        diagnosis_code = str(linked_dx.get("code", "")).strip().upper()
        if diagnosis_code and diagnosis_code not in diagnosis_by_code:
            normalized_dx = dict(linked_dx)
            normalized_dx["code"] = diagnosis_code
            diagnosis_by_code[diagnosis_code] = normalized_dx
    return diagnosis_by_code


def _build_linked_diagnoses(input_codes, evidence_codes_by_output, original_diagnosis_by_code):
    linked_diagnoses = []
    used_evidence_codes = set()
    seen_codes = set()

    for output_code, evidence_codes in evidence_codes_by_output.items():
        normalized_output = str(output_code or "").strip().upper()
        if not normalized_output or normalized_output in seen_codes:
            continue

        normalized_evidence_codes = [str(code or "").strip().upper() for code in evidence_codes if str(code or "").strip()]
        used_evidence_codes.update(normalized_evidence_codes)
        linked_diagnoses.append(_build_linked_diagnosis(normalized_output, normalized_evidence_codes, original_diagnosis_by_code))
        seen_codes.add(normalized_output)

    # Preserve inputs the mapper did not use as evidence so validation/regeneration can still process them.
    for input_code in input_codes:
        if input_code not in used_evidence_codes and input_code not in seen_codes:
            linked_diagnoses.append(dict(original_diagnosis_by_code[input_code]))
            seen_codes.add(input_code)

    return linked_diagnoses


def _build_linked_diagnosis(output_code, evidence_codes, original_diagnosis_by_code):
    if evidence_codes == [output_code] and output_code in original_diagnosis_by_code:
        return dict(original_diagnosis_by_code[output_code])

    return {
        "code": output_code,
        "justification": _build_combo_justification(output_code, evidence_codes, original_diagnosis_by_code),
    }


def _build_combo_justification(output_code, evidence_codes, original_diagnosis_by_code):
    evidence_text = _format_evidence_codes(evidence_codes)
    lines = [f"Combo code from {evidence_text}."]

    for evidence_code in evidence_codes:
        evidence_diagnosis = original_diagnosis_by_code.get(evidence_code, {})
        evidence_justification = str(evidence_diagnosis.get("justification", "") or "").strip()
        if evidence_justification:
            lines.append(f"{evidence_code}: {evidence_justification}")

    return "\n".join(lines)


def _format_evidence_codes(evidence_codes):
    if len(evidence_codes) <= 1:
        return ", ".join(evidence_codes)
    return ", ".join(evidence_codes[:-1]) + f" and {evidence_codes[-1]}"


def _dump_to_dictionary(x):
    return x.model_dump() if hasattr(x, "model_dump") else (vars(x) if hasattr(x, "__dict__") and not isinstance(x, dict) else x)


def _build_association_snapshot(em_associations, procedure_associations):
    return {
        "em_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (em_associations or [])],
        "procedure_associations": [copy.deepcopy(_dump_to_dictionary(association)) for association in (procedure_associations or [])],
    }
