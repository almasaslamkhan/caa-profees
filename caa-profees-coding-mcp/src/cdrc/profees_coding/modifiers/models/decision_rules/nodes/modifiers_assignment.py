from __future__ import annotations

from typing import Callable, Dict, List

from cdrc.profees_coding.modifiers.models.utils import addon_codes_mod_50, codes_for_95, inherently_bilateral_codes, is_anesthesia_cpt, is_em_code

CHECKLIST_GATE = "gate_qualifying_service_present"
CHECKLIST_SAME_DAY = "same_day_same_provider"
CHECKLIST_DISTINCT = "distinct_em_work"
CHECKLIST_DIAGNOSTIC_ONLY = "diagnostic_only_scenario"
CHECKLIST_DIAGNOSTIC_EXCEPTION = "diagnostic_only_policy_exception"
CHECKLIST_SAME_PROBLEM = "same_problem_only"
CHECKLIST_PAIRED_SITE = "paired_organ_or_site_mentioned"
CHECKLIST_LEFT = "laterality_left_explicit"
CHECKLIST_RIGHT = "laterality_right_explicit"
CHECKLIST_BILATERAL = "bilateral_explicit"
CHECKLIST_CODE_SPECIFIES_LATERALITY = "code_already_specifies_laterality"
CHECKLIST_PAYER_REQUIRES_LATERALITY = "payer_requires_laterality_over_50"
CHECKLIST_ADDON_CODE = "addon_code_for_50"
CHECKLIST_INHERENTLY_BILATERAL = "inherently_bilateral_code"

CHECKLIST_95_AUDIO_VIDEO = "telehealth_audio_video_confirmed"
CHECKLIST_95_AUDIO_ONLY = "telehealth_audio_only_confirmed"
CHECKLIST_95_SEPARATE_SITES = "telehealth_patient_and_provider_separate_sites"
CHECKLIST_95_SYNCHRONOUS = "telehealth_synchronous_interaction_confirmed"
CHECKLIST_SERVICE_PERFORMED_DOS = "service_performed_on_dos"
CHECKLIST_BILLING_PROVIDER_PERFORMED = "billing_provider_performed_service"

CHECKLIST_OFF_CAMPUS_OUTPATIENT = "off_campus_outpatient_hospital_confirmed"
CHECKLIST_FACILITY_PO = "facility_excepted_off_campus_pbd"
CHECKLIST_FACILITY_PN = "facility_nonexcepted_off_campus_pbd"

CHECKLIST_RESIDENT = "resident_involved_explicit"
CHECKLIST_TEACHING_ATTESTATION = "teaching_physician_attestation_present"
CHECKLIST_COSIGN_ONLY = "cosign_only_without_attestation"
CHECKLIST_MEDICAL_STUDENT_ONLY = "medical_student_only_mentioned"

CHECKLIST_PT_CONTEXT = "therapy_pt_plan_of_care_context"
CHECKLIST_OT_CONTEXT = "therapy_ot_plan_of_care_context"
CHECKLIST_SLP_CONTEXT = "therapy_slp_plan_of_care_context"
CHECKLIST_PTA_GE_10 = "pta_involvement_ge_10_percent"
CHECKLIST_OTA_GE_10 = "ota_involvement_ge_10_percent"
CHECKLIST_THERAPY_CANCELLED = "therapy_service_cancelled_or_not_performed"
CHECKLIST_THERAPY_NON_OUTPATIENT = "therapy_non_outpatient_or_bundled_setting"
CHECKLIST_THERAPY_CONFLICT = "therapy_discipline_conflict_in_text"

CHECKLIST_ASA_P1 = "asa_p1_explicit"
CHECKLIST_ASA_P2 = "asa_p2_explicit"
CHECKLIST_ASA_P3 = "asa_p3_explicit"
CHECKLIST_ASA_P4 = "asa_p4_explicit"
CHECKLIST_ASA_P5 = "asa_p5_explicit"
CHECKLIST_ASA_P6 = "asa_p6_explicit"
CHECKLIST_NO_SYSTEMIC_DISEASE = "no_systemic_disease_documented"
CHECKLIST_MILD_DISEASE = "mild_systemic_disease_supported"
CHECKLIST_SEVERE_DISEASE = "severe_systemic_disease_supported"
CHECKLIST_LIFE_THREAT = "life_threatening_instability_supported"
CHECKLIST_MORIBUND = "moribund_survival_without_surgery_unlikely"
CHECKLIST_BRAIN_DEAD_DONOR = "brain_dead_organ_donor_documented"
CHECKLIST_CRNA = "crna_performed_service"
CHECKLIST_MEDICAL_DIRECTION = "physician_medical_direction_documented"
CHECKLIST_CONCURRENCY_2_4 = "concurrency_two_to_four_documented"
CHECKLIST_CONCURRENCY_OVER_4 = "concurrency_over_four_or_supervision_only"

CHECKLIST_QW_ELIGIBLE = "qw_eligible_lab_code"
CHECKLIST_POC_TEST = "point_of_care_or_in_office_test_performed"
CHECKLIST_SAME_DAY_RESULT = "same_day_test_result_present"
CHECKLIST_REFERENCE_LAB = "reference_or_outside_lab_performed"
CHECKLIST_LAB_PERFORMED_DOS = "lab_test_performed_on_dos"
CHECKLIST_CLIA_WAIVED_SITE = "clia_waived_site_or_certificate"
CHECKLIST_MEDICARE = "medicare_payer_context"
CHECKLIST_MEDICAID = "medicaid_payer_context"
CHECKLIST_NON_MEDICAID = "non_medicaid_payer_context_explicit"
CHECKLIST_T1015 = "t1015_present_on_claim"
CHECKLIST_JCODE_ADMIN_DOS = "jcode_administered_on_dos"
CHECKLIST_STATUTORY_EXCLUSION = "statutory_exclusion_or_nonbenefit_supported"
CHECKLIST_COMFORT_ONLY = "comfort_or_convenience_only_use"
CHECKLIST_MEDICARE_BENEFIT = "qualifying_medicare_benefit_category_supported"
CHECKLIST_L3260_ORTHOTIC = "l3260_therapeutic_orthotic_use"
CHECKLIST_L3260_COMFORT = "l3260_comfort_or_footwear_use"
CHECKLIST_QUALIFYING_WOUND = "qualifying_wound_or_injury_present"
CHECKLIST_NONCOVERED_COSMETIC = "cosmetic_or_explicit_noncovered_language"

PT_CODE_RANGES = (
    (97010, 97039),
    (97110, 97118),
    (97161, 97164),
    (97150, 97158),
    (97530, 97535),
    (97750, 97763),
)
OT_CODE_RANGES = (
    (97165, 97168),
    (97530, 97535),
    (97750, 97763),
)
SLP_SPECIFIC_CODES = {
    "92507",
    "92508",
    "92521",
    "92522",
    "92523",
    "92524",
    "92526",
    "92597",
    "92610",
    "92611",
    "96125",
}
MULTI_DISCIPLINE_THERAPY_CODES = {
    "97110",
    "97112",
    "97116",
    "97129",
    "97130",
    "97530",
    "97535",
    "97750",
    "97755",
    "97760",
    "97761",
}
NON_QW_COLLECTION_ONLY_CODES = {"36415", "99000", "99001"}
MEDICARE_EXCLUDED_FOOTWEAR_CODES = {"L3260"}

INHERENTLY_BILATERAL_SET = {str(code).strip().upper() for code in inherently_bilateral_codes}
ADDON_CODE_SET = {str(code).strip().upper() for code in addon_codes_mod_50}
CODES_FOR_95_SET = {str(code).strip().upper() for code in codes_for_95}


def _flag(features, name: str) -> bool:
    return bool(features.get(name, {}).get("support", False))


def _normalize_code(code: str) -> str:
    return str(code or "").strip().upper()


def _is_numeric_cpt(code: str) -> bool:
    return _normalize_code(code).isdigit()


def _is_lab_test_code(code: str) -> bool:
    normalized = _normalize_code(code)
    if normalized in NON_QW_COLLECTION_ONLY_CODES:
        return False
    if not normalized.isdigit():
        return False
    return 80000 <= int(normalized) <= 89999


def _is_j_code(code: str) -> bool:
    normalized = _normalize_code(code)
    return normalized.startswith("J") and len(normalized) >= 2


def _is_therapy_code_for_pt(code: str) -> bool:
    normalized = _normalize_code(code)
    if normalized in MULTI_DISCIPLINE_THERAPY_CODES:
        return True
    if not normalized.isdigit():
        return False
    value = int(normalized)
    return any(start <= value <= end for start, end in PT_CODE_RANGES)


def _is_therapy_code_for_ot(code: str) -> bool:
    normalized = _normalize_code(code)
    if normalized in MULTI_DISCIPLINE_THERAPY_CODES:
        return True
    if not normalized.isdigit():
        return False
    value = int(normalized)
    return any(start <= value <= end for start, end in OT_CODE_RANGES)


def _is_therapy_code_for_slp(code: str) -> bool:
    normalized = _normalize_code(code)
    return normalized in SLP_SPECIFIC_CODES or normalized in {"97129", "97130"}


def _select_p_status(features) -> str | None:
    if _flag(features, CHECKLIST_ASA_P6) or _flag(features, CHECKLIST_BRAIN_DEAD_DONOR):
        return "P6"
    if _flag(features, CHECKLIST_ASA_P5) or _flag(features, CHECKLIST_MORIBUND):
        return "P5"
    if _flag(features, CHECKLIST_ASA_P4) or _flag(features, CHECKLIST_LIFE_THREAT):
        return "P4"
    if _flag(features, CHECKLIST_ASA_P3) or _flag(features, CHECKLIST_SEVERE_DISEASE):
        return "P3"
    if _flag(features, CHECKLIST_ASA_P2) or _flag(features, CHECKLIST_MILD_DISEASE):
        return "P2"
    if _flag(features, CHECKLIST_ASA_P1) or _flag(features, CHECKLIST_NO_SYSTEMIC_DISEASE):
        return "P1"
    return None


def assign_default(code, features, **kwargs):
    del code, features, kwargs
    return False


def assign_25(code, features, **kwargs):
    if not is_em_code(code):
        return False
    if not _flag(features, CHECKLIST_GATE):
        return False
    if not _flag(features, CHECKLIST_SAME_DAY):
        return False
    if not _flag(features, CHECKLIST_DISTINCT):
        return False
    if _flag(features, CHECKLIST_SAME_PROBLEM):
        return False
    if _flag(features, CHECKLIST_DIAGNOSTIC_ONLY) and not _flag(features, CHECKLIST_DIAGNOSTIC_EXCEPTION):
        return False
    return True


def assign_50(code, features, **kwargs):
    normalized = _normalize_code(code)
    if is_em_code(normalized):
        return False
    if _flag(features, CHECKLIST_ADDON_CODE) or normalized in ADDON_CODE_SET:
        return False
    if _flag(features, CHECKLIST_INHERENTLY_BILATERAL) or normalized in INHERENTLY_BILATERAL_SET:
        return False
    if _flag(features, CHECKLIST_PAYER_REQUIRES_LATERALITY):
        return False
    if not _flag(features, CHECKLIST_BILATERAL):
        return False
    return True


def assign_LT(code, features, **kwargs):
    normalized = _normalize_code(code)
    if is_em_code(normalized):
        return False
    if _flag(features, CHECKLIST_CODE_SPECIFIES_LATERALITY):
        return False
    if _flag(features, CHECKLIST_INHERENTLY_BILATERAL) or normalized in INHERENTLY_BILATERAL_SET:
        return False
    if not _flag(features, CHECKLIST_PAIRED_SITE):
        return False
    if not _flag(features, CHECKLIST_LEFT):
        return False
    if _flag(features, CHECKLIST_BILATERAL) and not _flag(features, CHECKLIST_PAYER_REQUIRES_LATERALITY):
        return False
    return True


def assign_RT(code, features, **kwargs):
    normalized = _normalize_code(code)
    if is_em_code(normalized):
        return False
    if _flag(features, CHECKLIST_CODE_SPECIFIES_LATERALITY):
        return False
    if _flag(features, CHECKLIST_INHERENTLY_BILATERAL) or normalized in INHERENTLY_BILATERAL_SET:
        return False
    if not _flag(features, CHECKLIST_PAIRED_SITE):
        return False
    if not _flag(features, CHECKLIST_RIGHT):
        return False
    if _flag(features, CHECKLIST_BILATERAL) and not _flag(features, CHECKLIST_PAYER_REQUIRES_LATERALITY):
        return False
    return True


def assign_95(code, features, **kwargs):
    normalized = _normalize_code(code)
    if normalized not in CODES_FOR_95_SET:
        return False
    if _flag(features, CHECKLIST_95_AUDIO_ONLY):
        return False
    return all(
        [
            _flag(features, CHECKLIST_95_AUDIO_VIDEO),
            _flag(features, CHECKLIST_95_SEPARATE_SITES),
            _flag(features, CHECKLIST_95_SYNCHRONOUS),
            _flag(features, CHECKLIST_SERVICE_PERFORMED_DOS),
            _flag(features, CHECKLIST_BILLING_PROVIDER_PERFORMED),
        ]
    )


def assign_PO(code, features, **kwargs):
    del code, kwargs
    return _flag(features, CHECKLIST_OFF_CAMPUS_OUTPATIENT) and _flag(features, CHECKLIST_FACILITY_PO) and not _flag(features, CHECKLIST_FACILITY_PN)


def assign_PN(code, features, **kwargs):
    del code, kwargs
    return _flag(features, CHECKLIST_OFF_CAMPUS_OUTPATIENT) and _flag(features, CHECKLIST_FACILITY_PN) and not _flag(features, CHECKLIST_FACILITY_PO)


def assign_GC(code, features, **kwargs):
    del code, kwargs
    if not _flag(features, CHECKLIST_RESIDENT):
        return False
    if not _flag(features, CHECKLIST_TEACHING_ATTESTATION):
        return False
    if _flag(features, CHECKLIST_COSIGN_ONLY):
        return False
    if _flag(features, CHECKLIST_MEDICAL_STUDENT_ONLY):
        return False
    return True


def assign_GP(code, features, **kwargs):
    if not _is_therapy_code_for_pt(code):
        return False
    if _flag(features, CHECKLIST_THERAPY_CANCELLED) or _flag(features, CHECKLIST_THERAPY_NON_OUTPATIENT):
        return False
    if _flag(features, CHECKLIST_THERAPY_CONFLICT):
        return False
    return _flag(features, CHECKLIST_PT_CONTEXT)


def assign_GO(code, features, **kwargs):
    if not _is_therapy_code_for_ot(code):
        return False
    if _flag(features, CHECKLIST_THERAPY_CANCELLED) or _flag(features, CHECKLIST_THERAPY_NON_OUTPATIENT):
        return False
    if _flag(features, CHECKLIST_THERAPY_CONFLICT):
        return False
    return _flag(features, CHECKLIST_OT_CONTEXT)


def assign_GN(code, features, **kwargs):
    if not _is_therapy_code_for_slp(code):
        return False
    if _flag(features, CHECKLIST_THERAPY_CANCELLED) or _flag(features, CHECKLIST_THERAPY_NON_OUTPATIENT):
        return False
    if _flag(features, CHECKLIST_THERAPY_CONFLICT):
        return False
    return _flag(features, CHECKLIST_SLP_CONTEXT)


def assign_CQ(code, features, **kwargs):
    return assign_GP(code, features, **kwargs) and _flag(features, CHECKLIST_PTA_GE_10)


def assign_CO(code, features, **kwargs):
    return assign_GO(code, features, **kwargs) and _flag(features, CHECKLIST_OTA_GE_10)


def assign_P1(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P1"


def assign_P2(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P2"


def assign_P3(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P3"


def assign_P4(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P4"


def assign_P5(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P5"


def assign_P6(code, features, **kwargs):
    return is_anesthesia_cpt(code) and _select_p_status(features) == "P6"


def assign_QX(code, features, **kwargs):
    if not is_anesthesia_cpt(code):
        return False
    return _flag(features, CHECKLIST_CRNA) and _flag(features, CHECKLIST_MEDICAL_DIRECTION)


def assign_QK(code, features, **kwargs):
    if not is_anesthesia_cpt(code):
        return False
    if not _flag(features, CHECKLIST_MEDICAL_DIRECTION):
        return False
    if not _flag(features, CHECKLIST_CONCURRENCY_2_4):
        return False
    if _flag(features, CHECKLIST_CONCURRENCY_OVER_4):
        return False
    return True


def assign_QW(code, features, **kwargs):
    if not _is_lab_test_code(code):
        return False
    if not _flag(features, CHECKLIST_QW_ELIGIBLE):
        return False
    if not _flag(features, CHECKLIST_LAB_PERFORMED_DOS):
        return False
    if _flag(features, CHECKLIST_REFERENCE_LAB):
        return False
    if not (_flag(features, CHECKLIST_POC_TEST) or _flag(features, CHECKLIST_CLIA_WAIVED_SITE)):
        return False
    return True


def assign_90(code, features, **kwargs):
    normalized = _normalize_code(code)
    if normalized == "36415":
        return _flag(features, CHECKLIST_REFERENCE_LAB) and not _flag(features, CHECKLIST_POC_TEST)
    if not _is_lab_test_code(normalized):
        return False
    if not _flag(features, CHECKLIST_REFERENCE_LAB):
        return False
    if not _flag(features, CHECKLIST_LAB_PERFORMED_DOS):
        return False
    if _flag(features, CHECKLIST_POC_TEST):
        return False
    return True


def assign_UD(code, features, **kwargs):
    if not _is_j_code(code):
        return False
    if not _flag(features, CHECKLIST_T1015):
        return False
    if not _flag(features, CHECKLIST_JCODE_ADMIN_DOS):
        return False
    if _flag(features, CHECKLIST_NON_MEDICAID):
        return False
    return True


def assign_GY(code, features, **kwargs):
    normalized = _normalize_code(code)
    if _flag(features, CHECKLIST_NON_MEDICAID) and not _flag(features, CHECKLIST_MEDICARE):
        return False
    if _flag(features, CHECKLIST_MEDICARE_BENEFIT):
        return False
    if normalized in MEDICARE_EXCLUDED_FOOTWEAR_CODES and _flag(features, CHECKLIST_L3260_ORTHOTIC):
        return False
    if _flag(features, CHECKLIST_QUALIFYING_WOUND):
        return False
    return any(
        [
            _flag(features, CHECKLIST_STATUTORY_EXCLUSION),
            _flag(features, CHECKLIST_NONCOVERED_COSMETIC),
            _flag(features, CHECKLIST_COMFORT_ONLY),
            _flag(features, CHECKLIST_L3260_COMFORT),
        ]
    )


ASSIGNMENT_FUNCTIONS: Dict[str, Callable] = {
    "25": assign_25,
    "50": assign_50,
    "LT": assign_LT,
    "RT": assign_RT,
    "95": assign_95,
    "PO": assign_PO,
    "PN": assign_PN,
    "GC": assign_GC,
    "GP": assign_GP,
    "GO": assign_GO,
    "GN": assign_GN,
    "CQ": assign_CQ,
    "CO": assign_CO,
    "P1": assign_P1,
    "P2": assign_P2,
    "P3": assign_P3,
    "P4": assign_P4,
    "P5": assign_P5,
    "P6": assign_P6,
    "QK": assign_QK,
    "QX": assign_QX,
    "QW": assign_QW,
    "90": assign_90,
    "GY": assign_GY,
    "UD": assign_UD,
}


def modifier_applicable_to_code(modifier: str, code: str) -> bool:
    normalized_modifier = _normalize_code(modifier)
    normalized_code = _normalize_code(code)

    if normalized_modifier == "25":
        return is_em_code(normalized_code)
    if normalized_modifier in {"50", "LT", "RT"}:
        return not is_em_code(normalized_code)
    if normalized_modifier == "95":
        return normalized_code in CODES_FOR_95_SET
    if normalized_modifier in {"GP", "CQ"}:
        return _is_therapy_code_for_pt(normalized_code)
    if normalized_modifier in {"GO", "CO"}:
        return _is_therapy_code_for_ot(normalized_code)
    if normalized_modifier == "GN":
        return _is_therapy_code_for_slp(normalized_code)
    if normalized_modifier in {"P1", "P2", "P3", "P4", "P5", "P6", "QK", "QX"}:
        return is_anesthesia_cpt(normalized_code)
    if normalized_modifier == "QW":
        return _is_lab_test_code(normalized_code)
    if normalized_modifier == "90":
        return _is_lab_test_code(normalized_code) or normalized_code == "36415"
    if normalized_modifier == "UD":
        return _is_j_code(normalized_code)
    return True


def build_unassigned_reasons(modifier: str, code: str, features) -> List[str]:
    mod = _normalize_code(modifier)
    normalized_code = _normalize_code(code)

    if mod == "25":
        reasons = []
        if not _flag(features, CHECKLIST_GATE):
            reasons.append("No qualifying same-day service was documented.")
        if not _flag(features, CHECKLIST_SAME_DAY):
            reasons.append("Same-day or same-provider relationship was not supported.")
        if not _flag(features, CHECKLIST_DISTINCT):
            reasons.append("Distinct E/M work beyond the other service was not supported.")
        if _flag(features, CHECKLIST_SAME_PROBLEM):
            reasons.append("The E/M appears limited to the same problem as the other service.")
        if _flag(features, CHECKLIST_DIAGNOSTIC_ONLY) and not _flag(features, CHECKLIST_DIAGNOSTIC_EXCEPTION):
            reasons.append("Only diagnostic/specimen services were documented without a policy exception.")
        return reasons

    if mod == "50":
        reasons = []
        if _flag(features, CHECKLIST_ADDON_CODE) or normalized_code in ADDON_CODE_SET:
            reasons.append("Add-on codes should not receive modifier 50.")
        if _flag(features, CHECKLIST_INHERENTLY_BILATERAL) or normalized_code in INHERENTLY_BILATERAL_SET:
            reasons.append("The code is inherently bilateral.")
        if _flag(features, CHECKLIST_PAYER_REQUIRES_LATERALITY):
            reasons.append("Payer direction favored LT/RT instead of modifier 50.")
        if not _flag(features, CHECKLIST_BILATERAL):
            reasons.append("Bilateral performance was not explicitly documented.")
        return reasons

    if mod in {"LT", "RT"}:
        reasons = []
        if _flag(features, CHECKLIST_CODE_SPECIFIES_LATERALITY):
            reasons.append("The code description already specifies laterality.")
        if _flag(features, CHECKLIST_INHERENTLY_BILATERAL) or normalized_code in INHERENTLY_BILATERAL_SET:
            reasons.append("The code is inherently bilateral.")
        if not _flag(features, CHECKLIST_PAIRED_SITE):
            reasons.append("A paired organ or site was not clearly documented.")
        if mod == "LT" and not _flag(features, CHECKLIST_LEFT):
            reasons.append("Left-sided performance was not explicitly documented.")
        if mod == "RT" and not _flag(features, CHECKLIST_RIGHT):
            reasons.append("Right-sided performance was not explicitly documented.")
        if _flag(features, CHECKLIST_BILATERAL) and not _flag(features, CHECKLIST_PAYER_REQUIRES_LATERALITY):
            reasons.append("Bilateral documentation favored modifier 50 over unilateral laterality.")
        return reasons

    if mod == "95":
        reasons = []
        if _flag(features, CHECKLIST_95_AUDIO_ONLY):
            reasons.append("The encounter was documented as audio-only.")
        if not _flag(features, CHECKLIST_95_AUDIO_VIDEO):
            reasons.append("Real-time audio and video were not both supported.")
        if not _flag(features, CHECKLIST_95_SEPARATE_SITES):
            reasons.append("Separate patient and provider locations were not supported.")
        if not _flag(features, CHECKLIST_95_SYNCHRONOUS):
            reasons.append("A synchronous telehealth encounter was not clearly documented.")
        if not _flag(features, CHECKLIST_SERVICE_PERFORMED_DOS):
            reasons.append("The service was not clearly performed on the date of service.")
        if not _flag(features, CHECKLIST_BILLING_PROVIDER_PERFORMED):
            reasons.append("The billed service was not clearly attributable to the billing provider.")
        return reasons

    if mod in {"PO", "PN"}:
        reasons = []
        if not _flag(features, CHECKLIST_OFF_CAMPUS_OUTPATIENT):
            reasons.append("Authoritative facility context for off-campus hospital outpatient care was not supported.")
        if mod == "PO" and not _flag(features, CHECKLIST_FACILITY_PO):
            reasons.append("Excepted off-campus PBD status was not documented.")
        if mod == "PN" and not _flag(features, CHECKLIST_FACILITY_PN):
            reasons.append("Non-excepted off-campus PBD status was not documented.")
        return reasons

    if mod == "GC":
        reasons = []
        if not _flag(features, CHECKLIST_RESIDENT):
            reasons.append("Resident involvement was not documented.")
        if not _flag(features, CHECKLIST_TEACHING_ATTESTATION):
            reasons.append("Teaching physician attestation was not documented.")
        if _flag(features, CHECKLIST_COSIGN_ONLY):
            reasons.append("Only a co-sign was documented without a teaching attestation.")
        if _flag(features, CHECKLIST_MEDICAL_STUDENT_ONLY):
            reasons.append("The record suggested medical-student-only involvement.")
        return reasons

    if mod in {"GP", "GO", "GN"}:
        reasons = []
        if _flag(features, CHECKLIST_THERAPY_CANCELLED):
            reasons.append("The therapy service was cancelled or not performed.")
        if _flag(features, CHECKLIST_THERAPY_NON_OUTPATIENT):
            reasons.append("The record suggested a non-outpatient or bundled therapy setting.")
        if _flag(features, CHECKLIST_THERAPY_CONFLICT):
            reasons.append("Conflicting therapy discipline signals were present.")
        if mod == "GP" and not _flag(features, CHECKLIST_PT_CONTEXT):
            reasons.append("PT plan-of-care context was not supported.")
        if mod == "GO" and not _flag(features, CHECKLIST_OT_CONTEXT):
            reasons.append("OT plan-of-care context was not supported.")
        if mod == "GN" and not _flag(features, CHECKLIST_SLP_CONTEXT):
            reasons.append("SLP plan-of-care context was not supported.")
        return reasons

    if mod == "CQ":
        reasons = build_unassigned_reasons("GP", code, features)
        if not _flag(features, CHECKLIST_PTA_GE_10):
            reasons.append("PTA involvement at or above the 10% threshold was not documented.")
        return reasons

    if mod == "CO":
        reasons = build_unassigned_reasons("GO", code, features)
        if not _flag(features, CHECKLIST_OTA_GE_10):
            reasons.append("OTA involvement at or above the 10% threshold was not documented.")
        return reasons

    if mod in {"P1", "P2", "P3", "P4", "P5", "P6"}:
        chosen = _select_p_status(features)
        if not chosen:
            return ["No supported ASA physical-status level was extracted."]
        return [f"Highest supported ASA physical-status level was {chosen}, not {mod}."]

    if mod == "QX":
        reasons = []
        if not _flag(features, CHECKLIST_CRNA):
            reasons.append("CRNA performance was not documented.")
        if not _flag(features, CHECKLIST_MEDICAL_DIRECTION):
            reasons.append("Physician medical direction was not documented.")
        return reasons

    if mod == "QK":
        reasons = []
        if not _flag(features, CHECKLIST_MEDICAL_DIRECTION):
            reasons.append("Physician medical direction was not documented.")
        if not _flag(features, CHECKLIST_CONCURRENCY_2_4):
            reasons.append("Concurrency of two to four cases was not documented.")
        if _flag(features, CHECKLIST_CONCURRENCY_OVER_4):
            reasons.append("The record suggested supervision-only or more than four concurrent cases.")
        return reasons

    if mod == "QW":
        reasons = []
        if not _flag(features, CHECKLIST_QW_ELIGIBLE):
            reasons.append("The line was not supported as a QW-eligible waived test code.")
        if not _flag(features, CHECKLIST_LAB_PERFORMED_DOS):
            reasons.append("The lab test was not clearly performed on the date of service.")
        if _flag(features, CHECKLIST_REFERENCE_LAB):
            reasons.append("The test appeared to be performed by an outside/reference lab.")
        if not (_flag(features, CHECKLIST_POC_TEST) or _flag(features, CHECKLIST_CLIA_WAIVED_SITE)):
            reasons.append("In-office waived testing context was not documented.")
        return reasons

    if mod == "90":
        reasons = []
        if not _flag(features, CHECKLIST_REFERENCE_LAB):
            if normalized_code == "36415":
                reasons.append("36415 requires explicit linkage between the draw/specimen and a send-out or reference lab.")
            else:
                reasons.append("Reference-lab performance was not documented.")
        if normalized_code != "36415" and not _flag(features, CHECKLIST_LAB_PERFORMED_DOS):
            reasons.append("The lab test was not clearly performed on the date of service.")
        if _flag(features, CHECKLIST_POC_TEST):
            if normalized_code == "36415":
                reasons.append("The record supported in-office/POC handling rather than a send-out specimen.")
            else:
                reasons.append("The record supported in-office/POC performance instead of a send-out lab.")
        return reasons

    if mod == "UD":
        reasons = []
        if not _flag(features, CHECKLIST_T1015):
            reasons.append("The encounter-level T1015 trigger was not present.")
        if not _flag(features, CHECKLIST_JCODE_ADMIN_DOS):
            reasons.append("Administration of the J-code on the date of service was not documented.")
        if _flag(features, CHECKLIST_NON_MEDICAID):
            reasons.append("The payer context explicitly conflicted with the dataset UD rule.")
        return reasons

    if mod == "GY":
        reasons = []
        if _flag(features, CHECKLIST_MEDICARE_BENEFIT):
            reasons.append("A Medicare benefit category was supported.")
        if _flag(features, CHECKLIST_QUALIFYING_WOUND):
            reasons.append("A qualifying wound or injury supported covered supply use.")
        if normalized_code in MEDICARE_EXCLUDED_FOOTWEAR_CODES and _flag(features, CHECKLIST_L3260_ORTHOTIC):
            reasons.append("L3260 was documented as therapeutic orthotic/stabilization use.")
        if not any(
            [
                _flag(features, CHECKLIST_STATUTORY_EXCLUSION),
                _flag(features, CHECKLIST_NONCOVERED_COSMETIC),
                _flag(features, CHECKLIST_COMFORT_ONLY),
                _flag(features, CHECKLIST_L3260_COMFORT),
            ]
        ):
            reasons.append("No statutorily excluded or non-benefit use was clearly documented.")
        return reasons

    return [f"Checklist did not support modifier {mod}."]
