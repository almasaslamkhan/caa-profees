from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.decision_rules.data_classes import CONFIG, PROMPTS
from cdrc.profees_coding.modifiers.models.utils import (
    addon_codes_mod_50,
    codes_for_95,
    get_cpt_codes,
    get_modifiers_in_scope,
    inherently_bilateral_codes,
    is_em_code,
)

from .modifiers_assignment import ASSIGNMENT_FUNCTIONS, assign_default, build_unassigned_reasons, modifier_applicable_to_code

MEDICARE_PAYER_KEYWORDS = {
    "MEDICARE",
    "MEDICARE FFS",
    "CMS",
    "MEDICARE PART A",
    "MEDICARE PART B",
}
MEDICAID_PAYER_KEYWORDS = {
    "MEDICAID",
    "MEDI-CAL",
    "MEDICAL ASSISTANCE",
    "TITLE XIX",
    "MCO MEDICAID",
}
WAIVED_CLIA_KEYWORDS = {"WAIVED", "CLIA WAIVED", "CERTIFICATE OF WAIVER", "COW"}
OFF_CAMPUS_POS_CODES = {"19"}
THERAPY_HINTS = {
    "PT": ("physical therapy", "pt eval", "pt treatment", "pta", "under pt plan of care"),
    "OT": ("occupational therapy", "ot eval", "ot treatment", "ota", "under ot plan of care", "hand therapy"),
    "SLP": ("speech therapy", "speech pathology", "slp", "swallow therapy", "speech eval"),
}
GUIDELINE_VERSION_KEYS_BY_MODIFIER = {
    "25": "25",
    "50": "50_LT_RT",
    "LT": "50_LT_RT",
    "RT": "50_LT_RT",
    "95": "95",
    "PO": "PO-PN",
    "PN": "PO-PN",
    "GC": "GC",
    "GP": "GP_GO_GN_CQ_CO",
    "GO": "GP_GO_GN_CQ_CO",
    "GN": "GP_GO_GN_CQ_CO",
    "CQ": "GP_GO_GN_CQ_CO",
    "CO": "GP_GO_GN_CQ_CO",
    "P1": "P1-P6",
    "P2": "P1-P6",
    "P3": "P1-P6",
    "P4": "P1-P6",
    "P5": "P1-P6",
    "P6": "P1-P6",
    "QK": "QK-QX",
    "QX": "QK-QX",
    "QW": "QW_90",
    "90": "QW_90",
    "GY": "GY_UD",
    "UD": "GY_UD",
}


class ChecklistAnswer(str, Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class ChecklistItemId(str, Enum):
    GATE_QUALIFYING_SERVICE = "gate_qualifying_service_present"
    SAME_DAY_SAME_PROVIDER = "same_day_same_provider"
    DISTINCT_EM_WORK = "distinct_em_work"
    DIAGNOSTIC_ONLY_SCENARIO = "diagnostic_only_scenario"
    DIAGNOSTIC_ONLY_POLICY_EXCEPTION = "diagnostic_only_policy_exception"
    SAME_PROBLEM_ONLY = "same_problem_only"
    PAIRED_ORGAN_OR_SITE = "paired_organ_or_site_mentioned"
    LATERALITY_LEFT = "laterality_left_explicit"
    LATERALITY_RIGHT = "laterality_right_explicit"
    BILATERAL_EXPLICIT = "bilateral_explicit"
    CODE_SPECIFIES_LATERALITY = "code_already_specifies_laterality"
    PAYER_REQUIRES_LATERALITY_OVER_50 = "payer_requires_laterality_over_50"
    ADDON_CODE_FOR_50 = "addon_code_for_50"
    INHERENTLY_BILATERAL_CODE = "inherently_bilateral_code"

    TELEHEALTH_AUDIO_VIDEO = "telehealth_audio_video_confirmed"
    TELEHEALTH_AUDIO_ONLY = "telehealth_audio_only_confirmed"
    TELEHEALTH_SEPARATE_SITES = "telehealth_patient_and_provider_separate_sites"
    TELEHEALTH_SYNCHRONOUS = "telehealth_synchronous_interaction_confirmed"
    SERVICE_PERFORMED_DOS = "service_performed_on_dos"
    BILLING_PROVIDER_PERFORMED = "billing_provider_performed_service"
    OFF_CAMPUS_OUTPATIENT = "off_campus_outpatient_hospital_confirmed"
    FACILITY_PO = "facility_excepted_off_campus_pbd"
    FACILITY_PN = "facility_nonexcepted_off_campus_pbd"
    RESIDENT_INVOLVED = "resident_involved_explicit"
    TEACHING_ATTESTATION = "teaching_physician_attestation_present"
    COSIGN_ONLY = "cosign_only_without_attestation"
    MEDICAL_STUDENT_ONLY = "medical_student_only_mentioned"

    THERAPY_PT_CONTEXT = "therapy_pt_plan_of_care_context"
    THERAPY_OT_CONTEXT = "therapy_ot_plan_of_care_context"
    THERAPY_SLP_CONTEXT = "therapy_slp_plan_of_care_context"
    PTA_GE_10 = "pta_involvement_ge_10_percent"
    OTA_GE_10 = "ota_involvement_ge_10_percent"
    THERAPY_CANCELLED = "therapy_service_cancelled_or_not_performed"
    THERAPY_NON_OUTPATIENT = "therapy_non_outpatient_or_bundled_setting"
    THERAPY_CONFLICT = "therapy_discipline_conflict_in_text"

    ASA_P1 = "asa_p1_explicit"
    ASA_P2 = "asa_p2_explicit"
    ASA_P3 = "asa_p3_explicit"
    ASA_P4 = "asa_p4_explicit"
    ASA_P5 = "asa_p5_explicit"
    ASA_P6 = "asa_p6_explicit"
    NO_SYSTEMIC_DISEASE = "no_systemic_disease_documented"
    MILD_DISEASE = "mild_systemic_disease_supported"
    SEVERE_DISEASE = "severe_systemic_disease_supported"
    LIFE_THREAT = "life_threatening_instability_supported"
    MORIBUND = "moribund_survival_without_surgery_unlikely"
    BRAIN_DEAD_DONOR = "brain_dead_organ_donor_documented"
    CRNA_PERFORMED = "crna_performed_service"
    MEDICAL_DIRECTION = "physician_medical_direction_documented"
    CONCURRENCY_2_4 = "concurrency_two_to_four_documented"
    CONCURRENCY_OVER_4 = "concurrency_over_four_or_supervision_only"

    QW_ELIGIBLE = "qw_eligible_lab_code"
    POC_TEST = "point_of_care_or_in_office_test_performed"
    SAME_DAY_RESULT = "same_day_test_result_present"
    REFERENCE_LAB = "reference_or_outside_lab_performed"
    LAB_PERFORMED_DOS = "lab_test_performed_on_dos"
    CLIA_WAIVED_SITE = "clia_waived_site_or_certificate"
    MEDICARE_PAYER = "medicare_payer_context"
    MEDICAID_PAYER = "medicaid_payer_context"
    NON_MEDICAID_PAYER = "non_medicaid_payer_context_explicit"
    T1015_PRESENT = "t1015_present_on_claim"
    JCODE_ADMIN_DOS = "jcode_administered_on_dos"
    STATUTORY_EXCLUSION = "statutory_exclusion_or_nonbenefit_supported"
    COMFORT_ONLY = "comfort_or_convenience_only_use"
    MEDICARE_BENEFIT = "qualifying_medicare_benefit_category_supported"
    L3260_ORTHOTIC = "l3260_therapeutic_orthotic_use"
    L3260_COMFORT = "l3260_comfort_or_footwear_use"
    QUALIFYING_WOUND = "qualifying_wound_or_injury_present"
    NONCOVERED_COSMETIC = "cosmetic_or_explicit_noncovered_language"


class ModifierChecklistItem(BaseModel):
    item_id: ChecklistItemId = Field(..., description="Checklist item identifier tied to the modifier and evaluation rules.")
    answer: ChecklistAnswer = Field(default=ChecklistAnswer.UNCLEAR, description="yes/no/unclear answer for this checklist item.")
    evidence: Optional[str] = Field(default=None, description="Short evidence snippet supporting the answer.")
    justification: Optional[str] = Field(default=None, description="Brief reasoning for the answer.")
    applies_to: List[str] = Field(
        default_factory=list, description="CPT codes this checklist item applies to, or ['ALL'] for universal applicability."
    )

    @field_validator("item_id", mode="before")
    @classmethod
    def _coerce_item_id(cls, value: Any) -> ChecklistItemId:
        if isinstance(value, ChecklistItemId):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if normalized in ChecklistItemId.__members__:
                return ChecklistItemId[normalized]
            upper = normalized.upper()
            if upper in ChecklistItemId.__members__:
                return ChecklistItemId[upper]
            for item in ChecklistItemId:
                if normalized.lower() == item.value.lower():
                    return item
        return value

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> ChecklistAnswer:
        if isinstance(value, ChecklistAnswer):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "y", "true", "t", "1"}:
                return ChecklistAnswer.YES
            if normalized in {"no", "n", "false", "f", "0"}:
                return ChecklistAnswer.NO
        return ChecklistAnswer.UNCLEAR

    @field_validator("applies_to", mode="before")
    @classmethod
    def _coerce_applies_to(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            if "," in cleaned:
                return [part.strip() for part in cleaned.split(",") if part.strip()]
            return [cleaned]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        cleaned = str(value).strip()
        return [cleaned] if cleaned else []


class ModifierChecklistExtraction(BaseModel):
    checklist: List[ModifierChecklistItem] = Field(default_factory=list, description="Extracted checklist items used to decide modifiers in scope.")


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code")
    evidence: Optional[str] = Field(None, description="Snippet from the encounter that supports the decision")
    justification: Optional[str] = Field(None, description="Reason for applying or rejecting the modifier")
    assigned: bool = Field(..., description="True if the modifier should be applied")
    cpt_codes: List[str] = Field(default_factory=list, description="CPT codes associated with this decision")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]
    payer_name: Optional[List[str]]
    encounter_metadata: Optional[Any]
    health_plan: Optional[List[Any]]
    facility_info: Optional[List[Any]]
    provider_info: Optional[List[Any]]
    lab_info: Optional[List[Any]]
    anaesthesia_staff_info: Optional[List[Any]]
    anaesthesia_concurrent_staff_info: Optional[List[Any]]
    anaesthesia_record_info: Optional[List[Any]]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    rules: Optional[ModifierChecklistExtraction]


RULE_GROUPS: List[dict[str, Any]] = [
    {
        "name": "em_laterality",
        "modifiers": {"25", "50", "LT", "RT"},
        "purpose": "Extract only the facts needed to decide same-day E/M modifier 25 and laterality modifiers 50/LT/RT.",
        "items": [
            ("gate_qualifying_service_present", "YES when a qualifying same-day service exists in addition to the E/M under review."),
            (
                "same_day_same_provider",
                "YES when the E/M and the other qualifying service were performed on the same DOS by the same physician/QHP or clinic staff under that provider.",
            ),
            ("distinct_em_work", "YES when documentation supports significant, separately identifiable E/M work beyond the other service."),
            ("diagnostic_only_scenario", "YES when the only other services are diagnostics or specimen collection."),
            (
                "diagnostic_only_policy_exception",
                "YES only when a named payer policy explicitly requires modifier 25 for a diagnostic-only scenario.",
            ),
            (
                "same_problem_only",
                "YES when the E/M addresses only the same problem as the minor procedure and reflects typical pre/intra/post work.",
            ),
            ("paired_organ_or_site_mentioned", "YES when the procedure involves a paired organ or site."),
            ("laterality_left_explicit", "YES when left-sided performance is explicitly documented."),
            ("laterality_right_explicit", "YES when right-sided performance is explicitly documented."),
            ("bilateral_explicit", "YES when bilateral performance is explicitly documented."),
            ("code_already_specifies_laterality", "YES when the CPT/HCPCS description already specifies left, right, or bilateral."),
            ("payer_requires_laterality_over_50", "YES only when the record explicitly indicates payer preference for LT/RT instead of 50."),
            ("addon_code_for_50", "YES when the code is an add-on code that should not receive modifier 50."),
            ("inherently_bilateral_code", "YES when the code is inherently bilateral and should not receive 50/LT/RT."),
        ],
        "context_fields": ("payer_name",),
    },
    {
        "name": "telehealth_facility_teaching",
        "modifiers": {"95", "PO", "PN", "GC"},
        "purpose": "Extract only the facts needed to decide telehealth modifier 95, facility modifiers PO/PN, and teaching modifier GC.",
        "items": [
            ("telehealth_audio_video_confirmed", "YES when real-time audio and video are both supported for the encounter."),
            ("telehealth_audio_only_confirmed", "YES when the encounter is clearly audio-only."),
            ("telehealth_patient_and_provider_separate_sites", "YES when patient and provider are documented at different physical sites."),
            ("telehealth_synchronous_interaction_confirmed", "YES when a synchronous telehealth encounter is clearly documented."),
            ("service_performed_on_dos", "YES when the billed service was performed on the documented date of service."),
            (
                "billing_provider_performed_service",
                "YES when the billed service is attributable to the billing/rendering provider rather than another department/entity.",
            ),
            ("off_campus_outpatient_hospital_confirmed", "YES when authoritative facility metadata supports off-campus hospital outpatient context."),
            (
                "facility_excepted_off_campus_pbd",
                "YES only when authoritative facility metadata explicitly supports excepted/grandfathered off-campus PBD status.",
            ),
            (
                "facility_nonexcepted_off_campus_pbd",
                "YES only when authoritative facility metadata explicitly supports non-excepted off-campus PBD status.",
            ),
            ("resident_involved_explicit", "YES when resident involvement in the billed service is explicitly documented."),
            ("teaching_physician_attestation_present", "YES when a teaching physician attestation/direction statement is explicitly documented."),
            ("cosign_only_without_attestation", "YES when there is only a cosign and no substantive teaching attestation."),
            ("medical_student_only_mentioned", "YES when the record supports only medical-student involvement rather than resident participation."),
        ],
        "context_fields": ("payer_name", "encounter_metadata", "facility_info", "provider_info"),
    },
    {
        "name": "therapy",
        "modifiers": {"GP", "GO", "GN", "CQ", "CO"},
        "purpose": "Extract only the facts needed to decide outpatient therapy discipline modifiers and assistant modifiers.",
        "items": [
            ("therapy_pt_plan_of_care_context", "YES when the encounter supports outpatient PT plan-of-care context."),
            ("therapy_ot_plan_of_care_context", "YES when the encounter supports outpatient OT plan-of-care context."),
            ("therapy_slp_plan_of_care_context", "YES when the encounter supports outpatient SLP plan-of-care context."),
            ("pta_involvement_ge_10_percent", "YES when a PTA furnished at least 10% of the total timed or untimed service."),
            ("ota_involvement_ge_10_percent", "YES when an OTA furnished at least 10% of the total timed or untimed service."),
            (
                "therapy_service_cancelled_or_not_performed",
                "YES when the therapy service was cancelled, deferred, or clearly not performed on this DOS.",
            ),
            ("therapy_non_outpatient_or_bundled_setting", "YES when the record explicitly indicates a non-outpatient or bundled therapy setting."),
            (
                "therapy_discipline_conflict_in_text",
                "YES when the record contains conflicting PT/OT/SLP discipline signals that cannot be reconciled.",
            ),
        ],
        "context_fields": ("encounter_metadata", "provider_info"),
    },
    {
        "name": "anesthesia",
        "modifiers": {"P1", "P2", "P3", "P4", "P5", "P6", "QK", "QX"},
        "purpose": "Extract only the facts needed to decide anesthesia physical-status modifiers and anesthesia medical direction modifiers.",
        "items": [
            ("asa_p1_explicit", "YES when ASA I / P1 is explicitly documented."),
            ("asa_p2_explicit", "YES when ASA II / P2 is explicitly documented."),
            ("asa_p3_explicit", "YES when ASA III / P3 is explicitly documented."),
            ("asa_p4_explicit", "YES when ASA IV / P4 is explicitly documented."),
            ("asa_p5_explicit", "YES when ASA V / P5 is explicitly documented."),
            ("asa_p6_explicit", "YES when ASA VI / P6 is explicitly documented."),
            ("no_systemic_disease_documented", "YES when the record clearly describes a normal healthy patient without systemic disease."),
            ("mild_systemic_disease_supported", "YES when the record supports mild, controlled systemic disease without functional limitation."),
            ("severe_systemic_disease_supported", "YES when the record supports severe systemic disease short of constant threat to life."),
            (
                "life_threatening_instability_supported",
                "YES when the record supports severe systemic disease that is a constant threat to life at the time of anesthesia.",
            ),
            ("moribund_survival_without_surgery_unlikely", "YES when the record supports moribund status with survival unlikely without surgery."),
            ("brain_dead_organ_donor_documented", "YES when brain death and organ-donor procurement are explicitly documented."),
            ("crna_performed_service", "YES when the CRNA is documented as performing the anesthesia service for this case."),
            (
                "physician_medical_direction_documented",
                "YES when physician medical direction is explicitly documented rather than mere availability/supervision.",
            ),
            (
                "concurrency_two_to_four_documented",
                "YES when the physician is explicitly documented as directing two to four concurrent anesthesia cases.",
            ),
            (
                "concurrency_over_four_or_supervision_only",
                "YES when the record suggests more than four concurrent cases or supervision-only/availability without medical direction.",
            ),
        ],
        "context_fields": ("provider_info", "anaesthesia_staff_info", "anaesthesia_concurrent_staff_info", "anaesthesia_record_info"),
    },
    {
        "name": "lab_coverage",
        "modifiers": {"QW", "90", "GY", "UD"},
        "purpose": "Extract only the facts needed to decide QW, 90, GY, and UD.",
        "items": [
            ("qw_eligible_lab_code", "YES when the billed lab code is supported as a QW-eligible waived test code."),
            ("point_of_care_or_in_office_test_performed", "YES when the test was clearly performed in-office/POC/rapid/in-house."),
            ("same_day_test_result_present", "YES when same-day test results are documented in the record."),
            (
                "reference_or_outside_lab_performed",
                "YES when the record clearly supports send-out/reference-lab performance for the billed lab test; for 36415, YES only when the draw/specimen is explicitly tied to a send-out/reference lab.",
            ),
            ("lab_test_performed_on_dos", "YES when the lab test was actually performed/resulted on the DOS rather than only ordered."),
            ("clia_waived_site_or_certificate", "YES when structured context supports a CLIA-waived site/certificate."),
            ("medicare_payer_context", "YES when payer context supports Medicare."),
            ("medicaid_payer_context", "YES when payer context supports Medicaid."),
            ("non_medicaid_payer_context_explicit", "YES when the payer context explicitly conflicts with Medicaid."),
            ("t1015_present_on_claim", "YES when T1015 is present anywhere on the claim for this DOS."),
            ("jcode_administered_on_dos", "YES when a J-code drug/supply was documented as administered/performed on the DOS."),
            (
                "statutory_exclusion_or_nonbenefit_supported",
                "YES when the service/item is explicitly supported as statutorily excluded or outside a Medicare benefit category.",
            ),
            ("comfort_or_convenience_only_use", "YES when the item/service is documented as comfort, convenience, hygiene, or personal-use only."),
            (
                "qualifying_medicare_benefit_category_supported",
                "YES when the documentation supports a covered Medicare benefit category for the item/service.",
            ),
            ("l3260_therapeutic_orthotic_use", "YES when L3260 is used for therapeutic orthotic/stabilization/offloading after injury or surgery."),
            ("l3260_comfort_or_footwear_use", "YES when L3260 is used only as comfort footwear/support without therapeutic orthotic intent."),
            ("qualifying_wound_or_injury_present", "YES when a qualifying wound, injury, or covered dressing indication is documented."),
            ("cosmetic_or_explicit_noncovered_language", "YES when cosmetic or explicitly non-covered language is documented."),
        ],
        "context_fields": ("payer_name", "lab_info", "facility_info"),
    },
]


def _get_model_kwargs(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _get_line_type(code: str) -> str:
    return "em" if is_em_code(code) else "procedure"


def _apply_modifiers(items: List[Any], modifiers_by_code: Dict[str, List[Any]], item_type: Any, clear_linked: bool = False):
    output = []
    for item in items or []:
        item_data = _dump(item).copy()
        code = item_data.get("code") or item_data.get("cpt_code") or getattr(item, "code", None) or getattr(item, "cpt_code", None)
        normalized_code = _normalize_code(code)
        item_data["modifiers"] = modifiers_by_code.get(normalized_code, [])
        if clear_linked and "linked_diagnosis" in item_data:
            item_data["linked_diagnosis"] = []
        output.append(item_type(**item_data))
    return output


def _dedupe_modifiers(mods: List[Modifier]) -> List[Modifier]:
    seen = set()
    output = []
    for modifier in mods:
        code = _normalize_code(getattr(modifier, "code", None))
        if not code or code in seen:
            continue
        seen.add(code)
        output.append(modifier)
    return output


def _features_for_code(checklist: List[ModifierChecklistItem], code: str) -> Dict[str, Dict[str, Any]]:
    normalized_code = _normalize_code(code)
    features: Dict[str, Dict[str, Any]] = {}
    for item in checklist or []:
        raw_id = getattr(item, "item_id", "") or ""
        item_id = raw_id.value if isinstance(raw_id, Enum) else str(raw_id)
        item_id = item_id.strip()
        if not item_id:
            continue
        applies_to = [_normalize_code(c) for c in (item.applies_to or []) if str(c).strip()]
        applies = not applies_to or any(target in {"ALL", "*"} for target in applies_to) or normalized_code in applies_to
        if not applies:
            continue
        support = item.answer == ChecklistAnswer.YES
        existing = features.get(item_id)
        if existing:
            existing["support"] = existing["support"] or support
            if item.evidence:
                existing["evidence"].append(item.evidence)
            if item.justification:
                existing["justification"].append(item.justification)
            continue
        features[item_id] = {
            "support": support,
            "evidence": [item.evidence] if item.evidence else [],
            "justification": [item.justification] if item.justification else [],
        }
    return features


def _build_assigned_justification(features: Dict[str, Dict[str, Any]], relevant_items: Optional[Iterable[str]] = None) -> Optional[str]:
    allowed = set(relevant_items or [])
    parts = []
    for name, meta in features.items():
        if allowed and name not in allowed:
            continue
        if not meta.get("support"):
            continue
        reason = " | ".join([value for value in meta.get("justification", []) if value]).strip()
        evidence = " | ".join([value for value in meta.get("evidence", []) if value]).strip()
        if reason and evidence:
            parts.append(f"{name}: {reason} (evidence: {evidence})")
        elif reason:
            parts.append(f"{name}: {reason}")
        elif evidence:
            parts.append(f"{name}: evidence: {evidence}")
        else:
            parts.append(f"{name}: supported")
    return " | ".join(parts) if parts else None


def _group_for_modifier(modifier: str) -> Optional[dict[str, Any]]:
    normalized = _normalize_code(modifier)
    for group in RULE_GROUPS:
        if normalized in group["modifiers"]:
            return group
    return None


def _selected_rule_groups(scope: List[str]) -> List[dict[str, Any]]:
    normalized_scope = {_normalize_code(modifier) for modifier in scope}
    return [group for group in RULE_GROUPS if group["modifiers"].intersection(normalized_scope)]


def _build_group_rule_definitions(group: dict[str, Any]) -> str:
    lines = []
    for item_id, instruction in group["items"]:
        lines.append(f"- `{item_id}`: {instruction}")
    return "\n".join(lines)


def _serialize_structured_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(indent=2)
    if hasattr(value, "model_dump"):
        return str(value.model_dump())
    return str(value)


def _render_additional_info(state: ModifierGraphState, fields: Iterable[str]) -> str:
    sections: List[str] = []
    for field_name in fields:
        value = state.get(field_name)
        if not value:
            continue
        label = field_name.replace("_", " ").title()
        sections.append(f"- {label}:")
        if isinstance(value, list):
            for item in value:
                sections.append(_serialize_structured_value(item))
        else:
            sections.append(_serialize_structured_value(value))
    return "\n".join(sections)


def _parse_prompt_versions(config: RunnableConfig) -> dict[str, str]:
    raw = config["configurable"].get(CONFIG.PROMPT_VERSIONS)
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    return {}


def _resolve_guideline_version(modifier: str, config: RunnableConfig, default_version: str) -> str:
    modifier_code = _normalize_code(modifier)
    prompt_versions = _parse_prompt_versions(config)
    family_key = GUIDELINE_VERSION_KEYS_BY_MODIFIER.get(modifier_code)
    return (
        prompt_versions.get(modifier_code)
        or (prompt_versions.get(family_key) if family_key else None)
        or prompt_versions.get("ALL")
        or default_version
    )


def _sorted_guideline_versions(guidelines_doc: Optional[dict[str, Any]]) -> List[str]:
    if not isinstance(guidelines_doc, dict):
        return []

    def version_sort_key(value: str) -> tuple[int, int | str]:
        if value.startswith("v") and value[1:].isdigit():
            return (1, int(value[1:]))
        return (0, value)

    return sorted((str(key) for key in guidelines_doc.keys()), key=version_sort_key, reverse=True)


def _lookup_modifier_guideline_doc(
    guidelines_doc: Optional[dict[str, Any]],
    modifier: str,
    config: RunnableConfig,
    default_version: str,
) -> Optional[dict[str, Any]]:
    modifier_code = _normalize_code(modifier)
    prompt_versions = _parse_prompt_versions(config)
    candidate_versions = [
        _resolve_guideline_version(modifier, config, default_version),
        prompt_versions.get("ALL"),
        default_version,
        *_sorted_guideline_versions(guidelines_doc),
    ]

    seen_versions = set()
    for version in candidate_versions:
        if not version or version in seen_versions:
            continue
        seen_versions.add(version)
        version_doc = guidelines_doc.get(version) if isinstance(guidelines_doc, dict) else None
        if not isinstance(version_doc, dict):
            continue
        modifier_doc = version_doc.get(modifier_code)
        if isinstance(modifier_doc, dict):
            return modifier_doc
    return None


def _join_guideline_sections(values: List[str], separator: str = "\n-----------------------------\n") -> str:
    if not values:
        return ""
    unique_values = list(dict.fromkeys(values))
    return separator.join(unique_values)


def _build_group_guidance_strings(
    guidelines_doc: Optional[dict[str, Any]],
    config: RunnableConfig,
    modifiers_in_scope: List[str],
    default_version: str,
) -> tuple[str, str, str, str]:
    if not guidelines_doc:
        return "", "", "", ""

    guidelines: List[str] = []
    compliance_checks: List[str] = []
    checklist: List[str] = []
    examples: List[str] = []

    for modifier in modifiers_in_scope:
        modifier_doc = _lookup_modifier_guideline_doc(guidelines_doc, modifier, config, default_version)
        if not isinstance(modifier_doc, dict):
            continue
        if modifier_doc.get("guidelines"):
            guidelines.append(str(modifier_doc["guidelines"]))
        if modifier_doc.get("compliance_checks"):
            compliance_checks.append(str(modifier_doc["compliance_checks"]))
        if modifier_doc.get("checklist"):
            checklist.append(str(modifier_doc["checklist"]))
        if modifier_doc.get("examples"):
            examples.append(str(modifier_doc["examples"]))

    return (
        _join_guideline_sections(guidelines),
        _join_guideline_sections(compliance_checks, separator="\n"),
        _join_guideline_sections(checklist, separator="\n"),
        _join_guideline_sections(examples, separator="\n"),
    )


def _base_all_item(item_id: str, support: bool, evidence: Optional[str] = None, justification: Optional[str] = None) -> ModifierChecklistItem:
    return ModifierChecklistItem(
        item_id=item_id,
        answer=ChecklistAnswer.YES if support else ChecklistAnswer.NO,
        evidence=evidence,
        justification=justification,
        applies_to=["ALL"],
    )


def _base_code_item(
    item_id: str, code: str, support: bool, evidence: Optional[str] = None, justification: Optional[str] = None
) -> ModifierChecklistItem:
    return ModifierChecklistItem(
        item_id=item_id,
        answer=ChecklistAnswer.YES if support else ChecklistAnswer.NO,
        evidence=evidence,
        justification=justification,
        applies_to=[_normalize_code(code)],
    )


def _extract_clia_waived_flag(state: ModifierGraphState) -> Optional[ModifierChecklistItem]:
    def extract_clia_value(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        if isinstance(raw, str):
            cleaned = raw.strip()
            return cleaned or None
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                found = extract_clia_value(item)
                if found:
                    return found
            return None
        if isinstance(raw, dict):
            for key in (
                "clia_certificate_type",
                "facility_clia_certificate_type",
                "performing_lab_clia_type",
                "lab_clia_certificate_type",
                "clia_type",
                "certificate_type",
            ):
                if key in raw:
                    found = extract_clia_value(raw.get(key))
                    if found:
                        return found
            return None
        return None

    structured_candidates: List[Any] = []
    structured_candidates.extend(state.get("procedure") or [])
    structured_candidates.extend(state.get("facility_info") or [])
    structured_candidates.extend(state.get("lab_info") or [])
    for key in ("clia_certificate_type", "facility_clia_certificate_type", "performing_lab_clia_type", "lab_clia_certificate_type", "clia_type"):
        if state.get(key):
            structured_candidates.append(state.get(key))

    value = None
    for candidate in structured_candidates:
        value = extract_clia_value(_dump(candidate))
        if value:
            break

    if not value:
        return None

    normalized = value.strip().upper()
    support = normalized in WAIVED_CLIA_KEYWORDS or "WAIV" in normalized
    return _base_all_item(
        "clia_waived_site_or_certificate",
        support,
        evidence=f"Structured CLIA type: {value}",
        justification="Derived from structured CLIA certificate metadata.",
    )


def _derive_static_checklist_items(state: ModifierGraphState) -> List[ModifierChecklistItem]:
    items: List[ModifierChecklistItem] = []
    payer_names = [str(value).strip() for value in (state.get("payer_name") or []) if str(value).strip()]
    payer_names_upper = [value.upper() for value in payer_names]

    has_medicare = any(any(keyword in payer for keyword in MEDICARE_PAYER_KEYWORDS) for payer in payer_names_upper)
    has_medicaid = any(any(keyword in payer for keyword in MEDICAID_PAYER_KEYWORDS) for payer in payer_names_upper)
    has_non_medicaid = bool(payer_names) and not has_medicaid

    items.append(_base_all_item("medicare_payer_context", has_medicare, evidence=", ".join(payer_names) or None))
    items.append(_base_all_item("medicaid_payer_context", has_medicaid, evidence=", ".join(payer_names) or None))
    items.append(_base_all_item("non_medicaid_payer_context_explicit", has_non_medicaid, evidence=", ".join(payer_names) or None))

    cpt_codes = [_normalize_code(code) for code in get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure")) if str(code).strip()]
    items.append(_base_all_item("t1015_present_on_claim", "T1015" in cpt_codes, evidence="T1015 in CPT list" if "T1015" in cpt_codes else None))

    for procedure in state.get("procedure") or []:
        data = _dump(procedure)
        code = _normalize_code(data.get("code") or data.get("cpt_code"))
        if not code:
            continue
        description = str(data.get("description") or "").strip()
        description_lower = description.lower()
        items.append(
            _base_code_item(
                "addon_code_for_50",
                code,
                code in {value.strip().upper() for value in addon_codes_mod_50},
                evidence=description or None,
                justification="Derived from maintained add-on code list.",
            )
        )
        items.append(
            _base_code_item(
                "inherently_bilateral_code",
                code,
                code in {value.strip().upper() for value in inherently_bilateral_codes},
                evidence=description or None,
                justification="Derived from maintained inherently bilateral code list.",
            )
        )
        code_has_laterality = any(token in description_lower for token in (" left", " right", "bilateral", "left ", "right "))
        items.append(
            _base_code_item(
                "code_already_specifies_laterality",
                code,
                code_has_laterality,
                evidence=description or None,
                justification="Derived from the procedure description.",
            )
        )

    facility_info = state.get("facility_info") or []
    pos_19_supported = False
    evidence = None
    for facility in facility_info:
        data = _dump(facility)
        pos_code = str(data.get("pos_code") or "").strip()
        if pos_code in OFF_CAMPUS_POS_CODES:
            pos_19_supported = True
            evidence = f'pos_code="{pos_code}"'
            break
    items.append(_base_all_item("off_campus_outpatient_hospital_confirmed", pos_19_supported, evidence=evidence))

    clia_item = _extract_clia_waived_flag(state)
    if clia_item is not None:
        items.append(clia_item)

    return items


def _merge_checklist_items(items: Iterable[ModifierChecklistItem]) -> List[ModifierChecklistItem]:
    merged: Dict[tuple[str, tuple[str, ...]], ModifierChecklistItem] = {}
    for item in items:
        key = (
            item.item_id.value if isinstance(item.item_id, ChecklistItemId) else str(item.item_id),
            tuple(sorted(_normalize_code(value) for value in (item.applies_to or []) if str(value).strip())),
        )
        existing = merged.get(key)
        if not existing:
            merged[key] = item
            continue
        answers = {existing.answer, item.answer}
        if ChecklistAnswer.YES in answers:
            existing.answer = ChecklistAnswer.YES
        elif ChecklistAnswer.NO in answers:
            existing.answer = ChecklistAnswer.NO
        else:
            existing.answer = ChecklistAnswer.UNCLEAR
        if item.evidence and item.evidence not in (existing.evidence or ""):
            existing.evidence = " | ".join([part for part in [existing.evidence, item.evidence] if part])
        if item.justification and item.justification not in (existing.justification or ""):
            existing.justification = " | ".join([part for part in [existing.justification, item.justification] if part])
    return list(merged.values())


@track_tokens("model_usage")
def rules_extraction_node(state: ModifierGraphState, config: RunnableConfig) -> dict:
    prompt_version = config["configurable"][CONFIG.PROMPT_VERSION]
    prompt_template = config["configurable"][CONFIG.PROMPTS][PROMPTS.DECISION_RULES][prompt_version]
    guidelines_path = config["configurable"].get("guidelines_path")
    scope = get_modifiers_in_scope(config["configurable"].get("modifiers_in_scope"))
    selected_groups = _selected_rule_groups(scope)
    if not selected_groups:
        return {"rules": ModifierChecklistExtraction(checklist=[])}
    guidelines_doc = load_from_yaml(guidelines_path) if guidelines_path else None

    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**_get_model_kwargs(config))
    parser = PydanticOutputParserWithRetries(pydantic_object=ModifierChecklistExtraction, correction_model=model, num_retries=3)
    cpt_codes = get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))
    aggregated_items: List[ModifierChecklistItem] = []

    for group in selected_groups[:5]:
        group_scope = [modifier for modifier in scope if _normalize_code(modifier) in group["modifiers"]]
        modifier_guidelines, compliance_checks, checklist, examples = _build_group_guidance_strings(
            guidelines_doc=guidelines_doc,
            config=config,
            modifiers_in_scope=group_scope,
            default_version=prompt_version,
        )
        prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(
            format_instructions=parser.get_format_instructions(),
            modifier_guidelines=modifier_guidelines,
            compliance_checks=compliance_checks,
            checklist=checklist,
            examples=examples,
            group_name=group["name"],
            group_purpose=group["purpose"],
            group_rule_definitions=_build_group_rule_definitions(group),
        )
        chain = prompt | model | parser
        response = chain.invoke(
            {
                "soap_note": state.get("text"),
                "cpt_code": ",".join(cpt_codes),
                "orders": state.get("orders_display"),
                "additional_info": _render_additional_info(state, group["context_fields"]),
            }
        )
        aggregated_items.extend((response.checklist if response else []) or [])

    aggregated_items.extend(_derive_static_checklist_items(state))
    return {"rules": ModifierChecklistExtraction(checklist=_merge_checklist_items(aggregated_items))}


def rule_engine_node(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    extracted_rules = state.get("rules")
    checklist = (extracted_rules.checklist if extracted_rules else []) or []
    cpt_codes = [_normalize_code(code) for code in get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure")) if str(code).strip()]
    modifiers_by_code: Dict[str, List[Modifier]] = defaultdict(list)
    unassigned_by_mod: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"reasons": [], "codes": []})
    scope = get_modifiers_in_scope(config["configurable"].get("modifiers_in_scope"))
    scope_set = {_normalize_code(modifier) for modifier in scope if str(modifier).strip()}
    modifiers_to_check = sorted(scope_set or {_normalize_code(modifier) for modifier in ASSIGNMENT_FUNCTIONS.keys() if str(modifier).strip()})

    for code in cpt_codes:
        features = _features_for_code(checklist, code)
        line_type = _get_line_type(code)
        for rule_mod in modifiers_to_check:
            if not modifier_applicable_to_code(rule_mod, code):
                continue
            assigner = ASSIGNMENT_FUNCTIONS.get(rule_mod, assign_default)
            assigned = assigner(code, features, line_type=line_type, state=state)
            if assigned:
                group = _group_for_modifier(rule_mod)
                relevant_items = [item_id for item_id, _ in (group["items"] if group else [])]
                justification = _build_assigned_justification(features, relevant_items=relevant_items)
                modifiers_by_code[code].append(Modifier(code=rule_mod, justification=justification))
                continue
            reason_parts = build_unassigned_reasons(rule_mod, code, features)
            reason_text = " ".join(reason_parts) if reason_parts else f"Checklist did not support modifier {rule_mod}."
            unassigned_by_mod[rule_mod]["reasons"].append(f"CPT {code}: {reason_text}")
            unassigned_by_mod[rule_mod]["codes"].append(code)

    em_modifiers: Dict[str, List[Modifier]] = {}
    proc_modifiers: Dict[str, List[Modifier]] = {}
    for code, modifiers in modifiers_by_code.items():
        if is_em_code(code):
            em_modifiers[code] = _dedupe_modifiers(modifiers)
        else:
            proc_modifiers[code] = _dedupe_modifiers(modifiers)

    unassigned_modifiers: List[ModifierJustification] = []
    for mod_code, payload in unassigned_by_mod.items():
        codes = sorted(set(payload["codes"]))
        if not codes:
            continue
        unique_reasons = " | ".join(dict.fromkeys(payload["reasons"]))
        unassigned_modifiers.append(
            ModifierJustification(
                code=mod_code,
                evidence=None,
                justification=unique_reasons,
                assigned=False,
                cpt_codes=codes,
            )
        )

    return {
        "em": _apply_modifiers(state.get("em"), em_modifiers, EMItem),
        "procedure": _apply_modifiers(state.get("procedure"), proc_modifiers, Procedure, clear_linked=True),
        "unassigned_modifiers": unassigned_modifiers,
    }


def build_decision_rules_subgraph():
    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("rules_extraction", rules_extraction_node)
    workflow.add_node("rule_engine", rule_engine_node)
    workflow.add_edge(START, "rules_extraction")
    workflow.add_edge("rules_extraction", "rule_engine")
    workflow.add_edge("rule_engine", END)
    return workflow.compile()
