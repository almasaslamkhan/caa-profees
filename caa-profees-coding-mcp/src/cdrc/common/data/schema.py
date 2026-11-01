import re
from itertools import chain
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.utils import StringEnum

# Regexes for validation
RE_CPT = re.compile(r"^(?:\d{4}[A-Z]|\d{5})$")
RE_HCPCS = re.compile(r"^[A-Z]\d{4}$")
RE_ICD10 = re.compile(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")
RE_MODIFIER = re.compile(r"^[A-Z0-9]{2}$")
RE_NOTE_YYYY_MM = re.compile(r"^\d{4}\.(0[1-9]|1[0-2])$")


class ProblemComplexityType(StringEnum):
    # Minimal complexity
    SELF_LIMITED = "self-limited or minor problem"
    # Low complexity
    STABLE_CHRONIC = "stable, chronic illness"
    ACUTE_UNCOMPLICATED = "acute, uncomplicated illness or injury"
    STABLE_ACUTE = "stable, acute illness"
    ACUTE_UNCOMPLICATED_INPATIENT = "acute, uncomplicated illness or injury requiring hospital inpatient or observation level of care"
    # Moderate complexity
    CHRONIC_EXACERBATION = "chronic illness with exacerbation, progression, or side effects of treatment"
    UNDIAGNOSED_UNCERTAIN = "undiagnosed new problem with uncertain prognosis"
    ACUTE_SYSTEMIC = "acute illness with systemic symptoms"
    ACUTE_COMPLICATED = "acute, complicated injury"
    # High complexity
    CHRONIC_SEVERE = "chronic illness with severe exacerbation, progression, or side effects of treatment"
    LIFE_THREATENING = "acute or chronic illness or injury that poses a threat to life or bodily function"


class ComplexityLevel(StringEnum):
    MINIMAL = "Minimal"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class AgeCategories(StringEnum):
    INVALID = "invalid_age"
    INFANT = "infant"
    EARLY_CHILDHOOD = "early_childhood"
    LATE_CHILDHOOD = "late_childhood"
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young"
    ADULT = "adult"
    SENIOR_ADULT = "senior"


class PatientType(StringEnum):
    NEW = "New"
    ESTABLISHED = "Established"


# TODO: remove comments once we have more encounter types to consider
class EncounterType(StringEnum):
    OUTPATIENT = "Office or Other Outpatient Service"
    PREVENTIVE_MEDICINE = "Preventive Medicine"
    EMERGENCY_DEPARTMENT = "Emergency Department"
    EMERGENCY_OTHER_SERVICES = "Other Emergency Services"
    CRITICAL_CARE = "Critical Care Services"
    INPATIENT_INITIAL = "Initial Hospital Inpatient or Observation Care"
    INPATIENT_SUBSEQUENT = "Subsequent Hospital Inpatient or Observation Care"
    INPATIENT_SAME_DAY = "Hospital Inpatient or Observation Care Services (Including Admission and Discharge Services)"
    INPATIENT_DISCHARGE = "Hospital Inpatient or Observation Discharge Services"
    NEWBORN_INITIAL = "Initial Hospital or Birthing Center Care, Normal Newborn"
    NEWBORN_SUBSEQUENT = "Subsequent Hospital Care, Normal Newborn"
    NEWBORN_SAME_DAY = "Initial and Discharge Same Date Hospital or Birthing Center Care, Normal Newborn"
    NEWBORN_OTHER_SETTING = "Initial Care, Normal Newborn, Other Setting"
    # CONSULTATION = "Consultation"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    OTHERS = "Others"


class ComplexityDetails(BaseModel):
    level: ComplexityLevel = Field(..., description="The level of complexity")
    detailed_justification: Optional[str] = Field(None, description="The detailed justification for the level of complexity")
    justification: Optional[str] = Field(None, description="The brief justification for the level of complexity. Please limit to 10 words")
    confidence: Optional[float] = Field(-1, description="The confidence in the level of complexity")


class EncounterTypeDetails(BaseModel):
    encounter_type: EncounterType = Field(..., description="The encounter type")
    justification: Optional[str] = Field(None, description="The justification for the chosen encounter type.  Please limit to 10 words")


# Added this for handling multiple codes
class EncounterTypesDetails(BaseModel):
    encounter_types: List[EncounterTypeDetails]


class EncounterMetaData(BaseModel):
    client_name: Optional[str] = Field(None, description="Client name")
    type: Optional[str] = Field(None, alias="type", description="Source-specific encounter categorization")
    admit_time: Optional[str] = Field(None, description="Episode admission date or timestamp")
    discharge_time: Optional[str] = Field(None, description="Episode discharge date or timestamp")
    total_days: Optional[int] = Field(None, alias="days", description="Total number of days in the episode")
    service_dates: Optional[List[str]] = Field(None, description="List of service dates included in the episode")

    @model_validator(mode="before")
    def _coerce_days(cls, v):
        if not isinstance(v, dict):
            return v
        days = v.get("days")
        if isinstance(days, str) and days.isdigit():
            v["days"] = int(days)
        return v

    @model_validator(mode="before")
    def _coerce_service_dates(cls, v):
        if not isinstance(v, dict):
            return v
        service_dates = v.get("service_dates")
        if service_dates is None:
            return v
        if isinstance(service_dates, str):
            v["service_dates"] = [service_dates]
        elif isinstance(service_dates, (set, tuple)):
            v["service_dates"] = list(service_dates)
        elif isinstance(service_dates, list):
            v["service_dates"] = [str(date) for date in service_dates if date]
        else:
            v["service_dates"] = [str(service_dates)]
        return v


class EncounterDocument(BaseModel):
    document_id: Optional[Union[int, str]] = Field(None, description="Source document identifier")
    service_date: Optional[str] = Field(None, description="Service date associated with the document")
    document_type: Optional[str] = Field(None, description="Document type or section label")
    document_status: Optional[str] = Field(None, description="Document status")
    text: Optional[str] = Field(None, description="Document body text")
    time_created: Optional[str] = Field(None, description="Document creation timestamp")
    signature_time: Optional[str] = Field(None, description="Document signature timestamp")
    attestation_time: Optional[str] = Field(None, description="Document attestation timestamp")
    provider_participants: Optional[List[Dict[str, Any]]] = Field(None, description="Document provider participants")
    document_type_normalized: Optional[str] = Field(
        None,
        description="Lower-cased document type used for internal logic",
        exclude=True,
    )

    @model_validator(mode="before")
    def _normalize_bool(cls, v):
        if not isinstance(v, dict):
            return v
        value = v.get("is_physician_note")
        practitioner = v.get("practitioner")
        if practitioner:
            v.setdefault("practitioner_id", practitioner)
            v.setdefault("practitioner_name", practitioner)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "true", "1"}:
                v["is_physician_note"] = True
            elif lowered in {"no", "false", "0"}:
                v["is_physician_note"] = False
        doc_type = v.get("document_type")
        if isinstance(doc_type, str) and doc_type.strip():
            v["document_type_normalized"] = doc_type.strip().lower()
        return v


CPT_CODES_MAPPING = {
    "Office or Other Outpatient Service": [
        "NOT_SUPPORTED",
        "99201",
        "99202",
        "99203",
        "99204",
        "99205",
        "99211",
        "99212",
        "99213",
        "99214",
        "99215",
    ],
    "Consultation": ["NOT_SUPPORTED", "99242", "99243", "99244", "99245"],
    "Preventive Medicine": [
        "NOT_SUPPORTED",
        "INVALID_AGE",
        "99381",
        "99382",
        "99383",
        "99384",
        "99385",
        "99386",
        "99387",
        "99391",
        "99392",
        "99393",
        "99394",
        "99395",
        "99396",
        "99397",
    ],
    "Initial Hospital Inpatient or Observation Care": [
        "NOT_SUPPORTED",
        "99221",
        "99222",
        "99223",
    ],
    "Subsequent Hospital Inpatient or Observation Care": [
        "NOT_SUPPORTED",
        "99231",
        "99232",
        "99233",
        "99224",
        "99225",
        "99226",
    ],
    "Hospital Inpatient or Observation Care Services (Including Admission and Discharge Services)": [
        "NOT_SUPPORTED",
        "99234",
        "99235",
        "99236",
    ],
    "Hospital Inpatient or Observation Discharge Services": [
        "NOT_SUPPORTED",
        "99238",
        "99239",
    ],
    "Emergency Department": [
        "NOT_SUPPORTED",
        "99281",
        "99282",
        "99283",
        "99284",
        "99285",
    ],
    "Other Emergency Services": [
        "NOT_SUPPORTED",
        "99288",
    ],
    "Critical Care Services": [
        "NOT_SUPPORTED",
        "99291",
        "99292",
    ],
    "Initial Hospital or Birthing Center Care, Normal Newborn": [
        "NOT_SUPPORTED",
        "INVALID_AGE",
        "99460",
    ],
    "Initial Care, Normal Newborn, Other Setting": [
        "NOT_SUPPORTED",
        "INVALID_AGE",
        "99461",
    ],
    "Subsequent Hospital Care, Normal Newborn": [
        "NOT_SUPPORTED",
        "INVALID_AGE",
        "99462",
    ],
    "Initial and Discharge Same Date Hospital or Birthing Center Care, Normal Newborn": [
        "NOT_SUPPORTED",
        "INVALID_AGE",
        "99463",
    ],
    "NOT_SUPPORTED": ["NOT_SUPPORTED"],
    "Others": ["NOT_SUPPORTED"],
}

VALID_CPT_CODES = list(chain.from_iterable(CPT_CODES_MAPPING.values()))
VALID_ENCOUNTER_TYPES = list(CPT_CODES_MAPPING.keys())


class Problem(BaseModel):
    name: str = Field(..., description="The name of the problem")
    type: Optional[ProblemComplexityType] = Field(None, description="The type of problem complexity")


class OrderType(StringEnum):
    MEDICATION = "Medication Order"
    NON_MEDICATION = "Non Medication Order"


class OrderDisplay(BaseModel):
    hna_mnemonic: Optional[str] = None
    ordered_as: Optional[str] = None
    clinical_display_line: Optional[Union[str, None]] = Field(None, description="The clinical display line")


class Order(BaseModel):
    order_type: OrderType = Field(..., description="Medication or non-medication")
    status: str = Field(..., description="Order status")
    catalog_type: str = Field(..., description="Catalog type")
    time_created: Optional[str] = Field(None, description="Order creation timestamp")
    time_last_updated: Optional[str] = Field(None, description="Order last updated timestamp")
    display: OrderDisplay = Field(..., description="Order description and details")


class HealthPlan(BaseModel):
    health_plan: Optional[str] = Field(None, description="The health plan of the patient")
    active_status: Optional[str] = Field(None, description="The active status of the health plan of the patient")
    health_plan_effective_date: Optional[str] = Field(None, description="The health plan effective date of the patient")


class FacilityInfo(BaseModel):
    patient_loc: Optional[str] = Field(None, description="Patient location")
    nurse_unit_loc: Optional[str] = Field(None, description="Nurse unit location")
    building_loc: Optional[str] = Field(None, description="Building location")
    facility_loc: Optional[str] = Field(None, description="Facility location")
    ccn_name: Optional[str] = Field(None, description="Name associated to the CCN.")
    ccn_number: Optional[float] = Field(None, description="CCNs identifying number")
    pos_code: Optional[int] = Field(
        None, description="Point of service code used on professional claims to specify the entity where service(s) were rendered."
    )
    cms_submission_type: Optional[str] = Field(None, description="Encounter associated encounter type")
    beg_effective_dt_tm: Optional[str] = Field(None, description="Beginning effective date")
    end_effective_dt_tm: Optional[str] = Field(None, description="Ending effective date")
    is_excepted: Optional[bool] = Field(None, description="Whether the facility is excepted or non-excepted")


class ProviderInfo(BaseModel):
    physician_ind: Optional[int] = Field(None, description="Set to 1, if the personnel is a physician. Otherwise, set to 0.")
    position: Optional[str] = Field(
        None, description="The position is used to determine the applications and tasks the personnel is authorized to use."
    )
    physician_status: Optional[str] = Field(
        None, description="Physician status code identifies the status of the physician. (For Example: In, In Surgery, Out, etc.)"
    )
    encntr_prsnl_r: Optional[str] = Field(None, description="Relationship of the prsnl to the encounter")


class LabInfo(BaseModel):
    facility_location: Optional[str] = Field(None, description="Facility location")
    service_location: Optional[str] = Field(None, description="Service location")
    lab_name: Optional[str] = Field(None, description="Lab name")
    clia_number_txt: Optional[str] = Field(None, description="The Clinical Laboratory Improvement Act number associated with the service resource.")


class AnaesthesiaStaffInfo(BaseModel):
    role_performed: Optional[str] = Field(None, description="Role performed")
    signing_indicator: Optional[int] = Field(None, description="Signing attendee indicator")
    in_time: Optional[str] = Field(None, description="Begin date and time of the attendee for the case.")
    out_time: Optional[str] = Field(None, description="End date and time of the attendee for the case.")


class AnaesthesiaConcurrentStaffInfo(BaseModel):
    relationship: Optional[str] = Field(None, description="Relationship of the personnel to the encounter")
    concurrent_encounter_count: Optional[int] = Field(None, description="Number of concurrent encounter the person is involved in")


class AnaesthesiaRecordInfo(BaseModel):
    asa_display_value: Optional[str] = Field(None, description="The value of the result")
    asa_structured_status: Optional[str] = Field(None, description="Structured nomenclature status")
    result_date: Optional[str] = Field(None, description="Event end date time")


class AWVHistory(BaseModel):
    hcpcs_code: Optional[str] = Field(None, description="HCPCS code from past claims")
    claim_date: Optional[str] = Field(None, description="The date of the claim")


class AdditionalInput(BaseModel):
    note_time: Optional[str] = Field(None, description="YYYY.MM")
    note_type: Optional[str] = None
    physician_code: Optional[Union[str, List[str]]] = None
    physician_specialty: Optional[str] = None
    coder_code: Optional[Union[str, List[str]]] = None
    customer_name: Optional[str] = None
    coding_mode: Optional[str] = Field(None, description="Coding mode indicator (e.g., concurrent)")
    concurrent_coding_rule: Optional[str] = Field(None, description="Rule guiding concurrent coding")
    coding_mode: Optional[str] = Field(None, description="Coding mode indicator (e.g., concurrent)")
    concurrent_coding_rule: Optional[str] = Field(None, description="Rule guiding concurrent coding")

    @model_validator(mode="before")
    def _validate_time_and_codes(cls, v):
        if not isinstance(v, dict):
            return v
        # Normalize codes to list[str]
        for key in ["physician_code", "coder_code"]:
            val = v.get(key)
            if isinstance(val, str):
                # split by comma and strip
                v[key] = [x.strip() for x in val.split(",") if x.strip()]
        return v


class InputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    encounter_metadata: Optional[EncounterMetaData] = Field(
        None,
        alias="encounter_information",
        description="Inpatient or longitudinal encounter metadata, including admission and discharge context.",
    )
    text: Optional[str] = Field(None, description="The clinical note text")
    soap: Optional[dict] = Field(default_factory=dict, description="SOAP note formatted in the draft note format of notegen")
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(
        default_factory=str, description="The list of orders for the encounter displayed as a single string for convenience."
    )
    health_plan: Optional[List[HealthPlan]] = Field(None, description="The health plan status of the patient")
    awv_history: Optional[List[AWVHistory]] = Field(None, description="A list of past AWV claims")
    facility_info: Optional[List[FacilityInfo]] = Field(None, description="Expected vs non-expected off-campus information")
    provider_info: Optional[List[ProviderInfo]] = Field(
        None, description="Clinical personnel and their associated professional roles for a patient visit"
    )
    lab_info: Optional[List[LabInfo]] = Field(
        None,
        description="Specific laboratory performing clinical work, its associated CLIA certification, and whether the work was performed internally or by an outside reference lab",
    )
    anaesthesia_staff_info: Optional[List[AnaesthesiaStaffInfo]] = Field(None, description="Anaesthesia staff information")
    anaesthesia_concurrent_staff_info: Optional[List[AnaesthesiaConcurrentStaffInfo]] = Field(
        None, description="Information about staffs concurrently involved in other encounters."
    )
    anaesthesia_record_info: Optional[List[AnaesthesiaRecordInfo]] = Field(None, description="Anaesthesia records information")
    encounter_date: Optional[str] = Field(None, description="The date of the encounter")
    appointment_type: Optional[str] = Field(None, description="Appointment type associated with the encounter, when available")
    patient_type: Optional[PatientType] = Field(None, description="The type of patient, e.g., 'New' or 'Established'")
    patient_age: Optional[Union[int, str]] = Field(None, description="The age of the patient")
    patient_sex: Optional[str] = Field(None, description="Sex as provided (free text to avoid over-constraining)")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    source: Optional[str] = Field(None, description="The source of the clinical note, e.g., 'Synthetic'")
    transcript: Optional[str] = Field(
        None,
        description="The verbatim transcription of the clinical encounter, including dialogue, observations,"
        "and key details captured during the visit.",
    )
    documents: Optional[List[EncounterDocument]] = Field(
        default_factory=list,
        description="Supporting clinical documents associated with the encounter or inpatient stay.",
    )
    tags: Optional[List[str]] = Field(None, description="Tags to compute metrics across different test types.")
    additional_input: Optional[AdditionalInput] = Field(None, description="Downstream metadata such as coder/physician codes")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")

    class Config:
        use_enum_values = True  # Automatically use enum values as strings
        populate_by_name = True

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def to_json(self):
        return self.model_dump(exclude_none=False)

    @model_validator(mode="before")
    def parse_patient_age(self):
        age = self.get("patient_age")
        if isinstance(age, str) and not age.isdigit():
            if age == ">=90":
                self["patient_age"] = 91
            else:
                self["patient_age"] = -1  # Defaults to -1
        elif age is not None:
            self["patient_age"] = int(age)
        return self

    @model_validator(mode="before")
    def parse_soap_note(self):
        def _is_inpatient_encounter(metadata):
            if not metadata:
                return False
            if isinstance(metadata, dict):
                encounter_type = metadata.get("type") or metadata.get("encounter_type")
            else:
                encounter_type = getattr(metadata, "type", None) or getattr(metadata, "encounter_type", None)
            if not encounter_type:
                return False
            return "inpatient" in str(encounter_type).strip().lower()

        soap = self.get("soap")
        text = self.get("text")
        documents = self.get("documents")
        metadata = self.get("encounter_metadata")
        if not soap and not text:
            if documents:
                pass
            elif _is_inpatient_encounter(metadata):
                # Allow inpatient encounters to rely solely on document bundles.
                self["documents"] = documents or []
            else:
                raise ValueError("InputSchema requires at least one of text, soap, or documents.")

        if soap and not text:
            self["text"] = format_soap_note(soap)
        documents = self.get("documents")
        if documents and not isinstance(documents, list):
            self["documents"] = [documents]
        if documents:
            try:
                self["documents"] = [EncounterDocument.model_validate(doc) for doc in documents]
            except (ValidationError, TypeError):
                self["documents"] = [EncounterDocument.model_validate(dict(doc)) for doc in documents]
        return self

    @model_validator(mode="before")
    def handle_unexpected_types(self):
        for field in ["vitals", "allergies", "medications"]:
            if not isinstance(self.get(field), list):
                self[field] = [self.get(field, "")]
            else:
                # account for the case when one of them is retrieved from facade as a dictionary
                if any(not isinstance(elem, str) for elem in self.get(field, [])):
                    self[field] = [str(elem) for elem in self.get(field, [])]

        if not isinstance(self.get("all_patient_problems"), list):
            if isinstance(self.get("all_patient_problems"), str):
                self["all_patient_problems"] = [{"name": elem} for elem in self.get("all_patient_problems", "").split("|")]
            else:
                self["all_patient_problems"] = [{"name": str(self.get("all_patient_problems", ""))}]
        else:
            try:
                self["all_patient_problems"] = [Problem(**elem) for elem in self.get("all_patient_problems", [])]
            except (ValidationError, TypeError):
                self["all_patient_problems"] = [{"name": str(elem)} for elem in self.get("all_patient_problems", [])]
        return self

    @model_validator(mode="after")
    def format_orders_display(self):
        if hasattr(self, "orders"):
            self.orders_display = format_orders(self.orders)
        return self


class ICD10CM(BaseModel):
    code: str = Field(..., description="ICD-10-CM code")
    is_primary: bool = Field(..., description="Whether this diagnosis is primary for its context")
    system: Literal["ICD-10-CM"] = "ICD-10-CM"
    description: str = Field(..., description="Diagnosis text")
    justification: Optional[str] = Field(None, description="Rationale for association")

    @model_validator(mode="before")
    def _validate_icd10(cls, v):
        if not isinstance(v, dict):
            return v
        code = v.get("code")
        system = v.get("system")
        if code and not RE_ICD10.match(code):
            raise ValueError(f"Invalid ICD-10-CM code: {code}")
        if system and system != "ICD-10-CM":
            raise ValueError(f"Unsupported system for diagnosis: {system}")
        return v


class LinkedDiagnosis(ICD10CM):
    pass


class Modifier(BaseModel):
    code: str = Field(..., description="CPT/HCPCS modifier (2 chars)")
    description: str = None
    justification: Optional[str] = None

    @model_validator(mode="before")
    def _validate_modifier(cls, v):
        if not isinstance(v, dict):
            return v
        c = v.get("code")
        if c and not RE_MODIFIER.match(c):
            raise ValueError(f"Invalid modifier code: {c}")
        return v

    @model_validator(mode="after")
    def fill_description(self):
        if not self.description:
            try:  # lazy import to avoid circular dependencies during module import
                from cdrc.profees_coding.modifiers.models.utils import MODIFIERS_IN_SCOPE

                self.description = MODIFIERS_IN_SCOPE.get(self.code, f"Modifier {self.code}")
            except Exception:
                self.description = f"Modifier {self.code}"
        return self


class ChargeDecision(BaseModel):
    billing_provider_id: Optional[str] = Field(None, description="Billing provider identifier")
    billing_provider_role: Optional[str] = Field(None, description="Billing provider role")
    billing_rationale: Optional[str] = Field(None, description="Rationale for the billing decision")


class Procedure(BaseModel):
    description: str = Field(..., description="CPT/HCPCS procedure code display")
    code: str = Field(..., description="CPT/HCPCS procedure code")
    system: Literal["CPT", "HCPCS"]
    justification: Optional[str] = None
    modifiers: Optional[List[Modifier]] = Field(default_factory=list)
    linked_diagnosis: Optional[List[LinkedDiagnosis]] = Field(default_factory=list)

    @model_validator(mode="before")
    def _validate_proc(cls, v):
        if not isinstance(v, dict):
            return v
        code = v.get("code")
        system = v.get("system")
        if system == "CPT":
            if not RE_CPT.match(code or ""):
                raise ValueError(f"Invalid CPT code: {code}")
        elif system == "HCPCS":
            if not RE_HCPCS.match(code or ""):
                raise ValueError(f"Invalid HCPCS code: {code}")
        else:
            raise ValueError(f"Unsupported system for procedure: {system}")
        return v


class EMTimeSpent(BaseModel):
    time_spent_minutes: int = Field(..., description="Total time spent in minutes")
    evidence: Optional[str] = Field(None, description="Narrative evidence for time spent")
    justification: Optional[str] = None

    @model_validator(mode="before")
    def _coerce_minutes(cls, v):
        if not isinstance(v, dict):
            return v
        mins = v.get("time_spent_minutes")
        if isinstance(mins, str):
            if mins.isdigit():
                v["time_spent_minutes"] = int(mins)
            else:
                # best-effort parse; fallback to -1
                try:
                    v["time_spent_minutes"] = int(re.findall(r"\d+", mins)[0])
                except Exception:
                    v["time_spent_minutes"] = -1
        return v


class EMItem(BaseModel):
    code: str = Field(..., description="E/M CPT code")
    description: Optional[str] = None
    source: Optional[Literal["MDM", "Time", "Age", "NOT_SUPPORTED"]] = Field(None, description="E/M coding basis (MDM or Time or Age).")
    system: Literal["CPT"] = "CPT"
    justification: Optional[str] = None
    modifiers: Optional[List[Modifier]] = Field(default_factory=list)
    encounter_type: Optional[EncounterTypeDetails] = None
    problem_complexity: Optional[ComplexityDetails] = None
    data_complexity: Optional[ComplexityDetails] = None
    risk_of_complications: Optional[ComplexityDetails] = None
    pc_critic_response: Optional[str] = None
    rc_critic_response: Optional[str] = None
    dc_critic_response: Optional[str] = None
    mdm_complexity: Optional[ComplexityDetails] = None
    time_spent: Optional[EMTimeSpent] = None
    linked_diagnosis: Optional[List[LinkedDiagnosis]] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, description="Confidence score for the CPT code assignment.")

    @model_validator(mode="before")
    def _val(cls, v):
        if not isinstance(v, dict):
            return v
        if v.get("system") and v["system"] != "CPT":
            raise ValueError("E/M system must be CPT")
        code = v.get("code")
        if code not in VALID_CPT_CODES:
            raise ValueError(f"Invalid CPT code: {code}")
        return v


class OutputSchema(BaseModel):
    service_date: Optional[str] = Field(None, description="Specific service date for this coding output")
    procedure: Optional[List[Procedure]] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: Optional[List[EMItem]] = Field(default_factory=list, description="List of E/M items")
    diagnosis: Optional[List[ICD10CM]] = Field(default_factory=list, description="List of diagnoses")
    charge_decision: Optional[ChargeDecision] = Field(None, description="Charge decision metadata")

    @model_validator(mode="before")
    def validate_cpt_list(self):
        cpt = self.get("cpt")
        if cpt is None:
            return self

        if isinstance(cpt, dict):
            cpt = [cpt]
            self["cpt"] = cpt

        for item in cpt:
            code = item.get("code") if isinstance(item, dict) else getattr(item, "code", None)
            if code not in VALID_CPT_CODES:
                raise ValueError(f"Invalid CPT code: {code}")

            type_ = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if type_ is None:
                match = [k for k, v in CPT_CODES_MAPPING.items() if code in v]
                if not match:
                    raise ValueError(f"Cannot infer encounter type for CPT code: {code}")
                if isinstance(item, dict):
                    item["type"] = match[0]
                else:
                    setattr(item, "type", match[0])
            elif type_ not in CPT_CODES_MAPPING:
                raise ValueError(f"Invalid encounter type: {type_}")
            elif code not in CPT_CODES_MAPPING[type_]:
                raise ValueError(f"Invalid CPT code for encounter type: {code}")

        return self


class InpatientOutput(BaseModel):
    gold: List[OutputSchema] = Field(..., description="Per-diem coding outputs for each service date")

    # comment out for preventative45 data
    # @model_validator(mode="before")
    # def validate_encounter(self):
    #     encounter_types = self.get("encounter_type")
    #     if isinstance(encounter_types, dict):  # Single CPT as dict
    #         # convert it into a list.
    #         encounter_types = [encounter_types]
    #         self["encounter_type"] = encounter_types

    #     for i, item in enumerate(encounter_types):
    #         encounter_type = item.get("encounter_type") if isinstance(item, dict) else getattr(item, "encounter_type", None)
    #         if encounter_type not in VALID_ENCOUNTER_TYPES:
    #             raise ValueError(f"Invalid Encounter Type: {encounter_type}")

    #     return self

    @model_validator(mode="before")
    def handle_null_mdm_complexity(self):
        try:
            if self.get("mdm_complexity") and self.get("mdm_complexity").get("level") is None:
                self["mdm_complexity"] = None
        except Exception:
            pass
        return self

    class Config:
        use_enum_values = True  # Automatically use enum values as strings


class UnifiedCaseBundle(BaseModel):
    """
    One record that can carry inputs and the predictions/links from all five projects.
    Use this as your warehouse / API contract for end-to-end cases.
    """

    version: str = "v1"
    example_id: str

    # raw inputs (optional to store here, or store elsewhere and reference by example_id)
    input_payload: Optional[Dict] = None

    # outputs by project (each is optional so you can store incrementally)
    em: Optional[List[EMItem]] = None
    diagnoses: Optional[List[ICD10CM]] = None
    procedures: Optional[List[Procedure]] = None


def format_soap_note(soap):
    """Dynamic formatting of soap note"""

    soap_str = ""

    for k, v in soap.items():
        k = k.capitalize().replace("_", " ")
        if isinstance(v, str):
            if v:
                soap_str += f"{k}:\n{''.join(['-'] * len(k))}\n{v}\n"
        elif isinstance(v, dict):
            soap_str += f"{k}:\n{''.join(['-'] * len(k))}\n"
            for kk, vv in v.items():
                if kk == "cc":
                    kk = "Chief Complaint"
                elif kk == "hpi":
                    kk = "History of Present Illness"
                kk = kk.capitalize().replace("_", " ")
                if isinstance(vv, list) and all(isinstance(elem, str) for elem in vv):
                    vv = "\n".join(vv)
                elif isinstance(vv, list) and all(
                    isinstance(elem, dict) for elem in vv
                ):  # to unpack diagnoses which is sometimes in assessment_and_plan section
                    try:
                        vv = "\n" + "\n\n".join(["\n".join(elem.values()) for elem in vv])
                    except Exception:
                        pass

                if isinstance(vv, list) and all(isinstance(elem, str) for elem in vv):
                    vv = "\n".join(vv)
                elif isinstance(vv, list) and all(
                    isinstance(elem, dict) for elem in vv
                ):  # to unpack diagnoses which is sometimes in assessment_and_plan section
                    try:
                        vv = "\n" + "\n\n".join(["\n".join(elem.values()) for elem in vv])
                    except Exception:
                        pass

                if vv:
                    soap_str += f"{kk}: {vv}\n"
        else:
            raise ValueError(f"Unknown type in SOAP note dictionary: {v} of type {type(v)}")
        soap_str += "\n"
        soap_str = soap_str.replace("<br/>", " ")
        soap_str = soap_str.replace("<br/>", " ")
    return soap_str


def format_orders(orders: List[Order]) -> str:
    if not orders:
        return ""

    seen_mnemonics: Set[str] = set()

    # Separate into non-medication and medication orders
    non_med_orders = [order for order in orders if order.order_type == "Non Medication Order"]
    med_orders = [order for order in orders if order.order_type == "Medication Order"]
    other_orders = [order for order in orders if order.order_type not in ["Non Medication Order", "Medication Order"]]
    other_orders = [order for order in orders if order.order_type not in ["Non Medication Order", "Medication Order"]]

    output = []

    # Add non-medication orders first
    if non_med_orders:
        output.append("Non-Medication Orders:")
        for order in non_med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status})")
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add medication orders
    if med_orders:
        if output:
            output.append("")
        output.append("Medication Orders:")
        for order in med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(
                    f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status}) (ordered as: {order.display.ordered_as})"
                )
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add other orders
    if other_orders:
        if output:
            output.append("")
        output.append("Other Orders:")
        for order in other_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(
                    f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status}) (ordered as: {order.display.ordered_as})"
                )
                seen_mnemonics.add(order.display.hna_mnemonic)

    return "\n".join(output)
