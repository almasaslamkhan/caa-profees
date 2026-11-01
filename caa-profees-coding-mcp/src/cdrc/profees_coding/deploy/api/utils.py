import asyncio
import gc
import os
import sys
import time
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Union

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator

from cdrc.common.data.schema import (
    ICD10CM,
    AnaesthesiaConcurrentStaffInfo,
    AnaesthesiaRecordInfo,
    AnaesthesiaStaffInfo,
    AWVHistory,
    EMItem,
    EncounterDocument,
    EncounterMetaData,
    FacilityInfo,
    HealthPlan,
    LabInfo,
    LinkedDiagnosis,
    Modifier,
    Procedure,
    ProviderInfo,
)
from cdrc.common.utils import StringEnum
from cdrc.profees_coding.em.data.schema import (  # Problem,
    ComplexityDetails,
    ComplexityDetailsOld,
    CptEmCode,
    CptEmCodeOld,
    EncounterTypeDetails,
    Order,
    OrderDisplay,
    OrderType,
    PatientType,
    format_orders,
)
from cdrc.profees_coding.em.models.medprompt_detailed_critics.preprocessing import MDMResult, MDMResultOld

AWV_HCPCS_CODES = {"G0402", "G0438", "G0439"}
DEPRESSION_SCREENING_BUNDLED_AWV_CODES = {"G0402", "G0438"}
MEDICARE_DEPRESSION_SCREENING_CODES = {"96127"}
BASE_OFFICE_EM_CODES = {
    "99202",
    "99203",
    "99204",
    "99205",
    "99211",
    "99212",
    "99213",
    "99214",
    "99215",
}
PREVENTIVE_EM_CODES = {
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
}


def normalize_nested_models(data: dict, fields_to_normalize: list):
    for field in fields_to_normalize:
        if isinstance(data.get(field), BaseModel):
            data[field] = data[field].model_dump()
    return data


def _item_value(item: Any, *field_names: str) -> Any:
    for field_name in field_names:
        if isinstance(item, dict):
            if field_name in item and item[field_name] is not None:
                return item[field_name]
        else:
            value = getattr(item, field_name, None)
            if value is not None:
                return value
    return None


def _item_code(item: Any) -> str:
    code = item.get("code") if isinstance(item, dict) else getattr(item, "code", None)
    return str(code or "").strip().upper()


def exclude_preventive_em_when_awv_present(
    em_items: list[dict[str, Any]],
    procedure_items: list[dict[str, Any]],
    *,
    request_id: str,
) -> list[dict[str, Any]]:
    if not any(_item_code(item) in AWV_HCPCS_CODES for item in procedure_items):
        return em_items

    filtered_em_items = [item for item in em_items if _item_code(item) not in PREVENTIVE_EM_CODES]
    removed_count = len(em_items) - len(filtered_em_items)
    if removed_count:
        logger.info(
            "{} Excluded {} preventive E/M code(s) because an AWV HCPCS code is present.",
            request_id,
            removed_count,
        )
    return filtered_em_items


def exclude_depression_screening_when_awv_present(
    procedure_items: list[dict[str, Any]],
    *,
    request_id: str,
) -> list[dict[str, Any]]:
    if not any(_item_code(item) in DEPRESSION_SCREENING_BUNDLED_AWV_CODES for item in procedure_items):
        return procedure_items

    filtered_procedure_items = [item for item in procedure_items if _item_code(item) not in MEDICARE_DEPRESSION_SCREENING_CODES]
    removed_count = len(procedure_items) - len(filtered_procedure_items)
    if removed_count:
        logger.info(
            "{} Excluded {} depression screening code(s) because G0402 or G0438 is present.",
            request_id,
            removed_count,
        )
    return filtered_procedure_items


def exclude_g2211_when_base_em_missing(
    em_items: list[dict[str, Any]],
    procedure_items: list[dict[str, Any]],
    *,
    request_id: str,
) -> list[dict[str, Any]]:
    if not any(_item_code(item) == "G2211" for item in procedure_items):
        return procedure_items

    base_em_present = any(_item_code(item) in BASE_OFFICE_EM_CODES for item in [*em_items, *procedure_items])
    if base_em_present:
        return procedure_items

    filtered_procedure_items = [item for item in procedure_items if _item_code(item) != "G2211"]
    removed_count = len(procedure_items) - len(filtered_procedure_items)
    if removed_count:
        logger.info(
            "{} Excluded {} G2211 code(s) because no base office E/M code is present.",
            request_id,
            removed_count,
        )
    return filtered_procedure_items


def _normalize_service_date(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if len(raw) >= 10:
        candidate = raw[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            pass

    sanitized = raw.replace("Z", "")
    if "+" in sanitized:
        sanitized = sanitized.split("+", 1)[0]
    if " " in sanitized and "T" not in sanitized:
        sanitized = sanitized.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(sanitized).date().isoformat()
    except ValueError:
        return ""


def _document_service_date(document: Any) -> str:
    return _normalize_service_date(_item_value(document, "service_date", "time_created"))


def _is_inpatient_encounter(encounter_metadata: Optional[EncounterMetaData]) -> bool:
    if encounter_metadata is None:
        return False
    encounter_type = getattr(encounter_metadata, "type", None)
    if not encounter_type:
        return False
    return "inpatient" in encounter_type.strip().lower()


def filter_orders_by_inpatient_window(orders: Sequence[Any], encounter_metadata: Optional[EncounterMetaData]) -> list[Any]:
    if encounter_metadata is None:
        return list(orders or [])
    admit_date = _normalize_service_date(getattr(encounter_metadata, "admit_time", None))
    discharge_date = _normalize_service_date(getattr(encounter_metadata, "discharge_time", None))
    if admit_date == "" and discharge_date == "":
        return list(orders or [])

    def _in_window(order: Any) -> bool:
        timestamp = _item_value(order, "time_created", "time_last_updated")
        date_value = _normalize_service_date(timestamp)
        if date_value == "":
            return True
        if admit_date and date_value < admit_date:
            return False
        if discharge_date and date_value > discharge_date:
            return False
        return True

    return [order for order in orders or [] if _in_window(order)]


def filter_documents_by_inpatient_window(documents: Sequence[Any], encounter_metadata: Optional[EncounterMetaData]) -> list[Any]:
    if encounter_metadata is None:
        return list(documents or [])
    admit_date = _normalize_service_date(getattr(encounter_metadata, "admit_time", None))
    discharge_date = _normalize_service_date(getattr(encounter_metadata, "discharge_time", None))
    if admit_date == "" and discharge_date == "":
        return list(documents or [])

    def _in_window(document: Any) -> bool:
        timestamp = _item_value(document, "service_date", "time_created", "signature_time")
        date_value = _normalize_service_date(timestamp)
        if date_value == "":
            return True
        if admit_date and date_value < admit_date:
            return False
        if discharge_date and date_value > discharge_date:
            return False
        return True

    return [document for document in documents or [] if _in_window(document)]


def select_documents_for_latest_service_date(documents: Sequence[Any]) -> list[Any]:
    document_list = [document for document in (documents or []) if document is not None]
    if not document_list:
        return []

    last_date = max((_document_service_date(document) for document in document_list), default="")
    if last_date == "":
        return document_list
    return [document for document in document_list if _document_service_date(document) == last_date]


def selected_inpatient_service_date(documents: Sequence[Any]) -> str:
    selected_documents = select_documents_for_latest_service_date(documents)
    if not selected_documents:
        return ""
    return _document_service_date(selected_documents[0])


def select_orders_for_service_date(orders: Sequence[Any], service_date: str) -> list[Any]:
    if not orders:
        return []
    if not service_date:
        return list(orders)

    selected_orders = []
    for order in orders:
        order_date = _normalize_service_date(_item_value(order, "time_created", "time_last_updated"))
        if order_date == service_date:
            selected_orders.append(order)
    return selected_orders


def documents_to_note(documents: Sequence[Any]) -> str:
    if not documents:
        return ""

    lines = []
    for document in select_documents_for_latest_service_date(documents):
        service_date = _item_value(document, "service_date")
        prefix = f"[{service_date}] " if service_date else ""
        document_type = _item_value(document, "document_type", "type") or "Document"
        text = _item_value(document, "text") or ""
        line = f"{prefix}{document_type}: {text}".strip()
        if line:
            lines.append(line)

    return "\n\n".join(lines)


class InFlight:
    count = 0
    last_ts = time.time()
    lock = asyncio.Lock()


def register_inflight_middleware(app):
    @app.middleware("http")
    async def _inflight_mw(request, call_next):
        async with InFlight.lock:
            InFlight.count += 1
            InFlight.last_ts = time.time()
        try:
            return await call_next(request)
        finally:
            async with InFlight.lock:
                InFlight.count -= 1
                InFlight.last_ts = time.time()


# def setup_memory_management(app, logger, emit: Optional[Callable] = None, config: Optional[GCConfig] = None):
#     """
#     Call this once from app.py after creating the FastAPI app and logger.
#     Registers middleware and schedules background tasks on startup.
#     """
#     cfg = config or GCConfig()

#     register_inflight_middleware(app)

#     @app.on_event("startup")
#     async def _start_background_tasks():
#         asyncio.create_task(periodic_memory_emitter(logger, emit, cfg.MEM_EMIT_INTERVAL_SEC))
#         asyncio.create_task(soft_idle_gc_loop(cfg, logger, emit))
#         asyncio.create_task(hard_gc_loop(cfg, logger, emit))


# ------------------ Define API request schemas ------------------
class Period(BaseModel):
    start: Optional[str] = Field(None, description="Start time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ) or 'Unknown'.")
    end: Optional[str] = Field(None, description="End time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ) or 'Unknown'.")

    @field_validator("start", "end")
    @classmethod
    def validate_iso8601(cls, value):
        if value is None or value.strip().lower() == "unknown":
            return "Unknown"  # Allow "Unknown" as a valid value

        try:
            datetime.fromisoformat(value.replace("Z", "")).date()
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 format for date: {value}")
        return value

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, end, info: ValidationInfo):
        """Ensures 'end' is after 'start' if both are valid ISO 8601 dates."""
        start = info.data.get("start")  # Get start value

        if not start or start.lower() == "unknown" or not end or end.lower() == "unknown":
            return end  # Skip validation if either is 'Unknown'

        try:
            start_dt = datetime.fromisoformat(start.replace("Z", ""))
            end_dt = datetime.fromisoformat(end.replace("Z", ""))
            if end_dt < start_dt:
                raise ValueError(f"End time {end} must be after start time {start}.")
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 format for start or end: {start}, {end}. Error: {e}")

        return end


# Enum for API Input Patient Type
class InputPatientType(StringEnum):
    NEW = "New"
    ESTABLISHED = "Established"
    UNKNOWN = "Unknown"


# Define ProposedAction Orders
class PAOrders(BaseModel):
    verifiedOrder: bool = Field(..., description="The type of order source. True: actual order; False: PA order.")
    order_type: str = Field(..., description="The type of order being placed.")
    status: Optional[str] = Field(
        "Draft Order", description="The status of the input order."
    )  # ToDo: should review the status after 25.03 as it is not available and would always be default now.
    catalog_type: str = Field(..., description="Catalog type for the order.")
    display_line: str = Field(..., description="Display line text for the order.")
    mnemonic: str = Field(..., description="Mnemonic for quick reference of the order.")


# Define ProposedAction Orders
class Orders(BaseModel):
    order_type: str = Field(..., description="The type of order being placed.")
    status: Optional[str] = Field(
        "Draft Order", description="The status of the input order."
    )  # ToDo: should review the status after 25.03 as it is not available and would always be default now.
    catalog_type: str = Field(..., description="Catalog type for the order.")
    display_line: str = Field(..., description="Display line text for the order.")
    mnemonic: str = Field(..., description="Mnemonic for quick reference of the order.")
    time_created: Optional[str] = Field(None, description="Order creation timestamp in ISO 8601 format.")
    time_last_updated: Optional[str] = Field(None, description="Order update timestamp in ISO 8601 format.")

    @model_validator(mode="before")
    @classmethod
    def normalize_order_payload(cls, values: Any):
        if not isinstance(values, dict):
            return values

        if "display" in values and ("display_line" not in values or "mnemonic" not in values):
            display = values.get("display") or {}
            if isinstance(display, dict):
                values.setdefault("display_line", display.get("ordered_as") or display.get("clinical_display_line") or "")
                values.setdefault("mnemonic", display.get("hna_mnemonic") or "")
        return values


# Define the Subjective section
class Subjective(BaseModel):
    cc: Optional[str] = Field("", description="Chief complaint of the patient.")
    hpi: Optional[str] = Field("", description="History of present illness.")
    pastMedicalHistory: Optional[str] = Field("", description="Past medical history.")
    surgicalHistory: Optional[str] = Field("", description="Surgical history.")
    familyHistory: Optional[str] = Field("", description="Family medical history.")
    socialHistory: Optional[str] = Field("", description="Social history including lifestyle factors.")
    immunizationHistory: Optional[str] = Field("", description="Immunization history.")
    listOfAllergies: Optional[str] = Field("", description="List of known allergies.")
    listOfCurrentRx: Optional[str] = Field("", description="List of current medications.")

    @model_validator(mode="after")
    def at_least_one_field_not_empty(cls, model):
        # At least one field must be non-empty (after stripping whitespace)
        if not any(getattr(model, name).strip() for name in model.model_fields):
            logger.warning("Subjective section has all empty fields.")
            return model
            raise ValueError("At least one field in the Subjective section must be non-empty.")
        return model


# Define the Objective section
class Objective(BaseModel):
    vitalSigns: Optional[str] = Field("", description="Patient's vital signs.")
    laboratoryResults: Optional[str] = Field("", description="Lab test results.")
    physicalExamination: Optional[str] = Field("", description="Findings from physical examination.")

    @model_validator(mode="after")
    def at_least_one_field_not_empty(cls, model):
        # model.model_fields is a dict of field names -> field info
        if not any(getattr(model, name).strip() for name in model.model_fields):
            logger.warning("Objective section has all empty fields.")
        return model


# Define the Diagnosis section
class Diagnosis(BaseModel):
    id: Optional[str] = Field("", description="ID of the diagnosis.")
    name: Optional[str] = Field("", description="Name of the diagnosis.")
    text: Optional[str] = Field("", description="Text of the diagnosis.")

    def __str__(self):
        if self.name and self.text:
            return f"{self.name}: {self.text}"
        elif self.name:
            return self.name
        elif self.text:
            return self.text
        return ""


# Define the Assessment and Plan section
class AssessmentAndPlan(BaseModel):
    summary: Optional[str] = Field("", description="Summary of the assessment and plan.")
    text: Optional[str] = Field("", description="Detailed assessment and plan notes.")
    diagnoses: Optional[List[Diagnosis]] = Field([], description="List of diagnoses.")

    @model_validator(mode="after")
    def at_least_one_field_not_empty(cls, model):
        for name, value in model.model_dump().items():
            if isinstance(value, str) and value.strip():
                return model
            if isinstance(value, list) and len(value) > 0:
                return model
            if value not in ("", None, [], {}):  # general fallback
                return model
        logger.warning("AssessmentAndPlan section has all empty fields.")
        return model
        raise ValueError("At least one field in the AssessmentAndPlan section must be non-empty.")


# Define the draft SOAP Text structure
class AmbientData(BaseModel):
    subjective: Subjective = Field(default_factory=Subjective, description="Subjective section of the note.")
    reviewOfSystems: Optional[str] = Field("", description="Review of systems details.")
    objective: Optional[Objective] = Field(Objective(), description="Objective section of the note.")
    assessmentAndPlan: AssessmentAndPlan = Field(default_factory=AssessmentAndPlan, description="Assessment and plan details.")
    diagnosticsStudies: Optional[str] = Field("", description="Details of diagnostic studies performed.")


#  Define the Problem structure as model function Input
class Problem(BaseModel):
    description: Optional[str] = None
    code: Optional[str] = None
    name: Optional[str] = None
    status: Optional[Literal["future", "active", "inactive"]] = Field(None, description="Current status of the problem")
    priority: Optional[Literal["1", "2"]] = Field(None, description="Priority level")

    @model_validator(mode="before")
    @classmethod
    def normalize_problem(cls, values: Any):
        if not isinstance(values, dict):
            return values

        name = values.get("name") or values.get("description") or ""
        code = values.get("code") or ""

        if not code and name:
            if "(" in name and ")" in name:
                extracted = name.split("(")[-1].split(")")[0].strip()
                if extracted:
                    code = extracted
            values["name"] = name

        if not values.get("description") and name:
            values["description"] = name
        if code:
            values["code"] = code
        return values


class DeliveryVehicle(str, Enum):
    SCRATCHPAD = "Scratchpad"
    CAAM_V1 = "CAAM V1"
    CAAM_V2 = "CAAM V2"
    EHRM = "EHRM"
    CAAD = "CAAD"


# Define CPT EM Response schema/ Old Response Schema
class CptEmResponse(BaseModel):
    cpt: List[CptEmCodeOld] = Field(..., description="CPT Evaluation and Management code.")
    encounter_type: List[EncounterTypeDetails] = Field(..., description="Type of encounter.")
    mdm_results: Optional[MDMResultOld] = Field(None, description="Medical Decision Making (MDM) results.")
    problem_complexity: Optional[ComplexityDetailsOld] = Field(None, description="Complexity of the patient's problems.")
    data_complexity: Optional[ComplexityDetailsOld] = Field(None, description="Complexity of data reviewed.")
    risk_of_complications: Optional[ComplexityDetailsOld] = Field(None, description="Risk level of complications.")
    pc_critic_response: Optional[str] = Field(None, description="Problem complexity RCI response.")
    rc_critic_response: Optional[str] = Field(None, description="Risk complexity RCI response.")
    dc_critic_response: Optional[str] = Field(None, description="Data complexity RCI response.")
    # justification: Optional[str] = Field(None, description="Justification for the CPT code assigned.")
    # critic_response: Optional[str] = Field(None, description="RCI response for evaluation.")
    # confidence: Optional[float] = Field(None, description="Confidence score for the CPT code assignment.")


class CareSetting(StringEnum):
    AMBULATORY = "ambulatory"
    ED = "ed"
    INPATIENT = "inpatient"


def derive_care_setting(encounter_metadata: Optional[EncounterMetaData]) -> CareSetting:
    logger.info("Deriving care setting from encounter metadata: {}", encounter_metadata)
    if not encounter_metadata or not encounter_metadata.type:
        return CareSetting.AMBULATORY

    encounter_type = encounter_metadata.type.strip().lower()
    if encounter_type in {"ambulatory", "outpatient"}:
        return CareSetting.AMBULATORY
    if encounter_type in {"ed", "emergency", "emergency department"}:
        return CareSetting.ED
    if encounter_type in {"inpatient", "hospital"}:
        return CareSetting.INPATIENT
    return CareSetting.AMBULATORY


# Define Profees Response schema
# ----------------- Top-level ProfeesResponse -----------------
class ProfeesResponse(BaseModel):
    procedure: List[Procedure] = Field(..., description="List of procedure codes")
    em: List[EMItem] = Field(..., description="List of E/M codes")
    diagnosis: List[ICD10CM] = Field(..., description="List of diagnoses")


class InputData_Old(BaseModel):
    ambient_note: AmbientData = Field(..., description="Structured ambient note data.")
    orders: List[PAOrders] = Field(..., description="List of orders related to the encounter.")
    patient_type: InputPatientType = Field(..., description="Type of patient: 'New', 'Established', or 'Unknown'.")
    patient_birthdate: str = Field(..., description="Patient's birthdate in ISO 8601 format with UTC (YYYY-MM-DDTHH:MM:SS.sssZ) or 'Unknown'.")
    current_encounter_period: Period = Field(..., description="Start and end period of the current encounter.")

    @field_validator("patient_birthdate")
    @classmethod
    def validate_patient_birthdate(cls, value):
        """Ensures patient birthdate is a valid ISO 8601 date and not in the future."""
        if value is None or value.strip().lower() == "unknown":
            return "Unknown"  # Allow "Unknown" as a valid value

        try:
            # Convert to datetime, ignore time, and extract only the date part
            birthdate = datetime.fromisoformat(value.replace("Z", "")).date()
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 format for patient_birthdate: {value}")

        # Ensure birthdate is not in the future
        if birthdate > date.today():
            raise ValueError(f"Patient birthdate {value} cannot be in the future.")

        return value  # Return the original string


# Define request schemas
class InputData(BaseModel):
    encounter_id: str = Field("sdcfsdfse", description="Encounter id.")
    encounter_metadata: Optional[EncounterMetaData] = Field(None, description="Encounter metadata from upstream sources.")
    documents: Optional[List[EncounterDocument]] = Field(None, description="Encounter documents for inpatient/ED cases.")
    ambient_note: Optional[AmbientData] = Field(None, description="Structured ambient note data.")
    soap_text: Optional[str] = Field(None, description="SOAP note in plain text format.")
    appointment_type: Optional[str] = Field(None, description="Appointment type associated with the encounter, when available.")
    orders: List[Orders] = Field(..., description="List of orders related to the encounter.")
    appointment_type: Optional[str] = Field(None, description="Type of appointment.")
    patient_type: InputPatientType = Field(..., description="Type of patient: 'New', 'Established', or 'Unknown'.")
    patient_birthdate: str = Field(..., description="Patient's birthdate in ISO 8601 format with UTC (YYYY-MM-DDTHH:MM:SS.sssZ) or 'Unknown'.")
    patient_sex: str = Field(..., description="Patient's sex.")
    current_encounter_period: Period = Field(..., description="Start and end period of the current encounter.")
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

    problem_list: Optional[List[Problem]] = Field(None, description="List of all patient problems.")
    soap_text_previous: Optional[str] = Field(None, description="Previous SOAP note in plain text format.")
    orders_previous: Optional[List[Orders]] = Field(None, description="List of previous orders related to the encounter.")
    response_previous: Optional[ProfeesResponse] = Field(None, description="Previous Profees model response for the encounter.")

    # Ensure at least one of ambient_note or soap is provided
    @model_validator(mode="before")
    def normalize_payload(self):
        if not isinstance(self, dict):
            return self

        encounter_metadata = self.get("encounter_metadata")

        if not self.get("patient_birthdate"):
            self["patient_birthdate"] = "Unknown"

        if not self.get("current_encounter_period") and isinstance(encounter_metadata, dict):
            start = encounter_metadata.get("admit_time")
            end = encounter_metadata.get("discharge_time")
            service_dates = encounter_metadata.get("service_dates") or []

            def _date_to_iso(date_value: str, end_of_day: bool = False) -> str:
                if not date_value:
                    return "Unknown"
                if "T" in date_value:
                    return date_value
                return f"{date_value}T23:59:59Z" if end_of_day else f"{date_value}T00:00:00Z"

            if not start and service_dates:
                start = _date_to_iso(service_dates[0])
            if not end and service_dates:
                end = _date_to_iso(service_dates[-1], end_of_day=True)

            self["current_encounter_period"] = {"start": start or "Unknown", "end": end or start or "Unknown"}

        soap_payload = self.get("soap")
        if soap_payload and not self.get("ambient_note") and not self.get("soap_text"):
            subjective = soap_payload.get("subjective") or {}
            review_of_systems = soap_payload.get("review_of_systems") or ""
            objective = soap_payload.get("objective") or {}
            assessment_and_plan = soap_payload.get("assessment_and_plan") or {}

            ambient_note = {
                "subjective": {
                    "cc": subjective.get("cc", ""),
                    "hpi": subjective.get("hpi", ""),
                    "pastMedicalHistory": subjective.get("past_medical_history", subjective.get("pastMedicalHistory", "")),
                    "surgicalHistory": subjective.get("surgical_history", subjective.get("surgicalHistory", "")),
                    "familyHistory": subjective.get("family_history", subjective.get("familyHistory", "")),
                    "socialHistory": subjective.get("social_history", subjective.get("socialHistory", "")),
                    "immunizationHistory": subjective.get("immunization_history", subjective.get("immunizationHistory", "")),
                    "listOfAllergies": subjective.get("list_of_allergies", subjective.get("listOfAllergies", "")),
                    "listOfCurrentRx": subjective.get("list_of_current_rx", subjective.get("listOfCurrentRx", "")),
                },
                "reviewOfSystems": review_of_systems,
                "objective": {
                    "vitalSigns": objective.get("vital_signs", objective.get("vitalSigns", "")),
                    "laboratoryResults": objective.get("laboratory_results", objective.get("laboratoryResults", "")),
                    "physicalExamination": objective.get("physical_examination", objective.get("physicalExamination", "")),
                },
            }

            ap_text = assessment_and_plan.get("text") or ""
            if isinstance(ap_text, dict):
                ambient_note["assessmentAndPlan"] = {
                    "summary": ap_text.get("summary", ""),
                    "text": ap_text.get("text", ""),
                    "diagnoses": ap_text.get("diagnoses", []),
                }
            else:
                ambient_note["assessmentAndPlan"] = {
                    "summary": assessment_and_plan.get("summary", ""),
                    "text": ap_text or "",
                    "diagnoses": assessment_and_plan.get("diagnoses", []),
                }

            self["ambient_note"] = ambient_note

        if self.get("documents") and not self.get("soap_text") and not self.get("ambient_note"):
            lines = []
            for doc in self.get("documents") or []:
                if not isinstance(doc, dict):
                    continue
                prefix = ""
                if doc.get("service_date"):
                    prefix = f"[{doc.get('service_date')}] "
                doc_type = doc.get("type") or "Document"
                text = doc.get("text") or ""
                lines.append(f"{prefix}{doc_type}: {text}".strip())
            self["soap_text"] = "\n\n".join([line for line in lines if line])

        # Some upstream evaluators provide only raw "text". Reuse it as SOAP text
        # so validation and downstream API execution can proceed consistently.
        if not self.get("soap_text") and not self.get("ambient_note"):
            raw_text = self.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                self["soap_text"] = raw_text

        encounter_metadata_obj = self.get("encounter_metadata")
        encounter_type = None
        if isinstance(encounter_metadata_obj, dict):
            encounter_type = encounter_metadata_obj.get("type")
        elif encounter_metadata_obj is not None:
            encounter_type = getattr(encounter_metadata_obj, "type", None)

        encounter_type_str = encounter_type.strip().lower() if isinstance(encounter_type, str) else ""
        if encounter_type_str == "inpatient":
            if not self.get("documents") and not self.get("soap_text") and not self.get("ambient_note"):
                raise ValueError("Inpatient encounters must include documents or note text")
        else:
            if not self.get("ambient_note") and not self.get("soap_text"):
                raise ValueError("Either SOAP note Dictionary or text must be provided")

        return self

    @field_validator("patient_birthdate")
    @classmethod
    def validate_patient_birthdate(cls, value):
        """Ensures patient birthdate is a valid ISO 8601 date and not in the future."""
        if value is None or value.strip().lower() == "unknown":
            return "Unknown"  # Allow "Unknown" as a valid value

        try:
            # Convert to datetime, ignore time, and extract only the date part
            birthdate = datetime.fromisoformat(value.replace("Z", "")).date()
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 format for patient_birthdate: {value}")

        # Ensure birthdate is not in the future
        if birthdate > date.today():
            raise ValueError(f"Patient birthdate {value} cannot be in the future.")

        return value  # Return the original string


def health_plan_payer_names(data: InputData) -> list[str]:
    payer_names: list[str] = []
    for health_plan in data.health_plan or []:
        payer_name = health_plan.get("health_plan") if isinstance(health_plan, dict) else getattr(health_plan, "health_plan", None)
        if payer_name:
            payer_names.append(str(payer_name).strip())
    return [payer_name for payer_name in payer_names if payer_name]


def build_hcpcs_input(data: InputData, cda_note: str, patient_age: int, orders_display: str) -> dict[str, Any]:
    return {
        "example_id": data.encounter_id,
        "text": cda_note,
        "patient_age": patient_age,
        "patient_sex": data.patient_sex,
        "orders_display": orders_display,
        "encounter_metadata": data.encounter_metadata,
        "health_plan": data.health_plan,
        "awv_history": data.awv_history,
        "encounter_date": data.current_encounter_period.start,
    }


def build_icd_associate_input(
    data: InputData,
    cda_note: str,
    patient_age: int,
    orders_display: str,
    em_items: Sequence[Any],
    procedure_items: Sequence[Any],
) -> dict[str, Any]:
    return {
        "example_id": data.encounter_id,
        "text": cda_note,
        "patient_age": patient_age,
        "patient_sex": data.patient_sex,
        "orders": data.orders,
        "orders_display": orders_display,
        "all_patient_problems": data.problem_list,
        "em": em_items,
        "procedure": procedure_items,
    }


def build_modifiers_input(data: InputData, icd_associate_input: dict[str, Any], diagnosis_items: Sequence[Any]) -> dict[str, Any]:
    return {
        **icd_associate_input,
        "diagnosis": diagnosis_items,
        "payer_name": health_plan_payer_names(data),
        "encounter_metadata": data.encounter_metadata,
        "health_plan": data.health_plan,
        "facility_info": data.facility_info,
        "provider_info": data.provider_info,
        "lab_info": data.lab_info,
        "anaesthesia_staff_info": data.anaesthesia_staff_info,
        "anaesthesia_concurrent_staff_info": data.anaesthesia_concurrent_staff_info,
        "anaesthesia_record_info": data.anaesthesia_record_info,
    }


# ------------------ Define Model Input and Response Schemas ------------------
# Define model input schema
class ModelData(BaseModel):
    text: str = Field(..., description="Generated textual summary of the patient encounter.")
    patient_type: PatientType = Field(..., description="Type of patient (e.g., outpatient, inpatient).")
    patient_age: int = Field(..., description="Age of the patient in years.")
    # all_patient_problems: List[Problem] = Field(..., description="List of all patient problems.")
    orders_display: str = Field(..., description="Formatted display of orders.")
    medications: List[str] = Field(..., description="List of medications prescribed.")
    reason_for_visit: str = Field(..., description="Primary reason for the visit.")
    appointment_type: Optional[str] = Field(None, description="Appointment type associated with the encounter, when available.")
    service_date: Optional[str] = Field(None, description="Encounter service date in ISO format when available.")
    soap_text_previous: Optional[str] = Field(None, description="Previous SOAP note in plain text format.")
    orders_display_previous: Optional[str] = Field(None, description="List of previous orders related to the encounter.")
    response_previous: Optional[ProfeesResponse] = Field(None, description="Previous Profees model response for the encounter.")


# ------------------ Define API Functions ------------------
# Helper function to convert patient type from Api input to model input
def predict_patient_type(patient_type: InputPatientType) -> PatientType:
    """
    Predict the patient type based on the workflow logic.

    If the input patient type is 'Unknown', it defaults to 'New'.
    Otherwise, it returns the same patient type.
    """
    return PatientType.ESTABLISHED if patient_type == InputPatientType.UNKNOWN else PatientType(patient_type)


def calculate_patient_age(birthdate: str, encounter_start: str) -> int:
    try:
        if birthdate.strip().lower() == "unknown" or encounter_start.strip().lower() == "unknown":
            # If invalid input data, set age as -1 to trigger "INVALID_AGE" response for PREVENTIVE_MEDICINE.
            # Age has no impact to other encounter type.
            logger.warning(
                f"Invalid birthdate {birthdate.strip().lower()} or encounter start date {encounter_start.strip().lower()}, returning age as -1."
            )
            return -1
        # Replace 'Z' with '+00:00' to make it compatible with fromisoformat
        encounter_start = encounter_start.replace("Z", "+00:00")
        birthdate = birthdate.replace("Z", "+00:00")

        # Parse the dates
        birth_date = datetime.fromisoformat(birthdate)
        encounter_date = datetime.fromisoformat(encounter_start)

        # Calculate the age
        age = encounter_date.year - birth_date.year - ((encounter_date.month, encounter_date.day) < (birth_date.month, birth_date.day))
        return age
    except ValueError as e:
        raise ValueError(f"Error parsing dates: {e}")


def convert_orders_to_display(orders: List[Orders]) -> str:
    converted_orders = []

    for input_order in orders:
        # Determine Order Type
        normalized_order_type = (input_order.order_type or "").strip().lower()
        if normalized_order_type in {
            OrderType.MEDICATION.value.lower(),
            OrderType.NON_MEDICATION.value.lower(),
        }:
            order_type = OrderType.MEDICATION.value if normalized_order_type == OrderType.MEDICATION.value.lower() else OrderType.NON_MEDICATION.value
        else:
            order_type = OrderType.MEDICATION.value if input_order.catalog_type.lower() == "pharmacy" else OrderType.NON_MEDICATION.value

        # Create OrderDisplay Object
        order_display = OrderDisplay(
            hna_mnemonic=input_order.mnemonic,
            ordered_as=input_order.display_line,  # Assuming `display_line` is equivalent to `ordered_as`
            clinical_display_line=input_order.display_line,
        )

        # Create Full Order Object
        order = Order(
            order_type=order_type,
            status=input_order.status,
            catalog_type=input_order.catalog_type,
            time_created=input_order.time_created,
            time_last_updated=input_order.time_last_updated,
            display=order_display,
        )
        # ToDo: should review the status after 25.03 as it is not available now.

        converted_orders.append(order)

    return format_orders(converted_orders)


def convert_pa_orders_to_display(pa_orders: List[PAOrders]) -> str:
    converted_orders = []
    skip_types = {"e & m", "charges (evaluation and management)"}

    for pa_order in pa_orders:
        if not pa_order.verifiedOrder:
            continue

        catalog_type = (pa_order.catalog_type or "").strip().lower()
        if catalog_type in skip_types:
            continue

        # Determine Order Type
        order_type = "Medication Order" if pa_order.catalog_type.lower() == "pharmacy" else "Non Medication Order"

        # Create OrderDisplay Object
        order_display = OrderDisplay(
            hna_mnemonic=pa_order.mnemonic,
            ordered_as=pa_order.display_line,  # Assuming `display_line` is equivalent to `ordered_as`
            clinical_display_line=pa_order.display_line,
        )

        # Create Full Order Object
        order = Order(order_type=order_type, status=pa_order.status, catalog_type=pa_order.catalog_type, display=order_display)
        # ToDo: should review the status after 25.03 as it is not available now.

        converted_orders.append(order)

    return format_orders(converted_orders)


def CDA_note(note: AmbientData) -> str:
    try:
        subjective = note.subjective
        objective = note.objective
        assessment_and_plan = note.assessmentAndPlan
        review_of_systems = note.reviewOfSystems

        def add_section(label: str, content: Union[str, list, None]) -> str:
            """Formats a section with a label if content is present."""
            if isinstance(content, list):
                content = "\n".join(str(item).strip() for item in content if item)
            elif isinstance(content, str):
                content = content.strip()
            else:
                content = ""
            return f"{label}:\n{content}\n" if content else ""

        # Subjective Section
        subjective_section = "\n".join(
            filter(
                lambda x: x.strip(),
                [
                    add_section("Chief Complaint", subjective.cc),
                    add_section("History of Present Illness", subjective.hpi),
                    add_section("Past Medical History", subjective.pastMedicalHistory),
                    add_section("Surgical History", subjective.surgicalHistory),
                    add_section("Family History", subjective.familyHistory),
                    add_section("Social History", subjective.socialHistory),
                    add_section("Immunization History", subjective.immunizationHistory),
                    add_section("List of Allergies", subjective.listOfAllergies),
                    add_section("List of Current Medications", subjective.listOfCurrentRx),
                ],
            )
        ).strip()

        # Objective Section
        objective_section = "\n".join(
            filter(
                lambda x: x.strip(),
                [
                    add_section("Vital Signs", objective.vitalSigns),
                    add_section("Laboratory Results", objective.laboratoryResults),
                    add_section("Physical Examination", objective.physicalExamination),
                    add_section("Diagnostics Studies", note.diagnosticsStudies),
                ],
            )
        ).strip()

        # Assessment and Plan Section
        assessment_and_plan_section = "\n".join(
            filter(
                lambda x: x.strip(),
                [
                    add_section("Summary", assessment_and_plan.summary),
                    add_section("Text", assessment_and_plan.text),
                    add_section("Diagnoses", assessment_and_plan.diagnoses),
                ],
            )
        ).strip()

        # Review of Systems Section
        review_of_systems_section = review_of_systems.strip() if isinstance(review_of_systems, str) else ""

        # Combine all sections into the SOAP note
        SOAP = "\n\n".join(
            filter(
                lambda x: x.strip(),
                [
                    add_section("Subjective", subjective_section),
                    add_section("Review of Systems", review_of_systems_section),
                    add_section("Objective", objective_section),
                    add_section("Assessment and Plan", assessment_and_plan_section),
                ],
            )
        )

        return SOAP.strip()

    except Exception as e:
        raise ValueError(f"Error processing CDA Note: {e}")


def flatten_multiline(d: dict) -> str:
    lines = []
    for k, v in d.items():
        lines.append(f"{k}:")  # parent key on its own line

        # indent value on the next line(s)
        if isinstance(v, str):
            indented_value = "\n".join("  " + line for line in v.split("\n"))
            lines.append(indented_value)
        else:
            lines.append(f"  {v}")

        lines.append("")  # blank line between sections

    return "\n".join(lines)


def sanitize_for_prod(data):
    """
    PROD-safe sanitizer:
      - Recursively remove all 'justification' fields
      - Clamp patient_age >= 92 to exactly 92
      - Remove 'text' fields wherever they appear
    """
    # Handle dict
    if isinstance(data, dict):
        new = {}
        for k, v in data.items():

            # Remove ALL justification fields recursively
            if k == "justification":
                continue

            # Remove SOAP text fields
            if k == "text":
                continue

            # Clamp age at 92
            if k == "patient_age":
                try:
                    age_val = int(v)
                except (ValueError, TypeError):
                    new[k] = v
                else:
                    new[k] = 92 if age_val >= 92 else age_val
                continue

            # Recurse into child structures
            new[k] = sanitize_for_prod(v)

        return new

    # Handle list
    if isinstance(data, list):
        return [sanitize_for_prod(item) for item in data]

    # Primitives
    return data


# ------------------ Define Metric Names ------------------
class MetricNames(StringEnum):
    ORDERS_RECEIVED = "orders_received"
    API_REQUEST_RECEIVED = "api_request_received"
    API_REQUEST_SUCCESS = "api_request_success"
    API_REQUEST_SERVER_ERROR = "api_request_server_error"
    API_REQUEST_BAD_REQUEST_ERROR = "api_request_bad_request_error"
    API_REQUEST_ERROR_MESSAGE = "api_request_error_message"
    API_REQUEST_TIME = "api_request_time"
    # ------------------ Define LLM Metric Names ------------------
    LLM_TOTAL_TOKENS = "llm_total_tokens"
    LLM_PROMPT_TOKENS = "llm_prompt_tokens"
    LLM_COMPLETION_TOKENS = "llm_completion_tokens"
    LLM_PROMPT_TOKENS_CACHED = "llm_prompt_tokens_cached"
    LLM_REASONING_TOKENS = "llm_reasoning_tokens"
    LLM_TOTAL_COST = "llm_total_cost"
    LLM_SUCCESSFUL_REQUESTS = "llm_successful_requests"
    LLM_REQUEST_TIME = "llm_request_time"
    LLM_RATE_LIMIT_DELAYS = "llm_rate_limit_delays"
    # ------------------ Define Memory Metric Names ------------------
    API_GC_SOFT_TRIGGERED = "api_gc_soft_triggered"
    API_GC_HARD_TRIGGERED = "api_gc_hard_triggered"
