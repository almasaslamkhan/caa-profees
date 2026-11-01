from itertools import chain
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import (
    CPT_CODES_MAPPING,
    AgeCategories,
    ComplexityDetails,
    ComplexityLevel,
    EMItem,
    EMTimeSpent,
    EncounterType,
    EncounterTypeDetails,
    EncounterTypesDetails,
    InputSchema,
    Order,
    OrderDisplay,
    OrderType,
    PatientType,
    Problem,
    ProblemComplexityType,
)
from cdrc.common.utils import StringEnum


class ComplexityDetailsOld(BaseModel):
    level: ComplexityLevel = Field(..., description="The level of complexity")
    justification: Optional[str] = Field(None, description="The justification for the level of complexity")
    confidence: Optional[float] = Field(-1, description="The confidence in the level of complexity")


class ComplexityDetails(ComplexityDetails):
    pass


class EncounterTypeDetails(EncounterTypeDetails):
    pass


class EncounterTypesDetails(EncounterTypesDetails):
    pass


class EncounterTypeAmbulatoryDetails(BaseModel):
    encounter_type: Literal[EncounterType.OUTPATIENT, EncounterType.PREVENTIVE_MEDICINE, EncounterType.NOT_SUPPORTED]
    justification: Optional[str] = Field(None, description="The justification for the chosen encounter type.  Please limit to 10 words")


class EncounterTypesDetailsAmbulatory(BaseModel):
    encounter_types: List[EncounterTypeAmbulatoryDetails]


class EncounterTypeEDDetails(BaseModel):
    encounter_type: Literal[EncounterType.EMERGENCY_DEPARTMENT, EncounterType.EMERGENCY_OTHER_SERVICES, EncounterType.CRITICAL_CARE]
    justification: Optional[str] = Field(None, description="The justification for the chosen encounter type.  Please limit to 10 words")


class EncounterTypesDetailsED(BaseModel):
    encounter_types: List[EncounterTypeEDDetails]


class EncounterTypeInpatientDetails(BaseModel):
    encounter_type: Literal[
        EncounterType.INPATIENT_INITIAL,
        EncounterType.INPATIENT_SUBSEQUENT,
        EncounterType.INPATIENT_SAME_DAY,
        EncounterType.INPATIENT_DISCHARGE,
        EncounterType.CRITICAL_CARE,
        EncounterType.NEWBORN_INITIAL,
        EncounterType.NEWBORN_SUBSEQUENT,
        EncounterType.NEWBORN_SAME_DAY,
    ]
    justification: Optional[str] = Field(None, description="The justification for the chosen encounter type.  Please limit to 10 words")


class EncounterTypesDetailsInpatient(BaseModel):
    encounter_types: List[EncounterTypeInpatientDetails]


VALID_CPT_CODES = list(chain.from_iterable(CPT_CODES_MAPPING.values()))
VALID_ENCOUNTER_TYPES = list(CPT_CODES_MAPPING.keys())


class CptEmCodeOld(BaseModel):
    code: str = Field(..., description="The cpt code")
    modifier: Optional[str] = Field(None, description="The modifier of the cpt code if any")
    justification: Optional[str] = Field("", description="The justification for the selected code")
    type: EncounterType = Field(None, description="The encounter type")


class CptEmCode(EMItem):
    pass


class Problem(Problem):
    pass


class LabTestStatus(StringEnum):
    REVIEWED = "Reviewed"
    ORDERED = "Ordered"


class LabTest(BaseModel):
    name: str = Field(..., description="The name of the lab test")
    status: LabTestStatus = Field(..., description="The status of the test whether reviewed or ordered.")


class HistoryDetail(BaseModel):
    involved: bool = Field(..., description="Indicates if the element was involved (True/False).")
    details: Optional[str] = Field(None, description="Explanation or additional details when the element is involved.")


class OrderDisplay(OrderDisplay):
    pass


class Order(Order):
    pass


class InputSchema(InputSchema):
    """Alias of the shared InputSchema for Project EM code."""

    pass


class OutputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    em: List[EMItem] = Field(None, description="The EM item details")
    problems: Optional[List[Problem]] = Field(None, description="The problems the encounter is about.")
    unique_sources_reviewed: Optional[List[str]] = Field(None, description="The unique sources reviewed")
    unique_tests: Optional[List[LabTest]] = Field(None, description="The unique tests")
    independent_historians: Optional[HistoryDetail] = Field(None, description="The independent historian involvement")
    independent_interpretation: Optional[HistoryDetail] = Field(None, description="The independent interpretation")
    external_discussion: Optional[HistoryDetail] = Field(None, description="The external discussion")
    additional_output: Optional[Dict[Any, Any]] = Field(default_factory=dict, description="Additional output fields")

    @model_validator(mode="before")
    def move_unknown_fields_to_additional(cls, values):
        if not isinstance(values, dict):
            return values

        known_fields = set(cls.model_fields.keys())
        unknown_fields = {k: v for k, v in values.items() if k not in known_fields}

        # Remove unknown fields from the main dict
        for k in unknown_fields:
            values.pop(k)

        # Add them to additional_output
        if unknown_fields:
            values["additional_output"] = values.get("additional_output", {}) | unknown_fields

        return values

    # # TODO REMOVE ONCE WE FIX THE METRICS TO HANDLE LIST OF CPT CODES
    # @model_validator(mode="before")
    # def singular_cpt(self):
    #     if isinstance(self.get("cpt"), list):
    #         if len(self.get("cpt")) > 1:
    #             self["cpt"] = [elem for elem in self.get("cpt") if elem.get("code").startswith("993")][0]
    #         else:
    #             self["cpt"] = self.get("cpt")[0]
    #     return self

    @model_validator(mode="before")
    def _coerce_em_items(cls, v: Dict[str, Any]):
        if not isinstance(v, dict):
            return v

        em_val = v.get("em")
        if em_val is None:
            return v

        # List cases
        if isinstance(em_val, list):
            fixed: List[Dict[str, Any]] = []
            # case: list of (k,v) pairs representing ONE EM
            if em_val and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in em_val):
                v["em"] = [dict(em_val)]
                return v

            for item in em_val:
                if isinstance(item, dict):
                    encounter = item.get("encounter_type")
                    if isinstance(encounter, dict) and encounter.get("encounter_type") == "Inpatient":
                        encounter["encounter_type"] = EncounterType.INPATIENT_INITIAL.value
                    fixed.append(item)
                elif (
                    isinstance(item, (list, tuple))
                    and item
                    and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in item)
                ):
                    item_dict = dict(item)
                    encounter = item_dict.get("encounter_type")
                    if isinstance(encounter, dict) and encounter.get("encounter_type") == "Inpatient":
                        encounter["encounter_type"] = EncounterType.INPATIENT_INITIAL.value
                    fixed.append(item_dict)
                elif hasattr(item, "model_dump"):
                    fixed.append(item.model_dump())  # pydantic v2 model instance
                elif hasattr(item, "__dict__"):
                    fixed.append({k: getattr(item, k) for k in vars(item).keys() if not k.startswith("_")})
                else:
                    raise ValueError(f"Unsupported EM item shape: {type(item)}")
            v["em"] = fixed
            return v

        # Anything else → leave; pydantic will error (by design)
        return v

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

    output = []

    # Add non-medication orders first
    if non_med_orders:
        output.append("Non-Medication Orders:")
        for order in non_med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                if order.display.clinical_display_line:
                    output.append(
                        f"- {order.catalog_type}: {order.display.hna_mnemonic} | {order.display.clinical_display_line} (order status: {order.status})"
                    )
                else:
                    output.append(f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status})")
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add medication orders
    if med_orders:
        if output:
            output.append("")
        output.append("Medication Orders:")
        for order in med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                if order.display.clinical_display_line:
                    output.append(
                        f"- {order.catalog_type}: {order.display.hna_mnemonic} | {order.display.clinical_display_line} (order status: {order.status})"
                    )
                else:
                    output.append(f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status})")
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add other orders
    if other_orders:
        if output:
            output.append("")
        output.append("Other Orders:")
        for order in other_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                if order.display.clinical_display_line:
                    output.append(
                        f"- {order.catalog_type}: {order.display.hna_mnemonic} | {order.display.clinical_display_line} (order status: {order.status})"
                    )
                else:
                    output.append(f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status})")
                seen_mnemonics.add(order.display.hna_mnemonic)

    return "\n".join(output)
