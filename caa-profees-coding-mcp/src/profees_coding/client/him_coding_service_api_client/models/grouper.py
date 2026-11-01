from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.status import Status
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.codeable_concept import CodeableConcept
    from ..models.reference import Reference
    from ..models.sub_type import SubType


T = TypeVar("T", bound="Grouper")


@_attrs_define
class Grouper:
    """The grouper code details.

    Attributes:
        type_ (str | Unset): Classification tag. Common values include DIAGNOSIS, EM, PROCEDURE, PREVENTIVE. For
            program-specific classifications, this may also be "G-code" (e.g., under diagnosis) or "Z-code" (e.g., under
            preventive), depending on workflow needs.
        code (str | Unset):
        description (str | Unset):
        system (str | Unset):
        justification (str | Unset):
        status (Status | Unset):
        sub_type (SubType | Unset): Identifier for proposed or accepted code.
        sources (list[Reference] | Unset):
        coding_type (CodeableConcept | Unset): Concept - reference to a terminology or just text.
        risk_of_mortality (CodeableConcept | Unset): Concept - reference to a terminology or just text.
        severity_of_illness (CodeableConcept | Unset): Concept - reference to a terminology or just text.
        major_diagnostic_category (CodeableConcept | Unset): Concept - reference to a terminology or just text.
        amount (float | Unset):
        weight (float | Unset):
        low_trim (float | Unset):
        high_trim (float | Unset):
        length_of_stay_arithmetic_mean (float | Unset):
        length_of_stay_geometric_mean (float | Unset):
        hospital_acquired_condition_indicator (float | Unset):
    """

    type_: str | Unset = UNSET
    code: str | Unset = UNSET
    description: str | Unset = UNSET
    system: str | Unset = UNSET
    justification: str | Unset = UNSET
    status: Status | Unset = UNSET
    sub_type: SubType | Unset = UNSET
    sources: list[Reference] | Unset = UNSET
    coding_type: CodeableConcept | Unset = UNSET
    risk_of_mortality: CodeableConcept | Unset = UNSET
    severity_of_illness: CodeableConcept | Unset = UNSET
    major_diagnostic_category: CodeableConcept | Unset = UNSET
    amount: float | Unset = UNSET
    weight: float | Unset = UNSET
    low_trim: float | Unset = UNSET
    high_trim: float | Unset = UNSET
    length_of_stay_arithmetic_mean: float | Unset = UNSET
    length_of_stay_geometric_mean: float | Unset = UNSET
    hospital_acquired_condition_indicator: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        code = self.code

        description = self.description

        system = self.system

        justification = self.justification

        status: str | Unset = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        sub_type: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sub_type, Unset):
            sub_type = self.sub_type.to_dict()

        sources: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.sources, Unset):
            sources = []
            for sources_item_data in self.sources:
                sources_item = sources_item_data.to_dict()
                sources.append(sources_item)

        coding_type: dict[str, Any] | Unset = UNSET
        if not isinstance(self.coding_type, Unset):
            coding_type = self.coding_type.to_dict()

        risk_of_mortality: dict[str, Any] | Unset = UNSET
        if not isinstance(self.risk_of_mortality, Unset):
            risk_of_mortality = self.risk_of_mortality.to_dict()

        severity_of_illness: dict[str, Any] | Unset = UNSET
        if not isinstance(self.severity_of_illness, Unset):
            severity_of_illness = self.severity_of_illness.to_dict()

        major_diagnostic_category: dict[str, Any] | Unset = UNSET
        if not isinstance(self.major_diagnostic_category, Unset):
            major_diagnostic_category = self.major_diagnostic_category.to_dict()

        amount = self.amount

        weight = self.weight

        low_trim = self.low_trim

        high_trim = self.high_trim

        length_of_stay_arithmetic_mean = self.length_of_stay_arithmetic_mean

        length_of_stay_geometric_mean = self.length_of_stay_geometric_mean

        hospital_acquired_condition_indicator = self.hospital_acquired_condition_indicator

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if code is not UNSET:
            field_dict["code"] = code
        if description is not UNSET:
            field_dict["description"] = description
        if system is not UNSET:
            field_dict["system"] = system
        if justification is not UNSET:
            field_dict["justification"] = justification
        if status is not UNSET:
            field_dict["status"] = status
        if sub_type is not UNSET:
            field_dict["sub_type"] = sub_type
        if sources is not UNSET:
            field_dict["sources"] = sources
        if coding_type is not UNSET:
            field_dict["coding_type"] = coding_type
        if risk_of_mortality is not UNSET:
            field_dict["risk_of_mortality"] = risk_of_mortality
        if severity_of_illness is not UNSET:
            field_dict["severity_of_illness"] = severity_of_illness
        if major_diagnostic_category is not UNSET:
            field_dict["major_diagnostic_category"] = major_diagnostic_category
        if amount is not UNSET:
            field_dict["amount"] = amount
        if weight is not UNSET:
            field_dict["weight"] = weight
        if low_trim is not UNSET:
            field_dict["low_trim"] = low_trim
        if high_trim is not UNSET:
            field_dict["high_trim"] = high_trim
        if length_of_stay_arithmetic_mean is not UNSET:
            field_dict["length_of_stay_arithmetic_mean"] = length_of_stay_arithmetic_mean
        if length_of_stay_geometric_mean is not UNSET:
            field_dict["length_of_stay_geometric_mean"] = length_of_stay_geometric_mean
        if hospital_acquired_condition_indicator is not UNSET:
            field_dict["hospital_acquired_condition_indicator"] = hospital_acquired_condition_indicator

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.codeable_concept import CodeableConcept
        from ..models.reference import Reference
        from ..models.sub_type import SubType

        d = dict(src_dict)
        type_ = d.pop("type", UNSET)

        code = d.pop("code", UNSET)

        description = d.pop("description", UNSET)

        system = d.pop("system", UNSET)

        justification = d.pop("justification", UNSET)

        _status = d.pop("status", UNSET)
        status: Status | Unset
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = Status(_status)

        _sub_type = d.pop("sub_type", UNSET)
        sub_type: SubType | Unset
        if isinstance(_sub_type, Unset):
            sub_type = UNSET
        else:
            sub_type = SubType.from_dict(_sub_type)

        _sources = d.pop("sources", UNSET)
        sources: list[Reference] | Unset = UNSET
        if _sources is not UNSET:
            sources = []
            for sources_item_data in _sources:
                sources_item = Reference.from_dict(sources_item_data)

                sources.append(sources_item)

        _coding_type = d.pop("coding_type", UNSET)
        coding_type: CodeableConcept | Unset
        if isinstance(_coding_type, Unset):
            coding_type = UNSET
        else:
            coding_type = CodeableConcept.from_dict(_coding_type)

        _risk_of_mortality = d.pop("risk_of_mortality", UNSET)
        risk_of_mortality: CodeableConcept | Unset
        if isinstance(_risk_of_mortality, Unset):
            risk_of_mortality = UNSET
        else:
            risk_of_mortality = CodeableConcept.from_dict(_risk_of_mortality)

        _severity_of_illness = d.pop("severity_of_illness", UNSET)
        severity_of_illness: CodeableConcept | Unset
        if isinstance(_severity_of_illness, Unset):
            severity_of_illness = UNSET
        else:
            severity_of_illness = CodeableConcept.from_dict(_severity_of_illness)

        _major_diagnostic_category = d.pop("major_diagnostic_category", UNSET)
        major_diagnostic_category: CodeableConcept | Unset
        if isinstance(_major_diagnostic_category, Unset):
            major_diagnostic_category = UNSET
        else:
            major_diagnostic_category = CodeableConcept.from_dict(_major_diagnostic_category)

        amount = d.pop("amount", UNSET)

        weight = d.pop("weight", UNSET)

        low_trim = d.pop("low_trim", UNSET)

        high_trim = d.pop("high_trim", UNSET)

        length_of_stay_arithmetic_mean = d.pop("length_of_stay_arithmetic_mean", UNSET)

        length_of_stay_geometric_mean = d.pop("length_of_stay_geometric_mean", UNSET)

        hospital_acquired_condition_indicator = d.pop("hospital_acquired_condition_indicator", UNSET)

        grouper = cls(
            type_=type_,
            code=code,
            description=description,
            system=system,
            justification=justification,
            status=status,
            sub_type=sub_type,
            sources=sources,
            coding_type=coding_type,
            risk_of_mortality=risk_of_mortality,
            severity_of_illness=severity_of_illness,
            major_diagnostic_category=major_diagnostic_category,
            amount=amount,
            weight=weight,
            low_trim=low_trim,
            high_trim=high_trim,
            length_of_stay_arithmetic_mean=length_of_stay_arithmetic_mean,
            length_of_stay_geometric_mean=length_of_stay_geometric_mean,
            hospital_acquired_condition_indicator=hospital_acquired_condition_indicator,
        )

        grouper.additional_properties = d
        return grouper

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
