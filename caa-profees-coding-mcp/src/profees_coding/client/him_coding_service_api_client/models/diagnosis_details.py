from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DiagnosisDetails")


@_attrs_define
class DiagnosisDetails:
    """Diagnosis detailed information

    Attributes:
        sequence (float | Unset):
        present_on_admission (str | Unset):
        type_ (str | Unset):
    """

    sequence: float | Unset = UNSET
    present_on_admission: str | Unset = UNSET
    type_: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sequence = self.sequence

        present_on_admission = self.present_on_admission

        type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if sequence is not UNSET:
            field_dict["sequence"] = sequence
        if present_on_admission is not UNSET:
            field_dict["presentOnAdmission"] = present_on_admission
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        sequence = d.pop("sequence", UNSET)

        present_on_admission = d.pop("presentOnAdmission", UNSET)

        type_ = d.pop("type", UNSET)

        diagnosis_details = cls(
            sequence=sequence,
            present_on_admission=present_on_admission,
            type_=type_,
        )

        diagnosis_details.additional_properties = d
        return diagnosis_details

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
