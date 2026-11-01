from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EmEncounterType")


@_attrs_define
class EmEncounterType:
    """
    Attributes:
        encounter_type (str | Unset):  Example: Office or Other Outpatient Service.
        justification (str | Unset):  Example: Outpatient E/M visit determined by MDM..
    """

    encounter_type: str | Unset = UNSET
    justification: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        encounter_type = self.encounter_type

        justification = self.justification

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if encounter_type is not UNSET:
            field_dict["encounter_type"] = encounter_type
        if justification is not UNSET:
            field_dict["justification"] = justification

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        encounter_type = d.pop("encounter_type", UNSET)

        justification = d.pop("justification", UNSET)

        em_encounter_type = cls(
            encounter_type=encounter_type,
            justification=justification,
        )

        em_encounter_type.additional_properties = d
        return em_encounter_type

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
