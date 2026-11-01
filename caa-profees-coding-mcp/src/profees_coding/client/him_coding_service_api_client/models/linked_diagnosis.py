from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="LinkedDiagnosis")


@_attrs_define
class LinkedDiagnosis:
    """Diagnosis linked to a procedure or E/M code.

    Attributes:
        code (str | Unset):
        system (str | Unset):
        description (str | Unset):
        justification (str | Unset):
        is_primary (bool | Unset):
    """

    code: str | Unset = UNSET
    system: str | Unset = UNSET
    description: str | Unset = UNSET
    justification: str | Unset = UNSET
    is_primary: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        code = self.code

        system = self.system

        description = self.description

        justification = self.justification

        is_primary = self.is_primary

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if code is not UNSET:
            field_dict["code"] = code
        if system is not UNSET:
            field_dict["system"] = system
        if description is not UNSET:
            field_dict["description"] = description
        if justification is not UNSET:
            field_dict["justification"] = justification
        if is_primary is not UNSET:
            field_dict["is_primary"] = is_primary

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        code = d.pop("code", UNSET)

        system = d.pop("system", UNSET)

        description = d.pop("description", UNSET)

        justification = d.pop("justification", UNSET)

        is_primary = d.pop("is_primary", UNSET)

        linked_diagnosis = cls(
            code=code,
            system=system,
            description=description,
            justification=justification,
            is_primary=is_primary,
        )

        linked_diagnosis.additional_properties = d
        return linked_diagnosis

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
